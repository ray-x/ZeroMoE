// Qwen4Exp full graph: Graph building blocks: hyper-connection mix/combine, shared expert, GDN,
// full attention (prefill / decode), PLE embedding.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_qwen4exp_graph_internal.hpp.

#include "ggml_qwen4exp_graph_internal.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

using namespace qwen4exp_graph;

namespace qwen4exp_graph {

ggml_tensor *hc_mix(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                    ggml_tensor *norm, ggml_tensor *down, ggml_tensor *up,
                    ggml_tensor *inject, ggml_tensor **inject_out) {
  ggml_tensor *xn =
      ggml_rms_norm(ctx, x, g.engine->model_file.architecture_config.eps);
  xn = ggml_reshape_2d(ctx, xn, g.D * g.HC, g.S);
  xn = ggml_mul(ctx, xn, norm);
  ggml_tensor *lo = ggml_mul_mat(ctx, down, xn);
  lo = ggml_silu(ctx, ggml_scale(ctx, lo, 1.0f / static_cast<float>(g.HC)));
  ggml_tensor *gate_logits = ggml_mul_mat(ctx, up, lo);
  ggml_tensor *gate = ggml_sigmoid(ctx, gate_logits);
  ggml_tensor *gated = ggml_mul(ctx, xn, gate);
  ggml_tensor *g3 = ggml_reshape_3d(ctx, gated, g.D, g.HC, g.S);

  ggml_tensor *mixed =
      ggml_cont(ctx, ggml_view_2d(ctx, g3, g.D, g.S,
                                  ggml_row_size(g3->type, g.D) * g.HC, 0));
  for (int h = 1; h < g.HC; ++h) {
    ggml_tensor *s =
        ggml_view_2d(ctx, g3, g.D, g.S, ggml_row_size(g3->type, g.D) * g.HC,
                     ggml_row_size(g3->type, g.D) * h);
    mixed = ggml_add(ctx, mixed, s);
  }
  mixed = ggml_scale(ctx, mixed, 1.0f / static_cast<float>(g.HC));
  if (inject_out) {
    *inject_out = inject ? ggml_mul_mat(ctx, inject, xn) : nullptr;
  }
  return mixed;
}

ggml_tensor *hc_combine(FullGraph &g, ggml_context *ctx, ggml_tensor *residual,
                        ggml_tensor *block, ggml_tensor *inject) {
  ggml_tensor *w = ggml_scale(
      ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / g.HC)), 2.0f);
  w = ggml_reshape_3d(ctx, w, 1, g.HC, g.S);
  ggml_tensor *b = ggml_reshape_3d(ctx, block, g.D, 1, g.S);
  b = ggml_repeat_4d(ctx, b, g.D, g.HC, g.S, 1);
  return ggml_add(ctx, residual, ggml_mul(ctx, b, w));
}

ggml_tensor *build_shared(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                          int layer, ggml_tensor *gate, ggml_tensor *up,
                          ggml_tensor *down, ggml_tensor *scalar_gate) {
  if (!gate || !up || !down)
    return nullptr;
  ggml_tensor *gate_out = ggml_mul_mat(ctx, gate, cur);
  ggml_tensor *up_out = ggml_mul_mat(ctx, up, cur);
  ggml_tensor *act = ggml_swiglu_split(ctx, gate_out, up_out);
  ggml_tensor *out = ggml_mul_mat(ctx, down, act);
  if (scalar_gate) {
    ggml_tensor *sg = ggml_sigmoid(ctx, ggml_mul_mat(ctx, scalar_gate, cur));
    out = ggml_mul(ctx, out, sg);
  }
  (void)layer;
  return out;
}

// ggml-cuda's gated_delta_net kernel reads AND writes its per-head recurrent
// state in one fixed physical layout: M[col][i] = S[i][col] (col=v slower,
// i=k faster; see ggml-cuda/gated_delta_net.cu's own comment). This is the
// only layout the op ever accepts as input, so any device-resident state
// that round-trips straight back into the op (the full-graph decode
// self-loop) must stay in this "native" layout untouched.
//
// This codebase's own legacy CPU step (qwen4exp_gdn_delta_step) uses the
// opposite convention, st[k*hd+v] (k slower, v faster) -- i.e. exactly the
// per-head TRANSPOSE of the native layout. Converting between the two is a
// per-head square-matrix transpose, which is its own inverse, so the same
// helper handles both directions: legacy-in-for-native-consumption and
// native-out-for-legacy-consumption.
void gdn_state_transpose_host(float *dst, const float *src, int hd, int Hv) {
  for (int h = 0; h < Hv; ++h) {
    const float *s = src + static_cast<size_t>(h) * hd * hd;
    float *d = dst + static_cast<size_t>(h) * hd * hd;
    for (int a = 0; a < hd; ++a)
      for (int b = 0; b < hd; ++b)
        d[a * hd + b] = s[b * hd + a];
  }
}

ggml_tensor *build_gdn(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *w_qkv, ggml_tensor *w_gate,
                       ggml_tensor *w_alpha, ggml_tensor *w_beta,
                       ggml_tensor *w_dt, ggml_tensor *w_a, ggml_tensor *w_conv,
                       ggml_tensor *w_norm, ggml_tensor *w_out,
                       ggml_tensor *hist_in, ggml_tensor *state_in,
                       ggml_tensor **state_out, ggml_tensor **hist_out) {
  ggml_tensor *qkv = ggml_mul_mat(ctx, w_qkv, cur);
  auto diag = [&](const char *label, ggml_tensor *value) {
  };
  // These snapshots are deliberately taken before and inside the GDN node.
  // The existing attention-output dump only tells us that the discrepancy is
  // present after the recurrent block; these boundaries separate qkv GEMM,
  // convolution, GDN/state update, and the final output projection.
  diag("gdn_qkv", qkv);
  ggml_tensor *z = ggml_mul_mat(ctx, w_gate, cur);
  ggml_tensor *alpha = ggml_mul_mat(ctx, w_alpha, cur);
  ggml_tensor *beta_raw = ggml_mul_mat(ctx, w_beta, cur);
  ggml_tensor *glog =
      ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, w_dt)), w_a);
  ggml_tensor *beta = ggml_sigmoid(ctx, beta_raw);

  ggml_tensor *conv_in = ggml_concat(ctx, hist_in, ggml_transpose(ctx, qkv), 0);
  diag("gdn_conv_in", conv_in);
  ggml_tensor *conv = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_in, w_conv));
  diag("gdn_conv", conv);
  const size_t row = ggml_row_size(GGML_TYPE_F32, g.conv_dim);
  ggml_tensor *q =
      ggml_view_4d(ctx, conv, g.gdn_hd, g.Hk_gdn, g.S, 1,
                   ggml_row_size(GGML_TYPE_F32, g.gdn_hd), row, row * g.S, 0);
  ggml_tensor *k =
      ggml_view_4d(ctx, conv, g.gdn_hd, g.Hk_gdn, g.S, 1,
                   ggml_row_size(GGML_TYPE_F32, g.gdn_hd), row, row * g.S,
                   ggml_row_size(GGML_TYPE_F32, g.key_dim));
  ggml_tensor *v =
      ggml_view_4d(ctx, conv, g.gdn_hd, g.Hv_gdn, g.S, 1,
                   ggml_row_size(GGML_TYPE_F32, g.gdn_hd), row, row * g.S,
                   ggml_row_size(GGML_TYPE_F32, 2 * g.key_dim));
  q = ggml_l2_norm(ctx, q, g.engine->model_file.architecture_config.eps);
  k = ggml_l2_norm(ctx, k, g.engine->model_file.architecture_config.eps);
  glog = ggml_reshape_4d(ctx, glog, 1, g.Hv_gdn, g.S, 1);
  beta = ggml_reshape_4d(ctx, beta, 1, g.Hv_gdn, g.S, 1);
  ggml_tensor *gdn =
      ggml_gated_delta_net(ctx, q, k, v, glog, beta, state_in, 1);
  diag("gdn_output", gdn);

  const size_t hd_row = ggml_row_size(GGML_TYPE_F32, g.gdn_hd);
  const size_t value_row = ggml_row_size(GGML_TYPE_F32, g.value_dim);
  const size_t state_row = ggml_row_size(GGML_TYPE_F32, g.gdn_hd * g.gdn_hd);
  ggml_tensor *core = ggml_view_4d(ctx, gdn, g.gdn_hd, g.Hv_gdn, g.S, 1, hd_row,
                                   value_row, value_row * g.S, 0);
  diag("gdn_core", core);
  // *state_out is kept in the op's own NATIVE layout (ne[0]=k, ne[1]=v; see
  // the gdn_state_transpose_host comment above) because the full-graph
  // decode self-loop feeds this same tensor straight back into
  // ggml_gated_delta_net as next-token's state_in with no host round trip.
  // Any host-facing consumer (legacy CPU decode, cross-request state
  // persistence) must convert with gdn_state_transpose_host at the host
  // boundary instead of receiving a transposed tensor from here -- a
  // transpose baked in here only benefits the host path and silently
  // breaks the device-resident self-loop instead (this was the actual bug:
  // wrong values from the second decode token onward).
  ggml_tensor *state_raw =
      ggml_view_4d(ctx, gdn, g.gdn_hd, g.gdn_hd, g.Hv_gdn, 1, hd_row, state_row,
                   state_row * g.Hv_gdn,
                   static_cast<size_t>(g.value_dim) * g.S * sizeof(float));
  *state_out = ggml_cont(ctx, state_raw);
  ggml_tensor *norm = ggml_mul(
      ctx,
      ggml_rms_norm(ctx, core, g.engine->model_file.architecture_config.eps),
      w_norm);
  diag("gdn_norm", norm);
  ggml_tensor *z4 = ggml_reshape_4d(ctx, z, g.gdn_hd, g.Hv_gdn, g.S, 1);
  ggml_tensor *z_gate = ggml_sigmoid(ctx, z4);
  ggml_tensor *gated = ggml_mul(ctx, norm, z_gate);
  diag("gdn_gated", gated);
  ggml_tensor *out_input = ggml_reshape_2d(ctx, gated, g.value_dim, g.S);
  ggml_tensor *out = ggml_mul_mat(ctx, w_out, out_input);
  diag("gdn_output_projection", out);
  *hist_out =
      ggml_cont(ctx, ggml_view_3d(ctx, conv_in, g.d_conv - 1, g.conv_dim, 1,
                                  conv_in->nb[1], conv_in->nb[2],
                                  ggml_row_size(conv_in->type, g.S)));
  (void)layer;
  return out;
}

ggml_tensor *build_full_attention(FullGraph &g, ggml_context *ctx,
                                  ggml_tensor *cur, int layer, ggml_tensor *w_q,
                                  ggml_tensor *w_k, ggml_tensor *w_v,
                                  ggml_tensor *w_qn, ggml_tensor *w_kn,
                                  ggml_tensor *w_o, ggml_tensor **k_cache,
                                  ggml_tensor **v_cache) {
  ggml_tensor *qfull = ggml_mul_mat(ctx, w_q, cur);
  ggml_tensor *q =
      ggml_view_3d(ctx, qfull, g.hd, g.H, g.S, sizeof(float) * g.hd * 2,
                   sizeof(float) * g.hd * 2 * g.H, 0);
  q = ggml_mul(
      ctx, ggml_rms_norm(ctx, q, g.engine->model_file.architecture_config.eps),
      w_qn);
  ggml_tensor *k =
      ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w_k, cur), g.hd, g.Hkv, g.S);
  k = ggml_mul(
      ctx, ggml_rms_norm(ctx, k, g.engine->model_file.architecture_config.eps),
      w_kn);
  ggml_tensor *v =
      ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w_v, cur), g.hd, g.Hkv, g.S);
  const int n_rot = g.engine->model_file.architecture_config.rope_n_rot > 0
                        ? g.engine->model_file.architecture_config.rope_n_rot
                        : g.hd;
  // Qwen4Exp's upstream rope_type is IMROPE (interleaved M-RoPE), whose
  // cos/sin pairing is NEOX-ordered (rotate-half), not the adjacent-pair
  // (2j,2j+1) ordering GGML_ROPE_TYPE_NORMAL applies. For text-only decode
  // every M-RoPE section carries the same scalar position (no vision/video
  // tokens), so IMROPE's per-section theta selection is provably identical
  // to plain single-axis NEOX rope (ggml.h's own ggml_rope_multi comment:
  // "the theta for each dim is computed the same way as ggml_rope_ext, no
  // matter the section") -- ggml_rope_ext with GGML_ROPE_TYPE_NEOX is exact,
  // not an approximation, and needs no 4x-wide position tensor. Also use
  // this model's real rope.freq_base (10,000,000 for this checkpoint) --
  // plain ggml_rope()'s 5-arg wrapper silently hardcodes ggml's generic
  // 10,000 default, which is wrong for Qwen4Exp by 1000x.
  const float rope_freq_base =
      g.engine->model_file.get_config_float("rope.freq_base", 10000.0f);
  const auto &rope_c = g.engine->model_file.architecture_config;
  q = ggml_rope_ext(ctx, q, g.positions, nullptr, n_rot, GGML_ROPE_TYPE_NEOX,
                     rope_c.rope_n_ctx_orig, rope_freq_base,
                     rope_c.rope_freq_scale, rope_c.rope_ext_factor,
                     rope_c.rope_attn_factor, 32.0f, 1.0f);
  k = ggml_rope_ext(ctx, k, g.positions, nullptr, n_rot, GGML_ROPE_TYPE_NEOX,
                     rope_c.rope_n_ctx_orig, rope_freq_base,
                     rope_c.rope_freq_scale, rope_c.rope_ext_factor,
                     rope_c.rope_attn_factor, 32.0f, 1.0f);
  // The host dense KV cache stores the same normalized and rotated K/V values
  // that llama.cpp's graph writes to its cache.  Keep these F32 tensors before the
  // layout-only permute/cast used by flash attention.
  if (k_cache)
    *k_cache = k;
  if (v_cache)
    *v_cache = v;
  q = ggml_permute(ctx, q, 0, 2, 1, 3);
  k = ggml_permute(ctx, k, 0, 2, 1, 3);
  v = ggml_permute(ctx, v, 0, 2, 1, 3);
  // The CUDA MMA flash-attention implementation intentionally consumes Q as
  // F32 and converts/loads K and V as F16.  Casting Q to F16 selects the MMA
  // path and then trips its Q-type assertion during graph execution.
  if (k->type == GGML_TYPE_F32)
    k = ggml_cast(ctx, k, GGML_TYPE_F16);
  if (v->type == GGML_TYPE_F32)
    v = ggml_cast(ctx, v, GGML_TYPE_F16);
  ggml_tensor *attn = ggml_flash_attn_ext(
      ctx, q, k, v, g.attn_mask, 1.0f / std::sqrt(static_cast<float>(g.hd)),
      0.0f, 0.0f);
  ggml_tensor *gate =
      ggml_view_3d(ctx, qfull, g.hd, g.H, g.S, sizeof(float) * g.hd * 2,
                   sizeof(float) * g.hd * 2 * g.H, sizeof(float) * g.hd);
  gate = ggml_reshape_3d(ctx, ggml_cont_2d(ctx, gate, g.H * g.hd, g.S), g.hd,
                         g.H, g.S);
  attn = ggml_mul(ctx, attn, ggml_sigmoid(ctx, gate));
  attn = ggml_reshape_2d(ctx, attn, g.H * g.hd, g.S);
  (void)layer;
  return ggml_mul_mat(ctx, w_o, attn);
}

// Decode variant of build_full_attention().  K/V are appended directly to a
// device-resident cache with one set_rows node.  The graph therefore carries
// the cache update, attention, and output projection in the same scheduler
// submission; no host K/V readback is needed between tokens.
ggml_tensor *build_full_attention_decode(
    FullGraph &g, ggml_context *ctx, ggml_tensor *cur, int layer,
    ggml_tensor *w_q, ggml_tensor *w_k, ggml_tensor *w_v, ggml_tensor *w_qn,
    ggml_tensor *w_kn, ggml_tensor *w_o, ggml_tensor *k_storage,
    ggml_tensor *v_storage) {
  ggml_tensor *qfull = ggml_mul_mat(ctx, w_q, cur);
  ggml_tensor *q = ggml_view_3d(
      ctx, qfull, g.hd, g.H, 1, sizeof(float) * g.hd * 2,
      sizeof(float) * g.hd * 2 * g.H, 0);
  q = ggml_mul(
      ctx, ggml_rms_norm(ctx, q, g.engine->model_file.architecture_config.eps),
      w_qn);
  ggml_tensor *k = ggml_reshape_3d(
      ctx, ggml_mul_mat(ctx, w_k, cur), g.hd, g.Hkv, 1);
  k = ggml_mul(
      ctx, ggml_rms_norm(ctx, k, g.engine->model_file.architecture_config.eps),
      w_kn);
  ggml_tensor *v = ggml_reshape_3d(
      ctx, ggml_mul_mat(ctx, w_v, cur), g.hd, g.Hkv, 1);
  const int n_rot = g.engine->model_file.architecture_config.rope_n_rot > 0
                        ? g.engine->model_file.architecture_config.rope_n_rot
                        : g.hd;
  // See build_full_attention() above for why NEOX ordering + the model's
  // real rope.freq_base (not GGML_ROPE_TYPE_NORMAL + ggml's hardcoded
  // 10,000 default) is the correct, exact substitute for IMROPE here.
  const float rope_freq_base =
      g.engine->model_file.get_config_float("rope.freq_base", 10000.0f);
  const auto &rope_c = g.engine->model_file.architecture_config;
  q = ggml_rope_ext(ctx, q, g.positions, nullptr, n_rot, GGML_ROPE_TYPE_NEOX,
                     rope_c.rope_n_ctx_orig, rope_freq_base,
                     rope_c.rope_freq_scale, rope_c.rope_ext_factor,
                     rope_c.rope_attn_factor, 32.0f, 1.0f);
  k = ggml_rope_ext(ctx, k, g.positions, nullptr, n_rot, GGML_ROPE_TYPE_NEOX,
                     rope_c.rope_n_ctx_orig, rope_freq_base,
                     rope_c.rope_freq_scale, rope_c.rope_ext_factor,
                     rope_c.rope_attn_factor, 32.0f, 1.0f);

  ggml_tensor *k2 = ggml_reshape_2d(ctx, k, g.hd, g.Hkv);
  ggml_tensor *v2 = ggml_reshape_2d(ctx, v, g.hd, g.Hkv);
  ggml_tensor *kset = ggml_set_rows(ctx, k_storage, k2, g.decode_indices);
  ggml_tensor *vset = ggml_set_rows(ctx, v_storage, v2, g.decode_indices);
  ggml_tensor *kcache = ggml_reshape_4d(
      ctx, kset, g.hd, g.kv_capacity, g.Hkv, 1);
  ggml_tensor *vcache = ggml_reshape_4d(
      ctx, vset, g.hd, g.kv_capacity, g.Hkv, 1);

  q = ggml_permute(ctx, q, 0, 2, 1, 3);
  if (kcache->type == GGML_TYPE_F32)
    kcache = ggml_cast(ctx, kcache, GGML_TYPE_F16);
  if (vcache->type == GGML_TYPE_F32)
    vcache = ggml_cast(ctx, vcache, GGML_TYPE_F16);
  ggml_tensor *attn = ggml_flash_attn_ext(
      ctx, q, kcache, vcache, g.attn_mask,
      1.0f / std::sqrt(static_cast<float>(g.hd)), 0.0f, 0.0f);
  ggml_tensor *gate = ggml_view_3d(
      ctx, qfull, g.hd, g.H, 1, sizeof(float) * g.hd * 2,
      sizeof(float) * g.hd * 2 * g.H, sizeof(float) * g.hd);
  gate = ggml_reshape_3d(ctx, ggml_cont_2d(ctx, gate, g.H * g.hd, 1), g.hd,
                         g.H, 1);
  attn = ggml_mul(ctx, attn, ggml_sigmoid(ctx, gate));
  attn = ggml_reshape_2d(ctx, attn, g.H * g.hd, 1);
  (void)layer;
  return ggml_mul_mat(ctx, w_o, attn);
}

bool prepare_ple_embedding(FullGraph &g, std::vector<float> &embedding, int S,
                           int pos_base, const int *token_ids) {
  const auto &c = g.engine->model_file.architecture_config;
  if (!has_ple_layer(c) || !token_ids ||
      (pos_base != 0 && !(g.decode_mode && S == 1)))
    return !has_ple_layer(c);
  const GGUFTensor *table =
      model_tensor(*g.engine, "per_layer_token_embd.weight");
  if (!table || table->shape.size() != 2)
    return false;
  const int row_dim = static_cast<int>(table->shape[0]);
  const int rb = row_bytes(table->type, row_dim);
  if (rb <= 0 || row_dim != c.ple_n_embd_per_layer)
    return false;

  std::vector<int32_t> rows(static_cast<size_t>(S) * c.ple_n_heads);
  const int64_t eos = c.ple_eos_token_id;
  for (int i = 0; i < S; ++i) {
    int64_t ctx[8] = {};
    if (c.ple_ngram_size > 8)
      return false;
    ctx[0] = token_ids[i];
    bool cut = false;
    for (int s = 1; s < c.ple_ngram_size; ++s) {
      if (g.decode_mode) {
        const int64_t pos = pos_base + i;
        const int64_t back = s - i;
        const auto &history = g.engine->qwen4exp_ple_tok_hist;
        if (g.engine->qwen4exp_ple_next_pos == pos && back > 0 &&
            back <= static_cast<int64_t>(history.size())) {
          const size_t index = history.size() - static_cast<size_t>(back);
          ctx[s] = history[index];
        } else {
          ctx[s] = eos;
        }
      } else {
        const int j = i - s;
        ctx[s] = j >= 0 ? token_ids[j] : eos;
      }
      if (cut || ctx[s] == eos) {
        ctx[s] = eos;
        cut = true;
      }
    }
    for (int n = 2; n <= c.ple_ngram_size; ++n) {
      int base = (n - 2) * c.ple_heads_per_ngram;
      for (int h = 0; h < c.ple_heads_per_ngram; ++h) {
        int hi = base + h;
        rows[static_cast<size_t>(i) * c.ple_n_heads + hi] =
            static_cast<int32_t>(
                ple_ngram_row(ctx, n, c.ple_layer_multipliers.data(),
                              c.ple_head_vocab_sizes[static_cast<size_t>(hi)],
                              c.ple_head_offsets[static_cast<size_t>(hi)]));
      }
    }
  }

  std::unordered_map<int32_t, std::vector<float>> cache;
  void *raw_mem = nullptr;
  if (posix_memalign(&raw_mem, 4096, static_cast<size_t>(rb)) != 0 ||
      !raw_mem)
    return false;
  std::unique_ptr<uint8_t, decltype(&std::free)> raw_guard(
      static_cast<uint8_t *>(raw_mem), &std::free);
  auto *raw = raw_guard.get();
  const int fd = g.engine->model_file.fd_for_tensor(*table);
  for (int32_t r : rows) {
    if (cache.count(r))
      continue;
    const ssize_t got = direct_io::pread_direct_aware(
        fd, g.engine->model_file.direct_io_active_for(*table),
        g.engine->model_file.direct_io_alignment, raw, static_cast<size_t>(rb),
        table->offset + static_cast<uint64_t>(r) * static_cast<size_t>(rb));
    if (got != static_cast<ssize_t>(rb))
      return false;
    std::vector<float> row(static_cast<size_t>(row_dim));
    if (!dequant_row(table->type, raw, row.data(), row_dim))
      return false;
    cache.emplace(r, std::move(row));
  }
  embedding.resize(static_cast<size_t>(S) * c.ple_n_heads * row_dim);
  for (int i = 0; i < S; ++i) {
    for (int h = 0; h < c.ple_n_heads; ++h) {
      const auto &row =
          cache.at(rows[static_cast<size_t>(i) * c.ple_n_heads + h]);
      std::memcpy(embedding.data() +
                      (static_cast<size_t>(i) * c.ple_n_heads + h) * row_dim,
                  row.data(), static_cast<size_t>(row_dim) * sizeof(float));
    }
  }
  return true;
}

} // namespace qwen4exp_graph

#endif // MOESTREAM_USE_GGML_CUDA
