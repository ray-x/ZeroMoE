// GLM5Next full graph: Graph building blocks: hyper-connection ops, KDA, DSA indexer, DSA
// attention.
//
// Split out of the former single-file ggml_glm5next_graph.cpp; shared
// declarations live in ggml_glm5next_graph_impl.hpp.

#include "ggml_glm5next_graph_impl.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

using namespace glm5next_graph;

namespace glm5next_graph {

ggml_tensor *hc_affine(ggml_context *ctx, ggml_tensor *x, ggml_tensor *scale,
                       ggml_tensor *base) {
  return ggml_add(ctx, ggml_mul(ctx, x, scale), base);
}

ggml_tensor *hc_mean(FullGraph &g, ggml_context *ctx, ggml_tensor *x) {
  ggml_tensor *out = ggml_view_2d(ctx, x, g.D, g.S, x->nb[2], 0);
  for (int h = 1; h < g.HC; ++h)
    out = ggml_add(
        ctx, out,
        ggml_view_2d(ctx, x, g.D, g.S, x->nb[2], (size_t)h * x->nb[1]));
  return ggml_scale(ctx, out, 1.0f / (float)g.HC);
}

ggml_tensor *hc_contig(ggml_context *ctx, ggml_tensor *t) {
  return t && !ggml_is_contiguous(t) ? ggml_cont(ctx, t) : t;
}

// MOESTREAM_GLM5NEXT_FUSED_HC: unset/"1" = all fused ops; otherwise any of
// the letters c (hc_comb), p (hc_pre sum), o (hc_post) enables just those.
bool fused_hc_op(char which) {
  // Default ON. MOESTREAM_GLM5NEXT_UNFUSED_HC=1 restores the composed-op path.
  static const bool off = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_UNFUSED_HC") != nullptr;
  if (off)
    return false;
  static const char *e = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_FUSED_HC");
  if (!e || e[0] == '1')
    return true;
  return std::strchr(e, which) != nullptr;
}

ggml_tensor *hc_pre(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                    ggml_tensor *fn, ggml_tensor *scale, ggml_tensor *base,
                    ggml_tensor **post, ggml_tensor **comb,
                    ggml_tensor **pre_gate, ggml_tensor *collapse_carry) {
  const auto &c = g.engine->model_file.architecture_config;
  const int mix_dim = (2 + g.HC) * g.HC;
  ggml_tensor *flat = ggml_reshape_2d(ctx, x, g.D * g.HC, g.S);
  ggml_tensor *flat_norm = ggml_rms_norm(ctx, flat, c.eps);
  ggml_tensor *mixes = ggml_mul_mat(ctx, fn, flat_norm);
  if (g.engine->family_ops.family == ModelFamily::DeepSeek_V41 &&
      moestream::graphcfg::flag_on("MOESTREAM_DEEPSEEK_V41_NATIVE_F32_GRAPH"))
    ggml_prec_set_src(mixes, GGML_PREC_F32, 1);

  ggml_tensor *scale_pre = ggml_view_1d(ctx, scale, 1, 0);
  ggml_tensor *scale_post = ggml_view_1d(ctx, scale, 1, sizeof(float));
  ggml_tensor *scale_comb = ggml_view_1d(ctx, scale, 1, 2 * sizeof(float));
  ggml_tensor *base_pre = ggml_view_1d(ctx, base, g.HC, 0);
  ggml_tensor *base_post = ggml_view_1d(ctx, base, g.HC, g.HC * sizeof(float));
  ggml_tensor *base_comb =
      ggml_view_1d(ctx, base, g.HC * g.HC, 2 * g.HC * sizeof(float));

  ggml_tensor *pre = ggml_view_2d(ctx, mixes, g.HC, g.S, mixes->nb[1], 0);
  pre = ggml_scale_bias(
      ctx, ggml_sigmoid(ctx, hc_affine(ctx, pre, scale_pre, base_pre)), 1.0f,
      c.hc_eps);
  if (pre_gate)
    *pre_gate = pre;
  ggml_tensor *collapse = collapse_carry ? collapse_carry : pre;
  *post = ggml_view_2d(ctx, mixes, g.HC, g.S, mixes->nb[1],
                       (size_t)g.HC * mixes->nb[0]);
  *post = ggml_scale(
      ctx, ggml_sigmoid(ctx, hc_affine(ctx, *post, scale_post, base_post)),
      2.0f);
  const bool f_comb = fused_hc_op('c'), f_pre = fused_hc_op('p');
  if (f_comb) {
    *comb = ggml_dsv4_hc_comb(ctx, hc_contig(ctx, mixes), hc_contig(ctx, scale),
                              hc_contig(ctx, base), c.hc_eps,
                              (int32_t)c.hc_sinkhorn_iters);
  } else {
    *comb = ggml_view_2d(ctx, mixes, g.HC * g.HC, g.S, mixes->nb[1],
                         (size_t)2 * g.HC * mixes->nb[0]);
    *comb = hc_affine(ctx, *comb, scale_comb, base_comb);
    *comb = ggml_reshape_3d(ctx, *comb, g.HC, g.HC, g.S);
    ggml_tensor *eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx, eps, c.hc_eps);
    *comb = ggml_add(ctx, ggml_soft_max(ctx, *comb), eps);
    auto norm_cols = [&]() {
      ggml_tensor *p = ggml_cont(ctx, ggml_permute(ctx, *comb, 1, 0, 2, 3));
      ggml_tensor *sum = ggml_add(ctx, ggml_sum_rows(ctx, p), eps);
      sum = ggml_permute(ctx, sum, 1, 0, 2, 3);
      *comb = ggml_div(ctx, *comb, sum);
    };
    auto norm_rows = [&]() {
      ggml_tensor *sum = ggml_add(ctx, ggml_sum_rows(ctx, *comb), eps);
      *comb = ggml_div(ctx, *comb, sum);
    };
    norm_cols();
    for (int i = 1; i < c.hc_sinkhorn_iters; ++i) {
      norm_rows();
      norm_cols();
    }
  }
  if (f_pre)
    return ggml_dsv4_hc_pre(ctx, hc_contig(ctx, x), hc_contig(ctx, collapse));

  ggml_tensor *result = nullptr;
  for (int h = 0; h < g.HC; ++h) {
    ggml_tensor *xh =
        ggml_view_2d(ctx, x, g.D, g.S, x->nb[2], (size_t)h * x->nb[1]);
    ggml_tensor *ph =
        ggml_view_2d(ctx, collapse, 1, g.S, collapse->nb[1],
                     (size_t)h * collapse->nb[0]);
    ggml_tensor *term = ggml_mul(ctx, xh, ph);
    result = result ? ggml_add(ctx, result, term) : term;
  }
  return result;
}

ggml_tensor *hc_post(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                     ggml_tensor *residual, ggml_tensor *post,
                     ggml_tensor *comb) {
  if (fused_hc_op('o'))
    return ggml_dsv4_hc_post(ctx, hc_contig(ctx, x), hc_contig(ctx, residual),
                             hc_contig(ctx, post), hc_contig(ctx, comb));
  ggml_tensor *out = nullptr;
  for (int dst = 0; dst < g.HC; ++dst) {
    ggml_tensor *pd =
        ggml_view_2d(ctx, post, 1, g.S, post->nb[1], (size_t)dst * post->nb[0]);
    ggml_tensor *cur = ggml_mul(ctx, x, pd);
    for (int src = 0; src < g.HC; ++src) {
      ggml_tensor *rs = ggml_view_2d(ctx, residual, g.D, g.S, residual->nb[2],
                                     (size_t)src * residual->nb[1]);
      ggml_tensor *cw =
          ggml_view_2d(ctx, comb, 1, g.S, comb->nb[2],
                       (size_t)dst * comb->nb[0] + (size_t)src * comb->nb[1]);
      cur = ggml_add(ctx, cur, ggml_mul(ctx, rs, cw));
    }
    ggml_tensor *stream = ggml_reshape_3d(ctx, cur, g.D, 1, g.S);
    out = out ? ggml_concat(ctx, out, stream, 1) : stream;
  }
  return out;
}

ggml_tensor *build_kda(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *wq, ggml_tensor *wk,
                       ggml_tensor *wv, ggml_tensor *cq, ggml_tensor *ck,
                       ggml_tensor *cv, ggml_tensor *fa, ggml_tensor *fb,
                       ggml_tensor *ga, ggml_tensor *gb, ggml_tensor *beta_w,
                       ggml_tensor *a, ggml_tensor *dt, ggml_tensor *norm_w,
                       ggml_tensor *out_w, ggml_tensor *hist_in,
                       ggml_tensor *state_in, ggml_tensor **state_out,
                       ggml_tensor **hist_out) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_tensor *q = ggml_mul_mat(ctx, wq, cur);
  ggml_tensor *k = ggml_mul_mat(ctx, wk, cur);
  ggml_tensor *v = ggml_mul_mat(ctx, wv, cur);
  ggml_tensor *qkv = ggml_concat(ctx, ggml_concat(ctx, q, k, 0), v, 0);
  qkv = ggml_reshape_3d(ctx, qkv, 3 * g.d_inner, g.S, 1);

  ggml_tensor *conv_w = ggml_concat(
      ctx,
      ggml_concat(ctx, ggml_reshape_2d(ctx, cq, g.d_conv, g.d_inner),
                  ggml_reshape_2d(ctx, ck, g.d_conv, g.d_inner), 1),
      ggml_reshape_2d(ctx, cv, g.d_conv, g.d_inner), 1);
  ggml_tensor *conv_in = ggml_concat(ctx, hist_in, ggml_transpose(ctx, qkv), 0);
  ggml_tensor *conv = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_in, conv_w));
  const size_t row = ggml_row_size(conv->type, 3 * g.d_inner);
  const size_t head_row = ggml_row_size(conv->type, g.hd);
  q = ggml_view_4d(ctx, conv, g.hd, g.H, g.S, 1, head_row, row, row * g.S, 0);
  k = ggml_view_4d(ctx, conv, g.hd, g.H, g.S, 1, head_row, row, row * g.S,
                   ggml_row_size(conv->type, g.d_inner));
  v = ggml_view_4d(ctx, conv, g.hd, g.H, g.S, 1, head_row, row, row * g.S,
                   ggml_row_size(conv->type, 2 * g.d_inner));
  q = ggml_l2_norm(ctx, q, 1e-6f);
  k = ggml_l2_norm(ctx, k, 1e-6f);

  ggml_tensor *decay = ggml_mul_mat(ctx, fb, ggml_mul_mat(ctx, fa, cur));
  decay = ggml_add(ctx, decay, dt);
  decay = ggml_reshape_3d(ctx, decay, g.hd, g.H, g.S);
  decay = ggml_mul(ctx, decay, ggml_reshape_3d(ctx, a, 1, g.H, 1));
  decay = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, decay, -1.0f)),
                     c.kda_gate_lower_bound);
  decay = ggml_reshape_4d(ctx, decay, g.hd, g.H, g.S, 1);

  ggml_tensor *beta = ggml_mul_mat(ctx, beta_w, cur);
  beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, beta, 1, g.H, g.S, 1));
  ggml_tensor *scan =
      ggml_gated_delta_net(ctx, q, k, v, decay, beta, state_in, 1);

  const size_t value_row = ggml_row_size(scan->type, g.d_inner);
  const size_t state_row = ggml_row_size(scan->type, g.hd * g.hd);
  // ggml-cuda's gated_delta_net kernel stores the per-head state TRANSPOSED
  // relative to the [k][v] math convention this codebase's own legacy CPU
  // qwen4exp_gdn_delta_step (and, by the identical pattern, the KDA
  // recurrence) use -- confirmed directly in
  // ggml/src/ggml-cuda/gated_delta_net.cu's own comment ("state is stored
  // transposed: M[col][i] = S[i][col]", where col indexes v and i indexes
  // k), i.e. the raw view below has ne[0]=k (contiguous), ne[1]=v. Legacy's
  // own flat layout (glm5next_recur_state[l], engine.hpp) is st[k*hd+v] --
  // v contiguous, k slower -- the opposite axis order. Left unfixed, the
  // flat ggml_backend_tensor_get+memcpy in glm5next_full_graph_run silently
  // hands decode a transposed state matrix: same byte count, same total
  // size (hd==hd), no crash, just wrong values from the very first decode
  // token onward. Transpose+materialize here so the exported tensor's own
  // byte layout already matches what the flat copy-out assumes.
  ggml_tensor *state_raw =
      ggml_view_4d(ctx, scan, g.hd, g.hd, g.H, 1, head_row, state_row,
                   state_row * g.H, (size_t)g.d_inner * g.S * sizeof(float));
  *state_out = g.decode_mode ? ggml_cont(ctx, state_raw)
                             : ggml_cont(ctx, ggml_transpose(ctx, state_raw));
  ggml_tensor *core = ggml_view_4d(ctx, scan, g.hd, g.H, g.S, 1, head_row,
                                   value_row, value_row * g.S, 0);
  ggml_tensor *norm = ggml_mul(ctx, ggml_rms_norm(ctx, core, c.eps), norm_w);
  ggml_tensor *gate = ggml_mul_mat(ctx, gb, ggml_mul_mat(ctx, ga, cur));
  gate = ggml_reshape_3d(ctx, gate, g.hd, g.H, g.S);
  ggml_tensor *gated = ggml_mul(ctx, norm, ggml_sigmoid(ctx, gate));
  ggml_tensor *out =
      ggml_mul_mat(ctx, out_w, ggml_reshape_2d(ctx, gated, g.d_inner, g.S));
  *hist_out =
      ggml_cont(ctx, ggml_view_3d(ctx, conv_in, g.d_conv - 1, 3 * g.d_inner, 1,
                                  conv_in->nb[1], conv_in->nb[2],
                                  ggml_row_size(conv_in->type, g.S)));
  (void)layer;
  return out;
}

Glm5nextIndexerLayer *glm5next_indexer_state(FullGraph &g, int layer) {
  for (Glm5nextIndexerLayer &state : g.glm5next_indexer_layers)
    if (state.layer == layer)
      return &state;
  return nullptr;
}

bool glm5next_indexer_owner(const FullGraph &g, int layer) {
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  return model_tensor(*g.engine, prefix + "indexer.proj.weight") != nullptr;
}

ggml_tensor *glm5next_indexer_pool(ggml_context *ctx, ggml_tensor *keys,
                                   ggml_tensor *gates, ggml_tensor *ape,
                                   int key_length, int kpool, int n_pools) {
  if (!keys || !gates || !ape || key_length <= 0 || kpool <= 1 ||
      n_pools <= 0)
    return nullptr;
  const int64_t pooled_tokens = (int64_t)kpool * n_pools;
  const int64_t pooled_elements = (int64_t)key_length * pooled_tokens;
  const auto complete_pool_prefix = [&](ggml_tensor *t) -> ggml_tensor * {
    if (!t || t->ne[0] != key_length || t->ne[2] != 1 || t->ne[3] != 1 ||
        t->ne[1] < pooled_tokens)
      return nullptr;
    if (ggml_nelements(t) == pooled_elements)
      return t;
    // Prefill also caches key/gate rows for an incomplete tail so decode can
    // finish that pool later.  Pool summaries cover only complete KPool
    // blocks; trim the trailing token(s) before reshaping instead of asking
    // GGML to reshape key_length*S elements into key_length*kpool*n_pools.
    ggml_tensor *prefix =
        ggml_view_2d(ctx, t, key_length, pooled_tokens, t->nb[1], 0);
    return prefix ? ggml_cont(ctx, prefix) : nullptr;
  };
  keys = complete_pool_prefix(keys);
  gates = complete_pool_prefix(gates);
  if (!keys || !gates)
    return nullptr;
  ggml_tensor *key_blocks =
      ggml_reshape_3d(ctx, keys, key_length, kpool, n_pools);
  ggml_tensor *gate_blocks =
      ggml_reshape_3d(ctx, gates, key_length, kpool, n_pools);
  ggml_tensor *logits = ggml_add(ctx, gate_blocks, ape);
  logits = ggml_cont(ctx, ggml_permute(ctx, logits, 1, 0, 2, 3));
  ggml_tensor *prob = ggml_soft_max(ctx, logits);
  ggml_tensor *values =
      ggml_cont(ctx, ggml_permute(ctx, key_blocks, 1, 0, 2, 3));
  ggml_tensor *pooled = ggml_sum_rows(ctx, ggml_mul(ctx, prob, values));
  return ggml_reshape_2d(ctx, pooled, key_length, n_pools);
}

bool build_glm5next_indexer_prefill(FullGraph &g, ggml_context *ctx,
                                    ggml_tensor *cur, int layer) {
  const int key_length = g.indexer_key_length;
  const int kpool = g.indexer_kpool;
  const int n_pools = key_length > 0 && kpool > 1 ? g.S / kpool : 0;
  if (key_length <= 0 || kpool <= 1)
    return false;
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  ggml_tensor *w_key = resident_tensor(g, ctx, prefix + "indexer.attn_k.weight");
  ggml_tensor *w_norm =
      resident_tensor(g, ctx, prefix + "indexer.k_norm.weight");
  ggml_tensor *w_gate =
      resident_tensor(g, ctx, prefix + "indexer_compressor_gate.weight");
  ggml_tensor *w_ape =
      resident_tensor(g, ctx, prefix + "indexer_compressor_ape.weight");
  ggml_tensor *w_bias = nullptr;
  if (model_tensor(*g.engine, prefix + "indexer.k_norm.bias"))
    w_bias = resident_tensor(g, ctx, prefix + "indexer.k_norm.bias");
  if (!w_key || !w_norm || !w_gate || !w_ape ||
      (model_tensor(*g.engine, prefix + "indexer.k_norm.bias") && !w_bias))
    return false;

  const auto &c = g.engine->model_file.architecture_config;
  const float norm_eps = g.engine->model_file.get_config_float(
      "attention.layer_norm_epsilon", c.eps);
  ggml_tensor *keys = ggml_norm(ctx, ggml_mul_mat(ctx, w_key, cur), norm_eps);
  keys = ggml_mul(ctx, keys, w_norm);
  if (w_bias)
    keys = ggml_add(ctx, keys, w_bias);
  ggml_tensor *gates = ggml_mul_mat(ctx, w_gate, cur);
  ggml_tensor *packed = ggml_concat(ctx, keys, gates, 0);
  ggml_tensor *pooled = n_pools > 0
                            ? glm5next_indexer_pool(ctx, keys, gates, w_ape,
                                                    key_length, kpool, n_pools)
                            : nullptr;
  if (!packed || (n_pools > 0 && !pooled))
    return false;
  g.glm5next_indexer_layers.push_back(
      {layer, packed, pooled, nullptr, nullptr});
  return true;
}

bool build_glm5next_indexer_decode(FullGraph &g, ggml_context *ctx,
                                   ggml_tensor *cur, ggml_tensor *qr,
                                   int layer, ggml_tensor **sel_out,
                                   ggml_tensor **mask_out) {
  Glm5nextIndexerLayer *state = glm5next_indexer_state(g, layer);
  if (!state || !state->key_gate_cache || !state->pooled_cache ||
      !g.decode_indices || !g.indexer_pool_mask ||
      !g.indexer_pool_member_ids || !g.indexer_pool_update_rows ||
      !g.indexer_pool_update_index ||
      (g.indexer_select_tail &&
       (!g.indexer_tail_indices || !g.indexer_tail_mask)))
    return false;
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  ggml_tensor *w_key = resident_tensor(g, ctx, prefix + "indexer.attn_k.weight");
  ggml_tensor *w_norm =
      resident_tensor(g, ctx, prefix + "indexer.k_norm.weight");
  ggml_tensor *w_bias = nullptr;
  if (model_tensor(*g.engine, prefix + "indexer.k_norm.bias"))
    w_bias = resident_tensor(g, ctx, prefix + "indexer.k_norm.bias");
  ggml_tensor *w_gate =
      resident_tensor(g, ctx, prefix + "indexer_compressor_gate.weight");
  ggml_tensor *w_ape =
      resident_tensor(g, ctx, prefix + "indexer_compressor_ape.weight");
  ggml_tensor *w_q =
      resident_tensor(g, ctx, prefix + "indexer.attn_q_b.weight");
  ggml_tensor *w_proj =
      resident_tensor(g, ctx, prefix + "indexer.proj.weight");
  if (!w_key || !w_norm || !w_gate || !w_ape || !w_q || !w_proj ||
      (model_tensor(*g.engine, prefix + "indexer.k_norm.bias") && !w_bias))
    return false;

  const auto &c = g.engine->model_file.architecture_config;
  const float norm_eps = g.engine->model_file.get_config_float(
      "attention.layer_norm_epsilon", c.eps);
  ggml_tensor *keys = ggml_norm(ctx, ggml_mul_mat(ctx, w_key, cur), norm_eps);
  keys = ggml_mul(ctx, keys, w_norm);
  if (w_bias)
    keys = ggml_add(ctx, keys, w_bias);
  ggml_tensor *gates = ggml_mul_mat(ctx, w_gate, cur);
  ggml_tensor *packed = ggml_concat(ctx, keys, gates, 0);
  ggml_tensor *raw_updated =
      ggml_set_rows(ctx, state->key_gate_cache, packed, g.decode_indices);
  if (!raw_updated)
    return false;
  ggml_tensor *pool_rows =
      ggml_get_rows(ctx, raw_updated, g.indexer_pool_update_rows);
  const int key_length = g.indexer_key_length;
  const size_t gate_offset = ggml_row_size(pool_rows->type, key_length);
  ggml_tensor *pool_keys =
      ggml_cont(ctx, ggml_view_2d(ctx, pool_rows, key_length, g.indexer_kpool,
                                  pool_rows->nb[1], 0));
  ggml_tensor *pool_gates =
      ggml_cont(ctx, ggml_view_2d(ctx, pool_rows, key_length, g.indexer_kpool,
                                  pool_rows->nb[1], gate_offset));
  ggml_tensor *pool = glm5next_indexer_pool(ctx, pool_keys, pool_gates, w_ape,
                                           key_length, g.indexer_kpool, 1);
  if (!pool)
    return false;
  ggml_tensor *pool_updated = ggml_set_rows(
      ctx, state->pooled_cache, pool, g.indexer_pool_update_index);

  ggml_tensor *indexer_q = ggml_mul_mat(ctx, w_q, qr);
  indexer_q = ggml_reshape_3d(ctx, indexer_q, key_length,
                              g.indexer_heads, 1);
  ggml_tensor *indexer_weights = ggml_mul_mat(ctx, w_proj, cur);
  indexer_weights = ggml_scale(
      ctx, indexer_weights,
      1.0f / std::sqrt((float)key_length * g.indexer_heads));

  const int pool_count = g.kv_capacity / g.indexer_kpool;
  ggml_tensor *pool_keys_all =
      ggml_reshape_3d(ctx, pool_updated, key_length, 1, pool_count);
  ggml_tensor *k_p = ggml_permute(ctx, pool_keys_all, 0, 2, 1, 3);
  ggml_tensor *q_p = ggml_permute(ctx, indexer_q, 0, 2, 1, 3);
  ggml_tensor *kq = ggml_mul_mat(ctx, k_p, q_p);
  kq = ggml_cont(ctx, ggml_permute(ctx, kq, 2, 1, 0, 3));
  ggml_tensor *weights_3d =
      ggml_reshape_3d(ctx, indexer_weights, g.indexer_heads, 1, 1);
  ggml_tensor *score = ggml_mul(ctx, ggml_relu(ctx, kq), weights_3d);
  score = ggml_sum_rows(ctx, score);
  score = ggml_cont(ctx, ggml_permute(ctx, score, 2, 1, 0, 3));
  score = ggml_add(ctx, score, g.indexer_pool_mask);
  const int n_top_pools =
      std::min(pool_count, g.indexer_top_k / g.indexer_kpool);
  if (n_top_pools <= 0)
    return false;
  ggml_tensor *top_pools = ggml_top_k(ctx, score, n_top_pools);
  ggml_tensor *selected_scores = ggml_get_rows(
      ctx, ggml_reshape_3d(ctx, score, 1, pool_count, 1), top_pools);
  ggml_tensor *selection_order = ggml_argsort(
      ctx, ggml_reshape_2d(ctx, selected_scores, n_top_pools, 1),
      GGML_SORT_ORDER_DESC);
  top_pools = ggml_get_rows(
      ctx,
      ggml_reshape_3d(ctx, ggml_cast(ctx, top_pools, GGML_TYPE_F32), 1,
                      n_top_pools, 1),
      selection_order);
  top_pools = ggml_cast(
      ctx, ggml_cont(ctx, ggml_reshape_2d(ctx, top_pools, n_top_pools, 1)),
      GGML_TYPE_I32);
  ggml_tensor *selected_rows =
      ggml_get_rows(ctx, g.indexer_pool_member_ids, top_pools);
  selected_rows = ggml_reshape_2d(
      ctx, selected_rows, n_top_pools * g.indexer_kpool, 1);

  ggml_tensor *pool_mask_2d =
      ggml_reshape_2d(ctx, g.indexer_pool_mask, 1, pool_count);
  ggml_tensor *selected_pool_mask =
      ggml_get_rows(ctx, pool_mask_2d, top_pools);
  selected_pool_mask = ggml_reshape_4d(ctx, selected_pool_mask, 1,
                                       n_top_pools, 1, 1);
  selected_pool_mask = ggml_repeat_4d(
      ctx, selected_pool_mask, g.indexer_kpool, n_top_pools, 1, 1);
  selected_pool_mask = ggml_reshape_2d(
      ctx, selected_pool_mask, n_top_pools * g.indexer_kpool, 1);
  ggml_tensor *selected = selected_rows;
  ggml_tensor *gather_mask = selected_pool_mask;
  const int tail_rows =
      g.indexer_select_tail ? g.indexer_kpool - 1 : 0;
  if (tail_rows > 0) {
    selected = ggml_concat(ctx, selected, g.indexer_tail_indices, 0);
    gather_mask = ggml_concat(ctx, gather_mask, g.indexer_tail_mask, 0);
  }
  const int n_selected = n_top_pools * g.indexer_kpool + tail_rows;
  *sel_out = selected;
  *mask_out = ggml_reshape_4d(ctx, gather_mask, n_selected, 1, 1, 1);
  return *mask_out != nullptr;
}

ggml_tensor *build_dsa(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *wqa, ggml_tensor *wqan,
                       ggml_tensor *wqb, ggml_tensor *wkva, ggml_tensor *wkvan,
                       ggml_tensor *wkb, ggml_tensor *wvb, ggml_tensor *wo,
                       ggml_tensor **kv_out,
                       ggml_tensor *kv_cache,
                       ggml_tensor **prev_indexer_selection,
                       ggml_tensor **prev_indexer_mask)  {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_tensor *qr = ggml_rms_norm(ctx, ggml_mul_mat(ctx, wqa, cur), c.eps);
  qr = ggml_mul(ctx, qr, wqan);
  ggml_tensor *q = ggml_mul_mat(ctx, wqb, qr);
  q = ggml_reshape_3d(ctx, q, g.qk_nope, g.H, g.S);
  ggml_tensor *kv = ggml_mul_mat(ctx, wkva, cur);
  kv = ggml_mul(ctx, ggml_rms_norm(ctx, kv, c.eps), wkvan);
  *kv_out = kv;

  if (!g.decode_mode && glm5next_indexer_owner(g, layer) &&
      !build_glm5next_indexer_prefill(g, ctx, cur, layer))
    return nullptr;

  q = ggml_permute(ctx, q, 0, 2, 1, 3);
  q = ggml_mul_mat(ctx, wkb, q);
  q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
  const int64_t kv_tokens = g.decode_mode
                                ? g.kv_capacity
                                : (((int64_t)g.S + 255) & ~((int64_t)255));
  ggml_tensor *latent = nullptr;
  ggml_tensor *kv_cache_updated = nullptr;
  if (g.decode_mode) {
    if (!kv_cache || !g.decode_indices)
      return nullptr;
    ggml_tensor *cache_row = ggml_reshape_2d(ctx, kv, g.kv_lora, 1);
    kv_cache_updated = ggml_set_rows(ctx, kv_cache, cache_row, g.decode_indices);
    if (!kv_cache_updated)
      return nullptr;
    latent = ggml_reshape_3d(ctx, kv_cache_updated, g.kv_lora, 1,
                             g.kv_capacity);
  } else {
    latent = ggml_reshape_3d(ctx, kv, g.kv_lora, 1, g.S);
  }
  if (!g.decode_mode && kv_tokens > g.S) {
    // ggml-cuda's 512-wide FATTN kernels require the KV sequence length to
    // be a multiple of FATTN_KQ_STRIDE (256).  llama.cpp's normal KV-cache path
    // already supplies a padded cache.  This standalone no-cache graph must
    // provide the same layout explicitly; zero padding is masked out below.
    latent = ggml_pad(ctx, latent, 0, 0, (int)(kv_tokens - g.S), 0);
  }

  // Keep this layout sequence identical to llama.cpp's build_attn_mha().
  // In particular, the FATTN CUDA dispatch expects the absorbed MQA key and
  // value to be permuted after q has been split into streams, and it expects
  // F32 latent K/V to be explicitly converted to F16.  The previous manual
  // path omitted those details, causing every DSA FLASH_ATTN_EXT node to
  // fall back to the CPU scheduler backend.
  const bool v_trans = latent->nb[1] > latent->nb[2];
  const int64_t n_stream = latent->ne[3];
  q = ggml_view_4d(ctx, q, q->ne[0], q->ne[1], q->ne[2] / n_stream, n_stream,
                   q->nb[1], q->nb[2], q->nb[3] / n_stream, 0);
  q = ggml_permute(ctx, q, 0, 2, 1, 3);
  ggml_tensor *k = ggml_permute(ctx, latent, 0, 2, 1, 3);
  ggml_tensor *v = ggml_permute(ctx, latent, 0, 2, 1, 3);
  if (v_trans)
    v = ggml_transpose(ctx, v);
  if (k->type == GGML_TYPE_F32)
    k = ggml_cast(ctx, k, GGML_TYPE_F16);
  if (v->type == GGML_TYPE_F32)
    v = ggml_cast(ctx, v, GGML_TYPE_F16);

  ggml_tensor *attn = nullptr;
  if (g.decode_mode && g.sparse_decode) {
    if (!prev_indexer_selection || !prev_indexer_mask)
      return nullptr;
    if (glm5next_indexer_owner(g, layer)) {
      ggml_tensor *selection = nullptr;
      ggml_tensor *selection_mask = nullptr;
      if (!build_glm5next_indexer_decode(g, ctx, cur, qr, layer, &selection,
                                         &selection_mask))
        return nullptr;
      *prev_indexer_selection = selection;
      *prev_indexer_mask = selection_mask;
    }
    ggml_tensor *selection = *prev_indexer_selection;
    ggml_tensor *selection_mask = *prev_indexer_mask;
    if (!selection || !selection_mask)
      return nullptr;
    ggml_tensor *gathered = ggml_get_rows(ctx, kv_cache_updated, selection);
    gathered = ggml_cast(ctx, gathered, GGML_TYPE_F32);
    const int64_t n_selected = selection->ne[0];
    ggml_tensor *k_g =
        ggml_reshape_4d(ctx, gathered, g.kv_lora, n_selected, 1, 1);
    ggml_tensor *kq = ggml_mul_mat(ctx, k_g, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(
        ctx, kq, selection_mask,
        1.0f / std::sqrt((float)(g.qk_nope > 0 ? g.qk_nope : 1)), 0.0f);
    ggml_tensor *v_t = ggml_cont(ctx, ggml_transpose(ctx, k_g));
    ggml_tensor *kqv = ggml_mul_mat(ctx, v_t, kq);
    kqv = ggml_mul_mat(ctx, wvb, kqv);
    attn = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
    attn = ggml_reshape_2d(ctx, attn, g.v_head * g.H, g.S);
  } else {
    attn = ggml_flash_attn_ext(
        ctx, q, k, v, g.attn_mask,
        1.0f / std::sqrt((float)(g.qk_nope > 0 ? g.qk_nope : 1)), 0.0f,
        0.0f);
  }
  if (!g.sparse_decode && g.dsa_layers.empty()) {
    LOG_INFO("[GGML-GLM5NEXT-FULL] DSA FATTN layer=%d support_cuda=%d "
             "q=(%lld,%lld,%lld,%lld)/%s k=(%lld,%lld,%lld,%lld)/%s "
             "v=(%lld,%lld,%lld,%lld)/%s mask=(%lld,%lld,%lld,%lld)/%s\n",
             layer, ggml_backend_supports_op(g.cuda, attn) ? 1 : 0,
             (long long)q->ne[0], (long long)q->ne[1], (long long)q->ne[2],
             (long long)q->ne[3], ggml_type_name(q->type), (long long)k->ne[0],
             (long long)k->ne[1], (long long)k->ne[2], (long long)k->ne[3],
             ggml_type_name(k->type), (long long)v->ne[0], (long long)v->ne[1],
             (long long)v->ne[2], (long long)v->ne[3], ggml_type_name(v->type),
             (long long)g.attn_mask->ne[0], (long long)g.attn_mask->ne[1],
             (long long)g.attn_mask->ne[2], (long long)g.attn_mask->ne[3],
             ggml_type_name(g.attn_mask->type));
  }
  // Same v_mla path as build_attn_mha(): FATTN [kv_lora,S,H] is interpreted
  // as [kv_lora,H,S] for WUV, then made contiguous before the 2D projection.
  ggml_tensor *value = attn;
  if (!(g.decode_mode && g.sparse_decode)) {
    value = ggml_permute(ctx, attn, 0, 2, 1, 3);
    value = ggml_mul_mat(ctx, wvb, value);
    value = ggml_permute(ctx, value, 0, 2, 1, 3);
    value = ggml_cont(ctx, value);
    value = ggml_reshape_2d(ctx, value, g.v_head * g.H, g.S);
  }
  (void)layer;
  return ggml_mul_mat(ctx, wo, value);
}

} // namespace glm5next_graph

#endif // MOESTREAM_USE_GGML_CUDA
