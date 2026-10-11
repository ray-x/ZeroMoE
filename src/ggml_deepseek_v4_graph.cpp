// DeepSeek-V4 full-graph builder and entry points (split out of ggml_glm5next_graph.cpp).

#include "glm5next_graph_internal.hpp"
#include "ggml_deepseek_v4_graph.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

namespace glm5next_graph {

// ---------------------------------------------------------------------------
// DeepSeek-V4 full prefill graph.
//
// V4 cannot reuse GLM5NEXT's KDA/absorbed-MLA builder: its real Flash file
// has raw SWA attention plus ratio-4 CSA and ratio-128 HCA layers.  The
// builder below keeps the same graph-owned dense/expert arenas and HC/MoE
// nodes, but expresses V4's compressed keys and attention sinks as GGML
// nodes.  For the normal Flash prompt sizes (well below CSA top-k * 4), all
// completed compressed blocks are visible, so the indexer only needs to be
// carried forward for the subsequent decode path; no host top-k loop is
// inserted into the prefill graph.
// ---------------------------------------------------------------------------

bool dsv4_env_enabled() {
  static const bool enabled = [] {
    return moestream::graphcfg::flag_on("MOESTREAM_DEEPSEEK_V4_FULL_GRAPH");
  }();
  return enabled;
}

bool dsv4_graph_diagnostic_enabled() {
  return moestream::graphcfg::flag_on("MOESTREAM_DEEPSEEK_V4_GRAPH_DIAGNOSTIC");
}

int dsv4_env_device_expert_layers() {
  const char *v = moestream::graphcfg::env("MOESTREAM_DEEPSEEK_V4_GRAPH_PRELOAD_LAYERS");
  if (!v || !v[0])
    return 0;
  char *end = nullptr;
  long n = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || n <= 0)
    return 0;
  return static_cast<int>(std::min<long>(n, 43));
}

ggml_tensor *dsv4_host_i32(FullGraph &g, ggml_context *ctx, int n,
                           const char *name, int32_t **out) {
  ggml_tensor *t =
      host_input(g, ctx, GGML_TYPE_I32, {n}, (size_t)n * sizeof(int32_t), name);
  if (t && out)
    *out = static_cast<int32_t *>(t->data);
  return t;
}

ggml_tensor *dsv4_host_i32_2d(FullGraph &g, ggml_context *ctx, int ne0, int ne1,
                              const char *name, int32_t **out) {
  ggml_tensor *t = host_input(g, ctx, GGML_TYPE_I32, {ne0, ne1},
                              (size_t)ne0 * ne1 * sizeof(int32_t), name);
  if (t && out)
    *out = static_cast<int32_t *>(t->data);
  return t;
}

ggml_tensor *dsv4_host_f32(FullGraph &g, ggml_context *ctx,
                           const std::vector<int64_t> &shape, const char *name,
                           float **out) {
  size_t n = 1;
  for (int64_t d : shape)
    n *= (size_t)d;
  ggml_tensor *t =
      host_input(g, ctx, GGML_TYPE_F32, shape, n * sizeof(float), name);
  if (t && out)
    *out = static_cast<float *>(t->data);
  return t;
}

struct Dsv4RopeParams {
  float theta = 10000.0f;
  float freq_scale = 1.0f;
  float ext_factor = 0.0f;
  float attn_factor = 1.0f;
  int n_ctx_orig = 0;
};

Dsv4RopeParams dsv4_rope_params(const FullGraph &g, int ratio) {
  const auto &c = g.engine->model_file.architecture_config;
  Dsv4RopeParams p;
  p.theta = ratio == 0 ? g.engine->model_file.get_config_float("rope.freq_base",
                                                               10000.0f)
                       : c.dsv4_compress_rope_freq_base;
  if (ratio == 0)
    return p;
  p.freq_scale = c.rope_freq_scale;
  p.ext_factor = c.rope_ext_factor;
  p.n_ctx_orig = c.rope_n_ctx_orig;
  // ggml's YaRN kernel multiplies the supplied attention factor by
  // (1+0.1*log(1/freq_scale)) when ext_factor is enabled.  DeepSeek-V4's
  // dsv4_mscale intentionally divides by that term instead, so pre-adjust
  // the argument to make the GGML operation produce the V4 value.
  if (p.ext_factor != 0.0f && p.freq_scale > 0.0f) {
    const float f = 1.0f + 0.1f * std::log(1.0f / p.freq_scale);
    p.attn_factor = c.rope_attn_factor / (f * f);
  } else {
    p.attn_factor = c.rope_attn_factor;
  }
  return p;
}

ggml_tensor *dsv4_rope_tail(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                            int nope, int rope, ggml_tensor *positions,
                            int ratio, bool inverse) {
  if (rope <= 0)
    return x;
  const Dsv4RopeParams p = dsv4_rope_params(g, ratio);
  ggml_tensor *head =
      ggml_view_3d(ctx, x, nope, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
  ggml_tensor *tail = ggml_view_3d(ctx, x, rope, x->ne[1], x->ne[2], x->nb[1],
                                   x->nb[2], (size_t)nope * x->nb[0]);
  ggml_tensor *rot =
      inverse ? ggml_rope_ext_back(ctx, tail, positions, nullptr, rope,
                                   GGML_ROPE_TYPE_NORMAL, p.n_ctx_orig, p.theta,
                                   p.freq_scale, p.ext_factor, p.attn_factor,
                                   32.0f, 1.0f)
              : ggml_rope_ext(ctx, tail, positions, nullptr, rope,
                              GGML_ROPE_TYPE_NORMAL, p.n_ctx_orig, p.theta,
                              p.freq_scale, p.ext_factor, p.attn_factor, 32.0f,
                              1.0f);
  return ggml_concat(ctx, head, rot, 0);
}

ggml_tensor *dsv4_compressed_keys(FullGraph &g, ggml_context *ctx,
                                  ggml_tensor *comp_kv, ggml_tensor *comp_score,
                                  ggml_tensor *comp_ape, ggml_tensor *comp_norm,
                                  int ratio, int key_dim, bool csa,
                                  ggml_tensor *mod_positions,
                                  ggml_tensor *block_positions) {
  const int blocks = g.S / ratio;
  if (blocks <= 0)
    return nullptr;
  const int width = csa ? 2 * key_dim : key_dim;
  const int n_read = blocks * ratio;

  // Match Llama.cpp's state compressor.  The IDs are block-major: for HCA
  // they select every prompt row once; for CSA they select the previous
  // four-row window for every block followed by the current four-row window.
  // This makes the following softmax operate over positions (the first
  // dimension after permute), rather than over key channels.
  const int n_indices = csa ? 2 * n_read : n_read;
  int32_t *index_data = nullptr;
  ggml_tensor *state_read_idxs = dsv4_host_i32(g, ctx, n_indices,
                                               csa ? "dsv4_graph_csa_read_idxs"
                                                   : "dsv4_graph_hca_read_idxs",
                                               &index_data);
  if (!state_read_idxs)
    return nullptr;
  if (!csa) {
    for (int i = 0; i < n_read; ++i)
      index_data[i] = i;
  } else {
    for (int b = 0; b < blocks; ++b) {
      for (int i = 0; i < ratio; ++i) {
        const int j = b * ratio + i;
        // The appended row at index S is the zero/-inf history used for the
        // first CSA block, exactly as dsv4_append_zero_row() in llama.cpp.
        index_data[j] = b == 0 ? g.S : (b - 1) * ratio + i;
        index_data[n_read + j] = j;
      }
    }
  }

  ggml_tensor *kv_source = comp_kv;
  ggml_tensor *score_source = comp_score;
  if (csa) {
    // Use a graph-derived row so this works for both the normal qh-wide CSA
    // compressor and the narrower indexer compressor.  It avoids a separate
    // fixed-width host zero buffer and is the same value-wise operation as
    // llama.cpp's dsv4_append_zero_row().
    ggml_tensor *kv_zero = ggml_reshape_2d(
        ctx, ggml_scale(ctx, ggml_view_1d(ctx, comp_kv, width, 0), 0.0f), width,
        1);
    ggml_tensor *score_zero = ggml_reshape_2d(
        ctx,
        ggml_scale_bias(ctx, ggml_view_1d(ctx, comp_score, width, 0), 0.0f,
                        -INFINITY),
        width, 1);
    kv_source = ggml_concat(ctx, comp_kv, kv_zero, 1);
    score_source = ggml_concat(ctx, comp_score, score_zero, 1);
  }

  ggml_tensor *kv_rows = ggml_get_rows(ctx, kv_source, state_read_idxs);
  ggml_tensor *score_rows = ggml_get_rows(ctx, score_source, state_read_idxs);
  ggml_tensor *values = nullptr;
  ggml_tensor *scores = nullptr;
  if (!csa) {
    ggml_tensor *kv = ggml_reshape_3d(ctx, kv_rows, key_dim, ratio, blocks);
    ggml_tensor *score =
        ggml_reshape_3d(ctx, score_rows, key_dim, ratio, blocks);
    values = ggml_cont(ctx, ggml_permute(ctx, kv, 1, 0, 2, 3));
    scores = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3));
  } else {
    // Each row has [previous/current] channels.  Split the gathered rows
    // before concatenating along the position dimension, as in llama.cpp's
    // build_overlap_compressed_kv_from_state().
    ggml_tensor *kv_prev = ggml_cont(
        ctx, ggml_view_2d(ctx, kv_rows, key_dim, n_read, kv_rows->nb[1], 0));
    ggml_tensor *score_prev =
        ggml_cont(ctx, ggml_view_2d(ctx, score_rows, key_dim, n_read,
                                    score_rows->nb[1], 0));
    const size_t second_half =
        (size_t)n_read * kv_rows->nb[1] + ggml_row_size(kv_rows->type, key_dim);
    const size_t second_score_half = (size_t)n_read * score_rows->nb[1] +
                                     ggml_row_size(score_rows->type, key_dim);
    ggml_tensor *kv_cur =
        ggml_cont(ctx, ggml_view_2d(ctx, kv_rows, key_dim, n_read,
                                    kv_rows->nb[1], second_half));
    ggml_tensor *score_cur =
        ggml_cont(ctx, ggml_view_2d(ctx, score_rows, key_dim, n_read,
                                    score_rows->nb[1], second_score_half));
    ggml_tensor *kv =
        ggml_concat(ctx, ggml_reshape_3d(ctx, kv_prev, key_dim, ratio, blocks),
                    ggml_reshape_3d(ctx, kv_cur, key_dim, ratio, blocks), 1);
    ggml_tensor *score = ggml_concat(
        ctx, ggml_reshape_3d(ctx, score_prev, key_dim, ratio, blocks),
        ggml_reshape_3d(ctx, score_cur, key_dim, ratio, blocks), 1);
    values = ggml_cont(ctx, ggml_permute(ctx, kv, 1, 0, 2, 3));
    scores = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3));
  }

  ggml_tensor *weights = ggml_soft_max(ctx, scores);
  ggml_tensor *comp = ggml_mul(ctx, values, weights);
  comp = ggml_sum_rows(ctx, comp);
  comp = ggml_cont(ctx, ggml_permute(ctx, comp, 1, 0, 2, 3));
  comp = ggml_mul(
      ctx,
      ggml_rms_norm(ctx, comp, g.engine->model_file.architecture_config.eps),
      comp_norm);
  comp = dsv4_rope_tail(g, ctx, comp, key_dim - g.qk_rope, g.qk_rope,
                        block_positions, ratio, false);
  (void)mod_positions;
  (void)comp_ape;
  return ggml_cont_3d(ctx, comp, key_dim, 1, blocks);
}

ggml_tensor *build_dsv4_attention(FullGraph &g, ggml_context *ctx,
                                  ggml_tensor *cur, int layer) {
  const auto &c = g.engine->model_file.architecture_config;
  const std::string p = "blk." + std::to_string(layer) + ".";
  const int qh = g.qk_nope + g.qk_rope;
  const int ratio =
      c.dsv4_compress_ratios.empty()
          ? 0
          : static_cast<int>(c.dsv4_compress_ratios[(size_t)layer]);
  const bool csa = ratio == 4;
  FullGraph::Dsv4LayerTensors lt;
  lt.layer = layer;
  lt.ratio = ratio;

  ggml_tensor *wqa = resident_tensor(g, ctx, p + "attn_q_a.weight");
  ggml_tensor *wqan = resident_tensor(g, ctx, p + "attn_q_a_norm.weight");
  ggml_tensor *wqb = resident_tensor(g, ctx, p + "attn_q_b.weight");
  ggml_tensor *wkv = resident_tensor(g, ctx, p + "attn_kv.weight");
  ggml_tensor *wkvn = resident_tensor(g, ctx, p + "attn_kv_a_norm.weight");
  ggml_tensor *woa = resident_tensor(g, ctx, p + "attn_output_a.weight");
  ggml_tensor *wob = resident_tensor(g, ctx, p + "attn_output_b.weight");
  if (!wqa || !wqan || !wqb || !wkv || !wkvn || !woa || !wob)
    return nullptr;

  ggml_tensor *q_lora = ggml_mul_mat(ctx, wqa, cur);
  q_lora = ggml_mul(ctx, ggml_rms_norm(ctx, q_lora, c.eps), wqan);
  ggml_tensor *q =
      ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wqb, q_lora), qh, g.H, g.S);
  q = ggml_rms_norm(ctx, q, c.eps); // V4's weightless per-head Q norm
  q = dsv4_rope_tail(g, ctx, q, g.qk_nope, g.qk_rope, g.positions, ratio,
                     false);

  ggml_tensor *kv =
      ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wkv, cur), qh, 1, g.S);
  kv = ggml_mul(ctx, ggml_rms_norm(ctx, kv, c.eps), wkvn);
  kv = dsv4_rope_tail(g, ctx, kv, g.qk_nope, g.qk_rope, g.positions, ratio,
                      false);
  lt.raw_kv = kv;

  ggml_tensor *compressed = nullptr;
  int ncompressed = 0;
  if (ratio != 0) {
    ggml_tensor *ckv = resident_tensor(g, ctx, p + "attn_compressor_kv.weight");
    ggml_tensor *cgate =
        resident_tensor(g, ctx, p + "attn_compressor_gate.weight");
    ggml_tensor *cape =
        resident_tensor(g, ctx, p + "attn_compressor_ape.weight");
    ggml_tensor *cnorm =
        resident_tensor(g, ctx, p + "attn_compressor_norm.weight");
    if (!ckv || !cgate || !cape || !cnorm)
      return nullptr;
    const int width = csa ? 2 * qh : qh;
    lt.comp_kv = ggml_mul_mat(ctx, ckv, cur);
    ggml_tensor *gate = ggml_mul_mat(ctx, cgate, cur);
    ggml_tensor *ape_rows =
        ggml_get_rows(ctx, cape, csa ? g.positions_mod4 : g.positions_mod128);
    lt.comp_score = ggml_add(ctx, gate, ape_rows);
    compressed = dsv4_compressed_keys(
        g, ctx, lt.comp_kv, lt.comp_score, cape, cnorm, ratio, qh, csa,
        csa ? g.positions_mod4 : g.positions_mod128,
        csa ? g.block_positions4 : g.block_positions128);
    ncompressed = g.S / ratio;
    lt.keys = compressed;

    if (csa) {
      ggml_tensor *ikv =
          resident_tensor(g, ctx, p + "indexer_compressor_kv.weight");
      ggml_tensor *igate =
          resident_tensor(g, ctx, p + "indexer_compressor_gate.weight");
      ggml_tensor *iape =
          resident_tensor(g, ctx, p + "indexer_compressor_ape.weight");
      ggml_tensor *inorm =
          resident_tensor(g, ctx, p + "indexer_compressor_norm.weight");
      if (!ikv || !igate || !iape || !inorm)
        return nullptr;
      const int idim = c.dsv4_indexer_key_length;
      lt.indexer_comp_kv = ggml_mul_mat(ctx, ikv, cur);
      lt.indexer_comp_score =
          ggml_add(ctx, ggml_mul_mat(ctx, igate, cur),
                   ggml_get_rows(ctx, iape, g.positions_mod4));
      lt.indexer_keys = dsv4_compressed_keys(
          g, ctx, lt.indexer_comp_kv, lt.indexer_comp_score, iape, inorm, 4,
          idim, true, g.positions_mod4, g.block_positions4);
    }
  }

  ggml_tensor *seq = kv;
  if (compressed) {
    seq = ggml_concat(ctx, seq,
                      ggml_reshape_3d(ctx, compressed, qh, 1, ncompressed), 2);
  }
  const int key_count = g.S + ncompressed;

  // Use GGML's native attention-sink input.  Appending a synthetic K/V
  // dimension would change V4's qh=512 head width to 513, which disables the
  // CUDA FATTN kernel on RTX 3090 and sends every attention layer to the CPU.
  ggml_tensor *sink_tensor = resident_tensor(g, ctx, p + "attn_sinks.weight");
  if (!sink_tensor || sink_tensor->type != GGML_TYPE_F32 ||
      sink_tensor->ne[0] != (int64_t)g.H)
    return nullptr;
  // FATTN's CUDA GQA path requires K->ne[1] to be a multiple of 256 for
  // qh=512.  Pad with zero K/V rows and mask those rows out.  The padding is
  // a graph-only workspace shape choice; it never enters the cache state.
  const int padded_key_count = ((key_count + 255) / 256) * 256;
  ggml_tensor *seq_padded = seq;
  if (padded_key_count > key_count) {
    float *pad_data = nullptr;
    ggml_tensor *pad =
        dsv4_host_f32(g, ctx, {qh, 1, padded_key_count - key_count},
                      "dsv4_graph_attention_padding", &pad_data);
    if (!pad)
      return nullptr;
    std::fill(pad_data, pad_data + (size_t)qh * (padded_key_count - key_count),
              0.0f);
    seq_padded = ggml_concat(ctx, seq, pad, 2);
  }
  ggml_tensor *real =
      ggml_repeat_4d(ctx, seq_padded, qh, g.H, padded_key_count, 1);

  float *mask_data = nullptr;
  ggml_tensor *mask =
      dsv4_host_f32(g, ctx, {padded_key_count, g.S, 1, 1},
                    "dsv4_graph_attention_mask_f32", &mask_data);
  if (!mask)
    return nullptr;
  const int window = g.sliding_window > 0 ? g.sliding_window : g.S;
  for (int qpos = 0; qpos < g.S; ++qpos) {
    for (int kpos = 0; kpos < padded_key_count; ++kpos) {
      bool visible = false;
      if (kpos < key_count && kpos < g.S)
        visible = kpos <= qpos && kpos >= std::max(0, qpos - window + 1);
      else if (kpos < key_count)
        visible = (kpos - g.S) < (qpos + 1) / std::max(1, ratio);
      mask_data[(size_t)qpos * padded_key_count + kpos] =
          visible ? 0.0f : -INFINITY;
    }
  }
  // FATTN's CUDA path expects an F16 additive mask.  Keep the host source as
  // F32 while constructing it, then cast once as part of the graph.
  ggml_tensor *mask_f16 = ggml_cast(ctx, mask, GGML_TYPE_F16);
  lt.mask = mask_f16;

  ggml_tensor *q_attn =
      ggml_permute(ctx, ggml_reshape_4d(ctx, q, qh, g.H, g.S, 1), 0, 2, 1, 3);
  ggml_tensor *k_attn = ggml_permute(ctx, real, 0, 2, 1, 3);
  ggml_tensor *v_attn = ggml_permute(ctx, real, 0, 2, 1, 3);
  if (k_attn->type == GGML_TYPE_F32)
    k_attn = ggml_cast(ctx, k_attn, GGML_TYPE_F16);
  if (v_attn->type == GGML_TYPE_F32)
    v_attn = ggml_cast(ctx, v_attn, GGML_TYPE_F16);
  ggml_tensor *attn =
      ggml_flash_attn_ext(ctx, q_attn, k_attn, v_attn, mask_f16,
                          1.0f / std::sqrt((float)qh), 0.0f, 0.0f);
  ggml_flash_attn_ext_add_sinks(attn, sink_tensor);
  ggml_tensor *value = ggml_permute(ctx, attn, 0, 2, 1, 3);
  ggml_tensor *context =
      ggml_cont(ctx, ggml_view_3d(ctx, value, qh, g.H, g.S, value->nb[1],
                                  value->nb[2], 0));
  context = dsv4_rope_tail(g, ctx, context, g.qk_nope, g.qk_rope, g.positions,
                           ratio, true);

  const int groups = std::max(1, c.o_group_count);
  const int rank = c.o_lora_rank;
  const int group_in = (qh * g.H) / groups;
  if (rank <= 0 || groups <= 0 || group_in * groups != qh * g.H ||
      woa->ne[0] != group_in || woa->ne[1] != (int64_t)(groups * rank))
    return nullptr;
  // Keep the grouped output projection as one batched GGML matmul.  This is
  // Llama.cpp's layout: [group_in, groups, tokens] is permuted to
  // [group_in, tokens, groups], projected by [group_in, rank, groups], then
  // restored and made contiguous for wo_b.  The previous loop emitted one
  // matmul and concat per group, creating eight extra graph boundaries.
  ggml_tensor *woa3 = ggml_reshape_3d(ctx, woa, group_in, rank, groups);
  // context is already llama.cpp's [head_dim, n_head, tokens] tensor.  Reshape it
  // directly so the group and token strides are derived from that layout;
  // flattening to 2D first loses the explicit head axis around rope_back.
  ggml_tensor *grouped_input =
      ggml_reshape_3d(ctx, context, group_in, groups, g.S);
  grouped_input = ggml_permute(ctx, grouped_input, 0, 2, 1, 3);
  ggml_tensor *grouped = ggml_mul_mat(ctx, woa3, grouped_input);
  grouped = ggml_permute(ctx, grouped, 0, 2, 1, 3);
  grouped = ggml_cont_2d(ctx, grouped, rank * groups, g.S);
  g.dsv4_layers.push_back(lt);
  return ggml_mul_mat(ctx, wob, grouped);
}

ggml_tensor *dsv4_hash_ids(FullGraph &g, ggml_context *ctx, int layer,
                           const int *token_ids) {
  const auto &c = g.engine->model_file.architecture_config;
  const std::string name =
      "blk." + std::to_string(layer) + ".ffn_gate_tid2eid.weight";
  const QuantizedTensor *table = resident(*g.engine, name);
  if (!table || table->meta.type != GGUFType::I32 ||
      table->meta.shape.size() < 2 ||
      table->meta.shape[0] != (uint64_t)c.topk || !token_ids)
    return nullptr;
  const int64_t n_vocab = (int64_t)table->meta.shape[1];
  const int32_t *src = reinterpret_cast<const int32_t *>(table->host_ptr);
  int32_t *dst = nullptr;
  // ggml_get_rows preserves the ID tensor's shape after its row dimension.
  // Keep IDs as [topk,S], matching the normal argsort_top_k output.  A flat
  // [topk*S] tensor is numerically equivalent for a host loop but changes the
  // routed expert tensor layout and can make MUL_MAT_ID select the wrong
  // token grouping on the CUDA backend.
  ggml_tensor *ids =
      dsv4_host_i32_2d(g, ctx, c.topk, g.S, "dsv4_graph_hash_ids", &dst);
  if (!ids)
    return nullptr;
  for (int s = 0; s < g.S; ++s) {
    const int tok = token_ids[s];
    if (tok < 0 || (int64_t)tok >= n_vocab)
      return nullptr;
    for (int k = 0; k < c.topk; ++k) {
      const int id = src[(size_t)tok * c.topk + k];
      if (id < 0 || id >= c.n_experts)
        return nullptr;
      dst[(size_t)s * c.topk + k] = id;
    }
  }
  return ids;
}

// Real, measured incident (2026-09-09, qwen4exp side, same ggml scheduler
// code path this file also uses): ggml_backend_sched_alloc_graph's internal
// ggml_gallocr_reserve_n_impl does not fail gracefully on a real cudaMalloc
// OOM -- it logs "cudaMalloc failed: out of memory" and then segfaults the
// whole process instead of returning a checkable failure. That happened when
// main.cpp's fallback VRAM cache reservation (a fixed 4GB, unaware of
// device_expert_layers' own VRAM use) left the scheduler's own reserve()
// short by ~1.8GB. main.cpp now sizes that fallback cache against real free
// VRAM and this family's own device_expert_layers estimate, but this check
// stays as a second, independent line of defense: refuse to even attempt the
// allocator call when real free VRAM on the graph's device is below a
// generous floor, and decline to the legacy path instead of risking the same
// crash again.
bool graph_alloc_vram_headroom_ok(const FullGraph &g, const char *log_tag) {
  size_t free_b = 0, total_b = 0;
  int prev_device = -1;
  cudaGetDevice(&prev_device);
  cudaSetDevice(g.device);
  const cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
  if (prev_device >= 0)
    cudaSetDevice(prev_device);
  size_t kMinGraphAllocHeadroomBytes =
      2ull * 1024ull * 1024ull * 1024ull; // 2 GiB
  if (const char *e = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_ALLOC_FLOOR_MIB"))
    kMinGraphAllocHeadroomBytes = (size_t)std::max(0, std::atoi(e)) << 20;
  if (err != cudaSuccess || free_b < kMinGraphAllocHeadroomBytes) {
    LOG_ERROR("%s refusing graph allocator reserve: device %d free=%.2f GiB "
              "< %.2f GiB safety floor (expert arena + dense arena already "
              "committed %.2f GiB) -- falling back to the legacy path\n",
              log_tag, g.device,
              err == cudaSuccess
                  ? static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0)
                  : -1.0,
              static_cast<double>(kMinGraphAllocHeadroomBytes) /
                  (1024.0 * 1024.0 * 1024.0),
              static_cast<double>(g.expert_capacity + g.dense_capacity) /
                  (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  return true;
}

bool build_dsv4_graph(FullGraph &g, const int *token_ids) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_init_params ip = {};
  ip.mem_size = kTensorContextBytes;
  ip.no_alloc = true;
  if (!MemoryGuard::try_reserve_host_bytes(kTensorContextBytes)) {
    LOG_WARN("[GGML-DSV4-FULL-MEM] host graph context refused: %.2f MiB "
             "(reserved=%.2f GiB, budget=%.2f GB)\n",
             (double)kTensorContextBytes / (1024.0 * 1024.0),
             (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
             (double)MemoryGuard::host_budget_bytes() / 1.0e9);
    return false;
  }
  g.context_reservation_bytes = kTensorContextBytes;
  g.ctx = ggml_init(ip);
  if (!g.ctx) {
    MemoryGuard::release_host_bytes(g.context_reservation_bytes);
    g.context_reservation_bytes = 0;
    return false;
  }
  g.graph = ggml_new_graph_custom(g.ctx, kGraphSize, false);
  if (!g.graph)
    return false;

  g.input =
      host_input(g, g.ctx, GGML_TYPE_F32, {g.D, g.HC, g.S},
                 (size_t)g.D * g.HC * g.S * sizeof(float), "dsv4_full_input");
  if (!g.input)
    return false;
  int32_t *pos_data = nullptr;
  g.positions = dsv4_host_i32(g, g.ctx, g.S, "dsv4_full_positions", &pos_data);
  if (!g.positions)
    return false;
  for (int s = 0; s < g.S; ++s)
    pos_data[s] = s;

  bool need4 = false, need128 = false, need_csa = false;
  for (int l = 0; l < g.n_layers; ++l) {
    const int ratio = c.dsv4_compress_ratios.empty()
                          ? 0
                          : (int)c.dsv4_compress_ratios[(size_t)l];
    need4 |= ratio == 4;
    need128 |= ratio == 128;
    need_csa |= ratio == 4;
  }
  auto make_mod = [&](int ratio, const char *name, ggml_tensor **out_t) {
    int32_t *data = nullptr;
    *out_t = dsv4_host_i32(g, g.ctx, g.S, name, &data);
    if (!*out_t)
      return false;
    for (int s = 0; s < g.S; ++s)
      data[s] = s % ratio;
    return true;
  };
  if (need4 && !make_mod(4, "dsv4_full_positions_mod4", &g.positions_mod4))
    return false;
  if (need128 &&
      !make_mod(128, "dsv4_full_positions_mod128", &g.positions_mod128))
    return false;
  auto make_blocks = [&](int ratio, const char *name, ggml_tensor **out_t) {
    const int blocks = g.S / ratio;
    if (blocks <= 0)
      return true;
    int32_t *data = nullptr;
    *out_t = dsv4_host_i32(g, g.ctx, blocks, name, &data);
    if (!*out_t)
      return false;
    for (int b = 0; b < blocks; ++b)
      data[b] = b * ratio;
    return true;
  };
  if (need4 &&
      !make_blocks(4, "dsv4_full_block_positions4", &g.block_positions4))
    return false;
  if (need128 &&
      !make_blocks(128, "dsv4_full_block_positions128", &g.block_positions128))
    return false;
  if (need_csa) {
    float *zv = nullptr;
    float *zs = nullptr;
    g.csa_zero_values = dsv4_host_f32(g, g.ctx, {g.qk_nope + g.qk_rope, 4},
                                      "dsv4_csa_zero_values", &zv);
    g.csa_zero_scores = dsv4_host_f32(g, g.ctx, {g.qk_nope + g.qk_rope, 4},
                                      "dsv4_csa_zero_scores", &zs);
    if (!g.csa_zero_values || !g.csa_zero_scores)
      return false;
    const size_t zero_n = (size_t)(g.qk_nope + g.qk_rope) * 4;
    std::fill(zv, zv + zero_n, 0.0f);
    std::fill(zs, zs + zero_n, -INFINITY);
  }

  ggml_tensor *res = g.input;
  for (int l = 0; l < g.n_layers; ++l) {
    if (MemoryGuard::allocation_blocked())
      return false;
    const std::string p = "blk." + std::to_string(l) + ".";
    ggml_tensor *post = nullptr;
    ggml_tensor *comb = nullptr;
    ggml_tensor *afn = resident_tensor(g, g.ctx, p + "hc_attn_fn.weight");
    ggml_tensor *abase = resident_tensor(g, g.ctx, p + "hc_attn_base.weight");
    ggml_tensor *ascale = resident_tensor(g, g.ctx, p + "hc_attn_scale.weight");
    if (!afn || !abase || !ascale)
      return false;
    ggml_tensor *residual = res;
    ggml_tensor *cur = hc_pre(g, g.ctx, res, afn, ascale, abase, &post, &comb);
    ggml_tensor *an = resident_tensor(g, g.ctx, p + "attn_norm.weight");
    if (!an)
      return false;
    cur = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, cur, c.eps), an);
    ggml_tensor *attn = build_dsv4_attention(g, g.ctx, cur, l);
    if (!attn || g.dsv4_layers.empty() || g.dsv4_layers.back().layer != l)
      return false;
    if (dsv4_graph_diagnostic_enabled())
      g.dsv4_stage_outputs.emplace_back(
          "layer=" + std::to_string(l) + " attention", attn);
    res = hc_post(g, g.ctx, attn, residual, post, comb);

    ggml_tensor *ffn_fn = resident_tensor(g, g.ctx, p + "hc_ffn_fn.weight");
    ggml_tensor *ffn_base = resident_tensor(g, g.ctx, p + "hc_ffn_base.weight");
    ggml_tensor *ffn_scale =
        resident_tensor(g, g.ctx, p + "hc_ffn_scale.weight");
    ggml_tensor *fn = resident_tensor(g, g.ctx, p + "ffn_norm.weight");
    if (!ffn_fn || !ffn_base || !ffn_scale || !fn)
      return false;
    post = comb = nullptr;
    cur = hc_pre(g, g.ctx, res, ffn_fn, ffn_scale, ffn_base, &post, &comb);
    cur = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, cur, c.eps), fn);
    ggml_tensor *forced_ids = l < c.hash_layer_count
                                  ? dsv4_hash_ids(g, g.ctx, l, token_ids)
                                  : nullptr;
    if (l < c.hash_layer_count && !forced_ids)
      return false;
    ggml_tensor *ffn = build_ffn(g, g.ctx, cur, l, forced_ids);
    if (!ffn)
      return false;
    if (dsv4_graph_diagnostic_enabled())
      g.dsv4_stage_outputs.emplace_back("layer=" + std::to_string(l) + " ffn",
                                        ffn);
    res = hc_post(g, g.ctx, ffn, res, post, comb);
    if (dsv4_graph_diagnostic_enabled()) {
      g.dsv4_stage_outputs.emplace_back(
          "layer=" + std::to_string(l) + " residual", res);
      g.dsv4_layer_outputs.push_back(res);
    }
  }

  // DeepSeek-V4's model-level HC head is the learned gated collapse.  The
  // surrounding generation loop still applies output_norm after this graph,
  // exactly as it does after the legacy hc_collapse_output path.
  ggml_tensor *flat = ggml_reshape_2d(g.ctx, res, g.D * g.HC, g.S);
  ggml_tensor *normed = ggml_rms_norm(g.ctx, flat, c.eps);
  ggml_tensor *head_fn = resident_tensor(g, g.ctx, "output_hc_fn.weight");
  ggml_tensor *head_base = resident_tensor(g, g.ctx, "output_hc_base.weight");
  ggml_tensor *head_scale = resident_tensor(g, g.ctx, "output_hc_scale.weight");
  if (!head_fn || !head_base || !head_scale)
    return false;
  ggml_tensor *head_logits = ggml_mul_mat(g.ctx, head_fn, normed);
  ggml_tensor *scale = ggml_view_1d(g.ctx, head_scale, 1, 0);
  ggml_tensor *base = ggml_view_1d(g.ctx, head_base, g.HC, 0);
  ggml_tensor *weights = ggml_scale_bias(
      g.ctx,
      ggml_sigmoid(g.ctx,
                   ggml_add(g.ctx, ggml_mul(g.ctx, head_logits, scale), base)),
      1.0f, c.hc_eps);
  ggml_tensor *out = nullptr;
  for (int h = 0; h < g.HC; ++h) {
    ggml_tensor *stream =
        ggml_view_2d(g.ctx, res, g.D, g.S, res->nb[2], (size_t)h * res->nb[1]);
    ggml_tensor *weight = ggml_view_2d(g.ctx, weights, 1, g.S, weights->nb[1],
                                       (size_t)h * weights->nb[0]);
    ggml_tensor *term = ggml_mul(g.ctx, stream, weight);
    out = out ? ggml_add(g.ctx, out, term) : term;
  }
  g.final_streams = res;
  g.output = out;
  ggml_set_output(g.final_streams);
  ggml_build_forward_expand(g.graph, g.final_streams);
  ggml_set_output(g.output);
  ggml_build_forward_expand(g.graph, g.output);
  for (ggml_tensor *layer_output : g.dsv4_layer_outputs) {
    ggml_set_output(layer_output);
    ggml_build_forward_expand(g.graph, layer_output);
  }
  for (const auto &lt : g.dsv4_layers) {
    ggml_set_output(lt.raw_kv);
    ggml_build_forward_expand(g.graph, lt.raw_kv);
    if (lt.comp_kv) {
      ggml_set_output(lt.comp_kv);
      ggml_build_forward_expand(g.graph, lt.comp_kv);
      ggml_set_output(lt.comp_score);
      ggml_build_forward_expand(g.graph, lt.comp_score);
      if (lt.keys) {
        ggml_set_output(lt.keys);
        ggml_build_forward_expand(g.graph, lt.keys);
      }
    }
    if (lt.indexer_comp_kv) {
      ggml_set_output(lt.indexer_comp_kv);
      ggml_build_forward_expand(g.graph, lt.indexer_comp_kv);
      ggml_set_output(lt.indexer_comp_score);
      ggml_build_forward_expand(g.graph, lt.indexer_comp_score);
      if (lt.indexer_keys) {
        ggml_set_output(lt.indexer_keys);
        ggml_build_forward_expand(g.graph, lt.indexer_keys);
      }
    }
  }
  if (!graph_alloc_vram_headroom_ok(g, "[GGML-DSV4-FULL]"))
    return false;
  if (!ggml_backend_sched_alloc_graph(g.sched, g.graph))
    return false;
  g.graph_workspace_bytes = ggml_backend_sched_get_buffer_size(g.sched, g.cuda);
  int cuda_nodes = 0, cpu_nodes = 0, transitions = 0;
  ggml_backend_t previous = nullptr;
  std::unordered_map<std::string, int> non_cuda;
  for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
    ggml_tensor *node = ggml_graph_node(g.graph, i);
    ggml_backend_t b = ggml_backend_sched_get_tensor_backend(g.sched, node);
    if (b == g.cuda)
      ++cuda_nodes;
    else if (b == g.cpu) {
      ++cpu_nodes;
      ++non_cuda[ggml_op_name(node->op)];
    }
    if (previous && previous != b)
      ++transitions;
    previous = b;
  }
  LOG_INFO("[GGML-DSV4-FULL] scheduler nodes cuda=%d cpu=%d transitions=%d\n",
           cuda_nodes, cpu_nodes, transitions);
  if (!non_cuda.empty()) {
    std::string detail;
    for (const auto &[name, count] : non_cuda) {
      if (!detail.empty())
        detail += ", ";
      detail += name + "=" + std::to_string(count);
    }
    LOG_INFO("[GGML-DSV4-FULL] non-CUDA nodes: %s\n", detail.c_str());
  }
  LOG_INFO("[GGML-DSV4-FULL] graph built layers=%d nodes=%d splits=%d "
           "workspace=%.2f GiB compressed=%zu\n",
           g.n_layers, ggml_graph_n_nodes(g.graph),
           ggml_backend_sched_get_n_splits(g.sched),
           (double)g.graph_workspace_bytes / (1024.0 * 1024.0 * 1024.0),
           std::count_if(g.dsv4_layers.begin(), g.dsv4_layers.end(),
                         [](const auto &x) { return x.ratio != 0; }));
  LOG_INFO(
      "[GGML-DSV4-FULL-MEM] host ledger=%.2f/%.2f GB; graph context=%.2f MiB; "
      "device workspace=%.2f GiB (GPU, outside host_ram_gb)\n",
      (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
      (double)MemoryGuard::host_budget_bytes() / 1.0e9,
      (double)g.context_reservation_bytes / (1024.0 * 1024.0),
      (double)g.graph_workspace_bytes / (1024.0 * 1024.0 * 1024.0));
  g.ready = true;
  return true;
}

std::unique_ptr<FullGraph> g_dsv4_graph;

}  // namespace glm5next_graph

using namespace glm5next_graph;

bool deepseek_v4_full_graph_enabled() { return dsv4_env_enabled(); }

int deepseek_v4_full_graph_device_expert_layers() {
  return dsv4_env_enabled() ? dsv4_env_device_expert_layers() : 0;
}

void deepseek_v4_full_graph_shutdown() { g_dsv4_graph.reset(); }

bool deepseek_v4_full_graph_prepare(StreamingEngine &engine, int S,
                                    int pos_base, const int *token_ids) {
  if (!dsv4_env_enabled() || S <= 1 || pos_base != 0 ||
      engine.family_ops.family != ModelFamily::DeepSeek_V4)
    return false;
  const auto &c = engine.model_file.architecture_config;
  // The graph deliberately attends every completed CSA block.  That is an
  // exact match while the visible block count stays below the model's
  // Lightning Indexer top-k; beyond that point the legacy path remains the
  // correctness fallback until dynamic in-graph top-k selection is added.
  if (c.dsv4_indexer_top_k > 0 && S > c.dsv4_indexer_top_k * 4) {
    LOG_WARN(
        "[GGML-DSV4-FULL] prompt S=%d exceeds CSA dense-selection limit %d; "
        "using legacy indexed attention\n",
        S, c.dsv4_indexer_top_k * 4);
    return false;
  }
  if (g_dsv4_graph && g_dsv4_graph->ready && g_dsv4_graph->engine == &engine &&
      g_dsv4_graph->S == S)
    return true;

  g_dsv4_graph.reset();
  auto g = std::make_unique<FullGraph>();
  g->engine = &engine;
  g->device = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  g->S = S;
  g->D = c.hidden;
  g->HC = std::max(1, c.hc_mult);
  g->n_layers = c.n_layers;
  g->H = c.n_heads;
  g->qk_nope = c.qk_nope;
  g->qk_rope = c.qk_rope;
  g->o_group_count = c.o_group_count;
  g->o_lora_rank = c.o_lora_rank;
  g->sliding_window = c.sliding_window;
  g->first_dense = c.first_dense;
  g->device_expert_layers =
      std::min(deepseek_v4_full_graph_device_expert_layers(),
               std::max(0, c.n_layers - c.first_dense));
  if (configure_partial_experts(*g))
    g->device_expert_layers = 0;
  g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(g->device));
  g->cpu = ggml_backend_cpu_init();
  if (g->cpu) {
    const char *e = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_CPU_THREADS");
    ggml_backend_cpu_set_n_threads(g->cpu, e ? std::max(1, std::atoi(e)) : 6);
  }
  if (!g->cuda || !g->cpu)
    return false;

  const size_t source = expert_source_bytes(engine, g->device_expert_layers);
  const size_t host_live = graph_host_live_bytes(engine);
  constexpr size_t kSourceSlack = 1ull << 20;
  // 10 GiB, per explicit user instruction after the 2026-09-09 crash.
  constexpr size_t kWatermark = 10ull * 1024ull * 1024ull * 1024ull;
  if (source == 0 || source > SIZE_MAX - kSourceSlack ||
      !MemoryGuard::try_reserve_host_bytes(source + kSourceSlack, kWatermark)) {
    LOG_WARN(
        "[GGML-DSV4-FULL] host watermark refused source=%.2f GiB + existing "
        "host=%.2f GiB + reserve=%.2f GiB; budget=%.2f GB MemAvailable=%ld "
        "MiB\n",
        (double)source / (1024.0 * 1024.0 * 1024.0),
        (double)host_live / (1024.0 * 1024.0 * 1024.0),
        (double)kWatermark / (1024.0 * 1024.0 * 1024.0),
        (double)MemoryGuard::host_budget_bytes() / 1.0e9,
        MemoryGuard::last_available_mb());
    return false;
  }
  g->host_reservation_bytes = source + kSourceSlack;
  LOG_INFO("[GGML-DSV4-FULL] host admission: source=%.2f GiB existing=%.2f GiB "
           "reserve=%.2f GiB budget=%.2f GB\n",
           (double)source / (1024.0 * 1024.0 * 1024.0),
           (double)host_live / (1024.0 * 1024.0 * 1024.0),
           (double)kWatermark / (1024.0 * 1024.0 * 1024.0),
           (double)MemoryGuard::host_budget_bytes() / 1.0e9);
  if (!init_expert_source(*g, source) || !preload_expert_source(*g) ||
      !init_expert_arena(*g) || !init_dense_arena(*g))
    return false;
  ggml_backend_t backends[] = {g->cuda, g->cpu};
  g->sched =
      ggml_backend_sched_new(backends, nullptr, 2, kGraphSize, false, true);
  if (!g->sched || !build_dsv4_graph(*g, token_ids)) {
    LOG_WARN(
        "[GGML-DSV4-FULL] graph construction declined; using legacy path\n");
    return false;
  }
  g_dsv4_graph = std::move(g);
  LOG_INFO("[GGML-DSV4-FULL] graph prepared for S=%d\n", S);
  return true;
}

bool deepseek_v4_full_graph_run(StreamingEngine &engine, float *hidden_states,
                                int S, int pos_base, const int *) {
  if (!dsv4_env_enabled() || !hidden_states || S <= 1 || pos_base != 0 ||
      !g_dsv4_graph || !g_dsv4_graph->ready ||
      g_dsv4_graph->engine != &engine || g_dsv4_graph->S != S)
    return false;
  FullGraph &g = *g_dsv4_graph;
  const auto &c = engine.model_file.architecture_config;
  const int window = g.sliding_window > 0 ? g.sliding_window : engine.max_t;

  // The graph starts a fresh MoEStream sequence.  Clear the state first so a
  // failed/short graph run cannot leave the following legacy decode mixed
  // with stale compressed rows from a prior request.
  for (const auto &lt : g.dsv4_layers) {
    if (lt.layer < 0 || lt.layer >= (int)engine.KVc.size() ||
        !engine.KVc[lt.layer] || lt.layer >= (int)engine.dsv4_compressed.size())
      return false;
    const int ratio = lt.ratio;
    if (ratio != 0 && engine.dsv4_compressed[lt.layer].ratio != ratio)
      return false;
    if (ratio != 0)
      engine.dsv4_compressed[lt.layer].reset();
  }
  if (g.host_inputs.empty() ||
      engine.hc_stream_buf.size() < (size_t)g.D * g.HC * S)
    return false;
  std::memcpy(g.host_inputs[0], engine.hc_stream_buf.data(),
              (size_t)g.D * g.HC * S * sizeof(float));

  // Same defense-in-depth backstop as glm5next_full_graph_run's own compute
  // call (see that call site's own comment for the real-hardware incident
  // that motivated it) -- DSV4's full graph hasn't been run on real hardware
  // yet at all, so there is no evidence either way here, but the gap this
  // closes (a fresh, synchronous MemAvailable check right before the one
  // uncounted compute step in this whole admission chain) applies equally.
  {
    const long avail_mb = MemoryGuard::read_available_mb();
    constexpr long kMinComputeHeadroomMb =
        10240; // 10 GiB, per explicit user instruction after the 2026-09-09
               // crash
    if (avail_mb >= 0 && avail_mb < kMinComputeHeadroomMb) {
      LOG_ERROR("[GGML-DSV4-FULL] refusing graph compute: MemAvailable=%ld MiB "
                "< %ld MiB safety floor -- falling back to the legacy path\n",
                avail_mb, kMinComputeHeadroomMb);
      return false;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  if (ggml_backend_sched_graph_compute_async(g.sched, g.graph) !=
      GGML_STATUS_SUCCESS)
    return false;
  ggml_backend_sched_synchronize(g.sched);
  const auto synced = std::chrono::steady_clock::now();
  ggml_backend_tensor_get(g.output, hidden_states, 0,
                          (size_t)g.D * S * sizeof(float));

  if (dsv4_graph_diagnostic_enabled()) {
    for (const auto &stage : g.dsv4_stage_outputs) {
      // Attention/FFN stages are [D,S] tensors, while residual stages are
      // [D,HC,S].  Reading every stage into the residual-sized buffer used
      // to over-read the former and triggered GGML's bounds assertion.
      const size_t bytes = ggml_nbytes(stage.second);
      if (bytes == 0 || bytes % sizeof(float) != 0) {
        LOG_ERROR("[GGML-DSV4-DIAG] %s has unsupported byte size=%zu\n",
                  stage.first.c_str(), bytes);
        return false;
      }
      std::vector<float> layer_values(bytes / sizeof(float));
      ggml_backend_tensor_get(stage.second, layer_values.data(), 0, bytes);
      size_t finite = 0;
      float min_value = INFINITY;
      float max_value = -INFINITY;
      for (float value : layer_values) {
        if (std::isfinite(value)) {
          ++finite;
          min_value = std::min(min_value, value);
          max_value = std::max(max_value, value);
        }
      }
      LOG_INFO("[GGML-DSV4-DIAG] %s shape=(%lld,%lld,%lld,%lld) finite=%zu/%zu "
               "min=%g max=%g\n",
               stage.first.c_str(), (long long)stage.second->ne[0],
               (long long)stage.second->ne[1], (long long)stage.second->ne[2],
               (long long)stage.second->ne[3], finite, layer_values.size(),
               min_value, max_value);
      if (finite != layer_values.size())
        break;
    }
  }

  for (const auto &lt : g.dsv4_layers) {
    const int l = lt.layer;
    const size_t raw_n = (size_t)(g.qk_nope + g.qk_rope) * S;
    std::vector<float> raw(raw_n);
    ggml_backend_tensor_get(lt.raw_kv, raw.data(), 0, raw_n * sizeof(float));
    for (int s = 0; s < S; ++s) {
      const int slot = s % std::max(1, window);
      std::memcpy(engine.KVc[l] + (size_t)slot * (g.qk_nope + g.qk_rope),
                  raw.data() + (size_t)s * (g.qk_nope + g.qk_rope),
                  (size_t)(g.qk_nope + g.qk_rope) * sizeof(float));
    }
    if (lt.ratio == 0)
      continue;
    auto &state = engine.dsv4_compressed[l];
    const int width =
        lt.ratio == 4 ? 2 * (g.qk_nope + g.qk_rope) : (g.qk_nope + g.qk_rope);
    std::vector<float> comp((size_t)width * S);
    std::vector<float> score((size_t)width * S);
    ggml_backend_tensor_get(lt.comp_kv, comp.data(), 0,
                            comp.size() * sizeof(float));
    ggml_backend_tensor_get(lt.comp_score, score.data(), 0,
                            score.size() * sizeof(float));
    for (int s = 0; s < S; ++s) {
      const int slot = s % state.state_size;
      std::memcpy(state.state_kv.data() + (size_t)slot * width,
                  comp.data() + (size_t)s * width,
                  (size_t)width * sizeof(float));
      std::memcpy(state.state_score.data() + (size_t)slot * width,
                  score.data() + (size_t)s * width,
                  (size_t)width * sizeof(float));
    }
    const int blocks = S / lt.ratio;
    if (blocks > 0 && lt.keys) {
      std::vector<float> keys((size_t)(g.qk_nope + g.qk_rope) * blocks);
      ggml_backend_tensor_get(lt.keys, keys.data(), 0,
                              keys.size() * sizeof(float));
      std::memcpy(state.keys.data(), keys.data(),
                  std::min(state.keys.size(), keys.size()) * sizeof(float));
    }
    if (lt.indexer_comp_kv) {
      const int idim = c.dsv4_indexer_key_length;
      const int iwidth = 2 * idim;
      std::vector<float> ikv((size_t)iwidth * S);
      std::vector<float> iscore((size_t)iwidth * S);
      ggml_backend_tensor_get(lt.indexer_comp_kv, ikv.data(), 0,
                              ikv.size() * sizeof(float));
      ggml_backend_tensor_get(lt.indexer_comp_score, iscore.data(), 0,
                              iscore.size() * sizeof(float));
      for (int s = 0; s < S; ++s) {
        const int slot = s % state.state_size;
        std::memcpy(state.indexer_state_kv.data() + (size_t)slot * iwidth,
                    ikv.data() + (size_t)s * iwidth,
                    (size_t)iwidth * sizeof(float));
        std::memcpy(state.indexer_state_score.data() + (size_t)slot * iwidth,
                    iscore.data() + (size_t)s * iwidth,
                    (size_t)iwidth * sizeof(float));
      }
      if (blocks > 0 && lt.indexer_keys) {
        std::vector<float> keys((size_t)idim * blocks);
        ggml_backend_tensor_get(lt.indexer_keys, keys.data(), 0,
                                keys.size() * sizeof(float));
        std::memcpy(state.indexer_keys.data(), keys.data(),
                    std::min(state.indexer_keys.size(), keys.size()) *
                        sizeof(float));
      }
    }
  }
  const auto done = std::chrono::steady_clock::now();
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  LOG_INFO(
      "[GGML-DSV4-FULL-TIME] sync=%.3f ms state_get=%.3f ms total=%.3f ms\n",
      ms(start, synced), ms(synced, done), ms(start, done));
  engine.deepseek_v4_full_graph_output_ready = true;
  return true;
}

#else

bool deepseek_v4_full_graph_enabled() { return false; }
int deepseek_v4_full_graph_device_expert_layers() { return 0; }
bool deepseek_v4_full_graph_prepare(StreamingEngine &, int, int, const int *) {
  return false;
}
bool deepseek_v4_full_graph_run(StreamingEngine &, float *, int, int,
                                const int *) {
  return false;
}
void deepseek_v4_full_graph_shutdown() {}

#endif
