// DeepSeek-V4.1 whole-prompt graph. The GPU prefill path uses GGML nodes
// for Engram projections, HC and compressed attention. Routed experts remain
// owned by the native DDR/VRAM caches in both execution paths.

#include "glm5next_graph_internal.hpp"
#include "dsv41_engram.hpp"
#include "ggml_deepseek_v41_graph.hpp"
#include "dsv41_trace.hpp"
#include "ggml_f32_quant_backend.hpp"
#include "ram_manager.hpp"
#include "async_prefetcher.hpp"
#include "scratch_buffers.hpp"
#include "moe_pipeline_internal.hpp"
#include "moe_router_service.hpp"
#include "prefill_warmup_manager.hpp"
#include <chrono>
#include <stdexcept>

#if defined(MOESTREAM_USE_GGML_CUDA)

namespace glm5next_graph {

namespace {

std::unique_ptr<FullGraph> g_dsv41_graph;

struct V41PrefillSite {
  StreamingEngine *engine = nullptr;
  int layer = -1;
  int S = 0;
  int pos_base = 0;
  const int *token_ids = nullptr;
};
std::vector<V41PrefillSite> g_v41_prefill_sites;
struct V41HcSite {
  StreamingEngine *engine = nullptr;
  int layer = -1;
  int S = 0;
  int pos_base = 0;
  int sub_block = 0;
};
std::vector<V41HcSite> g_v41_hc_sites;
struct V41TraceTensor {
  int layer = -1;
  const char *label = nullptr;
  ggml_tensor *tensor = nullptr;
};
std::vector<V41TraceTensor> g_v41_trace_tensors;

void v41_mark_trace(FullGraph &g, const char *label, int layer,
                    ggml_tensor *tensor) {
  if (!tensor || !dsv41_trace::enabled(layer, 0))
    return;
  g_v41_trace_tensors.push_back({layer, label, tensor});
  ggml_set_output(tensor);
  ggml_build_forward_expand(g.graph, tensor);
}

void v41_keep_tensor(FullGraph &g, ggml_tensor *tensor) {
  if (!tensor)
    return;
  ggml_set_output(tensor);
  ggml_build_forward_expand(g.graph, tensor);
}

void v41_moe_node(ggml_tensor *dst, const ggml_tensor *src, int ith, int,
                  void *ud) {
  if (ith != 0)
    return;
  const auto &site = *static_cast<const V41PrefillSite *>(ud);
  if (!site.engine || !site.token_ids || site.S <= 0 ||
      ggml_nelements(dst) != (size_t)site.engine->model_file
                                  .architecture_config.hidden * site.S ||
      ggml_nelements(src) != ggml_nelements(dst))
    throw std::runtime_error("invalid DeepSeek-V4.1 prefill MoE node");
  deepseek_v41_full_graph_moe_forward(
      *site.engine, site.layer, static_cast<const float *>(src->data), site.S,
      site.pos_base, static_cast<float *>(dst->data), site.token_ids);
}

void v41_attention_node(ggml_tensor *dst, const ggml_tensor *src, int ith,
                        int, void *ud) {
  if (ith != 0)
    return;
  const auto &site = *static_cast<const V41PrefillSite *>(ud);
  if (!site.engine || site.S <= 0 ||
      ggml_nelements(dst) != (size_t)site.engine->model_file
                                  .architecture_config.hidden * site.S ||
      ggml_nelements(src) != ggml_nelements(dst))
    throw std::runtime_error("invalid DeepSeek-V4.1 prefill attention node");
  deepseek_v41_full_graph_attention_forward(
      *site.engine, site.layer, static_cast<const float *>(src->data), site.S,
      site.pos_base, static_cast<float *>(dst->data));
}

void v41_engram_node(ggml_tensor *dst, const ggml_tensor *src, int ith, int,
                     void *ud) {
  if (ith != 0)
    return;
  const auto &site = *static_cast<const V41PrefillSite *>(ud);
  StreamingEngine &e = *site.engine;
  const auto &c = e.model_file.architecture_config;
  const size_t stream_elements = (size_t)site.S * c.hc_mult * c.hidden;
  if (!e.family_ops.layer_input_fn || !site.token_ids || !src || !src->data ||
      ggml_nelements(src) != stream_elements ||
      ggml_nelements(dst) != stream_elements ||
      e.hc_stream_buf.size() < stream_elements)
    throw std::runtime_error("DeepSeek-V4.1 graph Engram hook is not ready");
  std::memcpy(e.hc_stream_buf.data(), src->data,
              stream_elements * sizeof(float));
  e.family_ops.layer_input_fn(e, site.layer, site.S, site.pos_base,
                              site.token_ids);
  dsv41_trace::save(
      "post-input-hook", site.layer, site.pos_base,
      e.hc_stream_buf.data() + (size_t)(site.S - 1) * c.hc_mult * c.hidden,
      c.hidden);
  if (site.S > 1)
    dsv41_trace::save("post-input-hook-seq", site.layer, site.pos_base,
                      e.hc_stream_buf.data(), stream_elements);
  std::memcpy(dst->data, e.hc_stream_buf.data(),
              stream_elements * sizeof(float));
}

void v41_hc_pre_node(ggml_tensor *dst, int ith, int, void *ud) {
  if (ith != 0)
    return;
  const auto &site = *static_cast<const V41HcSite *>(ud);
  StreamingEngine &e = *site.engine;
  const auto &c = e.model_file.architecture_config;
  const size_t stream_elements = (size_t)site.S * c.hc_mult * c.hidden;
  if (!e.family_ops.hc_pre_fn || !dst->src[0] || !dst->src[0]->data ||
      ggml_nelements(dst->src[0]) != stream_elements ||
      e.hc_stream_buf.size() < stream_elements)
    throw std::runtime_error("DeepSeek-V4.1 graph HC pre state is not ready");
  std::memcpy(e.hc_stream_buf.data(), dst->src[0]->data,
              stream_elements * sizeof(float));
  e.family_ops.hc_pre_fn(e, site.layer, site.sub_block,
                         static_cast<float *>(dst->data), site.S, c.hidden);
  if (site.sub_block == 0)
    dsv41_trace::save("hc-attn-pre", site.layer, site.pos_base,
                      static_cast<const float *>(dst->data) +
                          (size_t)(site.S - 1) * c.hidden,
                      c.hidden);
}

void v41_hc_post_node(ggml_tensor *dst, int ith, int, void *ud) {
  if (ith != 0)
    return;
  const auto &site = *static_cast<const V41HcSite *>(ud);
  StreamingEngine &e = *site.engine;
  const auto &c = e.model_file.architecture_config;
  const size_t stream_elements = (size_t)site.S * c.hc_mult * c.hidden;
  const size_t hidden_elements = (size_t)site.S * c.hidden;
  if (!e.family_ops.hc_post_fn || !dst->src[0] || !dst->src[1] ||
      !dst->src[0]->data || !dst->src[1]->data ||
      ggml_nelements(dst->src[0]) != hidden_elements ||
      ggml_nelements(dst->src[1]) != stream_elements ||
      ggml_nelements(dst) != stream_elements ||
      e.hc_stream_buf.size() < stream_elements)
    throw std::runtime_error("DeepSeek-V4.1 graph HC post state is not ready");
  std::memcpy(e.hc_stream_buf.data(), dst->src[1]->data,
              stream_elements * sizeof(float));
  e.family_ops.hc_post_fn(
      e, site.layer, site.sub_block, static_cast<float *>(dst->data),
      static_cast<const float *>(dst->src[0]->data), site.S, c.hidden);
  const char *hidden_label =
      site.sub_block == 0 ? "post-attn" : "post-ffn";
  const char *streams_label =
      site.sub_block == 0 ? "post-attn-streams" : "post-ffn-streams";
  dsv41_trace::save(
      hidden_label, site.layer, site.pos_base,
      static_cast<const float *>(dst->data) + (size_t)(site.S - 1) * c.hidden,
      c.hidden);
  std::memcpy(dst->data, e.hc_stream_buf.data(),
              stream_elements * sizeof(float));
  dsv41_trace::save(streams_label, site.layer, site.pos_base,
                    static_cast<const float *>(dst->data) +
                        (size_t)(site.S - 1) * c.hc_mult * c.hidden,
                    (size_t)c.hc_mult * c.hidden);
}

bool v41_gpu_prefill_enabled() {
  return moestream::graphcfg::flag_on("MOESTREAM_DEEPSEEK_V41_GPU_PREFILL");
}

// Dense tensors already uploaded by the engine are non-owning graph views.
// This marker identifies their CUDA backend; it does not own their storage.
ggml_tensor *v41_weight(FullGraph &g, ggml_context *ctx,
                         const std::string &name) {
  const auto *q = resident(*g.engine, name);
  if (!q || !type_ok(q->meta.type)) {
    LOG_ERROR("[DSV41-GPU-PREFILL] unavailable weight %s\n", name.c_str());
    return nullptr;
  }
  if (!q->device_ptr) return resident_tensor(g, ctx, name);
  std::vector<int64_t> shape(q->meta.shape.begin(), q->meta.shape.end());
  auto *t = new_tensor(ctx, static_cast<ggml_type>(q->meta.type), shape);
  if (!t || ggml_nbytes(t) > q->meta.size_bytes) return nullptr;
  t->data = q->device_ptr;
  t->buffer = g.dense.buffer;
  ggml_set_name(t, name.c_str());
  return t;
}

bool v41_env_enabled() {
  return moestream::graphcfg::flag_on("MOESTREAM_DEEPSEEK_V41_FULL_GRAPH");
}

int v41_device_expert_layers() {
  const char *v = moestream::graphcfg::env(
      "MOESTREAM_DEEPSEEK_V41_GRAPH_PRELOAD_LAYERS");
  if (!v || !v[0])
    return 0;
  char *end = nullptr;
  const long n = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || n <= 0)
    return 0;
  return static_cast<int>(std::min<long>(n, 39));
}

// Diagnostic precision mode: retain F32 activations and attention values.
// Cast only dense tensors into reusable graph scratch; expert storage remains
// the original quantized native cache, with no graph expert allocation.
bool v41_native_f32() {
  return moestream::graphcfg::flag_on("MOESTREAM_DEEPSEEK_V41_NATIVE_F32_GRAPH");
}
bool v41_prefill_f32() {
  return v41_native_f32() || moestream::graphcfg::flag_on("MOESTREAM_DEEPSEEK_V41_PREFILL_F32");
}
ggml_tensor *v41_matmul(ggml_context *ctx, ggml_tensor *w, ggml_tensor *x) {
  if (v41_prefill_f32() && !v41_native_f32() && ggml_is_quantized(w->type))
    w = ggml_cast(ctx, w, GGML_TYPE_F32);
  auto *out = ggml_mul_mat(ctx, w, x);
  if (v41_prefill_f32()) ggml_prec_set_acc(out, GGML_PREC_F32);
  if (v41_native_f32() && ggml_is_quantized(w->type))
    ggml_prec_set_src(out, GGML_PREC_F32, 1);
  return out;
}

ggml_tensor *v41_host_f32(FullGraph &g, ggml_context *ctx,
                          const std::vector<int64_t> &shape,
                          const char *name, float **out = nullptr) {
  size_t n = 1;
  for (int64_t d : shape)
    n *= static_cast<size_t>(d);
  ggml_tensor *t = host_input(g, ctx, GGML_TYPE_F32, shape,
                              n * sizeof(float), name);
  if (t && out)
    *out = static_cast<float *>(t->data);
  return t;
}

ggml_tensor *v41_host_i32(FullGraph &g, ggml_context *ctx, int n,
                          const char *name, int32_t **out) {
  ggml_tensor *t = host_input(g, ctx, GGML_TYPE_I32, {n},
                              static_cast<size_t>(n) * sizeof(int32_t), name);
  if (t && out)
    *out = static_cast<int32_t *>(t->data);
  return t;
}

ggml_tensor *v41_rope_tail(ggml_context *ctx, ggml_tensor *x, int nope,
                           int rope, ggml_tensor *positions, int n_ctx_orig,
                           float theta, float freq_scale, float ext_factor,
                           float attn_factor, bool inverse = false) {
  if (rope <= 0)
    return x;
  ggml_tensor *head = ggml_view_3d(ctx, x, nope, x->ne[1], x->ne[2],
                                   x->nb[1], x->nb[2], 0);
  ggml_tensor *tail = ggml_view_3d(
      ctx, x, rope, x->ne[1], x->ne[2], x->nb[1], x->nb[2],
      static_cast<size_t>(nope) * x->nb[0]);
  ggml_tensor *rot = inverse
                         ? ggml_rope_ext_back(
                               ctx, tail, positions, nullptr, rope,
                               GGML_ROPE_TYPE_NORMAL, n_ctx_orig, theta,
                               freq_scale, ext_factor, attn_factor, 32.0f,
                               1.0f)
                         : ggml_rope_ext(ctx, tail, positions, nullptr, rope,
                                         GGML_ROPE_TYPE_NORMAL, n_ctx_orig,
                                         theta, freq_scale, ext_factor,
                                         attn_factor, 32.0f, 1.0f);
  return ggml_concat(ctx, head, rot, 0);
}

// V4.1's indexer key can be narrower than the attention latent, so pooling is
// kept separate from each output's RoPE width.
ggml_tensor *v41_compress_latent(ggml_context *ctx, ggml_tensor *kv,
                                 ggml_tensor *score, ggml_tensor *norm,
                                 int S, int ratio, int key_dim, float eps) {
  const int blocks = S / ratio;
  if (blocks <= 0) return nullptr;
  if (ratio == 1)
    return ggml_mul(ctx, ggml_rms_norm(ctx, kv, eps), norm);
  auto rows = [&](ggml_tensor *x) {
    auto *xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // [S, key_dim]
    xt = ggml_cont(ctx, ggml_view_2d(ctx, xt, ratio * blocks, key_dim,
                                    xt->nb[1], 0));
    xt = ggml_reshape_3d(ctx, xt, ratio, blocks, key_dim);
    return ggml_cont(ctx, ggml_permute(ctx, xt, 0, 2, 1, 3));
  };
  auto *values = rows(kv); // [ratio, key_dim, blocks]
  auto *weights = ggml_soft_max(ctx, rows(score));
  auto *pooled = ggml_sum_rows(ctx, ggml_mul(ctx, values, weights));
  pooled = ggml_reshape_2d(ctx, pooled, key_dim, blocks);
  return ggml_mul(ctx, ggml_rms_norm(ctx, pooled, eps), norm);
}

int nearest_source(const std::vector<int64_t> &sources, int layer,
                   const std::vector<int64_t> &ratios) {
  for (int i = layer; i >= 0; --i) {
    if (i < static_cast<int>(ratios.size()) && ratios[(size_t)i] ==
                                                    ratios[(size_t)layer] &&
        std::find(sources.begin(), sources.end(), i) != sources.end())
      return i;
  }
  return -1;
}

ggml_tensor *build_v41_engram(FullGraph &g, ggml_context *ctx,
                              ggml_tensor *res, int layer,
                              ggml_tensor *embedding) {
  const auto &c = g.engine->model_file.architecture_config;
  const int n_heads = c.dsv41_engram_n_heads;
  const int head_dim = c.dsv41_engram_head_dim;
  const int n_hash_cols = (c.dsv41_engram_max_ngram_size - 1) * n_heads;
  const std::string p = "blk." + std::to_string(layer) + ".";
  if (!embedding || n_hash_cols <= 0 || head_dim <= 0)
    return nullptr;
  ggml_tensor *wkv = v41_weight(g, ctx, p + "engram_wkv.weight");
  ggml_tensor *wq = v41_weight(g, ctx, p + "engram_q.weight");
  ggml_tensor *wk = v41_weight(g, ctx, p + "engram_k.weight");
  if (!wkv || !wq || !wk)
    return nullptr;

  ggml_tensor *kv = v41_matmul(ctx, wkv, embedding);
  ggml_tensor *key = ggml_view_3d(
      ctx, kv, g.D, g.HC, g.S, (size_t)g.D * sizeof(float), kv->nb[1], 0);
  ggml_tensor *value = ggml_view_2d(
      ctx, kv, g.D, g.S, kv->nb[1], (size_t)g.HC * g.D * kv->nb[0]);
  // Engram's per-stream gate multiplies Q and K elementwise. These are
  // quantized in this GGUF, and GGML does not implement Q2_K * Q2_K.
  // Cast only these small gate vectors before the elementwise operation.
  ggml_tensor *q = ggml_reshape_3d(
      ctx, ggml_cast(ctx, wq, GGML_TYPE_F32), g.D, g.HC, 1);
  ggml_tensor *k = ggml_reshape_3d(
      ctx, ggml_cast(ctx, wk, GGML_TYPE_F32), g.D, g.HC, 1);
  ggml_tensor *qk = ggml_repeat_4d(ctx, ggml_mul(ctx, q, k), g.D, g.HC,
                                  g.S, 1);
  ggml_tensor *key_sq = ggml_sum_rows(ctx, ggml_mul(ctx, key, key));
  ggml_tensor *res_sq = ggml_sum_rows(ctx, ggml_mul(ctx, res, res));
  ggml_tensor *dot = ggml_sum_rows(
      ctx, ggml_mul(ctx, ggml_mul(ctx, res, key), qk));
  const float inv_dim = 1.0f / static_cast<float>(g.D);
  const float inv_sqrt_dim = 1.0f / std::sqrt(static_cast<float>(g.D));
  key_sq = ggml_sqrt(ctx, ggml_scale_bias(
                             ctx, ggml_scale(ctx, key_sq, inv_dim), 1.0f,
                             c.eps));
  res_sq = ggml_sqrt(ctx, ggml_scale_bias(
                             ctx, ggml_scale(ctx, res_sq, inv_dim), 1.0f,
                             c.eps));
  ggml_tensor *score = ggml_scale(
      ctx, ggml_div(ctx, ggml_div(ctx, dot, res_sq), key_sq), inv_sqrt_dim);
  ggml_tensor *signed_sqrt = ggml_mul(
      ctx, ggml_sgn(ctx, score),
      ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, score), 1e-6f, INFINITY)));
  ggml_tensor *gate = ggml_sigmoid(ctx, signed_sqrt);
  ggml_tensor *value_streams = ggml_repeat_4d(
      ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, value), g.D, 1, g.S),
      g.D, g.HC, g.S, 1);
  return ggml_add(ctx, res, ggml_mul(ctx, value_streams, gate));
}

ggml_tensor *build_v41_attention(
    FullGraph &g, ggml_context *ctx, ggml_tensor *cur, int layer,
    std::unordered_map<int, ggml_tensor *> &kv_keys,
    std::unordered_map<int, ggml_tensor *> &index_keys,
    std::unordered_map<int, ggml_tensor *> &index_masks) {
  const auto &c = g.engine->model_file.architecture_config;
  const std::string p = "blk." + std::to_string(layer) + ".";
  const int qh = g.qk_nope + g.qk_rope;
  const int ratio = static_cast<int>(c.dsv4_compress_ratios[(size_t)layer]);
  const int kv_source = ratio == 0 ? -1 : nearest_source(
      c.dsv41_kv_source_layers, layer, c.dsv4_compress_ratios);
  const int index_source = ratio == 0 ? -1 : nearest_source(
      c.dsv41_index_source_layers, layer, c.dsv4_compress_ratios);
  if (ratio != 0 && (kv_source < 0 || index_source < 0)) {
    LOG_ERROR("[DSV41-GPU-PREFILL] missing source map layer=%d kv=%d index=%d\n", layer, kv_source, index_source);
    return nullptr;
  }

  ggml_tensor *wqa = v41_weight(g, ctx, p + "attn_q_a.weight");
  ggml_tensor *wqan = v41_weight(g, ctx, p + "attn_q_a_norm.weight");
  ggml_tensor *wqb = v41_weight(g, ctx, p + "attn_q_b.weight");
  ggml_tensor *wkv = v41_weight(g, ctx, p + "attn_kv.weight");
  ggml_tensor *wkvn = v41_weight(g, ctx, p + "attn_kv_a_norm.weight");
  ggml_tensor *woa = v41_weight(g, ctx, p + "attn_output_a.weight");
  ggml_tensor *wob = v41_weight(g, ctx, p + "attn_output_b.weight");
  ggml_tensor *sinks = v41_weight(g, ctx, p + "attn_sinks.weight");
  if (!wqa || !wqan || !wqb || !wkv || !wkvn || !woa || !wob || !sinks)
    return nullptr;

  v41_mark_trace(g, "attn-norm", layer, cur);
  const bool has_compression = ratio != 0;
  const float theta = has_compression
                          ? c.dsv4_compress_rope_freq_base
                          : g.engine->model_file.get_config_float(
                                "rope.freq_base", 10000.0f);
  const float ext_factor = has_compression ? c.rope_ext_factor : 0.0f;
  const float freq_scale = has_compression ? c.rope_freq_scale : 1.0f;
  float attn_factor = has_compression ? c.rope_attn_factor : 1.0f;
  if (has_compression && ext_factor != 0.0f && freq_scale > 0.0f) {
    const float f = 1.0f + 0.1f * std::log(1.0f / freq_scale);
    attn_factor /= f * f;
  }
  const int n_ctx_orig = has_compression ? c.rope_n_ctx_orig : 0;
  ggml_tensor *q_lora = v41_matmul(ctx, wqa, cur);
  q_lora = ggml_mul(ctx, ggml_rms_norm(ctx, q_lora, c.eps), wqan);
  v41_mark_trace(g, "q-lora", layer, q_lora);
  ggml_tensor *q = ggml_reshape_3d(
      ctx, v41_matmul(ctx, wqb, q_lora), qh, g.H, g.S);
  q = v41_rope_tail(ctx, q, g.qk_nope, g.qk_rope, g.positions,
                    n_ctx_orig, theta, freq_scale, ext_factor, attn_factor);
  v41_mark_trace(g, "query", layer, q);

  ggml_tensor *raw = ggml_reshape_3d(
      ctx, v41_matmul(ctx, wkv, cur), qh, 1, g.S);
  raw = ggml_mul(ctx, ggml_rms_norm(ctx, raw, c.eps), wkvn);
  raw = v41_rope_tail(ctx, raw, g.qk_nope, g.qk_rope, g.positions,
                      n_ctx_orig, theta, freq_scale, ext_factor,
                      attn_factor);
  v41_mark_trace(g, "raw-kv", layer, raw);

  FullGraph::Dsv41LayerTensors lt;
  lt.layer = layer;
  lt.ratio = ratio;
  lt.kv_source = kv_source;
  lt.index_source = index_source;
  lt.raw_kv = raw;

  ggml_tensor *compressed = nullptr;
  int ncompressed = 0;
  if (ratio != 0 && kv_source == layer) {
    ggml_tensor *wk = v41_weight(g, ctx, p + "attn_compressor_kv.weight");
    ggml_tensor *wn = v41_weight(g, ctx, p + "attn_compressor_norm.weight");
    if (!wk || !wn)
      return nullptr;
    lt.comp_kv = v41_matmul(ctx, wk, cur);
    if (ratio == 1) {
      lt.comp_score = nullptr;
      compressed = v41_compress_latent(ctx, lt.comp_kv, nullptr, wn, g.S,
                                       ratio, qh, c.eps);
    } else {
      ggml_tensor *wg = v41_weight(
          g, ctx, p + "attn_compressor_gate.weight");
      if (!wg)
        return nullptr;
      lt.comp_score = v41_matmul(ctx, wg, cur);
      compressed = v41_compress_latent(ctx, lt.comp_kv, lt.comp_score, wn,
                                       g.S, ratio, qh, c.eps);
    }
    if (!compressed)
      return nullptr;
    const int blocks = g.S / ratio;
    ggml_tensor *block_pos = nullptr;
    int32_t *block_data = nullptr;
    block_pos = v41_host_i32(g, ctx, blocks,
                             "dsv41_graph_block_positions", &block_data);
    if (!block_pos)
      return nullptr;
    for (int b = 0; b < blocks; ++b)
      block_data[b] = b * ratio;
    compressed = v41_rope_tail(ctx,
                               ggml_reshape_3d(ctx, compressed, qh, 1, blocks),
                               g.qk_nope, g.qk_rope, block_pos,
                               n_ctx_orig, theta, freq_scale, ext_factor,
                               attn_factor);
    lt.keys = compressed;
    kv_keys[layer] = compressed;

    // The indexer consumes the pooled latent before compressed RoPE, and its
    // source weights are attached to the KV owner in the V4.1 GGUF.
    ggml_tensor *w_ik = v41_weight(g, ctx, p + "indexer.attn_k.weight");
    ggml_tensor *w_ikn = v41_weight(g, ctx, p + "indexer.k_norm.weight");
    if (!w_ik || !w_ikn)
      return nullptr;
    ggml_tensor *unrotated = v41_compress_latent(
        ctx, lt.comp_kv, lt.comp_score, wn, g.S, ratio, qh, c.eps);
    if (!unrotated)
      return nullptr;
    ggml_tensor *ik = v41_matmul(ctx, w_ik, unrotated);
    ik = ggml_mul(ctx, ggml_rms_norm(ctx, ik, c.eps), w_ikn);
    const int idim = c.dsv4_indexer_key_length;
    if (idim < g.qk_rope)
      return nullptr;
    ik = v41_rope_tail(ctx, ggml_reshape_3d(ctx, ik, idim, 1, blocks),
                       idim - g.qk_rope, g.qk_rope, block_pos,
                       n_ctx_orig, theta, freq_scale, ext_factor,
                       attn_factor);
    lt.indexer_keys = ik;
    index_keys[layer] = ik;
    ncompressed = blocks;
  } else if (ratio != 0) {
    auto it = kv_keys.find(kv_source);
    auto index_it = index_keys.find(kv_source);
    if (it == kv_keys.end() || index_it == index_keys.end())
      return nullptr;
    compressed = it->second;
    ncompressed = static_cast<int>(compressed->ne[2]);
  }

  ggml_tensor *sequence = raw;
  if (compressed)
    sequence = ggml_concat(
        ctx, sequence,
        ggml_reshape_3d(ctx, compressed, qh, 1, ncompressed), 2);
  const int key_count = g.S + ncompressed;
  const int padded = v41_prefill_f32() ? key_count : ((key_count + 255) / 256) * 256;
  if (padded > key_count) {
    float *pad_data = nullptr;
    ggml_tensor *pad = v41_host_f32(
        g, ctx, {qh, 1, padded - key_count},
        "dsv41_graph_attention_padding", &pad_data);
    if (!pad)
      return nullptr;
    std::fill(pad_data, pad_data + (size_t)qh * (padded - key_count), 0.0f);
    sequence = ggml_concat(ctx, sequence, pad, 2);
  }
  ggml_tensor *real = ggml_reshape_4d(ctx, sequence, qh, 1, padded, 1);
  float *mask_data = nullptr;
  ggml_tensor *mask = v41_host_f32(
      g, ctx, {padded, g.S, 1, 1}, "dsv41_graph_attention_mask", &mask_data);
  if (!mask)
    return nullptr;
  const int window = g.sliding_window > 0 ? g.sliding_window : g.S;
  for (int qpos = 0; qpos < g.S; ++qpos) {
    for (int kpos = 0; kpos < padded; ++kpos) {
      bool visible = false;
      if (kpos < g.S)
        visible = kpos <= qpos && kpos >= std::max(0, qpos - window + 1);
      else if (kpos < key_count)
        visible = (kpos - g.S) < (qpos + 1) / std::max(1, ratio);
      mask_data[(size_t)qpos * padded + kpos] = visible ? 0.0f : -INFINITY;
    }
  }
  // Above top-k, select compressed blocks on the GPU, sharing the owner's
  // selection with later layers exactly as the native V4.1 source map does.
  if (ncompressed > c.dsv4_indexer_top_k) {
    ggml_tensor *compressed_mask = nullptr;
    if (index_source == layer) {
      const int idim = c.dsv4_indexer_key_length;
      const int ih = c.dsv4_indexer_head_count;
      auto *wq = v41_weight(g, ctx, p + "indexer.attn_q_b.weight");
      auto *ww = v41_weight(g, ctx, p + "indexer.proj.weight");
      auto ik = index_keys.find(kv_source);
      if (!wq || !ww || ik == index_keys.end()) return nullptr;
      auto *iq = ggml_reshape_3d(ctx, v41_matmul(ctx, wq, q_lora),
                                  idim, ih, g.S);
      iq = v41_rope_tail(ctx, iq, idim - g.qk_rope, g.qk_rope,
                         g.positions, n_ctx_orig, theta, freq_scale,
                         ext_factor, attn_factor);
      auto *iw = ggml_scale(ctx, v41_matmul(ctx, ww, cur),
                           1.0f / std::sqrt(float(idim * ih)));
      auto *k = ggml_permute(ctx, ik->second, 0, 2, 1, 3);
      auto *score = v41_matmul(ctx, k, ggml_permute(ctx, iq, 0, 2, 1, 3));
      score = ggml_cont(ctx, ggml_permute(ctx, score, 2, 1, 0, 3));
      score = ggml_sum_rows(ctx, ggml_mul(ctx, ggml_relu(ctx, score), iw));
      score = ggml_cont_2d(ctx, ggml_permute(ctx, score, 2, 1, 0, 3),
                           ncompressed, g.S);
      float *causal_data = nullptr;
      auto *causal = v41_host_f32(g, ctx, {ncompressed, g.S},
                                  "dsv41_indexer_causal", &causal_data);
      if (!causal) return nullptr;
      for (int t = 0; t < g.S; ++t)
        for (int kpos = 0; kpos < ncompressed; ++kpos)
          causal_data[(size_t)t * ncompressed + kpos] =
              kpos < (t + 1) / ratio ? 0.0f : -INFINITY;
      score = ggml_add(ctx, score, causal);
      const int topk = c.dsv4_indexer_top_k;
      auto *ids = ggml_cont(ctx, ggml_top_k(ctx, score, topk));
      // Keep the top-k I32 indices. ggml_set_rows accepts I32 directly;
      // converting them to I64 creates a GGML_OP_CPY that the CPU backend
      // cannot execute when a long-prompt graph split places it on CPU.
      auto *all = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                   1, ncompressed, g.S), -INFINITY);
      auto *zeros = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                     1, topk, g.S), 0.0f);
      compressed_mask = ggml_set_rows(ctx, all, zeros, ids);
      compressed_mask = ggml_reshape_2d(ctx, compressed_mask, ncompressed, g.S);
      compressed_mask = ggml_add(ctx, compressed_mask, causal);
      index_masks[layer] = compressed_mask;
    } else {
      auto it = index_masks.find(index_source);
      if (it == index_masks.end()) return nullptr;
      compressed_mask = it->second;
    }
    auto *raw_mask = ggml_cont(ctx, ggml_view_2d(ctx, mask, g.S, g.S,
                                                mask->nb[1], 0));
    auto *combined = ggml_concat(ctx, raw_mask, compressed_mask, 0);
    if (padded > key_count) {
      auto *padding = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                       padded - key_count, g.S), -INFINITY);
      combined = ggml_concat(ctx, combined, padding, 0);
    }
    mask = combined;
  }
  auto *q_attn = ggml_permute(ctx, ggml_reshape_4d(ctx, q, qh, g.H, g.S, 1), 0, 2, 1, 3);
  auto *k_attn = ggml_permute(ctx, real, 0, 2, 1, 3);
  ggml_tensor *context = nullptr;
  if (v41_prefill_f32()) {
    // F32 QK, softmax with an explicit sink, and F32 weighted values. This
    // removes both Q8 activation conversion and F16 KV rounding from the
    // diagnostic GPU trunk while retaining complete graph scheduling.
    auto *scores = ggml_scale(ctx, v41_matmul(ctx, k_attn, q_attn),
                             1.0f / std::sqrt(float(qh)));
    scores = ggml_add(ctx, scores, mask);
    auto *sink = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, sinks, 1, 1, g.H),
                               1, g.S, g.H, 1);
    auto *prob = ggml_soft_max(ctx, ggml_concat(ctx, scores, sink, 0));
    auto *weights = ggml_view_3d(ctx, prob, padded, g.S, g.H,
                                 prob->nb[1], prob->nb[2], 0);
    context = v41_matmul(ctx, ggml_cont(ctx, ggml_transpose(ctx, k_attn)), weights);
    context = ggml_cont(ctx, ggml_permute(ctx, context, 0, 2, 1, 3));
  } else {
    auto *mask_f16 = ggml_cast(ctx, mask, GGML_TYPE_F16);
    auto *v_attn = k_attn;
    if (k_attn->type == GGML_TYPE_F32) k_attn = ggml_cast(ctx, k_attn, GGML_TYPE_F16);
    if (v_attn->type == GGML_TYPE_F32) v_attn = ggml_cast(ctx, v_attn, GGML_TYPE_F16);
    auto *attn = ggml_flash_attn_ext(ctx, q_attn, k_attn, v_attn, mask_f16,
                                    1.0f / std::sqrt(float(qh)), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(attn, sinks);
    context = ggml_cont(ctx, attn);
  }
  context = v41_rope_tail(ctx, context, g.qk_nope, g.qk_rope, g.positions,
                          n_ctx_orig, theta, freq_scale, ext_factor,
                          attn_factor, true);
  v41_mark_trace(g, "attn-ctx", layer, context);

  const int groups = std::max(1, c.o_group_count);
  const int rank = c.o_lora_rank;
  const int group_in = qh * g.H / groups;
  if (rank <= 0 || group_in * groups != qh * g.H ||
      woa->ne[0] != group_in || woa->ne[1] != (int64_t)(groups * rank))
    return nullptr;
  ggml_tensor *woa3 = ggml_reshape_3d(ctx, woa, group_in, rank, groups);
  ggml_tensor *grouped_input =
      ggml_reshape_3d(ctx, context, group_in, groups, g.S);
  grouped_input = ggml_permute(ctx, grouped_input, 0, 2, 1, 3);
  ggml_tensor *grouped = v41_matmul(ctx, woa3, grouped_input);
  grouped = ggml_permute(ctx, grouped, 0, 2, 1, 3);
  grouped = ggml_cont_2d(ctx, grouped, rank * groups, g.S);
  lt.raw_kv = raw;
  g.dsv41_layers.push_back(std::move(lt));
  return v41_matmul(ctx, wob, grouped);
}

// Bind the native cache pools directly to routed GGML matmuls. Only route
// tables and activations belong to the graph. CPU nodes resolve real cache
// misses; all gate/up/SwiGLU/down/weighting/sums execute on the owning GPU.
struct V41CachePool {
  int device = -1;
  ggml_backend_t backend = nullptr;
  ggml_backend_buffer_t marker = nullptr;
  uint8_t *base = nullptr;
  size_t stride = 0;
  int slots = 0;
};
struct V41CacheHold {
  GpuTopology *topo;
  int layer, expert;
  const uint8_t *ptr;
};
struct V41CacheMoeSite {
  FullGraph *g = nullptr;
  int layer = -1;
  const int *tokens = nullptr;
  std::vector<V41CachePool> pools;
  std::vector<V41CacheHold> holds;
  void release() {
    for (auto &h : holds) h.topo->release_hold(h.layer, h.expert, h.ptr, false);
    holds.clear();
  }
  ~V41CacheMoeSite() { release(); }
};
std::vector<std::unique_ptr<V41CacheMoeSite>> g_v41_cache_sites;

void v41_cache_routes_node(ggml_tensor *dst, const ggml_tensor *src,
                            int ith, int, void *ud) {
  if (ith) return;
  const auto start = std::chrono::steady_clock::now();
  auto &site = *static_cast<V41CacheMoeSite *>(ud);
  auto &g = *site.g;
  auto &e = *g.engine;
  const auto &c = e.model_file.architecture_config;
  RAMSafetyManager *ram = nullptr;
  LockFreePrefetcher *prefetch = nullptr;
  LayerScratch *scratch = nullptr;
  if (!deepseek_v41_graph_runtime(e, ram, prefetch, scratch) ||
      !ram->vram_cache || ram->vram_cache->expert_compact_layout)
    throw std::runtime_error("DeepSeek cache graph needs bound uniform VRAM pools");
  // This node consumes the next layer's GPU HC output, which depends on
  // every preceding expert kernel. The prior layer's holds can now retire.
  if (site.layer > 0) g_v41_cache_sites[(size_t)site.layer - 1]->release();
  auto *topo = ram->vram_cache;
  topo->prefill_current_layer = site.layer;
  // LayerScratch was sized for this batch before graph execution. Reuse its
  // routed-output scratch for the router's zeroing step; the graph computes
  // the actual MoE output separately on the GPUs.
  if (scratch->moe_out.size() < (size_t)g.D * g.S)
    throw std::runtime_error("DeepSeek cache graph MoE scratch is too small");
  MoePipelineContext context{e, *ram, *prefetch, site.layer,
      static_cast<const float *>(src->data), g.S, scratch->moe_out.data(), *scratch,
      site.tokens, e.model_file.architecture_config, g.D, c.topk, c.n_experts,
      c.moe_inter, 0.0f, 0.0f};
  if (MoeRouterService::route(context))
    throw std::runtime_error("Unexpected integrated router in DeepSeek cache graph");
  if (dsv41_trace::enabled(site.layer, 0)) {
    const size_t count = (size_t)g.S * c.topk;
    std::vector<float> ids(count);
    for (size_t i = 0; i < count; ++i)
      ids[i] = float(scratch->tok_topk_ids[i]);
    const size_t last = (size_t)(g.S - 1) * c.topk;
    dsv41_trace::save("route-ids", site.layer, 0, ids.data() + last, c.topk);
    dsv41_trace::save("route-weights", site.layer, 0,
                      scratch->tok_topk_weights.data() + last, c.topk);
    if (g.S > 1) {
      dsv41_trace::save("route-ids-seq", site.layer, 0, ids.data(), count);
      dsv41_trace::save("route-weights-seq", site.layer, 0,
                        scratch->tok_topk_weights.data(), count);
    }
  }
  // This graph reads experts from the main VRAM cache directly. The native
  // prefill warmup instead fills the separate active-layer pool and launches
  // speculative main-cache promotions, neither of which its GPU nodes use.
  // Demand-load the exact routed set below and avoid competing uploads.
  const auto routed = std::chrono::steady_clock::now();
  std::vector<int> owner((size_t)c.n_experts, -1), physical(owner.size(), -1);
  int misses = 0;
  for (int expert = 0; expert < c.n_experts; ++expert) {
    if (!scratch->expert_freq[(size_t)expert]) continue;
    auto *loc = topo->find_expert(site.layer, expert);
    if (!loc || !loc->device_ptr || !loc->upload_ready) {
      auto *block = ram->get_expert(e, site.layer, expert);
      if (!block || !block->slab_buffer)
        throw std::runtime_error("DeepSeek cache graph missing DDR expert");
      ram->ram_pin(*block);
      const size_t bytes = block->gate_proj.meta.size_bytes +
                           block->up_proj.meta.size_bytes + block->down_proj.meta.size_bytes;
      const int preferred = site.pools[(size_t)(misses++ % site.pools.size())].device;
      loc = topo->upload_expert(site.layer, expert, block->slab_buffer, bytes, preferred);
      if (!loc) loc = topo->upload_expert(site.layer, expert, block->slab_buffer, bytes);
      ram->ram_unpin(*block);
    }
    if (!loc || !loc->device_ptr || !loc->upload_ready ||
        !topo->hold_for_compute(site.layer, expert, loc->device_ptr, false))
      throw std::runtime_error("DeepSeek cache graph failed to hold VRAM expert");
    site.holds.push_back({topo, site.layer, expert, loc->device_ptr});
    for (size_t b = 0; b < site.pools.size(); ++b) {
      const auto &pool = site.pools[b];
      if (loc->gpu_id != pool.device) continue;
      const size_t offset = (size_t)(loc->device_ptr - pool.base);
      if (offset % pool.stride || offset / pool.stride >= (size_t)pool.slots)
        throw std::runtime_error("DeepSeek cache graph physical slot out of bounds");
      owner[(size_t)expert] = (int)b;
      physical[(size_t)expert] = (int)(offset / pool.stride);
    }
    if (owner[(size_t)expert] < 0)
      throw std::runtime_error("DeepSeek cache graph expert is outside bound pools");
  }
  // The native F32 kernel writes zero for a negative route ID without
  // reading an expert. The ordinary GGML path uses held dummy slots because
  // its older kernels require nonnegative IDs.
  const bool skip_masked_routes = v41_native_f32();
  std::vector<std::vector<int>> dummy(site.pools.size());
  for (size_t b = 0; !skip_masked_routes && b < site.pools.size(); ++b) {
    const auto &pool = site.pools[b];
    std::lock_guard<std::recursive_mutex> lock(topo->cache_mutex);
    for (const auto &loc : topo->expert_vram_cache) {
      if (loc.gpu_id == pool.device && loc.layer_idx >= 0 && loc.upload_ready &&
          topo->hold_for_compute(loc.layer_idx, loc.expert_id, loc.device_ptr, false)) {
        site.holds.push_back({topo, loc.layer_idx, loc.expert_id, loc.device_ptr});
        dummy[b].push_back(int((loc.device_ptr - pool.base) / pool.stride));
        if (dummy[b].size() == (size_t)c.topk) break;
      }
    }
    if (dummy[b].size() < (size_t)c.topk)
      throw std::runtime_error("DeepSeek cache graph needs top-k valid dummy slots per GPU");
  }
  const auto uploaded = std::chrono::steady_clock::now();
  auto *data = static_cast<float *>(dst->data);
  const int width = g.D + int(site.pools.size()) * 2 * c.topk;
  for (int t = 0; t < g.S; ++t) {
    float *row = data + (size_t)t * width;
    std::memcpy(row, scratch->normed.data() + (size_t)t * g.D, g.D * sizeof(float));
    for (size_t b = 0; b < site.pools.size(); ++b) {
      float *ids = row + g.D + b * 2 * c.topk;
      float *weights = ids + c.topk;
      // The ordinary backend receives distinct held dummy IDs for masked
      // routes. The native F32 backend skips their matmuls entirely.
      std::vector<int> used;
      for (int k = 0; k < c.topk; ++k) {
        const size_t flat = (size_t)t * c.topk + k;
        const int expert = scratch->tok_topk_ids[flat];
        if (expert < 0 || expert >= c.n_experts)
          throw std::runtime_error("DeepSeek cache graph has an invalid route");
        const bool selected = owner[(size_t)expert] == (int)b;
        ids[k] = selected ? float(physical[(size_t)expert]) : -1.0f;
        weights[k] = selected ? scratch->tok_topk_weights[flat] : 0.0f;
        if (selected && !skip_masked_routes) used.push_back(physical[(size_t)expert]);
      }
      for (int k = 0; !skip_masked_routes && k < c.topk; ++k) {
        if (ids[k] >= 0) continue;
        auto it = std::find_if(dummy[b].begin(), dummy[b].end(), [&](int id) {
          return std::find(used.begin(), used.end(), id) == used.end();
        });
        if (it == dummy[b].end()) throw std::runtime_error("DeepSeek dummy ID set exhausted");
        ids[k] = float(*it);
        used.push_back(*it);
      }
    }
  }
  // Preserve the native prefill-to-decode retention signal without cleaning
  // the unrelated active-layer pool while graph weights are in flight.
  ram->stamp_prefill_demand(e, site.layer, scratch->expert_freq);
  const auto done = std::chrono::steady_clock::now();
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  LOG_INFO("[DSV41-CACHE-MOE] layer=%d tokens=%d held=%zu misses=%d "
           "route_ms=%.3f upload_ms=%.3f payload_ms=%.3f "
           "graph-owned expert bytes=0\n",
           site.layer, g.S, site.holds.size(), misses,
           ms(start, routed), ms(routed, uploaded), ms(uploaded, done));
}

ggml_tensor *build_v41_cache_moe(FullGraph &g, ggml_tensor *cur, int layer,
                                 const int *token_ids) {
  const auto &c = g.engine->model_file.architecture_config;
  auto &topo = g.engine->gpu->topology;
  if (topo.expert_compact_layout || !topo.expert_slab_size || c.topk <= 0) {
    LOG_ERROR("[DSV41-CACHE-MOE] layer=%d unsupported pool layout compact=%d stride=%zu topk=%d\n",
              layer, topo.expert_compact_layout ? 1 : 0,
              topo.expert_slab_size, c.topk);
    return nullptr;
  }
  size_t gb = 0, ub = 0, db = 0;
  const GGUFTensor *gm = nullptr, *um = nullptr, *dm = nullptr;
  if (!g.engine->expert_geometry_for(layer, 0, gb, ub, db, gm, um, dm)) {
    LOG_ERROR("[DSV41-CACHE-MOE] layer=%d expert geometry unavailable\n", layer);
    return nullptr;
  }
  for (int expert = 1; expert < c.n_experts; ++expert) {
    size_t gx, ux, dx;
    const GGUFTensor *gt, *ut, *dt;
    if (!g.engine->expert_geometry_for(layer, expert, gx, ux, dx, gt, ut, dt) ||
        gx != gb || ux != ub || dx != db || gt->type != gm->type ||
        ut->type != um->type || dt->type != dm->type) {
      LOG_ERROR("[DSV41-CACHE-MOE] layer=%d expert %d has mixed geometry\n",
                layer, expert);
      return nullptr;
    }
  }
  for (int other = 0; other < c.n_layers; ++other) {
    size_t gx, ux, dx;
    const GGUFTensor *gt, *ut, *dt;
    if (!g.engine->expert_geometry_for(other, 0, gx, ux, dx, gt, ut, dt) ||
        gx != gb || ux != ub || dx != db || gt->type != gm->type ||
        ut->type != um->type || dt->type != dm->type) {
      LOG_ERROR("[DSV41-CACHE-MOE] mixed layer geometry is not supported by whole-pool views\n");
      return nullptr;
    }
  }
  auto site = std::make_unique<V41CacheMoeSite>();
  site->g = &g; site->layer = layer; site->tokens = token_ids;
  for (int vi : topo.vault_gpus) {
    auto &dev = topo.devices[(size_t)vi];
    if (!dev.expert_cache_pool || dev.expert_cache_size < topo.expert_slab_size) continue;
    ggml_backend_t backend = dev.id == g.device ? g.cuda : g.gpu1;
    if (!backend) {
      LOG_ERROR("[DSV41-CACHE-MOE] layer=%d missing backend for GPU %d\n",
                layer, dev.id);
      return nullptr;
    }
    auto marker = ggml_backend_alloc_buffer(backend, 256);
    if (!marker) return nullptr;
    g.gpu1_buffers.push_back(marker); // marker only; the topology owns the pool
    ggml_backend_buffer_set_usage(marker, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    site->pools.push_back({dev.id, backend, marker, dev.expert_cache_pool,
                          topo.expert_slab_size, int(dev.expert_cache_size / topo.expert_slab_size)});
  }
  if (site->pools.empty()) {
    LOG_ERROR("[DSV41-CACHE-MOE] layer=%d has no bindable GPU expert pool (vault GPUs=%zu)\n",
              layer, topo.vault_gpus.size());
    return nullptr;
  }
  auto *ptr = site.get();
  g_v41_cache_sites.push_back(std::move(site));
  const int width = g.D + int(ptr->pools.size()) * 2 * c.topk;
  auto *payload = ggml_custom_4d(g.ctx, GGML_TYPE_F32, width, g.S, 1, 1,
                                 &cur, 1, [](ggml_tensor *dst, int ith, int nth, void *ud) {
    v41_cache_routes_node(dst, dst->src[0], ith, nth, ud);
  }, 1, ptr);
  if (!payload) {
    LOG_ERROR("[DSV41-CACHE-MOE] layer=%d route payload node allocation failed\n",
              layer);
    return nullptr;
  }
  ggml_backend_sched_set_tensor_backend(g.sched, payload, g.cpu);
  auto *x = ggml_cont(g.ctx, ggml_view_2d(g.ctx, payload, g.D, g.S, payload->nb[1], 0));
  auto *x3 = ggml_reshape_3d(g.ctx, x, g.D, 1, g.S);
  ggml_tensor *sum = nullptr;
  for (size_t b = 0; b < ptr->pools.size(); ++b) {
    const auto &pool = ptr->pools[b];
    auto tensor = [&](const GGUFTensor *m, size_t offset, int ne0, int ne1) {
      auto *t = ggml_new_tensor_3d(g.ctx, (ggml_type)m->type, ne0, ne1, pool.slots);
      if (pool.stride % t->nb[1]) return (ggml_tensor *)nullptr;
      t->nb[2] = pool.stride; t->nb[3] = pool.stride * pool.slots;
      t->data = pool.base + offset; t->buffer = pool.marker;
      ggml_set_name(t, ("dsv41_cache_view_gpu" + std::to_string(pool.device)).c_str());
      return t;
    };
    moe_routed::ExpertWeights ew;
    ew.gate = tensor(gm, 0, g.D, c.moe_inter);
    ew.up = tensor(um, gb, g.D, c.moe_inter);
    ew.down = tensor(dm, gb + ub, c.moe_inter, g.D);
    if (!ew.valid()) {
      LOG_ERROR("[DSV41-CACHE-MOE] layer=%d expert weight view invalid on GPU %d\n",
                layer, pool.device);
      return nullptr;
    }
    const size_t offset = (size_t)(g.D + b * 2 * c.topk) * sizeof(float);
    auto *ids = ggml_cont(g.ctx, ggml_view_2d(g.ctx, payload, c.topk, g.S, payload->nb[1], offset));
    ids = ggml_cast(g.ctx, ids, GGML_TYPE_I32);
    auto *weights = ggml_cont(g.ctx, ggml_view_2d(g.ctx, payload, c.topk, g.S,
                          payload->nb[1], offset + c.topk * sizeof(float)));
    weights = ggml_reshape_3d(g.ctx, weights, 1, c.topk, g.S);
    const float clamp = layer < (int)c.swiglu_clamp_exp.size() ? c.swiglu_clamp_exp[layer] : 0.0f;
    auto ffn = moe_routed::expert_ffn(g.ctx, ew, x3, ids, clamp, false);
    if (!ffn.down) {
      LOG_ERROR("[DSV41-CACHE-MOE] layer=%d GPU %d expert FFN node construction failed\n",
                layer, pool.device);
      return nullptr;
    }
    if (v41_native_f32())
      for (auto *node : {ffn.gate, ffn.up, ffn.down})
        ggml_prec_set_src(node, GGML_PREC_F32, 1);
    auto *weighted = ggml_mul(g.ctx, ffn.down, weights);
    auto *branch = moe_routed::sum_topk(g.ctx, weighted, g.D, g.S, c.topk);
    ggml_backend_sched_set_tensor_backend(g.sched, weighted, pool.backend);
    ggml_backend_sched_set_tensor_backend(g.sched, branch, pool.backend);
    sum = sum ? ggml_add(g.ctx, sum, branch) : branch;
  }
  const std::string p = "blk." + std::to_string(layer) + ".";
  auto *sg = v41_weight(g, g.ctx, p + "ffn_gate_shexp.weight");
  auto *su = v41_weight(g, g.ctx, p + "ffn_up_shexp.weight");
  auto *sd = v41_weight(g, g.ctx, p + "ffn_down_shexp.weight");
  if (!sg || !su || !sd) return nullptr;
  auto *gate = v41_matmul(g.ctx, sg, x);
  auto *up = v41_matmul(g.ctx, su, x);
  auto *shared = v41_matmul(g.ctx, sd, ggml_swiglu_split(g.ctx, gate, up));
  return ggml_add(g.ctx, sum, shared);
}

bool build_v41_gpu_trunk(FullGraph &g, const int *token_ids) {
  const auto &c = g.engine->model_file.architecture_config;
  if (!init_dense_arena(g)) { LOG_ERROR("[DSV41-GPU-PREFILL] dense binding arena failed\n"); return false; }
  g_v41_prefill_sites.clear();
  g_v41_prefill_sites.reserve((size_t)g.n_layers);
  g_v41_trace_tensors.clear();
  g_v41_cache_sites.clear();
  g_v41_cache_sites.reserve((size_t)g.n_layers);
  std::unordered_map<int, ggml_tensor *> kv_keys, index_keys, index_masks;
  auto *res = g.input;
  // The native V4.1 contract carries the prior FFN gate into attention.
  // Its first collapse selects the embedding's stream zero.
  float *seed_data = nullptr;
  auto *carry = v41_host_f32(g, g.ctx, {g.HC, g.S}, "dsv41_hc_seed", &seed_data);
  if (!carry) return false;
  std::fill(seed_data, seed_data + (size_t)g.HC * g.S, 0.0f);
  for (int t = 0; t < g.S; ++t) seed_data[(size_t)t * g.HC] = 1.0f;
  for (int l = 0; l < g.n_layers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    g_v41_prefill_sites.push_back({g.engine, l, g.S, 0, token_ids});
    auto *site = &g_v41_prefill_sites.back();
    if (std::find(c.dsv41_engram_layer_ids.begin(), c.dsv41_engram_layer_ids.end(), l)
          != c.dsv41_engram_layer_ids.end()) {
      std::vector<float> rows;
      if (!deepseek41_engram_embedding_rows(*g.engine, l, g.S, 0, token_ids, rows)) return false;
      float *data = nullptr;
      auto *emb = v41_host_f32(g, g.ctx,
          {(int64_t)(rows.size() / g.S), g.S}, "dsv41_engram_rows", &data);
      if (!emb) return false;
      std::memcpy(data, rows.data(), rows.size() * sizeof(float));
      g.dsv41_engram_embeddings.emplace_back(l, emb);
      res = build_v41_engram(g, g.ctx, res, l, emb);
      if (!res) { LOG_ERROR("[DSV41-GPU-PREFILL] layer=%d Engram build failed\n", l); return false; }
    }
    auto hc = [&](const char *stem, ggml_tensor *input, ggml_tensor *collapse,
                  ggml_tensor **post, ggml_tensor **comb, ggml_tensor **pre) {
      auto *fn = v41_weight(g, g.ctx, p + stem + "_fn.weight");
      auto *scale = v41_weight(g, g.ctx, p + stem + "_scale.weight");
      auto *base = v41_weight(g, g.ctx, p + stem + "_base.weight");
      if (!fn || !scale || !base) return (ggml_tensor *)nullptr;
      if (v41_prefill_f32() && !v41_native_f32() && ggml_is_quantized(fn->type))
        fn = ggml_cast(g.ctx, fn, GGML_TYPE_F32);
      return hc_pre(g, g.ctx, input, fn, scale, base, post, comb, pre, collapse);
    };
    ggml_tensor *post = nullptr, *comb = nullptr, *attn_pre = nullptr;
    auto *cur = hc("hc_attn", res, carry, &post, &comb, &attn_pre);
    v41_mark_trace(g, "hc-attn-pre", l, cur);
    auto *norm = v41_weight(g, g.ctx, p + "attn_norm.weight");
    if (!cur || !norm) { LOG_ERROR("[DSV41-GPU-PREFILL] layer=%d HC/attention norm build failed\n", l); return false; }
    cur = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, cur, c.eps), norm);
    auto *attn = build_v41_attention(g, g.ctx, cur, l, kv_keys, index_keys, index_masks);
    if (!attn) { LOG_ERROR("[DSV41-GPU-PREFILL] layer=%d attention build failed\n", l); return false; }
    v41_mark_trace(g, "attn-out-projected", l, attn);
    res = hc_post(g, g.ctx, attn, res, post, comb);
    v41_mark_trace(g, "post-attn-streams", l, res);
    cur = hc("hc_ffn", res, attn_pre, &post, &comb, &carry);
    if (!cur) return false;
    v41_mark_trace(g, "hc-ffn-pre", l, cur);
    ggml_tensor *ffn = nullptr;
    if (moestream::graphcfg::flag_on("MOESTREAM_DEEPSEEK_V41_CACHE_MOE_GRAPH")) {
      ffn = build_v41_cache_moe(g, cur, l, token_ids);
    } else {
      ffn = ggml_map_custom1(g.ctx, cur, v41_moe_node, 1, site);
      if (ffn) ggml_backend_sched_set_tensor_backend(g.sched, ffn, g.cpu);
    }
    if (!ffn) { LOG_ERROR("[DSV41-GPU-PREFILL] layer=%d MoE build failed\n", l); return false; }
    v41_mark_trace(g, "moe-out", l, ffn);
    res = hc_post(g, g.ctx, ffn, res, post, comb);
    v41_mark_trace(g, "post-ffn-streams", l, res);
  }
  g.final_streams = res;
  g.dsv41_final_pre_mix = carry;
  g.output = ggml_dsv4_hc_pre(g.ctx, ggml_cont(g.ctx, res), ggml_cont(g.ctx, carry));
  return g.output != nullptr;
}

bool build_v41_graph(FullGraph &g, const int *token_ids) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_init_params ip = {};
  ip.mem_size = kTensorContextBytes;
  ip.no_alloc = true;
  if (!MemoryGuard::try_reserve_host_bytes(kTensorContextBytes))
    return false;
  g.context_reservation_bytes = kTensorContextBytes;
  g.ctx = ggml_init(ip);
  if (!g.ctx)
    return false;
  g.graph = ggml_new_graph_custom(g.ctx, kGraphSize, false);
  if (!g.graph)
    return false;

  g.input = host_input(g, g.ctx, GGML_TYPE_F32, {g.D, g.HC, g.S},
                       (size_t)g.D * g.HC * g.S * sizeof(float),
                       "dsv41_full_input");
  if (!g.input)
    return false;
  int32_t *pos_data = nullptr;
  g.positions = v41_host_i32(g, g.ctx, g.S, "dsv41_full_positions", &pos_data);
  if (!g.positions)
    return false;
  for (int s = 0; s < g.S; ++s)
    pos_data[s] = s;

  if (v41_gpu_prefill_enabled()) {
    if (!build_v41_gpu_trunk(g, token_ids)) return false;
  } else {
  g_v41_prefill_sites.clear();
  g_v41_hc_sites.clear();
  g_v41_prefill_sites.reserve((size_t)g.n_layers);
  g_v41_hc_sites.reserve((size_t)g.n_layers * 2);
  ggml_tensor *res = g.input;
  for (int l = 0; l < g.n_layers; ++l) {
    if (MemoryGuard::allocation_blocked())
      return false;
    g_v41_prefill_sites.push_back({g.engine, l, g.S, 0, token_ids});
    V41PrefillSite *site = &g_v41_prefill_sites.back();
    g_v41_hc_sites.push_back({g.engine, l, g.S, 0, 0});
    V41HcSite *attn_hc_site = &g_v41_hc_sites.back();
    g_v41_hc_sites.push_back({g.engine, l, g.S, 0, 1});
    V41HcSite *ffn_hc_site = &g_v41_hc_sites.back();
    if (std::find(c.dsv41_engram_layer_ids.begin(),
                  c.dsv41_engram_layer_ids.end(), l) !=
        c.dsv41_engram_layer_ids.end()) {
      auto *engram = ggml_map_custom1(g.ctx, res, v41_engram_node, 1, site);
      if (!engram)
        return false;
      ggml_backend_sched_set_tensor_backend(g.sched, engram, g.cpu);
      res = engram;
    }

    ggml_tensor *attn_args[] = {res};
    ggml_tensor *cur = ggml_custom_4d(
        g.ctx, GGML_TYPE_F32, g.D, g.S, 1, 1, attn_args, 1,
        v41_hc_pre_node, 1, attn_hc_site);
    if (!cur)
      return false;
    ggml_backend_sched_set_tensor_backend(g.sched, cur, g.cpu);
    ggml_tensor *attn = ggml_map_custom1(
        g.ctx, cur, v41_attention_node, 1, site);
    if (!attn)
      return false;
    ggml_backend_sched_set_tensor_backend(g.sched, attn, g.cpu);
    v41_keep_tensor(g, attn);
    ggml_tensor *post_attn_args[] = {attn, res};
    auto *post_attn = ggml_custom_4d(
        g.ctx, GGML_TYPE_F32, g.D, g.HC, g.S, 1, post_attn_args, 2,
        v41_hc_post_node, 1, attn_hc_site);
    if (!post_attn)
      return false;
    ggml_backend_sched_set_tensor_backend(g.sched, post_attn, g.cpu);
    res = post_attn;

    ggml_tensor *ffn_args[] = {res};
    cur = ggml_custom_4d(g.ctx, GGML_TYPE_F32, g.D, g.S, 1, 1, ffn_args, 1,
                         v41_hc_pre_node, 1, ffn_hc_site);
    if (!cur)
      return false;
    ggml_backend_sched_set_tensor_backend(g.sched, cur, g.cpu);
    auto *ffn = ggml_map_custom1(g.ctx, cur, v41_moe_node, 1,
                                 site);
    if (!ffn)
      return false;
    v41_keep_tensor(g, ffn);
    ggml_backend_sched_set_tensor_backend(g.sched, ffn, g.cpu);
    ggml_tensor *post_ffn_args[] = {ffn, res};
    auto *post_ffn = ggml_custom_4d(
        g.ctx, GGML_TYPE_F32, g.D, g.HC, g.S, 1, post_ffn_args, 2,
        v41_hc_post_node, 1, ffn_hc_site);
    if (!post_ffn)
      return false;
    ggml_backend_sched_set_tensor_backend(g.sched, post_ffn, g.cpu);
    res = post_ffn;
  }
  g.final_streams = res;
  g.output = ggml_cont(
      g.ctx, ggml_view_2d(g.ctx, res, g.D, g.S, res->nb[2], 0));
  }
  ggml_set_output(g.final_streams);
  ggml_build_forward_expand(g.graph, g.final_streams);
  ggml_set_output(g.output);
  ggml_build_forward_expand(g.graph, g.output);
  if (g.dsv41_final_pre_mix) {
    ggml_set_output(g.dsv41_final_pre_mix);
    ggml_build_forward_expand(g.graph, g.dsv41_final_pre_mix);
  }
  for (const auto &lt : g.dsv41_layers) {
    ggml_set_output(lt.raw_kv);
    ggml_build_forward_expand(g.graph, lt.raw_kv);
    if (lt.comp_kv) {
      ggml_set_output(lt.comp_kv);
      ggml_build_forward_expand(g.graph, lt.comp_kv);
      if (lt.comp_score) {
        ggml_set_output(lt.comp_score);
        ggml_build_forward_expand(g.graph, lt.comp_score);
      }
      ggml_set_output(lt.keys);
      ggml_build_forward_expand(g.graph, lt.keys);
      ggml_set_output(lt.indexer_keys);
      ggml_build_forward_expand(g.graph, lt.indexer_keys);
    }
  }
  if (!graph_alloc_vram_headroom_ok(g, "[GGML-DSV41-FULL]"))
    return false;
  if (!ggml_backend_sched_alloc_graph(g.sched, g.graph))
    return false;
  if (g.expert.buffer || g.expert.capacity || g.expert_source_base ||
      g.expert_source_capacity || !g.expert_source_views.empty()) {
    LOG_ERROR("[GGML-DSV41-FULL] refusing graph with graph-owned MoE storage\n");
    return false;
  }
  g.graph_workspace_bytes = ggml_backend_sched_get_buffer_size(g.sched, g.cuda);
  LOG_INFO("[GGML-DSV41-PREFILL] gpu_trunk=%d tokens=%d dense_attention_f32=%d "
           "cache_owned_experts=1\n", int(v41_gpu_prefill_enabled()), g.S,
           int(v41_prefill_f32()));
  LOG_INFO("[GGML-DSV41-FULL] graph built layers=%d nodes=%d splits=%d "
           "workspace=%.2f GiB host_moe_callbacks=%d "
           "engram_tensor_inputs=%zu native_engram_layers=%zu "
           "graph-owned MoE bytes=0 "
           "(existing DDR/VRAM cache dispatcher)\n",
           g.n_layers, ggml_graph_n_nodes(g.graph),
           ggml_backend_sched_get_n_splits(g.sched),
           (double)g.graph_workspace_bytes / (1024.0 * 1024.0 * 1024.0),
           moestream::graphcfg::flag_on("MOESTREAM_DEEPSEEK_V41_CACHE_MOE_GRAPH")
               && v41_gpu_prefill_enabled() ? 0 : g.n_layers,
           g.dsv41_engram_embeddings.size(),
           std::count_if(c.dsv41_engram_layer_ids.begin(),
                         c.dsv41_engram_layer_ids.end(),
                         [&](int64_t l) { return l >= 0 && l < g.n_layers; }));
  g.ready = true;
  return true;
}

} // namespace

void v41_cache_moe_sites_reset() {
  for (auto &site : g_v41_cache_sites)
    if (site)
      site->release();
  g_v41_cache_sites.clear();
}

void v41_cache_moe_sites_release_holds() {
  for (auto &site : g_v41_cache_sites)
    if (site)
      site->release();
}

ggml_tensor *v41_build_cache_moe(FullGraph &g, ggml_tensor *cur, int layer,
                                 const int *token_ids) {
  return build_v41_cache_moe(g, cur, layer, token_ids);
}

} // namespace glm5next_graph

using namespace glm5next_graph;

bool deepseek_v41_full_graph_enabled() { return v41_env_enabled(); }

int deepseek_v41_full_graph_device_expert_layers() {
  // The cache owns all expert storage; reserve no graph preload arena.
  return 0;
}

void deepseek_v41_full_graph_shutdown() {
  deepseek_v41_full_graph_decode_shutdown();
  g_dsv41_graph.reset();
}

bool deepseek_v41_full_graph_prepare(StreamingEngine &engine, int S,
                                     int pos_base, const int *token_ids) {
  if (!v41_env_enabled() || S <= 1 || pos_base != 0 || !token_ids ||
      engine.family_ops.family != ModelFamily::DeepSeek_V41)
    return false;
  const auto &c = engine.model_file.architecture_config;
  if (c.first_dense != 0 || c.hc_mult <= 1 ||
      c.dsv4_compress_ratios.size() < (size_t)c.n_layers)
    return false;
  if (g_dsv41_graph && g_dsv41_graph->ready &&
      g_dsv41_graph->engine == &engine && g_dsv41_graph->S == S) {
    for (auto &site : g_v41_prefill_sites)
      site.token_ids = token_ids;
    for (const auto &[layer, tensor] : g_dsv41_graph->dsv41_engram_embeddings) {
      std::vector<float> rows;
      if (!deepseek41_engram_embedding_rows(engine, layer, S, pos_base,
                                            token_ids, rows) ||
          !tensor || rows.size() * sizeof(float) != ggml_nbytes(tensor))
        return false;
      std::memcpy(tensor->data, rows.data(), rows.size() * sizeof(float));
    }
    return true;
  }
  g_dsv41_graph.reset();
  g_v41_prefill_sites.clear();

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
  g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(g->device));
  g->cpu = ggml_backend_cpu_init();
  if (g->cpu) {
    const char *threads =
        moestream::graphcfg::env("MOESTREAM_DEEPSEEK_V41_CPU_THREADS");
    ggml_backend_cpu_set_n_threads(g->cpu,
                                   threads ? std::max(1, std::atoi(threads)) : 6);
  }
  if (!g->cuda || !g->cpu)
    return false;
  if (v41_native_f32() && !moestream_install_f32_quant_graph_hook()) {
    engine.qwen4exp_full_graph_failure = "Native F32 graph needs the optional GGML CUDA F32 quant hook";
    LOG_ERROR("[DSV41-GPU-PREFILL] %s\n", engine.qwen4exp_full_graph_failure.c_str());
    return false;
  }

  // GPU graph nodes borrow native dense/cache device pointers. The host
  // callback path remains available as the numerical reference.
  std::vector<ggml_backend_t> backends{g->cuda};
  if (v41_gpu_prefill_enabled() &&
      moestream::graphcfg::flag_on("MOESTREAM_DEEPSEEK_V41_CACHE_MOE_GRAPH")) {
    for (int vi : engine.gpu->topology.vault_gpus) {
      const auto &dev = engine.gpu->topology.devices[(size_t)vi];
      if (dev.id != g->device && dev.expert_cache_pool) {
        if (g->gpu1) return false; // this bridge supports the requested two GPUs
        g->gpu1 = static_cast<ggml_backend_t>(ggml_shared_backend(dev.id));
        if (!g->gpu1) return false;
        backends.push_back(g->gpu1);
      }
    }
  }
  backends.push_back(g->cpu);
  g->sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(),
                                    kGraphSize, false, true);
  if (!g->sched)
    return false;

  if (!build_v41_graph(*g, token_ids)) {
    if (v41_gpu_prefill_enabled()) {
      engine.qwen4exp_full_graph_failure = "DeepSeek GPU prefill graph construction failed";
      LOG_ERROR("[DSV41-GPU-PREFILL] %s; request stopped\n",
                engine.qwen4exp_full_graph_failure.c_str());
    }
    return false;
  }
  g_dsv41_graph = std::move(g);
  LOG_INFO("[GGML-DSV41-FULL] graph prepared for prompt S=%d\n", S);
  return true;
}

bool deepseek_v41_full_graph_run(StreamingEngine &engine, float *hidden_states,
                                 int S, int pos_base, const int *) {
  if (!v41_env_enabled() || !hidden_states || S <= 1 || pos_base != 0 ||
      !g_dsv41_graph || !g_dsv41_graph->ready ||
      g_dsv41_graph->engine != &engine || g_dsv41_graph->S != S)
    return false;
  std::unique_ptr<FullGraph> graph = std::move(g_dsv41_graph);
  FullGraph &g = *graph;
  struct FailureGuard {
    StreamingEngine &engine;
    bool done = false;
    ~FailureGuard() {
      if (!done && v41_gpu_prefill_enabled())
        engine.qwen4exp_full_graph_failure = "DeepSeek GPU prefill graph execution failed";
    }
  } failure{engine};
  struct PrefillSiteCleanup {
    ~PrefillSiteCleanup() {
      // Also runs on a compute exception: never release weights while a
      // CUDA split might still be reading them.
      if (!g_v41_cache_sites.empty()) {
        auto *g = g_v41_cache_sites.front()->g;
        ggml_backend_sched_synchronize(g->sched);
        g_v41_cache_sites.clear();
      }
      g_v41_prefill_sites.clear();
      g_v41_hc_sites.clear();
    }
  } cleanup_sites;
  if (g.expert.buffer || g.expert.capacity || g.expert_source_base ||
      g.expert_source_capacity || !g.expert_source_views.empty()) {
    LOG_ERROR("[GGML-DSV41-FULL] refusing run: graph-owned MoE storage found\n");
    return false;
  }
  if (engine.hc_stream_buf.size() < (size_t)g.D * g.HC * S ||
      g.host_inputs.empty())
    return false;
  std::memcpy(g.host_inputs[0], engine.hc_stream_buf.data(),
              (size_t)g.D * g.HC * S * sizeof(float));
  // Use the same live physical-memory floor as host allocations. The former
  // graph-only 8 GiB check admitted compute after the host had lost most of
  // the headroom required to survive large pinned DDR pools.
  const long kMinAvailableMiB = MemoryGuard::stop_limit_mb();
  const long avail_mb = MemoryGuard::read_available_mb();
  if (avail_mb >= 0 && avail_mb < kMinAvailableMiB) {
    LOG_ERROR("[GGML-DSV41-FULL] refusing graph compute: MemAvailable=%ld MiB "
              "< %ld MiB\n",
              avail_mb, kMinAvailableMiB);
    return false;
  }
  LOG_INFO("[GGML-DSV41-FULL] compute start S=%d nodes=%d\n", S,
           ggml_graph_n_nodes(g.graph));
  const ggml_status graph_status =
      ggml_backend_sched_graph_compute_async(g.sched, g.graph);
  ggml_backend_sched_synchronize(g.sched);
  LOG_INFO("[GGML-DSV41-FULL] compute complete status=%d\n",
           (int)graph_status);
  if (graph_status != GGML_STATUS_SUCCESS)
    return false;
  for (const auto &trace : g_v41_trace_tensors) {
    std::vector<float> values((size_t)ggml_nelements(trace.tensor));
    ggml_backend_tensor_get(trace.tensor, values.data(), 0, values.size() * sizeof(float));
    const size_t row = values.size() / (size_t)S;
    dsv41_trace::save(trace.label, trace.layer, 0, values.data() + (S - 1) * row, row);
    const std::string seq_label = std::string(trace.label) + "-seq";
    dsv41_trace::save(seq_label.c_str(), trace.layer, 0, values.data(), values.size());
  }

  ggml_backend_tensor_get(g.final_streams, engine.hc_stream_buf.data(), 0,
                          (size_t)g.D * g.HC * S * sizeof(float));
  if (g.dsv41_final_pre_mix) {
    const int qh = g.qk_nope + g.qk_rope;
    const int window = engine.sliding_window > 0 ? engine.sliding_window : engine.max_t;
    for (auto &state : engine.dsv41_compressed) state.reset();
    for (const auto &lt : g.dsv41_layers) {
      std::vector<float> raw((size_t)S * qh);
      ggml_backend_tensor_get(lt.raw_kv, raw.data(), 0, raw.size() * sizeof(float));
      for (int t = std::max(0, S - window); t < S; ++t)
        std::memcpy(engine.KVc[(size_t)lt.layer] + (size_t)(t % window) * qh,
                    raw.data() + (size_t)t * qh, (size_t)qh * sizeof(float));
      if (!lt.keys) continue;
      auto &state = engine.dsv41_compressed[(size_t)lt.layer];
      const int blocks = S / lt.ratio;
      if ((size_t)blocks * qh > state.keys.size() ||
          ggml_nbytes(lt.indexer_keys) > state.indexer_keys.size() * sizeof(float)) return false;
      ggml_backend_tensor_get(lt.keys, state.keys.data(), 0, (size_t)blocks * qh * sizeof(float));
      ggml_backend_tensor_get(lt.indexer_keys, state.indexer_keys.data(), 0, ggml_nbytes(lt.indexer_keys));
      const int tail = S % lt.ratio;
      state.pending_count = tail;
      if (tail) {
        const size_t offset = (size_t)(S - tail) * qh * sizeof(float);
        ggml_backend_tensor_get(lt.comp_kv, state.pending_kv.data(), offset,
                                (size_t)tail * qh * sizeof(float));
        if (lt.comp_score)
          ggml_backend_tensor_get(lt.comp_score, state.pending_score.data(), offset,
                                  (size_t)tail * qh * sizeof(float));
      }
    }
    engine.hc_final_pre_mix.resize((size_t)S * g.HC);
    ggml_backend_tensor_get(g.dsv41_final_pre_mix, engine.hc_final_pre_mix.data(), 0,
                            engine.hc_final_pre_mix.size() * sizeof(float));
    ggml_backend_tensor_get(g.output, hidden_states, 0, (size_t)S * g.D * sizeof(float));
    engine.deepseek_v4_full_graph_output_ready = true;
    failure.done = true;
    return true;
  }
  const size_t final_mix_n = (size_t)S * g.HC;
  if (engine.hc_final_pre_mix.size() != final_mix_n)
    return false;
  for (int s = 0; s < S; ++s) {
    float *out = hidden_states + (size_t)s * g.D;
    const float *streams = engine.hc_stream_buf.data() +
                           (size_t)s * g.HC * g.D;
    const float *mix = engine.hc_final_pre_mix.data() + (size_t)s * g.HC;
    std::fill(out, out + g.D, 0.0f);
    for (int h = 0; h < g.HC; ++h)
      for (int d = 0; d < g.D; ++d)
        out[d] += mix[h] * streams[(size_t)h * g.D + d];
  }
  engine.deepseek_v4_full_graph_output_ready = true;
  failure.done = true;
  return true;
}

#else

bool deepseek_v41_full_graph_enabled() { return false; }
int deepseek_v41_full_graph_device_expert_layers() { return 0; }
bool deepseek_v41_full_graph_prepare(StreamingEngine &, int, int, const int *) {
  return false;
}
bool deepseek_v41_full_graph_run(StreamingEngine &, float *, int, int,
                                 const int *) {
  return false;
}
void deepseek_v41_full_graph_shutdown() {}

#endif
