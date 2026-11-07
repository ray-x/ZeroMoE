// DeepSeek-V4.1 whole-prompt graph. The GPU prefill path uses GGML nodes
// for Engram projections, HC and compressed attention. Routed experts remain
// owned by the native DDR/VRAM caches in both execution paths.

#include "glm5next_graph_internal.hpp"
#include "graph_settings_global.hpp"
#include "dsv41_engram.hpp"
#include "ggml_deepseek_v41_graph.hpp"
#include "ggml_glm5next_graph.hpp"
#include "dsv41_trace.hpp"
#include "deepseek_cache_policy.hpp"
#include "dense_dispatch.hpp"
#include "ggml_f32_quant_backend.hpp"
#include "ram_manager.hpp"
#include "async_prefetcher.hpp"
#include "scratch_buffers.hpp"
#include "moe_pipeline_internal.hpp"
#include "moe_router_service.hpp"
#include "mmid_route_skip.hpp"
#include "vram_dynamic_slot_plan.hpp"
#include "prefill_warmup_manager.hpp"
#include "cpu_expert_executor.hpp"
#include <algorithm>
#include <omp.h>
#include <thread>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <future>
#include <stdexcept>
#include <unordered_set>

#if defined(ZEROMOE_USE_GGML_CUDA)

extern "C" cudaStream_t
zeromoe_ggml_backend_cuda_get_stream(ggml_backend_t backend);

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
  return zeromoe::graphsettings::active_settings().tun.gpu_prefill.value != 0;
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

bool v41_full_graph_enabled() {
  // The graph is the default execution path for DeepSeek-V4.1.  The separate
  // ZEROMOE_DEEPSEEK_V41_FULL_GRAPH_DECODE=0 opt-out is handled by the decode
  // adapter; prefill always uses the graph.
  return true;
}

// Diagnostic precision mode: retain F32 activations and attention values.
// Cast only dense tensors into reusable graph scratch; expert storage remains
// the original quantized native cache, with no graph expert allocation.
bool v41_native_f32() {
  return zeromoe::graphsettings::active_settings().tun.native_f32_graph.value != 0;
}
bool v41_prefill_f32() { return v41_native_f32(); }
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
  ggml_tensor *wkv = v41_weight(g, ctx, p + "engram_kv.weight");
  // The 1017 GGUF stores q_weight * k_weight as engram_gate.weight.
  ggml_tensor *gate_weight = v41_weight(g, ctx, p + "engram_gate.weight");
  if (!wkv || !gate_weight)
    return nullptr;

  ggml_tensor *kv = v41_matmul(ctx, wkv, embedding);
  ggml_tensor *key = ggml_view_3d(
      ctx, kv, g.D, g.HC, g.S, (size_t)g.D * sizeof(float), kv->nb[1], 0);
  ggml_tensor *value = ggml_view_2d(
      ctx, kv, g.D, g.S, kv->nb[1], (size_t)g.HC * g.D * kv->nb[0]);
  // This checkpoint pre-fuses the reference q_weight * k_weight product in
  // engram_gate.weight. Cast that small tensor to F32 before the gate math.
  ggml_tensor *gate_weight_f32 = ggml_reshape_3d(
      ctx, ggml_cast(ctx, gate_weight, GGML_TYPE_F32), g.D, g.HC, 1);
  ggml_tensor *qk =
      ggml_repeat_4d(ctx, gate_weight_f32, g.D, g.HC, g.S, 1);
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
  const int padded = ((key_count + 255) / 256) * 256;
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

    auto *mask_f16 = ggml_cast(ctx, mask, GGML_TYPE_F16);
    auto *v_attn = k_attn;
    if (k_attn->type == GGML_TYPE_F32) k_attn = ggml_cast(ctx, k_attn, GGML_TYPE_F16);
    if (v_attn->type == GGML_TYPE_F32) v_attn = ggml_cast(ctx, v_attn, GGML_TYPE_F16);
    auto *attn = ggml_flash_attn_ext(ctx, q_attn, k_attn, v_attn, mask_f16,
                                    1.0f / std::sqrt(float(qh)), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(attn, sinks);
    context = ggml_cont(ctx, attn);
  
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
  bool shared_dynamic = false;
};

static bool v41_cache_pool_slot(const V41CachePool &pool, const uint8_t *ptr,
                                int &physical) {
  if (!pool.base || !ptr || pool.stride == 0 || pool.slots <= 0 ||
      pool.stride > SIZE_MAX / (size_t)pool.slots)
    return false;
  const uintptr_t base = reinterpret_cast<uintptr_t>(pool.base);
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  if (address < base) return false;
  const size_t offset = (size_t)(address - base);
  const size_t span = pool.stride * (size_t)pool.slots;
  if (offset >= span || offset % pool.stride) return false;
  const size_t slot = offset / pool.stride;
  if (slot >= (size_t)pool.slots || slot > (size_t)std::numeric_limits<int>::max())
    return false;
  physical = (int)slot;
  return true;
}

static bool v41_cache_pool_matches_location(const V41CachePool &pool,
                                            const ExpertLocation &loc,
                                            int layer, bool compact) {
  if (loc.gpu_id != pool.device) return false;
  if (compact &&
      (pool.shared_dynamic ? loc.pool_layer != -1 : loc.pool_layer != layer))
    return false;
  int physical = -1;
  return v41_cache_pool_slot(pool, loc.device_ptr, physical);
}

struct V41CacheHold {
  GpuTopology *topo;
  int layer, expert;
  const uint8_t *ptr;
};
struct V41CacheMoeSite {
  FullGraph *g = nullptr;
  int layer = -1;
  const int *tokens = nullptr;
  const int *decode_pos = nullptr;
  bool cpu_cold_graph = false;
  bool cpu_cold_prefill = false;
  std::vector<V41CachePool> pools;
  std::vector<V41CacheHold> holds;
  std::future<void> cpu_job;
  std::vector<int> cpu_experts;
  std::vector<float> cpu_route_weights;
  std::vector<float> cpu_input;
  std::vector<float> cpu_output;
  float *cpu_output_data = nullptr;
  size_t cpu_output_elements = 0;
  double cpu_job_ms = 0.0;
  size_t cpu_ggml_grouped_experts = 0;
  size_t cpu_quant_grouped_experts = 0;
  size_t cpu_scalar_experts = 0;
  void release() {
    for (auto &h : holds) h.topo->release_hold(h.layer, h.expert, h.ptr, false);
    holds.clear();
  }
  ~V41CacheMoeSite() { release(); }
};
// Prefill and decode graphs own separate site lists: the decode graph is shut
// down and rebuilt around each prefill, and must not free the sites of a
// prefill graph that was prepared before it.
std::vector<std::unique_ptr<V41CacheMoeSite>> g_v41_cache_sites;
std::vector<std::unique_ptr<V41CacheMoeSite>> g_v41_decode_cache_sites;
std::vector<std::unique_ptr<V41CacheMoeSite>> &v41_sites_for(const FullGraph &g) {
  return g.decode_mode ? g_v41_decode_cache_sites : g_v41_cache_sites;
}

void v41_cache_routes_node(ggml_tensor *dst, const ggml_tensor *src,
                            int ith, int, void *ud) {
  if (ith) return;
  const auto start = std::chrono::steady_clock::now();
  auto *site_ptr = static_cast<V41CacheMoeSite *>(ud);
  auto &site = *site_ptr;
  auto &g = *site.g;
  auto &e = *g.engine;
  const auto &c = e.model_file.architecture_config;
  if (site.cpu_job.valid()) site.cpu_job.get();
  RAMSafetyManager *ram = nullptr;
  LockFreePrefetcher *prefetch = nullptr;
  LayerScratch *scratch = nullptr;
  if (!deepseek_v41_graph_runtime(e, ram, prefetch, scratch) ||
      !ram->vram_cache ||
      (!ram->vram_cache->expert_compact_layout &&
       !ram->vram_cache->expert_slab_size))
    throw std::runtime_error("DeepSeek cache graph needs bound VRAM slot pools");
  // This node consumes the next layer's GPU HC output, which depends on
  // every preceding expert kernel. The prior layer's holds can now retire.
  if (site.layer > 0) v41_sites_for(g)[(size_t)site.layer - 1]->release();
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
  const int trace_pos = site.decode_pos ? *site.decode_pos : 0;
  if (dsv41_trace::enabled(site.layer, trace_pos)) {
    const size_t count = (size_t)g.S * c.topk;
    std::vector<float> ids(count);
    for (size_t i = 0; i < count; ++i)
      ids[i] = float(scratch->tok_topk_ids[i]);
    const size_t last = (size_t)(g.S - 1) * c.topk;
    dsv41_trace::save("route-ids", site.layer, trace_pos, ids.data() + last, c.topk);
    dsv41_trace::save("route-weights", site.layer, trace_pos,
                      scratch->tok_topk_weights.data() + last, c.topk);
    if (g.S > 1) {
      dsv41_trace::save("route-ids-seq", site.layer, trace_pos, ids.data(), count);
      dsv41_trace::save("route-weights-seq", site.layer, trace_pos,
                        scratch->tok_topk_weights.data(), count);
    }
  }
  // This graph reads experts from the main VRAM cache directly. The native
  // prefill warmup instead fills the separate active-layer pool and launches
  // speculative main-cache promotions, neither of which its GPU nodes use.
  // Stage the routed GPU subset here; when enabled, DDR CPU work handles any
  // prefill overflow that cannot obtain a safe cache victim.
  const auto routed = std::chrono::steady_clock::now();
  std::vector<int> owner((size_t)c.n_experts, -1), physical(owner.size(), -1);
  struct RamPinGuard {
    RAMSafetyManager &ram;
    std::vector<MoEExpertBlock *> blocks;
    RamPinGuard(RAMSafetyManager &r, size_t n) : ram(r), blocks(n, nullptr) {}
    RamPinGuard(RamPinGuard &&other) noexcept
        : ram(other.ram), blocks(std::move(other.blocks)) {
      other.blocks.clear();
    }
    RamPinGuard(const RamPinGuard &) = delete;
    RamPinGuard &operator=(const RamPinGuard &) = delete;
    ~RamPinGuard() {
      for (auto *block : blocks)
        if (block) ram.ram_unpin(*block);
    }
    void release(size_t expert) {
      if (blocks[expert]) {
        ram.ram_unpin(*blocks[expert]);
        blocks[expert] = nullptr;
      }
    }
    MoEExpertBlock *take(size_t expert) {
      MoEExpertBlock *block = blocks[expert];
      blocks[expert] = nullptr;
      return block;
    }
  } ram_pins(*ram, (size_t)c.n_experts);
  RamPinGuard cpu_spill_pins(*ram, (size_t)c.n_experts);
  const bool async_upload = !site.cpu_cold_prefill &&
      zeromoe::graphsettings::active_settings().tun.cache_moe_async_upload.value != 0;
  const bool prefer_bounce_upload = async_upload &&
      zeromoe::graphsettings::active_settings().tun.cache_moe_async_bounce_upload.value != 0;
  std::vector<ExpertLocation *> pending_locations((size_t)c.n_experts, nullptr);
  std::vector<ExpertLocation *> held_locations((size_t)c.n_experts, nullptr);
  site.cpu_experts.clear();
  site.cpu_route_weights.clear();
  if (site.cpu_cold_graph) {
    if (site.cpu_cold_prefill) {
      // The router's native output scratch is no longer needed by this graph;
      // reuse it for the asynchronous DDR contribution instead of retaining
      // another D x prompt buffer for every layer site.
      site.cpu_output_data = scratch->moe_out.data();
      site.cpu_output_elements = (size_t)g.D * g.S;
      std::fill(site.cpu_output_data,
                site.cpu_output_data + (size_t)g.D * g.S, 0.0f);
    } else {
      if (site.cpu_output.size() != (size_t)g.D * g.S)
        site.cpu_output.resize((size_t)g.D * g.S);
      std::fill(site.cpu_output.begin(), site.cpu_output.end(), 0.0f);
      const float *normed = scratch->normed.data();
      site.cpu_input.assign(normed, normed + (size_t)g.D * g.S);
      site.cpu_output_data = site.cpu_output.data();
      site.cpu_output_elements = site.cpu_output.size();
    }
  }
  int misses = 0;
  int gpu_misses = 0;
  size_t decode_ddr_miss_ordinal = 0;
  int staged_misses = 0;
  // CPU-selected DDR misses have no VRAM location by design. Keep them out
  // of the async-upload completion validation below; their pinned DDR blocks
  // are consumed by the CPU lane and merged into the graph output separately.
  std::vector<uint8_t> cpu_routed((size_t)c.n_experts, 0);
  std::vector<MoEExpertBlock *> batch_blocks((size_t)c.n_experts, nullptr);
  std::vector<int> load_ids;
  if (site.cpu_cold_prefill) {
    // Protect all routed cache hits before uploading misses. Otherwise an
    // upload could evict a later hit that has not been visited yet.
    load_ids.reserve((size_t)c.n_experts);
    for (int expert = 0; expert < c.n_experts; ++expert) {
      if (!scratch->expert_freq[(size_t)expert]) continue;
      std::lock_guard<std::recursive_mutex> cache_lock(topo->cache_mutex);
      auto *loc = topo->find_expert(site.layer, expert);
      if (loc && loc->device_ptr && loc->upload_ready) {
        if (!topo->hold_for_compute(site.layer, expert, loc->device_ptr, false))
          throw std::runtime_error("DeepSeek prefill could not hold resident expert");
        site.holds.push_back({topo, site.layer, expert, loc->device_ptr});
        held_locations[(size_t)expert] = loc;
        if (topo->expert_compact_layout && loc->pool_layer != site.layer &&
            loc->pool_layer != -1)
          throw std::runtime_error("DeepSeek compact cache returned an expert from the wrong layer region");
        for (size_t b = 0; b < site.pools.size(); ++b) {
          const auto &pool = site.pools[b];
          int physical_slot = -1;
          if (!v41_cache_pool_matches_location(
                  pool, *loc, site.layer, topo->expert_compact_layout) ||
              !v41_cache_pool_slot(pool, loc->device_ptr, physical_slot))
            continue;
          if (physical_slot < 0)
            throw std::runtime_error("DeepSeek prefill resident slot is out of bounds");
          owner[(size_t)expert] = (int)b;
          physical[(size_t)expert] = physical_slot;
          break;
        }
        if (owner[(size_t)expert] < 0)
          throw std::runtime_error("DeepSeek prefill resident expert is outside bound pools");
      } else {
        load_ids.push_back(expert);
      }
    }
    std::stable_sort(load_ids.begin(), load_ids.end(), [&](int a, int b) {
      return scratch->expert_freq[(size_t)a] > scratch->expert_freq[(size_t)b];
    });
  } else if (!site.cpu_cold_prefill) {
    load_ids.reserve((size_t)c.topk);
    // Snapshot and hold the routed VRAM hits atomically with respect to the
    // cache LRU before claiming DDR misses. Without this early hold, an
    // asynchronous promotion/eviction can remove a route after this scan
    // classified it as resident. The later dispatch then treats it as a CPU
    // miss even though it was omitted from load_ids, leaving no pinned DDR
    // block to hand to the CPU worker.
    std::lock_guard<std::recursive_mutex> cache_lock(topo->cache_mutex);
    for (int expert = 0; expert < c.n_experts; ++expert) {
      if (!scratch->expert_freq[(size_t)expert]) continue;
      auto *loc = topo->find_expert(site.layer, expert);
      if (!loc || !loc->device_ptr || !loc->upload_ready) {
        load_ids.push_back(expert);
        continue;
      }
      if (!topo->hold_for_compute(site.layer, expert, loc->device_ptr, false))
        throw std::runtime_error(
            "DeepSeek decode could not hold a routed VRAM expert during DDR miss planning");
      site.holds.push_back({topo, site.layer, expert, loc->device_ptr});
      held_locations[(size_t)expert] = loc;
    }
  }
  if (!load_ids.empty()) {
    std::vector<MoEExpertBlock *> loaded(load_ids.size(), nullptr);
    ram->get_experts_batch(e, site.layer, load_ids.data(),
                           (int)load_ids.size(), loaded.data());
    for (size_t i = 0; i < load_ids.size(); ++i) {
      auto *block = loaded[i];
      if (!block || !block->slab_buffer)
        throw std::runtime_error("DeepSeek cache graph batch DDR load failed");
      ram->ram_pin(*block);
      ram_pins.blocks[(size_t)load_ids[i]] = block;
      batch_blocks[(size_t)load_ids[i]] = block;
    }
  }
  if (site.cpu_cold_prefill) {
    int staged = 0;
    for (int expert : load_ids) {
      ++misses;
      auto *block = batch_blocks[(size_t)expert];
      const size_t bytes = block->gate_proj.meta.size_bytes +
                           block->up_proj.meta.size_bytes + block->down_proj.meta.size_bytes;
      const int preferred = site.pools[(size_t)(staged % (int)site.pools.size())].device;
      ExpertLocation *loc = nullptr;
      {
        // Probe and upload under one cache lock. If no victim is available,
        // spill this expert to the DDR CPU lane without failing the graph.
        std::lock_guard<std::recursive_mutex> cache_lock(topo->cache_mutex);
        loc = topo->find_expert(site.layer, expert);
        if (!loc || !loc->device_ptr) {
          int target = preferred;
          auto *victim = topo->evict_lru(bytes, nullptr, nullptr, target, site.layer);
          if (!victim) {
            target = -1;
            victim = topo->evict_lru(bytes, nullptr, nullptr, target, site.layer);
          }
          if (victim)
            loc = topo->upload_expert(site.layer, expert, block->slab_buffer,
                                      bytes, target);
        }
        if (loc && loc->device_ptr && loc->upload_ready) {
          if (!topo->hold_for_compute(site.layer, expert, loc->device_ptr, false))
            throw std::runtime_error("DeepSeek prefill could not hold staged expert");
          site.holds.push_back({topo, site.layer, expert, loc->device_ptr});
          held_locations[(size_t)expert] = loc;
        } else {
          loc = nullptr;
        }
      }
      if (!loc) {
        site.cpu_experts.push_back(expert);
        cpu_spill_pins.blocks[(size_t)expert] =
            ram_pins.take((size_t)expert);
        if (!cpu_spill_pins.blocks[(size_t)expert])
          throw std::runtime_error(
              "DeepSeek prefill CPU spill has no pinned DDR expert block");
        continue;
      }
      ram_pins.release((size_t)expert);
      ++staged;
      ++staged_misses;
      if (topo->expert_compact_layout && loc->pool_layer != site.layer &&
          loc->pool_layer != -1)
        throw std::runtime_error("DeepSeek compact cache returned an expert from the wrong layer region");
      for (size_t b = 0; b < site.pools.size(); ++b) {
        const auto &pool = site.pools[b];
        int physical_slot = -1;
        if (!v41_cache_pool_matches_location(
                pool, *loc, site.layer, topo->expert_compact_layout) ||
            !v41_cache_pool_slot(pool, loc->device_ptr, physical_slot))
          continue;
        if (physical_slot < 0)
          throw std::runtime_error("DeepSeek prefill staged slot is out of bounds");
        owner[(size_t)expert] = (int)b;
        physical[(size_t)expert] = physical_slot;
        break;
      }
      if (owner[(size_t)expert] < 0)
        throw std::runtime_error("DeepSeek prefill staged expert is outside bound pools");
    }
  } else {
    for (int expert = 0; expert < c.n_experts; ++expert) {
      if (!scratch->expert_freq[(size_t)expert]) continue;
      ExpertLocation* loc = nullptr;
      if (async_upload) {
        // Keep identity lookup and hold atomic with respect to cache eviction.
        loc = held_locations[(size_t)expert];
        if (!loc) {
          std::lock_guard<std::recursive_mutex> cache_lock(topo->cache_mutex);
          loc = topo->find_expert(site.layer, expert);
          if (loc && loc->device_ptr && loc->upload_ready) {
            if (!topo->hold_for_compute(site.layer, expert, loc->device_ptr, false))
              throw std::runtime_error("DeepSeek async cache graph could not hold resident expert");
            site.holds.push_back({topo, site.layer, expert, loc->device_ptr});
            held_locations[(size_t)expert] = loc;
          }
        }
      } else {
        loc = held_locations[(size_t)expert];
        if (!loc) {
          std::lock_guard<std::recursive_mutex> cache_lock(topo->cache_mutex);
          loc = topo->find_expert(site.layer, expert);
        }
      }
      if (loc && loc->device_ptr && loc->upload_ready && ram_pins.blocks[(size_t)expert])
        ram_pins.release((size_t)expert);
      if (!loc || !loc->device_ptr || !loc->upload_ready) {
        bool cpu_miss = false;
        if (g.decode_mode && site.cpu_cold_graph) {
          const auto strategy = static_cast<zeromoe::deepseek_graph::DecodeMissStrategy>(
              e.dsv41_decode_moe_miss_strategy.load(std::memory_order_relaxed));
          cpu_miss = zeromoe::deepseek_graph::decode_miss_uses_cpu(
              strategy, decode_ddr_miss_ordinal++);
        }
        if (cpu_miss) {
          ++misses;
          site.cpu_experts.push_back(expert);
          cpu_routed[(size_t)expert] = 1;
          // Normally every CPU-selected miss is already pinned by the
          // batched load_ids pass. A concurrent cache promotion can make a
          // routed VRAM hit disappear between that classification and this
          // dispatch. Full-residency mode keeps an authoritative DDR source,
          // so recover the exact expert from that tier instead of failing the
          // entire request or attempting an H2D miss path.
          if (!ram_pins.blocks[(size_t)expert]) {
            auto *recovered = ram->get_expert(e, site.layer, expert);
            if (!recovered || !recovered->slab_buffer)
              throw std::runtime_error(
                  "DeepSeek CPU decode miss escaped the batched DDR set and "
                  "has no resident DDR source: layer=" +
                  std::to_string(site.layer) + " expert=" +
                  std::to_string(expert));
            ram->ram_pin(*recovered);
            ram_pins.blocks[(size_t)expert] = recovered;
            batch_blocks[(size_t)expert] = recovered;
            LOG_WARN("[DSV41-CACHE-MOE-CPU] layer=%d expert=%d recovered a DDR-resident CPU miss omitted by the initial cache snapshot\n",
                     site.layer, expert);
          }
          auto *block = ram_pins.blocks[(size_t)expert];
          if (!block || !block->slab_buffer)
            throw std::runtime_error(
                "DeepSeek CPU decode miss lost its pinned DDR source: layer=" +
                std::to_string(site.layer) + " expert=" +
                std::to_string(expert));
          cpu_spill_pins.blocks[(size_t)expert] =
              ram_pins.take((size_t)expert);
          continue;
        }
        ++misses;
        auto* block = batch_blocks[(size_t)expert];
        if (!block) {
          block = ram->get_expert(e, site.layer, expert);
          if (!block || !block->slab_buffer)
            throw std::runtime_error("DeepSeek cache graph missing DDR expert");
          ram->ram_pin(*block);
          ram_pins.blocks[(size_t)expert] = block;
        }
        const size_t bytes = block->gate_proj.meta.size_bytes + block->up_proj.meta.size_bytes +
                             block->down_proj.meta.size_bytes;
        const int preferred = site.pools[(size_t)(gpu_misses++ % site.pools.size())].device;
        if (async_upload && (!loc || !loc->device_ptr)) {
          bool upload_was_already_ready = false;
          {
            std::lock_guard<std::recursive_mutex> cache_lock(topo->cache_mutex);
            loc = topo->upload_expert_async(site.layer, expert, block->slab_buffer,
                                            bytes, preferred, block->slab_pinned,
                                            prefer_bounce_upload);
            if (loc) {
              // Protect the slot before another cache worker can observe its
              // completed event and evict it. The graph will not read the data
              // until the per-device copy-stream sync below.
              if (!topo->hold_for_compute(site.layer, expert, loc->device_ptr, false))
                throw std::runtime_error(
                    "DeepSeek async cache graph could not reserve uploaded expert");
              site.holds.push_back({topo, site.layer, expert, loc->device_ptr});
              held_locations[(size_t)expert] = loc;
              upload_was_already_ready = loc->upload_ready;
              if (!upload_was_already_ready) pending_locations[(size_t)expert] = loc;
            }
          }
          if (loc && upload_was_already_ready) ram_pins.release((size_t)expert);
          if (loc) continue;
        }
        if (async_upload) {
          // Synchronous fallback must also become non-evictable before the
          // cache lock is released; otherwise the later batch validation can
          // race a background promotion worker.
          {
            std::lock_guard<std::recursive_mutex> cache_lock(topo->cache_mutex);
            loc = topo->upload_expert(site.layer, expert, block->slab_buffer, bytes, preferred);
            if (!loc) loc = topo->upload_expert(site.layer, expert, block->slab_buffer, bytes);
            if (loc && loc->device_ptr && loc->upload_ready) {
              if (!topo->hold_for_compute(site.layer, expert, loc->device_ptr, false))
                throw std::runtime_error(
                    "DeepSeek async cache graph could not hold synchronous fallback expert");
              site.holds.push_back({topo, site.layer, expert, loc->device_ptr});
              held_locations[(size_t)expert] = loc;
            }
          }
          ram_pins.release((size_t)expert);
        } else {
          loc = topo->upload_expert(site.layer, expert, block->slab_buffer, bytes, preferred);
          if (!loc) loc = topo->upload_expert(site.layer, expert, block->slab_buffer, bytes);
          ram_pins.release((size_t)expert);
        }
      }
      if (async_upload) continue;
      if (topo->expert_compact_layout && loc &&
          loc->pool_layer != site.layer && loc->pool_layer != -1)
        throw std::runtime_error(
            "DeepSeek compact cache returned an expert from the wrong layer region");
      if (!loc || !loc->device_ptr || (!loc->upload_ready && !async_upload) ||
          (loc->upload_ready &&
           !topo->hold_for_compute(site.layer, expert, loc->device_ptr, false)))
        throw std::runtime_error("DeepSeek cache graph failed to hold VRAM expert");
      if (loc->upload_ready) {
        site.holds.push_back({topo, site.layer, expert, loc->device_ptr});
        held_locations[(size_t)expert] = loc;
      }
      for (size_t b = 0; b < site.pools.size(); ++b) {
        const auto& pool = site.pools[b];
        int physical_slot = -1;
        if (!v41_cache_pool_matches_location(
                pool, *loc, site.layer, topo->expert_compact_layout) ||
            !v41_cache_pool_slot(pool, loc->device_ptr, physical_slot))
          continue;
        if (physical_slot < 0)
          throw std::runtime_error("DeepSeek cache graph physical slot out of bounds");
        owner[(size_t)expert] = (int)b;
        physical[(size_t)expert] = physical_slot;
      }
      if (owner[(size_t)expert] < 0)
        if (!pending_locations[(size_t)expert])
          throw std::runtime_error("DeepSeek cache graph expert is outside bound pools");
    }
    if (async_upload) {
      // Order each device's GGML graph stream after its own cache uploads.
      // This leaves the host free and lets a ready GPU branch compute while
      // the other device is still receiving its cold experts.
      for (const auto& pool : site.pools) {
        const bool has_pending =
            std::any_of(pending_locations.begin(), pending_locations.end(),
                        [&](const auto* loc) { return loc && loc->gpu_id == pool.device; });
        if (!has_pending) continue;
        cudaSetDevice(pool.device);
        cudaStream_t graph_stream = zeromoe_ggml_backend_cuda_get_stream(pool.backend);
        if (!graph_stream)
          throw std::runtime_error("DeepSeek async cache graph stream is unavailable");
        for (int expert = 0; expert < c.n_experts; ++expert) {
          auto* pending = pending_locations[(size_t)expert];
          if (!pending || pending->gpu_id != pool.device) continue;
          if (!pending->upload_event ||
              cudaStreamWaitEvent(graph_stream, pending->upload_event, 0) != cudaSuccess)
            throw std::runtime_error("DeepSeek async cache graph could not order an upload");
        }
      }
      for (int expert = 0; expert < c.n_experts; ++expert) {
        if (!zeromoe::deepseek_graph::requires_vram_route_validation(
                scratch->expert_freq[(size_t)expert] != 0,
                cpu_routed[(size_t)expert] != 0))
          continue;
        auto* loc = held_locations[(size_t)expert];
        if (!loc) loc = topo->find_expert(site.layer, expert);
        auto* pending = pending_locations[(size_t)expert];
        const bool ordered_pending =
            pending && pending == loc && pending->upload_event && !loc->upload_ready;
        if (!loc || !loc->device_ptr || (!loc->upload_ready && !ordered_pending) ||
            (held_locations[(size_t)expert] && held_locations[(size_t)expert] != loc))
          throw std::runtime_error("DeepSeek async cache graph could not hold uploaded expert");
        // RAMSafetyManager observes the pending-copy event before it can reclaim
        // or regrow this source slab, so the graph can release its temporary RAM
        // pin after enqueueing the stream dependency.
        ram_pins.release((size_t)expert);
        owner[(size_t)expert] = -1;
        for (size_t b = 0; b < site.pools.size(); ++b) {
          const auto& pool = site.pools[b];
          int physical_slot = -1;
          if (!v41_cache_pool_matches_location(
                  pool, *loc, site.layer, topo->expert_compact_layout) ||
              !v41_cache_pool_slot(pool, loc->device_ptr, physical_slot))
            continue;
          if (physical_slot < 0)
            throw std::runtime_error("DeepSeek async cache graph slot is out of bounds");
          owner[(size_t)expert] = (int)b;
          physical[(size_t)expert] = physical_slot;
          break;
        }
        if (owner[(size_t)expert] < 0)
          throw std::runtime_error("DeepSeek async cache graph expert is outside bound pools");
      }
    }
  }
  // Quantized batched MUL_MAT_ID needs a valid same-layer expert ID for
  // masked routes. Build these placeholders before the CPU worker takes
  // ownership of the pinned DDR spill table. If a GPU has no resident
  // same-layer entry, temporarily stage one of its pinned DDR spill experts
  // into that GPU's already-allocated dynamic cache slot.
  const bool skip_masked_routes = v41_native_f32() ||
      (g.S == 1 && mmid_negative_skip_available());
  std::vector<std::vector<int>> dummy(site.pools.size());
  if (!skip_masked_routes) {
    for (size_t b = 0; b < site.pools.size(); ++b) {
      const auto &pool = site.pools[b];
      size_t selected_slots = 0;
      for (int expert = 0; expert < c.n_experts; ++expert) {
        if (scratch->expert_freq[(size_t)expert] && owner[(size_t)expert] == (int)b &&
            physical[(size_t)expert] >= 0) {
          dummy[b].push_back(physical[(size_t)expert]);
          ++selected_slots;
        }
      }

      size_t ready_same_layer = 0;
      {
        std::lock_guard<std::recursive_mutex> lock(topo->cache_mutex);
        for (const auto &loc : topo->expert_vram_cache) {
          if (dummy[b].size() >= (size_t)c.topk) break;
          if (loc.gpu_id != pool.device || loc.layer_idx != site.layer ||
              !loc.upload_ready ||
              !v41_cache_pool_matches_location(
                  pool, loc, site.layer, topo->expert_compact_layout))
            continue;
          ++ready_same_layer;
          if (!topo->hold_for_compute(loc.layer_idx, loc.expert_id,
                                      loc.device_ptr, false))
            continue;
          int physical_id = -1;
          if (!v41_cache_pool_slot(pool, loc.device_ptr, physical_id)) {
            topo->release_hold(loc.layer_idx, loc.expert_id,
                               loc.device_ptr, false);
            throw std::runtime_error(
                "DeepSeek same-layer dummy cache slot is out of bounds");
          }
          if (std::find(dummy[b].begin(), dummy[b].end(), physical_id) !=
              dummy[b].end()) {
            topo->release_hold(loc.layer_idx, loc.expert_id,
                               loc.device_ptr, false);
            continue;
          }
          site.holds.push_back(
              {topo, loc.layer_idx, loc.expert_id, loc.device_ptr});
          dummy[b].push_back(physical_id);
        }
      }

      size_t ddr_dummy_candidates = 0;
      if (dummy[b].empty() && site.cpu_cold_prefill) {
        std::vector<zeromoe::memory::DeepSeekDdrDummyCandidate>
            stage_candidates;
        size_t local_dynamic_slots = 0;
        {
          std::lock_guard<std::recursive_mutex> lock(topo->cache_mutex);
          for (size_t slot_index : topo->unpinned_slot_indices) {
            if (slot_index >= topo->expert_vram_cache.size()) continue;
            const auto &slot = topo->expert_vram_cache[slot_index];
            if (slot.gpu_id == pool.device && !slot.memcpy_broken)
              if (!topo->expert_compact_layout ||
                  (pool.shared_dynamic ? slot.pool_layer == -1
                                       : slot.pool_layer == site.layer))
                ++local_dynamic_slots;
          }
          for (int expert = 0; expert < c.n_experts; ++expert) {
            auto *block = cpu_spill_pins.blocks[(size_t)expert];
            if (!zeromoe::memory::is_pinned_ddr_dummy_candidate(
                    expert, block ? block->expert_id : -1,
                    owner[(size_t)expert] < 0, block != nullptr,
                    block && block->slab_buffer))
              continue;
            ++ddr_dummy_candidates;
            stage_candidates.push_back({
                expert, block->expert_id, owner[(size_t)expert] < 0,
                true, block->slab_buffer != nullptr,
                topo->find_expert(site.layer, expert) == nullptr});
          }
        }
        const auto stage_plan =
            zeromoe::memory::plan_deepseek_prefill_dummy_stage(
                site.cpu_cold_prefill, dummy[b].size(), local_dynamic_slots,
                stage_candidates);
        if (!stage_plan) {
          {
            // Per-filter breakdown of every slot, to show which condition
            // removes the shared dynamic slots from the dummy search.
            std::lock_guard<std::recursive_mutex> lock(topo->cache_mutex);
            std::vector<char> in_unpinned(topo->expert_vram_cache.size(), 0);
            for (size_t i : topo->unpinned_slot_indices)
              if (i < in_unpinned.size()) in_unpinned[i] = 1;
            size_t total = 0, on_gpu = 0, shared = 0, shared_unpinned = 0,
                   shared_pinned = 0, shared_held = 0, shared_broken = 0,
                   shared_occupied = 0, shared_noptr = 0;
            for (size_t i = 0; i < topo->expert_vram_cache.size(); ++i) {
              const auto &s = topo->expert_vram_cache[i];
              ++total;
              if (s.gpu_id != pool.device) continue;
              ++on_gpu;
              if (s.pool_layer != -1) continue;
              ++shared;
              shared_unpinned += in_unpinned[i];
              shared_pinned += s.pinned;
              shared_held += s.temp_pin_count > 0;
              shared_broken += s.memcpy_broken;
              shared_occupied += s.layer_idx >= 0;
              shared_noptr += s.device_ptr == nullptr;
            }
            LOG_ERROR("[DSV41-CACHE-MOE] slot census layer=%d GPU=%d total=%zu on_gpu=%zu shared=%zu in_unpinned_index=%zu pinned=%zu held=%zu memcpy_broken=%zu occupied=%zu null_ptr=%zu unpinned_index_size=%zu\n",
                      site.layer, pool.device, total, on_gpu, shared,
                      shared_unpinned, shared_pinned, shared_held,
                      shared_broken, shared_occupied, shared_noptr,
                      topo->unpinned_slot_indices.size());
          }
          LOG_ERROR("[DSV41-CACHE-MOE] layer=%d GPU=%d dummy preflight failed: %s; pool_slots=%d ready_same_layer=%zu pinned_DDR_candidates=%zu local_dynamic_slots=%zu compact=%d\n",
                    site.layer, pool.device, stage_plan.error.c_str(),
                    pool.slots, ready_same_layer, ddr_dummy_candidates,
                    local_dynamic_slots,
                    topo->expert_compact_layout ? 1 : 0);
          throw std::runtime_error(
              "DeepSeek cache graph cannot stage a same-layer dummy on GPU " +
              std::to_string(pool.device) + " at layer " +
              std::to_string(site.layer) + ": " + stage_plan.error);
        }
        for (int expert : stage_plan.expert_ids) {
          auto *block = cpu_spill_pins.blocks[(size_t)expert];
          const size_t bytes = block->gate_proj.meta.size_bytes +
                               block->up_proj.meta.size_bytes +
                               block->down_proj.meta.size_bytes;
          ExpertLocation *loc = nullptr;
          bool held = false;
          {
            std::lock_guard<std::recursive_mutex> lock(topo->cache_mutex);
            loc = topo->find_expert(site.layer, expert);
            if (loc && loc->gpu_id == pool.device && loc->device_ptr &&
                loc->upload_ready &&
                v41_cache_pool_matches_location(
                    pool, *loc, site.layer, topo->expert_compact_layout)) {
              held = topo->hold_for_compute(site.layer, expert,
                                            loc->device_ptr, false);
            } else if (!loc ||
                       !v41_cache_pool_matches_location(
                           pool, *loc, site.layer,
                           topo->expert_compact_layout)) {
              // Use the existing cache allocation. The DDR source remains
              // pinned by cpu_spill_pins while the CPU worker uses it.
              auto *victim = topo->evict_lru(
                  bytes, nullptr, nullptr, pool.device, site.layer);
              if (victim)
                loc = topo->upload_expert(site.layer, expert,
                                          block->slab_buffer, bytes,
                                          pool.device);
              held = loc && loc->gpu_id == pool.device && loc->device_ptr &&
                  loc->upload_ready &&
                  v41_cache_pool_matches_location(
                      pool, *loc, site.layer,
                      topo->expert_compact_layout) &&
                  topo->hold_for_compute(site.layer, expert,
                                         loc->device_ptr, false);
            }
            if (held) {
              int physical_id = -1;
              if (!v41_cache_pool_slot(pool, loc->device_ptr, physical_id)) {
                topo->release_hold(site.layer, expert, loc->device_ptr, false);
                throw std::runtime_error(
                    "DeepSeek staged dummy slot is outside its GPU pool");
              }
              site.holds.push_back(
                  {topo, site.layer, expert, loc->device_ptr});
              dummy[b].push_back(physical_id);
            }
          }
          if (!dummy[b].empty()) {
            LOG_INFO("[DSV41-CACHE-MOE] layer=%d GPU=%d staged dummy expert=%d from pinned DDR into dynamic VRAM slot=%d\n",
                     site.layer, pool.device, expert, dummy[b].back());
            break;
          }
        }
      }
      if (dummy[b].empty()) {
        LOG_ERROR("[DSV41-CACHE-MOE] layer=%d GPU=%d has no same-layer dummy slot: pool_slots=%d selected_slots=%zu ready_same_layer=%zu pinned_DDR_candidates=%zu compact=%d\n",
                  site.layer, pool.device, pool.slots, selected_slots,
                  ready_same_layer, ddr_dummy_candidates,
                  topo->expert_compact_layout ? 1 : 0);
        throw std::runtime_error(
            "DeepSeek cache graph needs a same-layer dummy slot on GPU " +
            std::to_string(pool.device) + " at layer " +
            std::to_string(site.layer));
      }
      if (dummy[b].size() < (size_t)c.topk)
        LOG_INFO("[DSV41-CACHE-MOE] layer=%d GPU=%d has %zu same-layer dummy slot(s); masked routes will reuse them\n",
                 site.layer, pool.device, dummy[b].size());
    }
  }

  if (site.cpu_cold_graph && !site.cpu_experts.empty()) {
    if (!site.cpu_cold_prefill) {
      site.cpu_route_weights.assign(site.cpu_experts.size(), 0.0f);
      for (size_t i = 0; i < site.cpu_experts.size(); ++i) {
        for (int k = 0; k < c.topk; ++k) {
          if (scratch->tok_topk_ids[(size_t)k] == site.cpu_experts[i])
            site.cpu_route_weights[i] += scratch->tok_topk_weights[(size_t)k];
        }
      }
    }
    const int layer = site.layer;
    const int dim = g.D;
    const int inter = c.moe_inter;
    const int token_count = g.S;
    const int topk = c.topk;
    const int *route_ids = scratch->tok_topk_ids.data();
    const float *route_weights = scratch->tok_topk_weights.data();
    const float *cpu_input_data = site.cpu_cold_prefill
        ? scratch->normed.data() : site.cpu_input.data();
    const int cpu_expert_count = c.n_experts;
    std::vector<int> cpu_expert_ids = site.cpu_experts;
    bool cpu_spill_ids_match = cpu_expert_ids.size() <= (size_t)cpu_expert_count;
    std::vector<uint8_t> cpu_spill_ids_seen((size_t)cpu_expert_count, 0);
    for (size_t i = 0; i < cpu_expert_ids.size(); ++i) {
      const int expert = cpu_expert_ids[i];
      if (expert < 0 || expert >= cpu_expert_count) {
        cpu_spill_ids_match = false;
        LOG_ERROR("[DSV41-CACHE-MOE-CPU] layer=%d spill ID invalid before worker: index=%zu id=%d range=[0,%d)\n",
                  layer, i, expert, cpu_expert_count);
        continue;
      }
      if (cpu_spill_ids_seen[(size_t)expert]) {
        cpu_spill_ids_match = false;
        LOG_ERROR("[DSV41-CACHE-MOE-CPU] layer=%d duplicate spill ID before worker: index=%zu id=%d\n",
                  layer, i, expert);
      }
      cpu_spill_ids_seen[(size_t)expert] = 1;
    }
    if (!site.cpu_cold_prefill && !cpu_spill_ids_match)
      throw std::runtime_error("DeepSeek decode CPU spill IDs failed validation");
    const bool cpu_prefill = site.cpu_cold_prefill;
    float *cpu_output_data = site.cpu_output_data;
    const float clamp = layer < (int)c.swiglu_clamp_exp.size()
        ? c.swiglu_clamp_exp[(size_t)layer] : 0.0f;
    site.cpu_job = std::async(std::launch::async, [&site, ram, &e, layer,
        dim, inter, token_count, topk, route_ids, route_weights,
        cpu_input_data, cpu_output_data, cpu_prefill, cpu_expert_count,
        cpu_expert_ids = std::move(cpu_expert_ids),
        cpu_spill_ids_match,
        cpu_spill_pins = std::move(cpu_spill_pins), clamp]() mutable {
      const auto cpu_begin = std::chrono::steady_clock::now();
      if (!cpu_prefill) {
        // Decode misses are bandwidth-bound single-token dots. SMT siblings
        // only add contention, so use one OpenMP thread per physical core.
        // This worker is a fresh std::async thread, so the setting applies
        // to its own team only (measured: decode +3.5%, prefill unchanged).
        static const int decode_threads =
            std::max(1, (int)std::thread::hardware_concurrency() / 2);
        omp_set_num_threads(decode_threads);
      }
      std::vector<int> worker_expert_ids;
      std::vector<MoEExpertBlock *> blocks;
      if (cpu_prefill) {
        if (cpu_spill_pins.blocks.size() != (size_t)cpu_expert_count)
          throw std::runtime_error("DeepSeek prefill pinned-block table has wrong size");
        worker_expert_ids.reserve((size_t)cpu_expert_count);
        blocks.reserve((size_t)cpu_expert_count);
        for (int expert = 0; expert < cpu_expert_count; ++expert) {
          auto *block = cpu_spill_pins.blocks[(size_t)expert];
          if (!block) continue;
          worker_expert_ids.push_back(expert);
          blocks.push_back(block);
        }
        bool spill_ids_match = cpu_spill_ids_match &&
            cpu_expert_ids.size() == worker_expert_ids.size();
        std::vector<uint8_t> listed((size_t)cpu_expert_count, 0);
        for (size_t i = 0; i < cpu_expert_ids.size(); ++i) {
          const int expert = cpu_expert_ids[i];
          if (expert < 0 || expert >= cpu_expert_count ||
              listed[(size_t)expert] ||
              !cpu_spill_pins.blocks[(size_t)expert]) {
            spill_ids_match = false;
            continue;
          }
          listed[(size_t)expert] = 1;
        }
        for (int expert : worker_expert_ids) {
          if (!listed[(size_t)expert]) {
            spill_ids_match = false;
            break;
          }
        }
        if (!spill_ids_match) {
          LOG_ERROR("[DSV41-CACHE-MOE-CPU] layer=%d spill ID snapshot disagrees with pinned DDR expert table; using table identities (listed=%zu pinned=%zu)\n",
                    layer, cpu_expert_ids.size(), worker_expert_ids.size());
        }
      } else {
        worker_expert_ids = std::move(cpu_expert_ids);
        blocks.assign(worker_expert_ids.size(), nullptr);
        for (size_t i = 0; i < worker_expert_ids.size(); ++i) {
          const int expert = worker_expert_ids[i];
          if (expert < 0 || expert >= cpu_expert_count)
            throw std::runtime_error(
                "DeepSeek CPU spill list invalid before DDR lookup: index=" +
                std::to_string(i) + " id=" + std::to_string(expert));
          blocks[i] = cpu_spill_pins.blocks[(size_t)expert];
          if (!blocks[i])
            throw std::runtime_error(
                "DeepSeek CPU spill block lost its DDR pin before worker start");
        }
      }
      const size_t count = worker_expert_ids.size();
      for (size_t i = 0; i < count; ++i) {
        auto *block = blocks[i];
        if (!block || !block->gate_proj.host_ptr || !block->up_proj.host_ptr ||
            !block->down_proj.host_ptr)
          throw std::runtime_error("DeepSeek CPU cold lane could not resolve DDR expert");
      }
      if (cpu_prefill) {
        struct Route {
          int token;
          float weight;
        };
        std::vector<uint8_t> is_cpu((size_t)cpu_expert_count, 0);
        std::vector<std::vector<Route>> routes(is_cpu.size());
        for (int expert : worker_expert_ids)
          is_cpu[(size_t)expert] = 1;
        for (int token = 0; token < token_count; ++token) {
          for (int k = 0; k < topk; ++k) {
            const size_t flat = (size_t)token * (size_t)topk + (size_t)k;
            const int expert = route_ids[flat];
            if (expert >= 0 && (size_t)expert < is_cpu.size() && is_cpu[(size_t)expert])
              routes[(size_t)expert].push_back({token, route_weights[flat]});
          }
        }
        std::vector<float> compact_input, expert_output;
        std::vector<float> gate, up;
        for (size_t i = 0; i < count; ++i) {
          const int expert = worker_expert_ids[i];
          const auto &expert_routes = routes[(size_t)expert];
          if (expert_routes.empty()) continue;
          const size_t n = expert_routes.size();
          compact_input.resize(n * (size_t)dim);
          expert_output.resize(n * (size_t)dim);
          gate.resize(n * (size_t)inter);
          up.resize(n * (size_t)inter);
          for (size_t r = 0; r < n; ++r) {
            const size_t source = (size_t)expert_routes[r].token * (size_t)dim;
            std::memcpy(compact_input.data() + r * (size_t)dim,
                        cpu_input_data + source,
                        (size_t)dim * sizeof(float));
          }
          const auto &block = *blocks[i];
          expert_ffn_cpu(compact_input.data(), expert_output.data(),
              block.gate_proj.host_ptr, block.gate_proj.meta,
              block.up_proj.host_ptr, block.up_proj.meta,
              block.down_proj.host_ptr, block.down_proj.meta,
              gate.data(), up.data(), (int)n, dim, inter, clamp, layer, expert);
          for (size_t r = 0; r < n; ++r) {
            const size_t target = (size_t)expert_routes[r].token * (size_t)dim;
            const float weight = expert_routes[r].weight;
            for (int d = 0; d < dim; ++d)
              cpu_output_data[target + (size_t)d] +=
                  weight * expert_output[r * (size_t)dim + (size_t)d];
          }
        }
        site.cpu_scalar_experts = count;
        site.cpu_job_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cpu_begin).count();
        return;
      }
      bool ggml_grouped = false;
      if (count >= 2 &&
          blocks[0]->gate_proj.meta.type == GGUFType::Q2_K &&
          blocks[0]->up_proj.meta.type == GGUFType::Q2_K &&
          blocks[0]->down_proj.meta.type == GGUFType::Q3_K) {
        std::vector<GgmlQwenCpuExpertRef> refs(count);
        for (size_t i = 0; i < count; ++i) {
          const auto &block = *blocks[i];
          refs[i] = {block.gate_proj.host_ptr, block.up_proj.host_ptr,
                     block.down_proj.host_ptr,
                     block.gate_proj.meta.size_bytes,
                     block.up_proj.meta.size_bytes,
                     block.down_proj.meta.size_bytes,
                     static_cast<uint32_t>(block.gate_proj.meta.type),
                     static_cast<uint32_t>(block.up_proj.meta.type),
                     static_cast<uint32_t>(block.down_proj.meta.type)};
        }
        ggml_grouped = GgmlQwenCpuMoeRunner::instance().run(
            site.cpu_input.data(), site.cpu_output.data(), refs.data(),
            static_cast<int>(count), site.cpu_route_weights.data(), dim,
            inter, static_cast<int>(count), clamp);
      }
      const bool quant_grouped = !ggml_grouped && count >= 1 &&
          expert_ffn_cpu_grouped_ddr(
              site.cpu_input.data(), site.cpu_output.data(), blocks,
              site.cpu_route_weights.data(), dim, inter, clamp, layer);
      const bool grouped = ggml_grouped || quant_grouped;
      site.cpu_ggml_grouped_experts = ggml_grouped ? count : 0;
      site.cpu_quant_grouped_experts = quant_grouped ? count : 0;
      site.cpu_scalar_experts = grouped ? 0 : count;
      if (!grouped) {
        std::vector<float> gate((size_t)inter), up((size_t)inter), down((size_t)dim);
        for (size_t i = 0; i < count; ++i) {
          const auto &block = *blocks[i];
          expert_ffn_cpu(site.cpu_input.data(), down.data(),
              block.gate_proj.host_ptr, block.gate_proj.meta,
              block.up_proj.host_ptr, block.up_proj.meta,
              block.down_proj.host_ptr, block.down_proj.meta,
              gate.data(), up.data(), 1, dim, inter, clamp, layer,
              worker_expert_ids[i]);
          const float weight = site.cpu_route_weights[i];
          for (int d = 0; d < dim; ++d)
            site.cpu_output[(size_t)d] += weight * down[(size_t)d];
        }
      }
      site.cpu_job_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
    });
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
        // Reusing an expert from this same layer is safe for a zero-weight
        // route. Avoid substituting another layer's bytes just to obtain a
        // distinct physical ID: quantized MUL_MAT_ID may produce NaNs from
        // that incompatible view, and NaN * 0 is still NaN.
        if (it == dummy[b].end()) it = dummy[b].begin();
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
           "staged=%d cpu_spill=%zu "
           "route_ms=%.3f upload_ms=%.3f payload_ms=%.3f "
           "graph-owned expert bytes=0\n",
           site.layer, g.S, site.holds.size(), misses, staged_misses,
           site.cpu_experts.size(),
           ms(start, routed), ms(routed, uploaded), ms(uploaded, done));
}

void v41_cache_cpu_merge_node(ggml_tensor *dst, int ith, int, void *ud) {
  if (ith) return;
  auto &site = *static_cast<V41CacheMoeSite *>(ud);
  double wait_ms = 0.0;
  if (site.cpu_job.valid()) {
    const auto wait_begin = std::chrono::steady_clock::now();
    site.cpu_job.get();
    wait_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wait_begin).count();
  LOG_INFO("[DSV41-CACHE-MOE-CPU] layer=%d mode=%s cold=%zu ggml_grouped=%zu "
           "quant_grouped=%zu scalar=%zu worker_ms=%.3f merge_wait_ms=%.3f\n",
           site.layer, site.cpu_cold_prefill ? "prefill-batched" : "decode",
           site.cpu_experts.size(),
           site.cpu_ggml_grouped_experts, site.cpu_quant_grouped_experts,
           site.cpu_scalar_experts, site.cpu_job_ms, wait_ms);
  }
  const ggml_tensor *src = dst->src[0];
  if (!src) throw std::runtime_error("DeepSeek CPU cold-lane merge is missing GPU output");
  const size_t bytes = ggml_nbytes(dst);
  if (bytes != ggml_nbytes(src) || !site.cpu_output_data ||
      bytes % sizeof(float) != 0)
    throw std::runtime_error("DeepSeek CPU cold-lane merge tensor size mismatch");
  const size_t graph_elements =
      static_cast<size_t>(ggml_nelements(dst));
  const size_t merge_elements =
      zeromoe::deepseek_graph::cpu_merge_element_count(
          site.cpu_cold_prefill, graph_elements, site.cpu_output_elements);
  if (graph_elements != bytes / sizeof(float) ||
      merge_elements != graph_elements ||
      site.cpu_output_elements != graph_elements)
    throw std::runtime_error(
        "DeepSeek CPU cold-lane merge output element count mismatch");
  std::memcpy(dst->data, src->data, bytes);
  auto *out = static_cast<float *>(dst->data);
  for (size_t i = 0; i < merge_elements; ++i)
    out[i] += site.cpu_output_data[i];

  // This merge node is the completion boundary for this layer's GPU and CPU
  // expert branches. The graph owns one site per layer, so retaining every
  // site's compute holds until the whole prompt finishes makes the shared
  // compact dynamic pool unusable after its first 128 uploads. Release this
  // layer's holds now: its output is materialized, and later layers may reuse
  // those preallocated LRU slots without moving or reallocating tensor data.
  const size_t released_holds = site.holds.size();
  site.release();
  LOG_INFO("[DSV41-CACHE-MOE-RELEASE] layer=%d cpu_prefill=%d merge_elements=%zu released=%zu slot holds after merged output; shared dynamic slots are reusable\n",
           site.layer, site.cpu_cold_prefill ? 1 : 0, merge_elements,
           released_holds);
}

ggml_tensor *build_v41_cache_moe(FullGraph &g, ggml_tensor *cur, int layer,
                                 const int *token_ids,
                                 const int *decode_pos = nullptr) {
  const auto &c = g.engine->model_file.architecture_config;
  auto &topo = g.engine->gpu->topology;
  if ((!topo.expert_compact_layout && !topo.expert_slab_size) || c.topk <= 0) {
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
  if (gb > std::numeric_limits<size_t>::max() - ub ||
      gb + ub > std::numeric_limits<size_t>::max() - db) {
    LOG_ERROR("[DSV41-CACHE-MOE] layer=%d expert byte geometry overflows size_t\n", layer);
    return nullptr;
  }
  const size_t required_bytes = gb + ub + db;
  auto site = std::make_unique<V41CacheMoeSite>();
  site->g = &g; site->layer = layer; site->tokens = token_ids;
  site->decode_pos = decode_pos;
  const bool cpu_cold_enabled =
      zeromoe::graphsettings::active_settings().tun.cache_moe_cpu_cold.value != 0;
  const bool cpu_cold_decode_enabled =
      zeromoe::graphsettings::active_settings().tun.cache_moe_cpu_cold_decode.value != 0;
  site->cpu_cold_prefill = !g.decode_mode && g.S > 1 && cpu_cold_enabled;
  site->cpu_cold_graph = zeromoe::deepseek_graph::use_cpu_cold_graph(
      g.decode_mode, g.S, cpu_cold_enabled, cpu_cold_decode_enabled);
  for (int vi : topo.vault_gpus) {
    auto &dev = topo.devices[(size_t)vi];
    if (!dev.expert_cache_pool) continue;
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
    if (topo.expert_compact_layout) {
      std::vector<const ExpertLocation *> slots;
      for (const auto &loc : topo.expert_vram_cache)
        if (loc.gpu_id == dev.id && loc.pool_layer == layer)
          slots.push_back(&loc);
      if (!slots.empty()) {
        std::sort(slots.begin(), slots.end(), [](const auto *a, const auto *b) {
          return a->offset < b->offset;
        });
        const size_t base_offset = slots.front()->offset;
        const size_t stride = slots.front()->size;
        const size_t declared_stride =
            layer >= 0 && (size_t)layer < topo.expert_layer_slab_bytes.size()
                ? topo.expert_layer_slab_bytes[(size_t)layer]
                : 0;
        bool valid = stride >= required_bytes && stride != 0 &&
                     stride == declared_stride &&
                     base_offset <= dev.expert_cache_size &&
                     slots.size() <=
                         (dev.expert_cache_size - base_offset) / stride;
        for (size_t i = 0; valid && i < slots.size(); ++i) {
          const auto &slot = *slots[i];
          valid = slot.pool_layer == layer && slot.size == stride &&
                  i <= (std::numeric_limits<size_t>::max() - base_offset) / stride &&
                  slot.offset == base_offset + i * stride &&
                  slot.offset <= dev.expert_cache_size &&
                  slot.size <= dev.expert_cache_size - slot.offset &&
                  slot.device_ptr == dev.expert_cache_pool + slot.offset;
        }
        if (!valid) {
          LOG_ERROR("[DSV41-CACHE-MOE] compact layer=%d GPU=%d slot region failed bounds/contiguity checks\n",
                    layer, dev.id);
          return nullptr;
        }
        site->pools.push_back({dev.id, backend, marker,
            dev.expert_cache_pool + base_offset, stride, (int)slots.size(),
            false});
      }

      // The global-stride tail can be borrowed by any layer during prefill
      // and decode. It has its own GGML MUL_MAT_ID view because its slot
      // stride differs from the compact per-layer regions on this GPU.
      std::vector<const ExpertLocation *> dynamic_slots;
      for (const auto &loc : topo.expert_vram_cache)
        if (loc.gpu_id == dev.id && loc.pool_layer == -1)
          dynamic_slots.push_back(&loc);
      if (!dynamic_slots.empty()) {
        std::sort(dynamic_slots.begin(), dynamic_slots.end(),
                  [](const auto *a, const auto *b) {
                    return a->offset < b->offset;
                  });
        const size_t base_offset = dynamic_slots.front()->offset;
        const size_t stride = topo.expert_max_slab_size;
        bool valid = stride >= required_bytes && stride != 0 &&
                     base_offset <= dev.expert_cache_size &&
                     dynamic_slots.size() <=
                         (dev.expert_cache_size - base_offset) / stride;
        for (size_t i = 0; valid && i < dynamic_slots.size(); ++i) {
          const auto &slot = *dynamic_slots[i];
          valid = slot.pool_layer == -1 && slot.size == stride &&
                  i <= (std::numeric_limits<size_t>::max() - base_offset) / stride &&
                  slot.offset == base_offset + i * stride &&
                  slot.offset <= dev.expert_cache_size &&
                  slot.size <= dev.expert_cache_size - slot.offset &&
                  slot.device_ptr == dev.expert_cache_pool + slot.offset;
        }
        if (!valid) {
          LOG_ERROR("[DSV41-CACHE-MOE] layer=%d GPU=%d shared dynamic pool failed bounds/contiguity checks\n",
                    layer, dev.id);
          return nullptr;
        }
        site->pools.push_back({dev.id, backend, marker,
            dev.expert_cache_pool + base_offset, stride,
            (int)dynamic_slots.size(), true});
      }
    } else {
      if (topo.expert_slab_size == 0 ||
          dev.expert_cache_size < topo.expert_slab_size) continue;
      site->pools.push_back({dev.id, backend, marker, dev.expert_cache_pool,
          topo.expert_slab_size,
          int(dev.expert_cache_size / topo.expert_slab_size), false});
    }
  }
  if (site->pools.empty()) {
    LOG_ERROR("[DSV41-CACHE-MOE] layer=%d has no bindable GPU expert pool (vault GPUs=%zu)\n",
              layer, topo.vault_gpus.size());
    return nullptr;
  }
  auto *ptr = site.get();
  v41_sites_for(g).push_back(std::move(site));
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
      if (!t || !pool.base || ne0 <= 0 || ne1 <= 0 ||
          pool.slots <= 0 || pool.stride == 0 ||
          pool.stride > std::numeric_limits<size_t>::max() / (size_t)pool.slots ||
          t->nb[1] == 0 || (size_t)ne1 > std::numeric_limits<size_t>::max() / t->nb[1])
        return (ggml_tensor *)nullptr;
      // GGML's CUDA MUL_MAT_ID expert-slice path requires the expert-plane
      // stride to be a whole number of matrix rows. Compact cache slots whose
      // byte size violates this can pass bounds checks yet produce invalid
      // FFN outputs, so decline graph construction and use native dispatch.
      if (pool.stride % t->nb[1] != 0) {
        LOG_ERROR("[DSV41-CACHE-MOE] compact layer=%d GPU=%d expert stride=%zu is not row-aligned (row stride=%zu)\n",
                  layer, pool.device, pool.stride, t->nb[1]);
        return (ggml_tensor *)nullptr;
      }
      const size_t plane_bytes = t->nb[1] * (size_t)ne1;
      if (offset > pool.stride || plane_bytes > pool.stride - offset) {
        LOG_ERROR("[DSV41-CACHE-MOE] layer=%d GPU=%d tensor plane exceeds compact slot: offset=%zu plane=%zu stride=%zu type=%u shape=[%d,%d]\n",
                  layer, pool.device, offset, plane_bytes, pool.stride,
                  (unsigned)m->type, ne0, ne1);
        return (ggml_tensor *)nullptr;
      }
      const uintptr_t tensor_address =
          reinterpret_cast<uintptr_t>(pool.base + offset);
      if ((tensor_address % 16) != 0 || (pool.stride % 16) != 0) {
        LOG_ERROR("[DSV41-CACHE-MOE] layer=%d GPU=%d tensor view lacks 16-byte alignment: addr_mod16=%zu stride_mod16=%zu stride=%zu offset=%zu type=%u shape=[%d,%d]\n",
                  layer, pool.device, tensor_address % 16, pool.stride % 16,
                  pool.stride, offset, (unsigned)m->type, ne0, ne1);
        return (ggml_tensor *)nullptr;
      }
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
      LOG_ERROR("[DSV41-CACHE-MOE] layer=%d expert weight view invalid on GPU %d: gate=%p up=%p down=%p stride=%zu slots=%d gate_bytes=%zu up_bytes=%zu down_bytes=%zu\n",
                layer, pool.device, (void *)ew.gate, (void *)ew.up,
                (void *)ew.down, pool.stride, pool.slots, gb, ub, db);
      return nullptr;
    }
    const size_t offset = (size_t)(g.D + b * 2 * c.topk) * sizeof(float);
    auto *ids = ggml_cont(g.ctx, ggml_view_2d(g.ctx, payload, c.topk, g.S, payload->nb[1], offset));
    ids = ggml_cast(g.ctx, ids, GGML_TYPE_I32);
    auto *weights = ggml_cont(g.ctx, ggml_view_2d(g.ctx, payload, c.topk, g.S,
                          payload->nb[1], offset + c.topk * sizeof(float)));
    weights = ggml_reshape_3d(g.ctx, weights, 1, c.topk, g.S);
    const float clamp = layer < (int)c.swiglu_clamp_exp.size() ? c.swiglu_clamp_exp[layer] : 0.0f;
    auto ffn = moe_routed::expert_ffn(g.ctx, ew, x3, ids, clamp, swiglu_clamp_fused());
    if (!ffn.down) {
      LOG_ERROR("[DSV41-CACHE-MOE] layer=%d GPU %d expert FFN node construction failed\n",
                layer, pool.device);
      return nullptr;
    }
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
  // Shared experts carry their own limit (swiglu_clamp_shexp); the graph
  // previously applied none.
  const float shexp_clamp = layer < (int)c.swiglu_clamp_shexp.size()
                                ? c.swiglu_clamp_shexp[(size_t)layer] : 0.0f;
  ggml_tensor *shared_act = nullptr;
  if (shexp_clamp > 1e-6f && swiglu_clamp_fused())
    shared_act = ggml_mul(g.ctx,
        ggml_silu(g.ctx, ggml_clamp(g.ctx, gate, -INFINITY, shexp_clamp)),
        ggml_clamp(g.ctx, up, -shexp_clamp, shexp_clamp));
  else if (shexp_clamp > 1e-6f)
    shared_act = ggml_clamp(g.ctx, ggml_swiglu_split(g.ctx, gate, up),
                            -shexp_clamp, shexp_clamp);
  else
    shared_act = ggml_swiglu_split(g.ctx, gate, up);
  auto *shared = v41_matmul(g.ctx, sd, shared_act);
  auto *combined = ggml_add(g.ctx, sum, shared);
  if (!ptr->cpu_cold_graph) return combined;
  ggml_tensor *merge_args[] = {combined};
  auto *merged = ggml_custom_4d(g.ctx, GGML_TYPE_F32, g.D, g.S, 1, 1,
      merge_args, 1, v41_cache_cpu_merge_node, 1, ptr);
  if (!merged) return nullptr;
  ggml_backend_sched_set_tensor_backend(g.sched, merged, g.cpu);
  return merged;
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
    if (zeromoe::graphsettings::active_settings().tun.cache_moe_graph.value != 0) {
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
  LOG_INFO("[GGML-DSV41-PREFILL] gpu_trunk=%d tokens=%d cache_owned_experts=1\n",
           int(v41_gpu_prefill_enabled()), g.S);
  LOG_INFO("[GGML-DSV41-FULL] graph built layers=%d nodes=%d splits=%d "
           "workspace=%.2f GiB host_moe_callbacks=%d "
           "engram_tensor_inputs=%zu native_engram_layers=%zu "
           "graph-owned MoE bytes=0 "
           "(existing DDR/VRAM cache dispatcher)\n",
           g.n_layers, ggml_graph_n_nodes(g.graph),
           ggml_backend_sched_get_n_splits(g.sched),
           (double)g.graph_workspace_bytes / (1024.0 * 1024.0 * 1024.0),
           zeromoe::graphsettings::active_settings().tun.cache_moe_graph.value != 0 &&
                   v41_gpu_prefill_enabled()
               ? 0 : g.n_layers,
           g.dsv41_engram_embeddings.size(),
           std::count_if(c.dsv41_engram_layer_ids.begin(),
                         c.dsv41_engram_layer_ids.end(),
                         [&](int64_t l) { return l >= 0 && l < g.n_layers; }));
  g.ready = true;
  return true;
}

} // namespace

void v41_cache_moe_sites_reset() {
  for (auto &site : g_v41_decode_cache_sites)
    if (site)
      site->release();
  g_v41_decode_cache_sites.clear();
}

void v41_cache_moe_sites_release_holds() {
  for (auto &site : g_v41_decode_cache_sites)
    if (site)
      site->release();
}

ggml_tensor *v41_build_cache_moe(FullGraph &g, ggml_tensor *cur, int layer,
                                 const int *token_ids,
                                 const int *decode_pos) {
  return build_v41_cache_moe(g, cur, layer, token_ids, decode_pos);
}

} // namespace glm5next_graph

using namespace glm5next_graph;

bool deepseek_v41_rebalance_prefill_dynamic_cache(
    StreamingEngine &engine, RAMSafetyManager &ram,
    const std::vector<uint64_t> &prefill_route_counts) {
  if (!engine.dsv41_strategy_test_isolation ||
      !engine.qwen_ram_preload_all_experts || !engine.gpu)
    return false;
  auto &topo = engine.gpu->topology;
  const size_t dynamic_slots =
      (size_t)std::max(0, topo.expert_compact_shared_dynamic_slots);
  const int n_experts = engine.model_file.architecture_config.n_experts;
  const int n_layers = engine.model_file.architecture_config.n_layers;
  if (!topo.expert_compact_layout || dynamic_slots == 0 || n_experts <= 0 ||
      n_layers <= 0 ||
      prefill_route_counts.size() < (size_t)n_layers * (size_t)n_experts)
    return false;

  struct Candidate { int layer; int expert; uint64_t count; };
  std::vector<std::vector<Candidate>> by_layer((size_t)n_layers);
  std::unordered_set<uint64_t> pinned;
  const auto key = [n_experts](int layer, int expert) {
    return ((uint64_t)(uint32_t)layer << 32) | (uint32_t)expert;
  };
  {
    std::lock_guard<std::recursive_mutex> lock(topo.cache_mutex);
    for (const auto &loc : topo.expert_vram_cache) {
      if (loc.pinned && loc.layer_idx >= 0 && loc.expert_id >= 0)
        pinned.insert(key(loc.layer_idx, loc.expert_id));
    }
  }
  size_t observed = 0;
  for (int layer = 0; layer < n_layers; ++layer) {
    auto &ranked = by_layer[(size_t)layer];
    for (int expert = 0; expert < n_experts; ++expert) {
      const uint64_t count = prefill_route_counts[
          (size_t)layer * (size_t)n_experts + (size_t)expert];
      if (!count) continue;
      ++observed;
      if (pinned.contains(key(layer, expert))) continue;
      ranked.push_back({layer, expert, count});
    }
    std::stable_sort(ranked.begin(), ranked.end(),
        [](const Candidate &a, const Candidate &b) {
          if (a.count != b.count) return a.count > b.count;
          return a.expert < b.expert;
        });
  }

  // Round-robin ranks across layers so one concentrated layer cannot use the
  // entire shared dynamic pool. The physical pool is global; its candidate
  // set is therefore a balanced extension of the pinned decode hot set.
  std::vector<Candidate> selected;
  selected.reserve(dynamic_slots);
  for (size_t rank = 0; selected.size() < dynamic_slots; ++rank) {
    bool added = false;
    for (int layer = 0; layer < n_layers && selected.size() < dynamic_slots;
         ++layer) {
      const auto &ranked = by_layer[(size_t)layer];
      if (rank >= ranked.size()) continue;
      selected.push_back(ranked[rank]);
      added = true;
    }
    if (!added) break;
  }
  if (selected.empty()) {
    LOG_WARN("[DSV41-PREFILL-DECODE-POOL] no non-pinned prompt experts to stage into the dynamic pool\n");
    return false;
  }

  std::string reason;
  if (!topo.reset_unpinned_expert_cache(dynamic_slots, &reason)) {
    LOG_WARN("[DSV41-PREFILL-DECODE-POOL] cannot reset shared dynamic slots: %s\n",
             reason.c_str());
    return false;
  }

  size_t uploaded = 0;
  size_t source_misses = 0;
  size_t uploaded_bytes = 0;
  for (const Candidate &candidate : selected) {
    if (topo.main_layer_has(candidate.layer, candidate.expert)) continue;
    auto *block = ram.get_expert(engine, candidate.layer, candidate.expert);
    if (!block || !block->slab_buffer) {
      ++source_misses;
      continue;
    }
    ram.ram_pin(*block);
    const size_t bytes = block->gate_proj.meta.size_bytes +
        block->up_proj.meta.size_bytes + block->down_proj.meta.size_bytes;
    auto *loc = topo.upload_expert(candidate.layer, candidate.expert,
                                   block->slab_buffer, bytes);
    const bool ready = loc && loc->device_ptr && loc->upload_ready &&
        (!topo.expert_compact_layout || loc->pool_layer == -1);
    ram.ram_unpin(*block);
    if (!ready) {
      ++source_misses;
      continue;
    }
    ++uploaded;
    uploaded_bytes += bytes;
  }
  LOG_INFO("[DSV41-PREFILL-DECODE-POOL] prompt experts=%zu selected=%zu uploaded=%zu bytes=%.3f GB DDR_source_misses=%zu; shared preallocated slots were reused\n",
           observed, selected.size(), uploaded,
           (double)uploaded_bytes / 1.0e9, source_misses);
  return uploaded > 0;
}

bool deepseek_v41_full_graph_enabled() { return v41_full_graph_enabled(); }

int deepseek_v41_full_graph_device_expert_layers() {
  // The cache owns all expert storage; reserve no graph preload arena.
  return 0;
}

void deepseek_v41_full_graph_shutdown() {
  deepseek_v41_full_graph_decode_shutdown();
  g_dsv41_graph.reset();
  g_v41_prefill_sites.clear();
  // DeepSeek's GraphCore borrows the shared GLM dense arena. Release it with
  // the other shared graph resources while the CUDA manager is still alive;
  // leaving the process-wide owner for static destruction calls cudaFree
  // after driver teardown.
  glm5next_full_graph_shutdown();
}

bool deepseek_v41_full_graph_prepare(StreamingEngine &engine, int S,
                                     int pos_base, const int *token_ids) {
  if (!v41_full_graph_enabled() || S <= 1 || pos_base != 0 || !token_ids ||
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
    ggml_backend_cpu_set_n_threads(g->cpu, engine.omp_max_threads);
  }
  if (!g->cuda || !g->cpu)
    return false;
  // GPU graph nodes borrow native dense/cache device pointers. The host
  // callback path remains available as the numerical reference.
  std::vector<ggml_backend_t> backends{g->cuda};
  if (v41_gpu_prefill_enabled() &&
      zeromoe::graphsettings::active_settings().tun.cache_moe_graph.value != 0) {
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
  if (!v41_full_graph_enabled() || !hidden_states || S <= 1 || pos_base != 0 ||
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

bool deepseek_v41_rebalance_prefill_dynamic_cache(
    StreamingEngine &, RAMSafetyManager &,
    const std::vector<uint64_t> &) {
  return false;
}

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
