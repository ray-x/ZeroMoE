// Persistent DeepSeek-V4.1 decode graph. MoE weights belong exclusively to
// RAMSafetyManager/GpuTopology: the graph has NO expert source/weight arena.
// Host graph nodes call the existing cache-aware MoE dispatcher, which binds
// live DDR/VRAM slots and holds them through compute. Both GGUF layouts work.
#include "ggml_deepseek_v41_graph.hpp"
#include "graph_config.hpp"
#include "dsv41_attention.hpp"
#include "glm5next_graph_internal.hpp"
#include "cache_backed_moe.hpp"
#include "ram_manager.hpp"
#include "async_prefetcher.hpp"
#include "scratch_buffers.hpp"
#include "dsv41_trace.hpp"
#include "layer_plan.hpp"
#include "dense_dispatch.hpp"
#include "moe_router.hpp"
#include <stdexcept>

#if defined(ZEROMOE_USE_GGML_CUDA)
namespace {
using namespace glm5next_graph;
constexpr int kDecodeNodes = 32768;
enum DecodeTimer : size_t {
  kAttentionHost,
  kNativeQkv,
  kCacheMoe,
  kHcAndEngram,
  kRouterPilot,
  kLogitsHead,
  kDecodeTimerCount,
};

struct DecodeGraph;
struct DecodeSite { DecodeGraph *owner; int layer; };
struct DecodeHcSite { DecodeGraph *owner; int layer; int sub_block; };
struct DecodeGraph : FullGraph {
  // GPU hyper-connections: previous FFN pre-gate fed to the first attention
  // collapse (host input), written back to the engine after each token.
  ggml_tensor *hc_carry_in = nullptr;
  RAMSafetyManager *ram = nullptr;
  LockFreePrefetcher *prefetch = nullptr;
  LayerScratch *scratch = nullptr;
  int pos = 0;
  int token = 0;
  bool cache_moe_graph = false;
  std::vector<DecodeSite> sites;
  std::vector<DecodeHcSite> hc_sites;
  std::vector<float> hc_hidden_scratch;
  std::vector<float> logits_normed;
  struct Trace { int layer; const char *label; ggml_tensor *value; };
  std::vector<Trace> traces;
  std::unordered_map<std::string, ggml_tensor *> weights;
  uint64_t runs = 0;
  double warm_compute_ms = 0;
  double first_compute_ms = 0;
  std::array<double, kDecodeTimerCount> node_ms_total{};
  std::array<double, kDecodeTimerCount> warm_node_ms_total{};
  std::array<double, kDecodeTimerCount> first_node_ms{};
  std::vector<std::array<double, 3>> layer_attention_ms;
  std::vector<std::array<double, 3>> layer_moe_ms;
  PerfCounters::Snapshot perf_at_first_run{};
};
std::unique_ptr<DecodeGraph> decode;
StreamingEngine *cache_moe_graph_failed_engine = nullptr;
StreamingEngine *runtime_engine = nullptr;
RAMSafetyManager *runtime_ram = nullptr;
zeromoe::graph::CacheBackedMoeRuntime runtime_moe;
LockFreePrefetcher *runtime_prefetch = nullptr;
LayerScratch *runtime_scratch = nullptr;

bool v41_grouped_outproj_gpu_enabled() {
  static const bool enabled = [] {
    const char *value =
        zeromoe::graphcfg::env("ZEROMOE_DEEPSEEK_V41_OUTPROJ_GPU");
    return !value || value[0] != '0';
  }();
  return enabled;
}

void mark_trace(DecodeGraph &g, const char *label, int layer,
                ggml_tensor *value) {
  if (!value || !dsv41_trace::directory() ||
      (dsv41_trace::selected_layer() >= 0 &&
       dsv41_trace::selected_layer() != layer))
    return;
  g.traces.push_back({layer, label, value});
  ggml_set_output(value);
  ggml_build_forward_expand(g.graph, value);
}

ggml_tensor *weight(DecodeGraph &g, const std::string &name) {
  if (name.find(".ffn_") != std::string::npos)
    throw std::logic_error("MoE weights must use existing cache storage");
  if (auto it = g.weights.find(name); it != g.weights.end())
    return it->second;
  const auto *q = resident(*g.engine, name);
  if (!q || !type_ok(q->meta.type))
    return nullptr;
  std::vector<int64_t> shape(q->meta.shape.begin(), q->meta.shape.end());
  auto *t = new_tensor(g.ctx, static_cast<ggml_type>(q->meta.type), shape);
  if (!t || ggml_nbytes(t) > q->meta.size_bytes)
    return nullptr;
  if (q->device_ptr) {
    // Non-owning CUDA binding, as for GLM's physical expert slot views.
    // The buffer handle identifies the backend; GpuTopology/CudaManager
    // retains ownership of the actual device allocation.
    t->data = q->device_ptr;
    t->buffer = g.dense.buffer;
  } else {
    // This is the first device placement of a dense weight, never a DDR
    // copy. resident_tensor binds the engine's existing host_ptr as source.
    t = resident_tensor(g, g.ctx, name);
    if (!t)
      return nullptr;
  }
  ggml_set_name(t, name.c_str());
  g.weights.emplace(name, t);
  return t;
}

void engram_node(ggml_tensor *dst, const ggml_tensor *a, int ith, int,
                  void *ud) {
  if (ith != 0) return;
  const auto t0 = std::chrono::steady_clock::now();
  auto &s = *static_cast<DecodeSite *>(ud);
  auto &g = *s.owner;
  auto &e = *g.engine;
  std::memcpy(e.hc_stream_buf.data(), a->data, ggml_nbytes(a));
  e.family_ops.layer_input_fn(e, s.layer, 1, g.pos, &g.token);
  std::memcpy(dst->data, e.hc_stream_buf.data(), ggml_nbytes(dst));
  dsv41_trace::save("post-input-hook", s.layer, g.pos,
                    static_cast<const float *>(dst->data), g.D);
  g.node_ms_total[kHcAndEngram] +=
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
}

void hc_pre_node(ggml_tensor *dst, int ith, int, void *ud) {
  if (ith != 0) return;
  const auto t0 = std::chrono::steady_clock::now();
  auto &s = *static_cast<DecodeHcSite *>(ud);
  auto &g = *s.owner;
  auto &e = *g.engine;
  const size_t stream_bytes = (size_t)g.D * g.HC * sizeof(float);
  if (!dst->src[0] || !dst->src[0]->data || !e.family_ops.hc_pre_fn ||
      ggml_nbytes(dst->src[0]) != stream_bytes ||
      e.hc_stream_buf.size() < (size_t)g.D * g.HC)
    throw std::runtime_error("DeepSeek-V4.1 decode HC pre state is not ready");
  std::memcpy(e.hc_stream_buf.data(), dst->src[0]->data, stream_bytes);
  e.family_ops.hc_pre_fn(e, s.layer, s.sub_block,
                         static_cast<float *>(dst->data), 1, g.D);
  const char *label = s.sub_block == 0 ? "hc-attn-pre" : "hc-ffn-pre";
  dsv41_trace::save(label, s.layer, g.pos,
                    static_cast<const float *>(dst->data), g.D);
  g.node_ms_total[kHcAndEngram] +=
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
}

// Preserve the legacy depth-1 router lookahead inside the full graph. The
// prediction is made from the just-combined stream-0 state and only queues
// cache-miss hints into the existing prefetcher; it never allocates expert
// storage or copies weights into the graph.
void prefetch_next_layer(DecodeGraph &g, int layer, const float *hidden) {
  if (!hidden || !g.ram || !g.prefetch || !g.scratch ||
      layer + 1 >= g.n_layers)
    return;
  auto &e = *g.engine;
  const auto &c = e.model_file.architecture_config;
  const int target = layer + 1;
  int k = std::min(pilot_k_for_layer(target), c.topk);
  k = std::min(adaptive_pilot_k_for_layer(target, k), 8);
  auto &scratch = *g.scratch;
  scratch.pilot_pred_count = 0;
  scratch.pilot_pred_layer = -1;
  scratch.pilot_pred_was_hint.fill(false);
  scratch.pilot_pred_was_ram_only.fill(false);
  if (k <= 0 || scratch.pilot_nrm.size() < (size_t)g.D ||
      scratch.pilot_logits.size() < (size_t)c.n_experts)
    return;

  const std::string p = "blk." + std::to_string(target) + ".";
  const auto norm = e.resident_layers.find(p + "ffn_norm.weight");
  const auto router = e.resident_layers.find(p + "ffn_gate_inp.weight");
  if (norm == e.resident_layers.end() || router == e.resident_layers.end() ||
      !norm->second.host_ptr || !router->second.host_ptr)
    return;
  rmsnorm_row(scratch.pilot_nrm.data(), hidden,
              reinterpret_cast<const float *>(norm->second.host_ptr), g.D,
              c.eps);
  dense_proj(e, p + "ffn_gate_inp.weight", scratch.pilot_nrm.data(),
             scratch.pilot_logits.data(), 1, g.D, c.n_experts);
  apply_gating_activation(c.gating_func, scratch.pilot_logits.data(),
                          c.n_experts);
  const auto bias = e.resident_layers.find(p + "exp_probs_b.bias");
  if (bias != e.resident_layers.end() && bias->second.host_ptr &&
      bias->second.meta.type == GGUFType::F32) {
    const auto *b = reinterpret_cast<const float *>(bias->second.host_ptr);
    for (int expert = 0; expert < c.n_experts; ++expert)
      scratch.pilot_logits[(size_t)expert] += b[expert];
  }

  scratch.pilot_pred_count = k;
  scratch.pilot_pred_layer = target;
  for (int rank = 0; rank < k; ++rank) {
    int best = 0;
    for (int expert = 1; expert < c.n_experts; ++expert)
      if (scratch.pilot_logits[(size_t)expert] >
          scratch.pilot_logits[(size_t)best])
        best = expert;
    scratch.pilot_logits[(size_t)best] = -2e30f;
    scratch.pilot_pred_ids[(size_t)rank] = best;
    const bool in_ram = g.ram->is_ram_resident(target, best);
    const bool in_vram = g.ram->vram_cache &&
        (g.ram->vram_cache->main_layer_has(target, best) ||
         g.ram->vram_cache->active_layer_has(target, best));
    scratch.pilot_pred_was_ram_only[(size_t)rank] = in_ram && !in_vram;
    scratch.pilot_pred_was_hint[(size_t)rank] = !in_ram && !in_vram;
    if (!in_ram && !in_vram)
      g.prefetch->push_hint(target, best);
  }
}

void hc_post_node(ggml_tensor *dst, int ith, int, void *ud) {
  if (ith != 0) return;
  const auto t0 = std::chrono::steady_clock::now();
  auto &s = *static_cast<DecodeHcSite *>(ud);
  auto &g = *s.owner;
  auto &e = *g.engine;
  const size_t stream_elements = (size_t)g.D * g.HC;
  if (!e.family_ops.hc_post_fn || !dst->src[0] || !dst->src[0]->data ||
      !dst->src[1] || !dst->src[1]->data ||
      ggml_nelements(dst->src[0]) != (size_t)g.D ||
      ggml_nelements(dst->src[1]) != stream_elements ||
      e.hc_stream_buf.size() < stream_elements ||
      g.hc_hidden_scratch.size() < (size_t)g.D)
    throw std::runtime_error("DeepSeek-V4.1 decode HC post state is not ready");
  std::memcpy(e.hc_stream_buf.data(), dst->src[1]->data,
              stream_elements * sizeof(float));
  e.family_ops.hc_post_fn(e, s.layer, s.sub_block,
                          g.hc_hidden_scratch.data(),
                          static_cast<const float *>(dst->src[0]->data), 1,
                          g.D);
  std::memcpy(dst->data, e.hc_stream_buf.data(),
              stream_elements * sizeof(float));
  const char *hidden_label =
      s.sub_block == 0 ? "post-attn" : "post-ffn";
  const char *stream_label =
      s.sub_block == 0 ? "post-attn-streams" : "post-ffn-streams";
  dsv41_trace::save(hidden_label, s.layer, g.pos,
                    g.hc_hidden_scratch.data(), (size_t)g.D);
  dsv41_trace::save(stream_label, s.layer, g.pos, e.hc_stream_buf.data(),
                    stream_elements);
  const auto pilot_start = std::chrono::steady_clock::now();
  if (s.sub_block == 1)
    prefetch_next_layer(g, s.layer, g.hc_hidden_scratch.data());
  const auto finished = std::chrono::steady_clock::now();
  g.node_ms_total[kHcAndEngram] +=
      std::chrono::duration<double, std::milli>(pilot_start - t0).count();
  if (s.sub_block == 1)
    g.node_ms_total[kRouterPilot] +=
        std::chrono::duration<double, std::milli>(finished - pilot_start).count();
}

void final_collapse_node(ggml_tensor *dst, int ith, int, void *ud) {
  if (ith != 0) return;
  const auto t0 = std::chrono::steady_clock::now();
  auto &g = *static_cast<DecodeGraph *>(ud);
  auto &e = *g.engine;
  if (!dst->src[0] || !dst->src[0]->data ||
      ggml_nelements(dst->src[0]) != (size_t)g.D * g.HC ||
      e.hc_final_pre_mix.size() != (size_t)g.HC)
    throw std::runtime_error("DeepSeek-V4.1 decode final HC mix is not ready");
  const auto *streams = static_cast<const float *>(dst->src[0]->data);
  const float *mix = e.hc_final_pre_mix.data();
  auto *out = static_cast<float *>(dst->data);
  std::fill(out, out + g.D, 0.0f);
  for (int h = 0; h < g.HC; ++h) {
    const float *stream = streams + (size_t)h * g.D;
    const float w = mix[h];
    for (int d = 0; d < g.D; ++d) out[d] += w * stream[d];
  }
  g.node_ms_total[kHcAndEngram] +=
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
}

// Match the engine's output head: GGML's quantized MUL_MAT rounds the
// activation to Q8, whereas dense_proj consumes F32 activations. Keep this
// node in the scheduler graph and borrow the same persistent dense weights.
void logits_node(ggml_tensor *dst, int ith, int, void *ud) {
  if (ith != 0) return;
  const auto t0 = std::chrono::steady_clock::now();
  auto &g = *static_cast<DecodeGraph *>(ud);
  auto &e = *g.engine;
  const auto &c = e.model_file.architecture_config;
  const auto *hidden = static_cast<const float *>(dst->src[0]->data);
  rmsnorm_proj(e, "output_norm.weight", hidden, g.logits_normed.data(),
               1, g.D, c.eps);
  dense_proj(e, "output.weight", g.logits_normed.data(),
             static_cast<float *>(dst->data), 1, g.D, c.vocab);
  g.node_ms_total[kLogitsHead] +=
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
}

void swa_node(ggml_tensor *dst, const ggml_tensor *a, int ith, int,
               void *ud) {
  if (ith != 0) return;
  auto &s = *static_cast<DecodeSite *>(ud);
  auto &g = *s.owner;
  dsv41_trace::save("attn-input", s.layer, g.pos,
                    static_cast<const float *>(a->data), g.D);
  const auto t0 = std::chrono::steady_clock::now();
  g.engine->family_ops.attention_fn(*g.engine, s.layer,
      static_cast<const float *>(a->data), 1, g.pos,
      static_cast<float *>(dst->data));
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  g.node_ms_total[kAttentionHost] += ms;
  g.layer_attention_ms[(size_t)s.layer][0] += ms;
}

void compressed_node(ggml_tensor *dst, int ith, int, void *ud) {
  if (ith != 0) return;
  const ggml_tensor *query = dst->src[0];
  const ggml_tensor *kv = dst->src[1];
  const ggml_tensor *norm_ql = dst->src[2];
  if (!query || !kv || !norm_ql || !query->data || !kv->data ||
      !norm_ql->data)
    throw std::runtime_error("DeepSeek-V4.1 compressed attention inputs are missing");
  auto &s = *static_cast<DecodeSite *>(ud);
  auto &g = *s.owner;
  auto *in = static_cast<const float *>(norm_ql->data);
  const auto t0 = std::chrono::steady_clock::now();
  deepseek41_attention_projected(*g.engine, s.layer, g.pos, in, in + g.D,
      static_cast<const float *>(query->data),
      static_cast<const float *>(kv->data), static_cast<float *>(dst->data));
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  g.node_ms_total[kAttentionHost] += ms;
  g.layer_attention_ms[(size_t)s.layer][0] += ms;
}

void compressed_context_node(ggml_tensor *dst, int ith, int, void *ud) {
  if (ith != 0) return;
  const ggml_tensor *query = dst->src[0];
  const ggml_tensor *kv = dst->src[1];
  const ggml_tensor *norm_ql = dst->src[2];
  if (!query || !kv || !norm_ql || !query->data || !kv->data ||
      !norm_ql->data)
    throw std::runtime_error("DeepSeek-V4.1 compressed attention inputs are missing");
  auto &s = *static_cast<DecodeSite *>(ud);
  auto &g = *s.owner;
  auto *in = static_cast<const float *>(norm_ql->data);
  const auto t0 = std::chrono::steady_clock::now();
  deepseek41_attention_projected_context(
      *g.engine, s.layer, g.pos, in, in + g.D,
      static_cast<const float *>(query->data),
      static_cast<const float *>(kv->data), static_cast<float *>(dst->data));
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  g.node_ms_total[kAttentionHost] += ms;
  g.layer_attention_ms[(size_t)s.layer][0] += ms;
}

void grouped_outproj_node(ggml_tensor *dst, int ith, int, void *ud) {
  if (ith != 0) return;
  auto &s = *static_cast<DecodeSite *>(ud);
  auto &g = *s.owner;
  auto &e = *g.engine;
  const auto t0 = std::chrono::steady_clock::now();
  const auto &c = e.model_file.architecture_config;
  const std::string p = "blk." + std::to_string(s.layer) + ".";
  const int qh = g.qk_nope + g.qk_rope;
  const int groups = std::max(1, c.o_group_count);
  const int rank = c.o_lora_rank;
  const int group_in = (g.H * qh) / groups;
  if (!dst->src[0] || !dst->src[0]->data ||
      ggml_nelements(dst->src[0]) != (size_t)g.H * qh || rank <= 0 ||
      group_in * groups != g.H * qh)
    throw std::runtime_error("DeepSeek-V4.1 grouped output projection shape is invalid");

  const std::string wa_name = p + "attn_output_a.weight";
  const std::string wb_name = p + "attn_output_b.weight";
  auto wa_it = g.weights.find(wa_name);
  auto wb_it = g.weights.find(wb_name);
  const auto *wa_meta = resident(e, wa_name);
  const auto *wb_meta = resident(e, wb_name);
  if (wa_it == g.weights.end() || wb_it == g.weights.end() || !wa_meta ||
      !wb_meta || !wa_meta->host_ptr)
    throw std::runtime_error("DeepSeek-V4.1 output projection weights are missing");

  std::vector<float> grouped((size_t)groups * rank);
  const auto *context = static_cast<const float *>(dst->src[0]->data);
  bool gpu_woa = false;
  if (wa_it->second && wa_it->second->data && e.gpu &&
      wa_it->second->type == GGML_TYPE_Q2_K) {
    const size_t row_bytes = ggml_row_size(wa_it->second->type, group_in);
    const size_t group_stride = row_bytes * (size_t)rank;
    gpu_woa = e.gpu->grouped_output_proj_q2k_host(
        wa_it->second->data, group_stride, groups, rank, group_in, context,
        grouped.data());
  }
  if (!gpu_woa) {
    static std::unordered_map<std::string, std::vector<float>> woa_cache;
    auto &weights = woa_cache[e.model_file.resolved_file_path + "#" + wa_name];
    const size_t n = (size_t)groups * rank * group_in;
    if (weights.empty()) {
      weights.resize(n);
      dequant_row(wa_meta->meta.type, wa_meta->host_ptr, weights.data(),
                  (int)n);
    }
#pragma omp parallel for schedule(static)
    for (int go = 0; go < groups * rank; ++go) {
      const int group = go / rank;
      const int row = go % rank;
      grouped[(size_t)go] = dot_f32_avx512(
          context + (size_t)group * group_in,
          weights.data() + (size_t)go * group_in, group_in);
    }
  }

  bool gpu_wob = false;
  if (e.gpu && wb_it->second && wb_it->second->data)
    gpu_wob = e.gpu->dense_matmul_host(
        wb_meta->meta.type, wb_it->second->data, grouped.data(),
        static_cast<float *>(dst->data), 1, groups * rank, g.D);
  if (!gpu_wob)
    dense_proj(e, wb_name, grouped.data(), static_cast<float *>(dst->data), 1,
               groups * rank, g.D);
  dsv41_trace::save("attn-out-projected", s.layer, g.pos,
                    static_cast<const float *>(dst->data), g.D);
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  g.node_ms_total[kAttentionHost] += ms;
  g.layer_attention_ms[(size_t)s.layer][0] += ms;
}

// Preserve the legacy Q2_K F32-activation projection here. ggml-cuda
// quantizes activations for Q2_K MUL_MAT; the accumulated Q/K drift changed
// routes by layer 8. dense_proj reads the engine's resident DDR weight bytes.
// This graph stores only the small projection outputs, never another weight.
void qkv_source_node(ggml_tensor *dst, int ith, int, void *ud) {
  if (ith != 0) return;
  auto &s = *static_cast<DecodeSite *>(ud);
  auto &g = *s.owner;
  const auto t0 = std::chrono::steady_clock::now();
  auto &e = *g.engine;
  const auto &c = e.model_file.architecture_config;
  const std::string p = "blk." + std::to_string(s.layer) + ".";
  const auto *an = resident(e, p + "attn_norm.weight");
  const auto *qa = resident(e, p + "attn_q_a.weight");
  const auto *qb = resident(e, p + "attn_q_b.weight");
  const auto *kvn = resident(e, p + "attn_kv_a_norm.weight");
  if (!an || !qa || !qb || !kvn || qa->meta.shape.size() < 2 ||
      !an->host_ptr || !qa->host_ptr || !qb->host_ptr || !kvn->host_ptr)
    throw std::runtime_error("missing resident Q-lora projection in V4.1 decode");
  const int ql = static_cast<int>(qa->meta.shape[1]);
  const int qh = c.qk_nope + c.qk_rope;
  const int query_n = g.H * qh;
  const auto *input = static_cast<const float *>(dst->src[0]->data);
  auto *out = static_cast<float *>(dst->data);
  auto *normed = out;
  auto *ql_out = normed + g.D;
  auto *query = ql_out + ql;
  auto *raw_kv = query + query_n;
  dsv41_trace::save("attn-input", s.layer, g.pos, input, g.D);
  rmsnorm_proj(e, p + "attn_norm.weight", input, normed, 1, g.D, c.eps);
  dsv41_trace::save("attn-norm", s.layer, g.pos, normed, g.D);
  dense_proj(e, p + "attn_q_a.weight", normed, ql_out, 1, g.D, ql);
  rmsnorm_proj(e, p + "attn_q_a_norm.weight", ql_out, ql_out, 1, ql,
               c.eps);
  dsv41_trace::save("q-lora", s.layer, g.pos, ql_out, ql);
  const auto qbit = g.weights.find(p + "attn_q_b.weight");
  if (!e.gpu || e.gpu->device_id != g.device || qbit == g.weights.end() ||
      !e.gpu->dense_matmul_host(qb->meta.type, qbit->second->data, ql_out,
                                query, 1, ql, query_n))
    throw std::runtime_error("V4.1 Q projection failed in native kernel");
  dsv41_trace::save("query", s.layer, g.pos, query, (size_t)g.H * qh);
  dense_proj(e, p + "attn_kv.weight", normed, raw_kv, 1, g.D, qh);
  rmsnorm_proj(e, p + "attn_kv_a_norm.weight", raw_kv, raw_kv, 1, qh,
               c.eps);
  dsv41_trace::save("raw-kv", s.layer, g.pos, raw_kv, qh);
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  g.node_ms_total[kNativeQkv] += ms;
}

void moe_node(ggml_tensor *dst, const ggml_tensor *a, int ith, int,
               void *ud) {
  if (ith != 0) return;
  auto &s = *static_cast<DecodeSite *>(ud);
  auto &g = *s.owner;
  // No model_tensor(), mmap(), expert_source(), malloc(), or weight copy
  // belongs here. The dispatcher acquires DDR claims/VRAM compute holds and
  // resolves packed or unpacked disk offsets only on a genuine cache miss.
  const auto t0 = std::chrono::steady_clock::now();
  runtime_moe.run(s.layer, static_cast<const float *>(a->data), 1,
                  static_cast<float *>(dst->data), &g.token);
  if (dsv41_trace::enabled(s.layer, g.pos) &&
      (int)g.scratch->tok_topk_ids.size() >=
          g.engine->model_file.architecture_config.topk &&
      (int)g.scratch->tok_topk_weights.size() >=
          g.engine->model_file.architecture_config.topk) {
    const int k = g.engine->model_file.architecture_config.topk;
    std::vector<float> ids((size_t)k);
    for (int i = 0; i < k; ++i)
      ids[(size_t)i] =
          static_cast<float>(g.scratch->tok_topk_ids[(size_t)i]);
    dsv41_trace::save("route-ids", s.layer, g.pos, ids.data(), ids.size());
    dsv41_trace::save("route-weights", s.layer, g.pos,
                      g.scratch->tok_topk_weights.data(), (size_t)k);
  }
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  g.node_ms_total[kCacheMoe] += ms;
  g.layer_moe_ms[(size_t)s.layer][0] += ms;
}

// V4.1 carries the previous FFN pre-mix into attention, and the current
// attention pre-mix into FFN. GLM's immediate per-block collapse is different.
ggml_tensor *hc(DecodeGraph &g, ggml_tensor *res, int layer, bool ffn,
                 ggml_tensor *carry, ggml_tensor **pre,
                 ggml_tensor **post, ggml_tensor **comb) {
  const auto &c = g.engine->model_file.architecture_config;
  const std::string p = "blk." + std::to_string(layer) +
                        (ffn ? ".hc_ffn_" : ".hc_attn_");
  auto *fn = weight(g, p + "fn.weight");
  auto *sc = weight(g, p + "scale.weight");
  auto *ba = weight(g, p + "base.weight");
  if (!fn || !sc || !ba) return nullptr;
  auto *mix = ggml_mul_mat(g.ctx, fn, ggml_rms_norm(g.ctx,
      ggml_reshape_2d(g.ctx, res, g.D * g.HC, 1), c.eps));
  auto affine = [&](int off, int scale_idx) {
    auto *x = ggml_view_2d(g.ctx, mix, g.HC, 1, mix->nb[1],
                           off * sizeof(float));
    auto *s = ggml_view_1d(g.ctx, sc, 1, scale_idx * sizeof(float));
    auto *b = ggml_view_1d(g.ctx, ba, g.HC, off * sizeof(float));
    return ggml_add(g.ctx, ggml_mul(g.ctx, x, s), b);
  };
  *pre = ggml_scale_bias(g.ctx, ggml_sigmoid(g.ctx, affine(0, 0)),
                          1.0f, c.hc_eps);
  *post = ggml_scale(g.ctx, ggml_sigmoid(g.ctx, affine(g.HC, 1)), 2.0f);
  *comb = ggml_dsv4_hc_comb(g.ctx, mix, sc, ba, c.hc_eps,
                            c.hc_sinkhorn_iters);
  return ggml_dsv4_hc_pre(g.ctx, res, carry);
}

ggml_tensor *attention(DecodeGraph &g, ggml_tensor *cur, int layer) {
  auto *site = &g.sites[layer];
  const auto &c = g.engine->model_file.architecture_config;
  const int ratio = (int)c.dsv4_compress_ratios[(size_t)layer];
  if (ratio == 0) {
    // Raw-SWA layers retain the complete native path. Their grouped output
    // projection and cache update are handled by dsv4_attention_gguf.
    auto *out = ggml_map_custom1(g.ctx, cur, swa_node, 1, site);
    if (!out) return nullptr;
    ggml_backend_sched_set_tensor_backend(g.sched, out, g.cpu);
    mark_trace(g, "attn-out-projected", layer, out);
    return out;
  }

  // Compressed layers keep the stateful cache/indexer update and grouped
  // output projection in the native path. Q_B uses its resident CUDA weight
  // binding; the native F32 grouped output math is retained for parity.
  const std::string p = "blk." + std::to_string(layer) + ".";
  const auto *qa = resident(*g.engine, p + "attn_q_a.weight");
  if (!qa || qa->meta.shape.size() < 2) return nullptr;
  const int ql = (int)qa->meta.shape[1];
  const int qh = g.qk_nope + g.qk_rope;
  const int query_n = g.H * qh;
  const int64_t qkv_n = (int64_t)g.D + ql + query_n + qh;
  if (!weight(g, p + "attn_q_b.weight")) return nullptr;
  ggml_tensor *qkv_src[] = {cur};
  auto *qkv = ggml_custom_4d(g.ctx, GGML_TYPE_F32, qkv_n, 1, 1, 1,
                              qkv_src, 1, qkv_source_node, 1, site);
  if (!qkv) return nullptr;
  ggml_backend_sched_set_tensor_backend(g.sched, qkv, g.cpu);
  auto *norm_ql = ggml_view_1d(g.ctx, qkv, (int64_t)g.D + ql, 0);
  auto *query = ggml_view_1d(g.ctx, qkv, query_n,
                             ((size_t)g.D + ql) * sizeof(float));
  auto *kv = ggml_view_1d(g.ctx, qkv, qh,
      ((size_t)g.D + ql + query_n) * sizeof(float));
  ggml_tensor *attn_args[] = {query, kv, norm_ql};
  ggml_tensor *out = nullptr;
  if (v41_grouped_outproj_gpu_enabled()) {
    if (!weight(g, p + "attn_output_a.weight") ||
        !weight(g, p + "attn_output_b.weight"))
      return nullptr;
    auto *context = ggml_custom_4d(
        g.ctx, GGML_TYPE_F32, g.H * qh, 1, 1, 1, attn_args, 3,
        compressed_context_node, 1, site);
    if (!context) return nullptr;
    ggml_backend_sched_set_tensor_backend(g.sched, context, g.cpu);
    ggml_tensor *output_args[] = {context};
    out = ggml_custom_4d(g.ctx, GGML_TYPE_F32, g.D, 1, 1, 1,
                         output_args, 1, grouped_outproj_node, 1, site);
  } else {
    out = ggml_custom_4d(g.ctx, GGML_TYPE_F32, g.D, 1, 1, 1,
                         attn_args, 3, compressed_node, 1, site);
  }
  if (!out) return nullptr;
  ggml_backend_sched_set_tensor_backend(g.sched, out, g.cpu);
  mark_trace(g, "attn-out-projected", layer, out);
  return out;
}

bool build(DecodeGraph &g) {
  const auto &c = g.engine->model_file.architecture_config;
  g.layer_attention_ms.assign((size_t)g.n_layers, {});
  g.layer_moe_ms.assign((size_t)g.n_layers, {});
  // Context and scheduler workspace are fixed for S=1, including new turns.
  const size_t context_bytes = 32ull << 20;
  if (!MemoryGuard::try_reserve_host_bytes(context_bytes)) return false;
  g.context_reservation_bytes = context_bytes;
  g.ctx = ggml_init({context_bytes, nullptr, true});
  if (!g.ctx) return false;
  g.graph = ggml_new_graph_custom(g.ctx, kDecodeNodes, false);
  if (!init_dense_arena(g)) return false;
  ggml_backend_t backends[] = {g.cuda, g.cpu, nullptr};
  int backend_count = 2;
  if (g.cache_moe_graph && g.gpu1) {
    backends[1] = g.gpu1;
    backends[2] = g.cpu;
    backend_count = 3;
  }
  g.sched = ggml_backend_sched_new(backends, nullptr, backend_count,
                                   kDecodeNodes, false, true);
  if (!g.sched) return false;
  g.input = host_input(g, g.ctx, GGML_TYPE_F32, {g.D, g.HC, 1},
                        g.D * g.HC * sizeof(float), "dsv41_decode_streams");
  if (!g.input) return false;
  g.hc_hidden_scratch.resize((size_t)g.D);
  g.logits_normed.resize((size_t)g.D);
  g.sites.reserve(g.n_layers);
  for (int l = 0; l < g.n_layers; ++l) g.sites.push_back({&g, l});
  g.hc_sites.reserve((size_t)g.n_layers * 2);
  for (int l = 0; l < g.n_layers; ++l) {
    g.hc_sites.push_back({&g, l, 0});
    g.hc_sites.push_back({&g, l, 1});
  }
  // Default on; ZEROMOE_DEEPSEEK_V41_DECODE_GPU_HC=0 selects the host nodes.
  const bool gpu_hc_decode =
      !zeromoe::graphcfg::env("ZEROMOE_DEEPSEEK_V41_DECODE_GPU_HC") ||
      zeromoe::graphcfg::flag_on("ZEROMOE_DEEPSEEK_V41_DECODE_GPU_HC");
  ggml_tensor *carry = nullptr;
  if (gpu_hc_decode) {
    g.hc_carry_in = host_input(g, g.ctx, GGML_TYPE_F32, {g.HC, 1},
                               g.HC * sizeof(float), "dsv41_decode_hc_carry");
    if (!g.hc_carry_in) return false;
    carry = g.hc_carry_in;
    LOG_INFO("[GGML-DSV41-DECODE] hyper-connections on GPU\n");
  }
  // The layer loop runs on the shared interfaces (layer_plan.hpp): DsvAttention and DsvFfn are the blocks,
  // DsvHcPlan is the DeepSeek-V4.1 residual structure (hyper-connection streams with a carried pre-mix, GPU
  // nodes or host nodes, and the engram hook at the start of a layer).
  struct DsvAttention final : zeromoe::graph::AttentionBlock {
    explicit DsvAttention(DecodeGraph &graph) : g(graph) {}
    DecodeGraph &g;
    bool declare_inputs() override { return true; }
    void set_inputs(int) override {}
    ggml_tensor *build(ggml_tensor *cur, int l) override { return attention(g, cur, l); }
  };
  struct DsvFfn final : zeromoe::graph::FfnBlock {
    explicit DsvFfn(DecodeGraph &graph) : g(graph) {}
    DecodeGraph &g;
    ggml_tensor *build(ggml_tensor *cur, int l) override {
      mark_trace(g, "hc-ffn-pre", l, cur);
      ggml_tensor *ffn = nullptr;
      if (g.cache_moe_graph) {
        ffn = v41_build_cache_moe(g, cur, l, &g.token, &g.pos);
        if (!ffn) {
          LOG_ERROR("[GGML-DSV41-DECODE] cache-backed MoE graph build failed at layer %d\n", l);
          return nullptr;
        }
      } else {
        ffn = ggml_map_custom1(g.ctx, cur, moe_node, 1, &g.sites[l]);
        ggml_backend_sched_set_tensor_backend(g.sched, ffn, g.cpu);
      }
      mark_trace(g, "moe-out", l, ffn);
      return ffn;
    }
  };
  struct DsvHcPlan final : zeromoe::graph::LayerPlan {
    DsvHcPlan(DecodeGraph &graph, const ArchitectureConfig &cfg, DsvAttention &attention, DsvFfn &ffn_block,
              bool gpu, ggml_tensor *carry_in)
        : g(graph), c(cfg), attn(attention), ffn(ffn_block), gpu_hc_decode(gpu), carry(carry_in) {}
    DecodeGraph &g;
    const ArchitectureConfig &c;
    DsvAttention &attn;
    DsvFfn &ffn;
    bool gpu_hc_decode;
    ggml_tensor *carry;  // the pre-mix carried between layers (GPU hyper-connections); read after the loop
    ggml_tensor *build_layer(ggml_tensor *res, int l) override {
    if (MemoryGuard::allocation_blocked()) return nullptr;
    mark_trace(g, "input", l,
               ggml_view_2d(g.ctx, res, g.D, 1, res->nb[2], 0));
    if (std::find(c.dsv41_engram_layer_ids.begin(),
                  c.dsv41_engram_layer_ids.end(), l) !=
        c.dsv41_engram_layer_ids.end()) {
      res = ggml_map_custom1(g.ctx, res, engram_node, 1, &g.sites[l]);
      ggml_backend_sched_set_tensor_backend(g.sched, res, g.cpu);
    }
    mark_trace(g, "post-input-hook", l,
               ggml_view_2d(g.ctx, res, g.D, 1, res->nb[2], 0));
    ggml_tensor *cur = nullptr, *attn_out = nullptr;
    ggml_tensor *gpu_post = nullptr, *gpu_comb = nullptr, *gpu_attn_pre = nullptr;
    // GPU hyper-connections (same ops as the prefill trunk): the host nodes
    // below cost ~0.7 ms each x 80 per token on CPU.
    auto gpu_hc = [&](const char *stem, ggml_tensor *input, ggml_tensor *collapse,
                      ggml_tensor **post, ggml_tensor **comb, ggml_tensor **pre)
        -> ggml_tensor * {
      const std::string p = "blk." + std::to_string(l) + ".";
      auto *fn = weight(g, p + stem + "_fn.weight");
      auto *scale = weight(g, p + stem + "_scale.weight");
      auto *base = weight(g, p + stem + "_base.weight");
      if (!fn || !scale || !base) return nullptr;
      return hc_pre(g, g.ctx, input, fn, scale, base, post, comb, pre, collapse);
    };
    if (gpu_hc_decode) {
      cur = gpu_hc("hc_attn", res, carry, &gpu_post, &gpu_comb, &gpu_attn_pre);
      if (!cur) return nullptr;
    } else {
      ggml_tensor *attn_hc_args[] = {res};
      cur = ggml_custom_4d(
          g.ctx, GGML_TYPE_F32, g.D, 1, 1, 1, attn_hc_args, 1,
          hc_pre_node, 1, &g.hc_sites[(size_t)l * 2]);
      if (!cur) return nullptr;
      ggml_backend_sched_set_tensor_backend(g.sched, cur, g.cpu);
    }
    attn_out = attn.build(cur, l);
    if (!attn_out) return nullptr;
    if (gpu_hc_decode) {
      res = hc_post(g, g.ctx, attn_out, res, gpu_post, gpu_comb);
      if (!res) return nullptr;
      cur = gpu_hc("hc_ffn", res, gpu_attn_pre, &gpu_post, &gpu_comb, &carry);
      if (!cur) return nullptr;
    } else {
      ggml_tensor *post_attn_args[] = {attn_out, res};
      res = ggml_custom_4d(
          g.ctx, GGML_TYPE_F32, g.D, g.HC, 1, 1, post_attn_args, 2,
          hc_post_node, 1, &g.hc_sites[(size_t)l * 2]);
      if (!res) return nullptr;
      ggml_backend_sched_set_tensor_backend(g.sched, res, g.cpu);

      ggml_tensor *ffn_hc_args[] = {res};
      cur = ggml_custom_4d(
          g.ctx, GGML_TYPE_F32, g.D, 1, 1, 1, ffn_hc_args, 1,
          hc_pre_node, 1, &g.hc_sites[(size_t)l * 2 + 1]);
      if (!cur) return nullptr;
      ggml_backend_sched_set_tensor_backend(g.sched, cur, g.cpu);
    }
    ggml_tensor *ffn_out = ffn.build(cur, l);
    if (!ffn_out) return nullptr;
    if (gpu_hc_decode) {
      res = hc_post(g, g.ctx, ffn_out, res, gpu_post, gpu_comb);
      if (!res) return nullptr;
    } else {
      ggml_tensor *post_ffn_args[] = {ffn_out, res};
      res = ggml_custom_4d(
          g.ctx, GGML_TYPE_F32, g.D, g.HC, 1, 1, post_ffn_args, 2,
          hc_post_node, 1, &g.hc_sites[(size_t)l * 2 + 1]);
      if (!res) return nullptr;
      ggml_backend_sched_set_tensor_backend(g.sched, res, g.cpu);
    }
    return res;
    }
  };
  DsvAttention attention_block(g);
  DsvFfn ffn_block(g);
  DsvHcPlan plan(g, c, attention_block, ffn_block, gpu_hc_decode, carry);
  auto *res = g.input;
  for (int l = 0; l < g.n_layers; ++l) {
    res = plan.build_layer(res, l);
    if (!res) return false;
  }
  carry = plan.carry;
  g.final_streams = res;
  if (gpu_hc_decode) {
    g.output = ggml_dsv4_hc_pre(g.ctx, ggml_cont(g.ctx, res), ggml_cont(g.ctx, carry));
    g.dsv41_final_pre_mix = carry;
    ggml_set_output(carry);
    ggml_build_forward_expand(g.graph, carry);
  } else {
    ggml_tensor *collapse_args[] = {res};
    g.output = ggml_custom_4d(g.ctx, GGML_TYPE_F32, g.D, 1, 1, 1,
                              collapse_args, 1, final_collapse_node, 1, &g);
    if (g.output) ggml_backend_sched_set_tensor_backend(g.sched, g.output, g.cpu);
  }
  if (!g.output) return false;
  mark_trace(g, "final-hidden", -1, g.output);
  if (!resident(*g.engine, "output_norm.weight") ||
      !resident(*g.engine, "output.weight")) return false;
  ggml_tensor *logits_args[] = {g.output};
  g.logits = ggml_custom_4d(g.ctx, GGML_TYPE_F32, c.vocab, 1, 1, 1,
                            logits_args, 1, logits_node, 1, &g);
  if (!g.logits) return false;
  ggml_backend_sched_set_tensor_backend(g.sched, g.logits, g.cpu);
  for (auto *t : {g.final_streams, g.output, g.logits}) {
    ggml_set_output(t);
    ggml_build_forward_expand(g.graph, t);
  }
  if (!graph_alloc_vram_headroom_ok(g, "[GGML-DSV41-DECODE]") ||
      !ggml_backend_sched_alloc_graph(g.sched, g.graph)) return false;
  g.graph_workspace_bytes = ggml_backend_sched_get_buffer_size(g.sched, g.cuda);
  LOG_INFO("[GGML-DSV41-DECODE] built nodes=%d splits=%d dense=%.3f GiB "
           "dense_capacity=%.3f GiB "
           "workspace=%.3f GiB; graph-owned MoE bytes=0 (existing DDR/VRAM "
           "cache dispatcher)\n", ggml_graph_n_nodes(g.graph),
           ggml_backend_sched_get_n_splits(g.sched), g.dense.used / double(1ull << 30),
           g.dense.capacity / double(1ull << 30),
           g.graph_workspace_bytes / double(1ull << 30));
  g.ready = true;
  return true;
}
} // namespace

bool deepseek_v41_full_graph_decode_enabled() {
  if (zeromoe::graphcfg::env("ZEROMOE_DEEPSEEK_V41_FULL_GRAPH_DECODE"))
    return zeromoe::graphcfg::flag_on("ZEROMOE_DEEPSEEK_V41_FULL_GRAPH_DECODE");
  return zeromoe::graphcfg::flag_on("ZEROMOE_DEEPSEEK_V41_FULL_GRAPH");
}

void deepseek_v41_bind_graph_runtime(StreamingEngine &e, RAMSafetyManager &r,
                                     LockFreePrefetcher &p, LayerScratch &s) {
  runtime_moe.bind(e, r, p, s);
  runtime_engine = &e; runtime_ram = &r;
  runtime_prefetch = &p; runtime_scratch = &s;
}

bool deepseek_v41_graph_runtime(StreamingEngine &e, RAMSafetyManager *&ram,
                                LockFreePrefetcher *&prefetch, LayerScratch *&scratch) {
  if (runtime_engine != &e) return false;
  ram = runtime_ram; prefetch = runtime_prefetch; scratch = runtime_scratch;
  return ram && prefetch && scratch;
}

void deepseek_v41_full_graph_moe_forward(StreamingEngine &e, int layer,
                                         const float *input, int S,
                                         int pos_base,
                                         float *output,
                                         const int *token_ids) {
  if (runtime_engine != &e || !runtime_ram || !runtime_prefetch ||
      !runtime_scratch || !input || !output || S <= 0 ||
      runtime_scratch->moe_out.size() < (size_t)S *
          e.model_file.architecture_config.hidden ||
      runtime_scratch->tok_topk_ids.size() < (size_t)S *
          e.model_file.architecture_config.topk)
    throw std::runtime_error("DeepSeek-V4.1 graph MoE runtime is not ready");
  runtime_moe.run(layer, input, S, output, token_ids);
  if (dsv41_trace::enabled(layer, pos_base)) {
    const size_t last = (size_t)(S - 1) * e.model_file.architecture_config.hidden;
    dsv41_trace::save("hc-ffn-pre", layer, pos_base, input + last,
                      e.model_file.architecture_config.hidden);
    dsv41_trace::save("moe-out", layer, pos_base, output + last,
                      e.model_file.architecture_config.hidden);
    const int topk = e.model_file.architecture_config.topk;
    const size_t route = (size_t)(S - 1) * topk;
    std::vector<float> ids((size_t)topk);
    for (int k = 0; k < topk; ++k)
      ids[(size_t)k] =
          static_cast<float>(runtime_scratch->tok_topk_ids[route + k]);
    dsv41_trace::save("route-ids", layer, pos_base, ids.data(), ids.size());
    dsv41_trace::save("route-weights", layer, pos_base,
                      runtime_scratch->tok_topk_weights.data() + route,
                      (size_t)topk);
    if (S > 1) {
      std::vector<float> all_ids((size_t)S * topk);
      for (size_t i = 0; i < all_ids.size(); ++i)
        all_ids[i] =
            static_cast<float>(runtime_scratch->tok_topk_ids[i]);
      dsv41_trace::save("route-ids-seq", layer, pos_base, all_ids.data(),
                        all_ids.size());
      dsv41_trace::save("route-weights-seq", layer, pos_base,
                        runtime_scratch->tok_topk_weights.data(),
                        all_ids.size());
      dsv41_trace::save("moe-out-seq", layer, pos_base, output,
                        (size_t)S * e.model_file.architecture_config.hidden);
    }
  }
}

void deepseek_v41_full_graph_attention_forward(StreamingEngine &e, int layer,
                                               const float *input, int S,
                                               int pos_base, float *output) {
  if (e.family_ops.family != ModelFamily::DeepSeek_V41 ||
      !e.family_ops.attention_fn || !input || !output || S <= 0 ||
      pos_base < 0)
    throw std::runtime_error("DeepSeek-V4.1 graph attention is not ready");
  e.family_ops.attention_fn(e, layer, input, S, pos_base, output);
}

bool deepseek_v41_full_graph_decode_prepare(StreamingEngine &e, int pos,
                                            const int *token_ids) {
  if (!deepseek_v41_full_graph_decode_enabled() || !token_ids ||
      e.mtp_enabled || e.family_ops.family != ModelFamily::DeepSeek_V41 ||
      pos < 0 || pos >= e.max_t) return false;
  const bool cache_moe_requested = zeromoe::graphcfg::flag_on(
      "ZEROMOE_DEEPSEEK_V41_CACHE_MOE_DECODE");
  if (cache_moe_requested && cache_moe_graph_failed_engine == &e)
    return false;
  // The caches stay engine-owned, so a new prompt/position does not rebuild
  // either tensor bindings or scheduler workspace. Only input values change.
  if (decode && decode->ready && decode->engine == &e) return true;
  const auto &c = e.model_file.architecture_config;
  if (c.first_dense != 0 || c.hc_mult <= 1 ||
      c.dsv4_compress_ratios.size() < static_cast<size_t>(c.n_layers))
    return false;
  decode.reset();
  auto g = std::make_unique<DecodeGraph>();
  g->engine = &e;
  g->device = e.gpu_device_id >= 0 ? e.gpu_device_id : 0;
  g->S = 1; g->decode_mode = true;
  g->D = c.hidden; g->HC = c.hc_mult;
  g->n_layers = c.n_layers;
  g->H = c.n_heads;
  g->qk_nope = c.qk_nope;
  g->qk_rope = c.qk_rope;
  g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(g->device));
  g->cpu = ggml_backend_cpu_init();
  if (!g->cuda || !g->cpu) return false;
  g->cache_moe_graph = cache_moe_requested;
  if (g->cache_moe_graph) {
    v41_cache_moe_sites_reset();
    if (e.gpu) {
      for (int vi : e.gpu->topology.vault_gpus) {
        const auto &device = e.gpu->topology.devices[(size_t)vi];
        if (device.id != g->device && device.expert_cache_pool &&
            device.expert_cache_size >= e.gpu->topology.expert_slab_size) {
          g->gpu1 = static_cast<ggml_backend_t>(ggml_shared_backend(device.id));
          break;
        }
      }
    }
    if (!g->gpu1 && e.gpu && e.gpu->topology.vault_gpus.size() > 1) {
      LOG_ERROR("[GGML-DSV41-DECODE] second-GPU backend unavailable for cache MoE graph\n");
      return false;
    }
  }
  // One host task calls the existing OpenMP expert/attention code; avoid
  // nesting that work inside a second GGML worker team.
  ggml_backend_cpu_set_n_threads(g->cpu, 1);
  if (!build(*g)) {
    if (g->cache_moe_graph) {
      v41_cache_moe_sites_reset();
      cache_moe_graph_failed_engine = &e;
    }
    return false;
  }
  decode = std::move(g);
  return true;
}

bool deepseek_v41_full_graph_decode_run(StreamingEngine &e, float *hidden,
                                        int pos, const int *ids,
                                        float *logits, int vocab) {
  if (!decode || !decode->ready || decode->engine != &e || !hidden || !ids ||
      runtime_engine != &e || !runtime_ram || !runtime_prefetch ||
      !runtime_scratch || pos < 0 || pos >= e.max_t) return false;
  auto &g = *decode;
  if (g.expert.buffer || g.expert.capacity || g.expert_source_base ||
      !g.expert_source_views.empty())
    throw std::logic_error("DeepSeek decode graph allocated duplicate MoE storage");
  if (e.hc_stream_buf.size() != static_cast<size_t>(g.D * g.HC) ||
      e.hc_pending_pre_mix.size() != static_cast<size_t>(g.HC)) return false;
  g.pos = pos; g.token = *ids;
  g.ram = runtime_ram; g.prefetch = runtime_prefetch; g.scratch = runtime_scratch;
  if (g.runs == 0) g.perf_at_first_run = perf().snapshot();
  const auto node_ms_before = g.node_ms_total;
  std::memcpy(g.input->data, e.hc_stream_buf.data(), g.D * g.HC * sizeof(float));
  if (g.hc_carry_in)
    std::memcpy(g.hc_carry_in->data, e.hc_pending_pre_mix.data(), g.HC * sizeof(float));
  const auto start = std::chrono::steady_clock::now();
  const auto status = ggml_backend_sched_graph_compute(g.sched, g.graph);
  if (g.cache_moe_graph)
    v41_cache_moe_sites_release_holds();
  if (status != GGML_STATUS_SUCCESS) {
    // Host cache nodes may already have published this token. Replaying a
    // legacy forward would apply state transitions twice: fail the request.
    throw std::runtime_error("DeepSeek-V4.1 decode graph compute failed");
  }
  ggml_backend_tensor_get(g.output, hidden, 0, g.D * sizeof(float));
  ggml_backend_tensor_get(g.final_streams, e.hc_stream_buf.data(), 0,
                           g.D * g.HC * sizeof(float));
  if (g.hc_carry_in && g.dsv41_final_pre_mix) {
    // The last FFN pre-gate is both the next token's attention carry and the
    // final collapse mix, exactly as the host hc_pre_fn leaves them.
    e.hc_final_pre_mix.resize((size_t)g.HC);
    ggml_backend_tensor_get(g.dsv41_final_pre_mix, e.hc_final_pre_mix.data(), 0,
                            g.HC * sizeof(float));
    std::memcpy(e.hc_pending_pre_mix.data(), e.hc_final_pre_mix.data(),
                g.HC * sizeof(float));
  }
  for (const auto &trace : g.traces) {
    if (!dsv41_trace::enabled(trace.layer, pos)) continue;
    const size_t n = (size_t)ggml_nelements(trace.value);
    std::vector<float> values(n);
    ggml_backend_tensor_get(trace.value, values.data(), 0,
                            values.size() * sizeof(float));
    dsv41_trace::save(trace.label, trace.layer, pos, values.data(),
                      values.size());
  }
  if (logits && vocab == g.logits->ne[0]) {
    ggml_backend_tensor_get(g.logits, logits, 0,
                             ggml_nbytes(g.logits));
    e.qwen4exp_full_graph_logits_ready = true;
  }
  e.deepseek_v4_full_graph_output_ready = true;
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  std::array<double, kDecodeTimerCount> node_ms{};
  for (size_t i = 0; i < node_ms.size(); ++i) {
    node_ms[i] = g.node_ms_total[i] - node_ms_before[i];
    if (g.runs == 0) g.first_node_ms[i] = node_ms[i];
    else g.warm_node_ms_total[i] += node_ms[i];
  }
  if (g.runs == 0) g.first_compute_ms = ms;
  else g.warm_compute_ms += ms;
  if (++g.runs <= 2)
    LOG_INFO("[GGML-DSV41-DECODE] run=%llu pos=%d compute=%.3f ms; "
             "persistent bindings and workspace reused; graph-owned MoE bytes=0\n",
             static_cast<unsigned long long>(g.runs), pos, ms);
  if (g.runs <= 2)
    LOG_INFO("[GGML-DSV41-DECODE-NODES] run=%llu attention_host=%.3f ms "
             "native_qkv=%.3f ms cache_moe=%.3f ms hc_engram=%.3f ms "
             "router_pilot=%.3f ms logits_head=%.3f ms\n",
             static_cast<unsigned long long>(g.runs), node_ms[0], node_ms[1],
             node_ms[2], node_ms[3], node_ms[4], node_ms[5]);
  return true;
}

void deepseek_v41_full_graph_decode_shutdown() {
  v41_cache_moe_sites_reset();
  v41_cache_moe_sites_reset();
  if (decode && decode->runs > 0)
    LOG_INFO("[GGML-DSV41-DECODE] summary runs=%llu first=%.3f ms "
             "warm_mean=%.3f ms (%llu runs); graph-owned MoE bytes=0\n",
             static_cast<unsigned long long>(decode->runs),
             decode->first_compute_ms,
             decode->runs > 1 ? decode->warm_compute_ms / (decode->runs - 1) : 0.0,
             static_cast<unsigned long long>(decode->runs - 1));
  if (decode && decode->runs > 0) {
    // Snapshot deltas exclude startup and prefill. Several subphase timers
    // overlap their enclosing dispatch timers, so report each separately.
    const auto delta = perf().snapshot() - decode->perf_at_first_run;
    const double ms_per_token = 1000.0 / decode->runs;
    LOG_INFO("[GGML-DSV41-DECODE-MOE] ram_get=%.3f ms/token "
             "ram_batch=%.3f ms/token upload_async=%.3f ms/token "
             "active_upload=%.3f ms/token phase3=%.3f ms/token "
             "resolve_location=%.3f ms/token ram_needed=%.3f ms/token "
             "disk_read=%.3f ms/token disk_bytes=%.3f GB "
             "gpu_upload_bytes=%.3f GB cache_misses=%llu\n",
             delta.t_ram_get_expert * ms_per_token,
             delta.t_ram_get_experts_batch * ms_per_token,
             delta.t_upload_expert_async * ms_per_token,
             delta.t_active_layer_upload * ms_per_token,
             delta.t_moe_phase3_dispatch * ms_per_token,
             delta.t_resolve_gpu_loc_total * ms_per_token,
             delta.t_ram_needed_dispatch_total * ms_per_token,
             delta.t_disk_read * ms_per_token,
             delta.bytes_disk_read / 1.0e9,
             delta.bytes_gpu_uploaded / 1.0e9,
             static_cast<unsigned long long>(delta.n_expert_cache_miss));
    const char *names[] = {"attention_host", "native_qkv", "cache_moe",
                           "hc_engram", "router_pilot", "logits_head"};
    for (size_t i = 0; i < decode->warm_node_ms_total.size(); ++i)
      LOG_INFO("[GGML-DSV41-DECODE-NODES] %s first=%.3f ms warm_mean=%.3f "
               "ms per token\n", names[i], decode->first_node_ms[i],
               decode->runs > 1
                   ? decode->warm_node_ms_total[i] / (decode->runs - 1)
                   : 0.0);
    std::vector<std::pair<double, int>> hottest;
    for (size_t l = 0; l < decode->layer_moe_ms.size(); ++l)
      hottest.emplace_back(decode->layer_moe_ms[l][0] / decode->runs,
                           (int)l);
    std::sort(hottest.begin(), hottest.end(), std::greater<>());
    LOG_INFO("[GGML-DSV41-DECODE-NODES] MoE host time per layer (top 5): "
             "L%d=%.3f L%d=%.3f L%d=%.3f L%d=%.3f L%d=%.3f ms\n",
             hottest.size() > 0 ? hottest[0].second : -1,
             hottest.size() > 0 ? hottest[0].first : 0.0,
             hottest.size() > 1 ? hottest[1].second : -1,
             hottest.size() > 1 ? hottest[1].first : 0.0,
             hottest.size() > 2 ? hottest[2].second : -1,
             hottest.size() > 2 ? hottest[2].first : 0.0,
             hottest.size() > 3 ? hottest[3].second : -1,
             hottest.size() > 3 ? hottest[3].first : 0.0,
             hottest.size() > 4 ? hottest[4].second : -1,
             hottest.size() > 4 ? hottest[4].first : 0.0);
  }
  decode.reset(); runtime_engine = nullptr; runtime_ram = nullptr;
  runtime_prefetch = nullptr; runtime_scratch = nullptr;
}
#else
bool deepseek_v41_graph_runtime(StreamingEngine &, RAMSafetyManager *&,
                                LockFreePrefetcher *&, LayerScratch *&) { return false; }
bool deepseek_v41_full_graph_decode_enabled() { return false; }
void deepseek_v41_bind_graph_runtime(StreamingEngine &, RAMSafetyManager &,
                                     LockFreePrefetcher &, LayerScratch &) {}
void deepseek_v41_full_graph_moe_forward(StreamingEngine &, int,
                                         const float *, int, int, float *,
                                         const int *) {}
void deepseek_v41_full_graph_attention_forward(StreamingEngine &, int,
                                               const float *, int, int,
                                               float *) {}
bool deepseek_v41_full_graph_decode_prepare(StreamingEngine &, int, const int *) {
  return false;
}
bool deepseek_v41_full_graph_decode_run(StreamingEngine &, float *, int,
                                        const int *, float *, int) { return false; }
void deepseek_v41_full_graph_decode_shutdown() {}
#endif
