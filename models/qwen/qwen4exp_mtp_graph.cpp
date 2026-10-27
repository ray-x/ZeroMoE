#include "qwen4exp_mtp_graph.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "moe_routed_ggml.hpp"
#include "qwen4exp_graph_ops.hpp"

#include <chrono>
#include <climits>
#include <cstring>
#include <limits>
#include <numeric>

namespace qwen4exp_mtp {
namespace {
using namespace zeromoe::graph;

struct Config {
  int D = 0, HC = 0, H = 0, Hkv = 0, hd = 0, rank = 0;
  int layer = 0, experts = 0, topk = 0, inter = 0, vocab = 0;
  float eps = 0;
  qwen4exp_ops::AttentionShape attention{};
  moe_routed::RouterSpec router;
};

bool read_config(const StreamingEngine &e, int capacity, Config &c,
                 std::string &error) {
  const auto &base = e.model_file;
  const auto &mtp = e.mtp_model_file;
  const auto is_qwen = [](const GGUFFile &f) {
    auto it = f.kv_string.find("general.architecture");
    return it != f.kv_string.end() && it->second == "qwen4exp";
  };
  if (!e.has_mtp_model || !is_qwen(base) || !is_qwen(mtp)) {
    error = "draft graph requires Qwen4Exp trunk and separate MTP metadata";
    return false;
  }
  c.layer = base.get_config_int("block_count", 0);
  if (c.layer <= 0 || c.layer == INT_MAX ||
      base.get_config_int("nextn_predict_layers", 0) != 0 ||
      mtp.get_config_int("nextn_predict_layers", 0) != 1 ||
      mtp.get_config_int("block_count", 0) != c.layer + 1) {
    error = "draft graph requires exactly one shared-file NextN block";
    return false;
  }
  const auto integer = [&](const char *key, int fallback = 0) {
    const int value = base.get_config_int(key, fallback);
    if (mtp.get_config_int(key, value) != value)
      error = std::string("MTP/trunk metadata mismatch: ") + key;
    return value;
  };
  const auto real = [&](const char *key, float fallback) {
    const float value = base.get_config_float(key, fallback);
    if (!std::isfinite(value) || mtp.get_config_float(key, value) != value)
      error = std::string("MTP/trunk float metadata mismatch: ") + key;
    return value;
  };
  c.D = integer("embedding_length");
  c.HC = integer("hyper_connection.count");
  c.rank = integer("hyper_connection.low_rank");
  c.H = integer("attention.head_count");
  c.Hkv = integer("attention.head_count_kv", c.H);
  c.hd = integer("attention.key_length", c.H > 0 ? c.D / c.H : 0);
  if (integer("attention.value_length", c.hd) != c.hd)
    error = "draft graph requires identical key/value head dimensions";
  c.experts = integer("expert_count");
  c.topk = integer("expert_used_count");
  c.inter = integer("expert_feed_forward_length");
  c.eps = real("attention.layer_norm_rms_epsilon", 1e-5f);
  auto head = resident(e, "output.weight");
  if (!head || head->meta.shape.size() != 2 ||
      head->meta.shape[0] != static_cast<uint64_t>(c.D) ||
      head->meta.shape[1] > INT_MAX) {
    error = "draft graph requires resident trunk output.weight [D,V]";
    return false;
  }
  c.vocab = static_cast<int>(head->meta.shape[1]);
  if (!error.empty()) return false;
  if (c.D <= 0 || c.HC <= 1 || c.rank <= 0 || c.H <= 0 || c.Hkv <= 0 ||
      c.H % c.Hkv || c.hd <= 0 || c.hd % 2 || c.inter <= 0 ||
      c.experts <= 0 || c.topk <= 0 || c.topk > c.experts || c.vocab <= 0 ||
      !std::isfinite(c.eps) || c.eps <= 0 || capacity <= 0 ||
      c.D > INT_MAX / c.HC || c.D > INT_MAX / 2 ||
      c.hd > INT_MAX / c.H / 2 ||
      ((int64_t{capacity} + 255) / 256) * 256 > INT_MAX / c.Hkv) {
    error = "invalid Qwen draft dimensions, epsilon or context capacity";
    return false;
  }
  const auto prefix = "blk." + std::to_string(c.layer) + ".";
  for (const auto &entry : mtp.tensors) {
    if (entry.first.starts_with(prefix) &&
        (entry.first.ends_with(".bias") ||
         entry.first.find("down_inject") != std::string::npos ||
         entry.first.find("nextn.embed_tokens") != std::string::npos ||
         entry.first.find("nextn.shared_head") != std::string::npos)) {
      error = "unsupported MTP checkpoint variant: " + entry.first;
      return false;
    }
  }
  auto ratios = mtp.get_config_int_array("attention.compress_ratios");
  if (ratios.empty()) ratios = base.get_config_int_array("attention.compress_ratios");
  if (!ratios.empty() && ratios.size() <= static_cast<size_t>(c.layer)) {
    error = "MTP attention compression metadata does not describe NextN block";
    return false;
  }
  const int64_t ratio = ratios.empty()
      ? mtp.get_config_int("attention.compress_ratios",
                          base.get_config_int("attention.compress_ratios", 0))
      : ratios[c.layer];
  const int selection_budget = mtp.get_config_int(
      "attention.indexer.top_k", base.get_config_int("attention.indexer.top_k", 0));
  // With every complete pool selected, QSA retains all causal cells (plus
  // its unfinished tail). Beyond this conservative bound dense is not QSA.
  if (ratio < 0 || (ratio > 0 &&
      (ratio <= 1 || selection_budget <= 0 || selection_budget % ratio != 0 ||
       capacity > selection_budget))) {
    error = "draft context exceeds dense-equivalent QSA prefix or invalid pooling";
    return false;
  }
  const auto scaling = base.get_config_string("rope.scaling.type", "none");
  if (mtp.get_config_string("rope.scaling.type", scaling) != scaling)
    error = "MTP/trunk positional scaling types differ";
  const float factor = real("rope.scaling.factor", 1.0f);
  const int n_rot = integer("rope.dimension_count", c.hd);
  if (!error.empty()) return false;
  if ((scaling != "none" && scaling != "yarn" && scaling != "linear") ||
      factor <= 0 || !std::isfinite(factor) || n_rot <= 0 || n_rot > c.hd ||
      n_rot % 2 ||
      real("rope.scaling.yarn_log_multiplier", 0.0f) != 0 ||
      real("rope.scaling.yarn_beta_fast", 32.0f) != 32.0f ||
      real("rope.scaling.yarn_beta_slow", 1.0f) != 1.0f ||
      real("attention.scale", 0.0f) != 0) {
    error = "unsupported Qwen draft positional or attention scaling";
    return false;
  }
  c.attention = {c.H, c.Hkv, c.hd, capacity, n_rot,
      integer("rope.scaling.original_context_length",
               base.get_config_int("context_length", capacity)),
      c.eps, real("rope.freq_base", 10000.0f),
      1.0f / factor, scaling == "yarn" ? 1.0f : 0.0f,
      real("rope.scaling.attn_factor", 1.0f)};
  c.router = {moe_routed::Gating::Softmax, c.experts, c.topk, 1, 0, 0,
               true, real("expert_weights_scale", 1.0f)};
  if (!error.empty()) return false;
  if (!std::isfinite(c.attention.freq_base) || c.attention.freq_base <= 0 ||
      !std::isfinite(c.attention.attn_factor) ||
      !std::isfinite(c.router.weights_scale)) {
    error = "non-finite draft attention or router metadata";
    return false;
  }
  return true;
}

std::shared_ptr<MtpGraphState> build(
    std::shared_ptr<StreamingEngine> owner, const Options &opts, const Config &c,
    std::string &error) {
  const auto build_started = std::chrono::steady_clock::now();
  auto g = std::make_shared<MtpGraphState>();
  g->owner = std::move(owner);
  g->engine = g->owner.get();
  g->device = opts.device;
  g->D = c.D; g->HC = c.HC; g->H = c.H; g->hd = c.hd;
  g->S = 1;
  g->kv_capacity = static_cast<int>(((int64_t{opts.capacity} + 255) / 256) * 256);
  g->decode_mode = true;
  g->mem_tag = "MTP-GRAPH-MEM";
  if (cudaSetDevice(opts.device) != cudaSuccess ||
      !(g->cuda = ggml_backend_cuda_init(opts.device)) ||
      !(g->cpu = ggml_backend_cpu_init())) {
    error = "cannot initialize draft CUDA/CPU backends";
    return nullptr;
  }
  constexpr size_t graph_size = 4096;
  g->ctx = g->make_context(
      graph_size * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false));
  g->weight_ctx = g->make_context(128 * ggml_tensor_overhead());
  g->state_ctx = g->make_context(8 * ggml_tensor_overhead());
  if (!g->ctx || !g->weight_ctx || !g->state_ctx) {
    error = "host budget refused draft tensor metadata";
    return nullptr;
  }
  std::map<std::string, ggml_tensor *> weights;
  std::vector<std::pair<ggml_tensor *, const QuantizedTensor *>> uploads;
  const std::string prefix = "blk." + std::to_string(c.layer) + ".";
  const auto add = [&](const std::string &suffix, std::vector<int64_t> dims,
                       bool norm = false, bool head = false) -> bool {
    const std::string name = head ? suffix : prefix + suffix;
    const auto *source = resident(*g->engine, name);
    const auto &file = head ? g->engine->model_file : g->engine->mtp_model_file;
    const auto meta = file.tensors.find(name);
    if (!source || meta == file.tensors.end() ||
        source->meta.type != meta->second.type ||
        source->meta.shape != meta->second.shape ||
        source->meta.size_bytes != meta->second.size_bytes) {
      error = "missing or mismatched resident draft tensor: " + name;
      return false;
    }
    const auto &shape = source->meta.shape;
    bool same = shape.size() == dims.size();
    for (size_t i = 0; same && i < dims.size(); ++i)
      same = shape[i] == static_cast<uint64_t>(dims[i]);
    if (!same && norm) {
      // Converters may store a grouped gain as [D,HC] or flattened [D*HC].
      same = (shape.size() == 1 || shape.size() == 2) &&
          std::accumulate(shape.begin(), shape.end(), uint64_t{1},
                          std::multiplies<uint64_t>()) ==
          static_cast<uint64_t>(std::accumulate(
              dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>()));
    }
    auto type = static_cast<ggml_type>(source->meta.type);
    switch (type) {
    case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16:
    case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0: case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S:
    case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS: break;
    default:
      error = "unsupported draft weight type: " + name;
      return false;
    }
    if (!same || dims[0] % ggml_blck_size(type) ||
        (norm && type != GGML_TYPE_F32)) {
      error = "invalid draft tensor geometry or non-F32 gain: " + name;
      return false;
    }
    auto *t = new_tensor(g->weight_ctx, type, dims);
    if (!t || ggml_nbytes(t) != source->meta.size_bytes) {
      error = "invalid draft tensor byte geometry: " + name;
      return false;
    }
    ggml_set_name(t, name.c_str());
    weights[suffix] = t;
    uploads.emplace_back(t, source);
    return true;
  };
  const int W = c.D * c.HC;
  if (!add("nextn.eh_proj.weight", {2 * c.D, c.D}) ||
      !add("nextn.enorm.weight", {c.D}, true) ||
      !add("nextn.hnorm.weight", {c.D, c.HC}, true)) return nullptr;
  for (const char *tag : {"hc_attn", "hc_ffn", "nextn.hc_head"}) {
    const std::string p = tag;
    if (!add(p + "_norm.weight", {W}, true) ||
        !add(p + "_down.weight", {W, c.rank}) ||
        !add(p + "_up.weight", {c.rank, W}) ||
        (p != "nextn.hc_head" &&
         !add(p + "_inject.weight", {W, c.HC}))) return nullptr;
  }
  if (!add("attn_q.weight", {c.D, 2 * c.hd * c.H}) ||
      !add("attn_k.weight", {c.D, c.hd * c.Hkv}) ||
      !add("attn_v.weight", {c.D, c.hd * c.Hkv}) ||
      !add("attn_output.weight", {c.hd * c.H, c.D}) ||
      !add("attn_q_norm.weight", {c.hd}, true) ||
      !add("attn_k_norm.weight", {c.hd}, true) ||
      !add("ffn_gate_inp.weight", {c.D, c.experts}) ||
      !add("ffn_gate_exps.weight", {c.D, c.inter, c.experts}) ||
      !add("ffn_up_exps.weight", {c.D, c.inter, c.experts}) ||
      !add("ffn_down_exps.weight", {c.inter, c.D, c.experts}) ||
      !add("output.weight", {c.D, c.vocab}, false, true)) return nullptr;
  auto shared = g->engine->mtp_model_file.tensors.find(prefix + "ffn_up_shexp.weight");
  if (shared == g->engine->mtp_model_file.tensors.end() &&
      (g->engine->mtp_model_file.get_config_int("expert_shared_count", 0) > 0 ||
       g->engine->mtp_model_file.tensors.contains(prefix + "ffn_gate_shexp.weight") ||
       g->engine->mtp_model_file.tensors.contains(prefix + "ffn_down_shexp.weight"))) {
    error = "missing draft shared expert up projection";
    return nullptr;
  }
  if (shared != g->engine->mtp_model_file.tensors.end()) {
    if (shared->second.shape.size() != 2 ||
        shared->second.shape[1] == 0 || shared->second.shape[1] > INT_MAX) {
      error = "invalid draft shared expert shape";
      return nullptr;
    }
    const auto width = static_cast<int64_t>(shared->second.shape[1]);
    if (!add("ffn_gate_shexp.weight", {c.D, width}) ||
        !add("ffn_up_shexp.weight", {c.D, width}) ||
        !add("ffn_down_shexp.weight", {width, c.D}) ||
        !add("ffn_gate_inp_shexp.weight", {c.D, 1}, true)) return nullptr;
  }
  auto *keys = ggml_new_tensor_2d(g->state_ctx, GGML_TYPE_F32, c.hd,
                                  static_cast<int64_t>(g->kv_capacity) * c.Hkv);
  auto *values = ggml_dup_tensor(g->state_ctx, keys);
  const auto buft = ggml_backend_get_default_buffer_type(g->cuda);
  const size_t weight_size =
      ggml_backend_alloc_ctx_tensors_from_buft_size(g->weight_ctx, buft);
  const size_t state_size =
      ggml_backend_alloc_ctx_tensors_from_buft_size(g->state_ctx, buft);
  size_t free_bytes = 0, total_bytes = 0;
  if (!weight_size || !state_size || weight_size > opts.device_budget ||
      state_size > opts.device_budget - weight_size ||
      cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      weight_size > free_bytes || state_size > free_bytes - weight_size) {
    error = "draft weights and KV do not fit admitted/free device memory";
    return nullptr;
  }
  g->dense.buffer = ggml_backend_alloc_ctx_tensors(g->weight_ctx, g->cuda);
  g->state.buffer = ggml_backend_alloc_ctx_tensors(g->state_ctx, g->cuda);
  if (!g->dense.buffer || !g->state.buffer) {
    error = "draft weight/KV allocation failed";
    return nullptr;
  }
  g->device_weight_bytes = ggml_backend_buffer_get_size(g->dense.buffer);
  g->device_state_bytes = ggml_backend_buffer_get_size(g->state.buffer);
  ggml_backend_buffer_set_usage(g->dense.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  double weight_upload_ms = 0.0, head_upload_ms = 0.0;
  for (const auto &[tensor, source] : uploads) {
    const auto upload_started = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(tensor, source->host_ptr, 0, ggml_nbytes(tensor));
    const double upload_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - upload_started).count();
    if (std::strcmp(ggml_get_name(tensor), "output.weight") == 0)
      head_upload_ms += upload_ms;
    else
      weight_upload_ms += upload_ms;
  }
  auto *ctx = g->ctx;
  g->embedding = g->host_input(ctx, GGML_TYPE_F32, {c.D, 1}, 0, "mtp_embedding");
  g->hidden = g->host_input(ctx, GGML_TYPE_F32, {W, 1}, 0, "mtp_hidden");
  g->positions = g->host_input(ctx, GGML_TYPE_I32, {1}, 0, "mtp_position");
  g->decode_indices = g->host_input(ctx, GGML_TYPE_I32, {c.Hkv}, 0, "mtp_kv_rows");
  g->attn_mask = g->host_input(ctx, GGML_TYPE_F16, {g->kv_capacity, 1}, 0, "mtp_mask");
  if (!g->embedding || !g->hidden || !g->positions || !g->decode_indices ||
      !g->attn_mask) {
    error = "host budget refused draft graph inputs";
    return nullptr;
  }
  const auto w = [&](const std::string &name) { return weights.at(name); };
  const qwen4exp_ops::HcShape hc{c.D, c.HC, 1, c.eps};
  const auto mix = [&](ggml_tensor *x, const std::string &p, ggml_tensor **inj) {
    return qwen4exp_ops::hc_mix(hc, ctx, x, w(p + "_norm.weight"),
        w(p + "_down.weight"), w(p + "_up.weight"),
        inj ? w(p + "_inject.weight") : nullptr, inj);
  };
  const auto diagnostic = [&](const std::string &name, ggml_tensor *tensor) {
    if (!opts.diagnostics) return;
    ggml_set_output(tensor);
    g->diagnostics[name] = tensor;
  };
  std::vector<ggml_tensor *> cpu_norms;
  auto *hn_rms = ggml_rms_norm(ctx, ggml_reshape_3d(ctx, g->hidden, c.D, c.HC, 1), c.eps);
  auto *hn = ggml_mul(ctx, hn_rms, w("nextn.hnorm.weight"));
  auto *en_rms = ggml_rms_norm(ctx, g->embedding, c.eps);
  auto *en = ggml_mul(ctx, en_rms, w("nextn.enorm.weight"));
  cpu_norms.insert(cpu_norms.end(), {hn_rms, en_rms});
  en = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, en, c.D, 1, 1), c.D, c.HC, 1, 1);
  auto *res = ggml_mul_mat(ctx, w("nextn.eh_proj.weight"),
                           ggml_concat(ctx, en, hn, 0));
  diagnostic("bridge", res);
  ggml_tensor *inject = nullptr;
  auto *cur = mix(res, "hc_attn", &inject);
  auto attention = c.attention;
  attention.capacity = g->kv_capacity;
  cur = qwen4exp_ops::attention_decode(attention, ctx, cur,
      w("attn_q.weight"), w("attn_k.weight"), w("attn_v.weight"),
      w("attn_q_norm.weight"), w("attn_k_norm.weight"), w("attn_output.weight"),
      keys, values, g->positions, g->decode_indices, g->attn_mask, &cpu_norms);
  res = qwen4exp_ops::hc_combine(hc, ctx, res, cur, inject);
  diagnostic("post_attention", res);
  cur = mix(res, "hc_ffn", &inject);
  auto *probs = moe_routed::gate_probs(ctx, c.router, w("ffn_gate_inp.weight"), cur);
  auto *ids = moe_routed::select_ids(ctx, c.router, probs, nullptr);
  auto *route = moe_routed::route_weights(ctx, c.router, probs, ids);
  diagnostic("router_probabilities", probs);
  diagnostic("router_weights", route);
  diagnostic("router_ids", ids);
  const moe_routed::ExpertWeights experts{nullptr, w("ffn_gate_exps.weight"),
      w("ffn_up_exps.weight"), w("ffn_down_exps.weight")};
  const auto ffn = moe_routed::expert_ffn(
      ctx, experts, ggml_reshape_3d(ctx, cur, c.D, 1, 1), ids);
  auto *moe = moe_routed::sum_topk(
      ctx, ggml_mul(ctx, ffn.down, route), c.D, 1, c.topk);
  if (shared != g->engine->mtp_model_file.tensors.end()) {
    auto *shexp = qwen4exp_ops::shared_expert(ctx, cur, w("ffn_gate_shexp.weight"),
        w("ffn_up_shexp.weight"), w("ffn_down_shexp.weight"),
        w("ffn_gate_inp_shexp.weight"));
    moe = ggml_add(ctx, moe, shexp);
  }
  res = qwen4exp_ops::hc_combine(hc, ctx, res, moe, inject);
  diagnostic("post_ffn", res);
  g->final_streams = ggml_reshape_2d(ctx, res, W, 1);
  g->output = mix(res, "nextn.hc_head", nullptr);
  g->logits = ggml_mul_mat(ctx, w("output.weight"), g->output);
  ggml_set_output(g->final_streams);
  ggml_set_output(g->logits);
  g->graph = ggml_new_graph_custom(ctx, graph_size, false);
  ggml_build_forward_expand(g->graph, g->final_streams);
  ggml_build_forward_expand(g->graph, g->logits);
  for (const auto &[name, tensor] : g->diagnostics)
    ggml_build_forward_expand(g->graph, tensor);
  ggml_backend_t backends[] = {g->cuda, g->cpu};
  g->sched = ggml_backend_sched_new(backends, nullptr, 2, graph_size, false, true);
  if (!g->sched) {
    error = "cannot create draft scheduler";
    return nullptr;
  }
  for (auto *node : {ffn.gate, ffn.up, ffn.down}) {
    if (!ggml_backend_supports_op(g->cuda, node)) {
      error = "CUDA backend does not support draft expert quantization";
      return nullptr;
    }
    ggml_backend_sched_set_tensor_backend(g->sched, node, g->cuda);
  }
  // The vocabulary head runs on the GPU from its own device copy. Borrowing the
  // host weight and multiplying on the CPU cost ~13.6 ms per draft step (the
  // rest of the draft step is ~0.4 ms), which made drafting dearer than a
  // target step. The copy is one head's worth of VRAM, charged to the draft.
  ggml_backend_sched_set_tensor_backend(g->sched, g->logits, g->cuda);
  if (opts.reference_cpu_norms)
    for (auto *norm : cpu_norms)
      ggml_backend_sched_set_tensor_backend(g->sched, norm, g->cpu);
  const auto *head_ptr = resident(*g->engine, "output.weight")->host_ptr;
  const auto head_meta = resident(*g->engine, "output.weight")->meta;
  auto *raw = g.get();
  g->bindings_valid = [raw, head_ptr, head_meta] {
    auto *head = resident(*raw->engine, "output.weight");
    return head && head->host_ptr == head_ptr &&
           head->meta.type == head_meta.type && head->meta.shape == head_meta.shape &&
           head->meta.size_bytes == head_meta.size_bytes;
  };
  g->prepare_position = [raw, heads = c.Hkv](int position) -> std::string {
    auto *p = static_cast<int32_t *>(raw->positions->data);
    auto *rows = static_cast<int32_t *>(raw->decode_indices->data);
    auto *mask = static_cast<ggml_fp16_t *>(raw->attn_mask->data);
    *p = position;
    for (int h = 0; h < heads; ++h) rows[h] = h * raw->kv_capacity + position;
    for (int i = 0; i < raw->kv_capacity; ++i)
      mask[i] = ggml_fp32_to_fp16(i <= position ? 0.0f : -INFINITY);
    return {};
  };
  const double build_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - build_started).count();
  LOG_INFO("[MTP-GRAPH-BUILD] weights=%zu state=%zu weight_upload_ms=%.2f "
           "head_upload_ms=%.2f build_ms=%.2f\n",
           g->device_weight_bytes, g->device_state_bytes, weight_upload_ms,
           head_upload_ms, build_ms);
  return g;
}
} // namespace

std::unique_ptr<zeromoe::graph::MtpGraphRuntime> create(
    std::shared_ptr<StreamingEngine> owner, const Options &options,
    std::string &error) {
  error.clear();
  if (!owner || options.device < 0 || !options.device_budget) {
    error = "draft graph requires an engine owner, device and explicit budget";
    return nullptr;
  }
  Config config;
  if (!read_config(*owner, options.capacity, config, error)) return nullptr;
  const MtpGraphShape shape{static_cast<size_t>(config.D),
      static_cast<size_t>(config.D) * config.HC,
      static_cast<size_t>(config.vocab), options.capacity};
  return MtpGraphRuntime::create(shape, options.device_budget,
      [owner, options, config](std::string &reason) {
        return build(owner, options, config, reason);
      }, error);
}
} // namespace qwen4exp_mtp
#endif
