// full_graph_models.cpp -- the FullGraphModel of each family and the registry.
//
// These classes are thin: each maps the common request onto the family's
// graph implementation (src/ggml_<family>_graph*.cpp), which keeps its own
// state.  Request validation and prefill/decode dispatch live in
// FullGraphModelBase.
#include "full_graph_model.hpp"

#include "engine.hpp"
#include "ggml_deepseek_v4_graph.hpp"
#include "ggml_deepseek_v41_graph.hpp"
#include "ggml_glm5next_graph.hpp"
#include "ggml_qwen4exp_graph.hpp"

#include <cstdlib>
#include <algorithm>

namespace {

class Qwen4ExpGraphModel final : public FullGraphModelBase<Qwen4ExpGraphModel> {
 public:
  ModelFamily family() const override { return ModelFamily::Qwen4Exp; }
  const char *name() const override { return "qwen4exp"; }

  bool enabled(const StreamingEngine &, ModelGraphPhase phase) const override {
    return phase == ModelGraphPhase::Decode ? qwen4exp_full_graph_decode_enabled()
                                            : qwen4exp_full_graph_enabled();
  }
  bool bootstrap(StreamingEngine &engine) const override {
    return qwen4exp_full_graph_bootstrap(engine);
  }
  void shutdown() const override { qwen4exp_full_graph_shutdown(); }
  int device_expert_layers() const override {
    return qwen4exp_full_graph_device_expert_layers();
  }

  moe_routed::RouterSpec router_spec(const ArchitectureConfig &c,
                                     int n_tokens) const override {
    return moe_routed::router_spec(c, n_tokens, /*honor_gating_func=*/false);
  }
  bool supports_cpu_lane() const override { return true; }

  // Tokens per prefill graph batch: MOESTREAM_QWEN4EXP_PREFILL_CHUNK, else the toml's `prefill_batch_tokens`, else the
  // whole prompt (0). Batches continue each other (recurrent state, dense KV rows and the PLE window are carried through
  // the engine), and the memory pre-check lowers a batch that does not fit to the one that does (suggested batch).
  int max_prefill_tokens(const StreamingEngine &e) const override {
    if (const char *v = std::getenv("MOESTREAM_QWEN4EXP_PREFILL_CHUNK"); v && v[0])
      return std::atoi(v);
    return e.prefill_batch_tokens > 0 ? e.prefill_batch_tokens : 0;
  }
  int suggested_prefill_batch(const StreamingEngine &) const override {
    return qwen4exp_full_graph_suggested_batch();
  }
  bool supports_prefill_continuation(const StreamingEngine &) const override { return true; }
  int min_prefill_batch() const override { return 8; } // batches this small still fit when little VRAM is free

  bool prefill_prepare(StreamingEngine &e, const ModelGraphRequest &r) const {
    return qwen4exp_full_graph_prepare(e, r.n_tokens, r.pos_base, r.token_ids);
  }
  bool prefill_run(StreamingEngine &e, const ModelGraphRequest &r) const {
    return qwen4exp_full_graph_run(e, r.hidden_states, r.n_tokens, r.pos_base,
                                   r.token_ids);
  }
  bool decode_prepare(StreamingEngine &e, const ModelGraphRequest &r) const {
    return qwen4exp_full_graph_decode_prepare(e, r.pos_base, r.token_ids);
  }
  bool decode_run(StreamingEngine &e, const ModelGraphRequest &r) const {
    // The generation loop installs a short-lived logits destination on the
    // engine; the request carries the same information for other callers.
    float *logits =
        r.logits ? r.logits : e.qwen4exp_full_graph_logits_target;
    const int vocab =
        r.vocab > 0 ? r.vocab : e.qwen4exp_full_graph_logits_target_size;
    return qwen4exp_full_graph_decode_run(e, r.hidden_states, r.pos_base,
                                          r.token_ids, logits, vocab);
  }
};

class Glm5NextGraphModel final : public FullGraphModelBase<Glm5NextGraphModel> {
 public:
  ModelFamily family() const override { return ModelFamily::GLM5NEXT; }
  const char *name() const override { return "glm5next"; }

  bool enabled(const StreamingEngine &engine,
               ModelGraphPhase phase) const override {
    if (engine.mtp_enabled || !glm5next_full_graph_enabled())
      return false;
    if (phase == ModelGraphPhase::Prefill) {
      // The batched sparse hot/cold MUL_MAT_ID split reaches a CUDA MMQ
      // route-compaction limitation for inactive branch slots, so graph
      // prefill stays opt-in until that is fixed upstream.
      const char *prefill = std::getenv("MOESTREAM_GLM5NEXT_FULL_GRAPH_PREFILL");
      return prefill && prefill[0] == '1';
    }
    return phase == ModelGraphPhase::Decode;
  }
  bool bootstrap(StreamingEngine &engine) const override {
    return glm5next_full_graph_bootstrap(engine);
  }
  void shutdown() const override { glm5next_full_graph_shutdown(); }
  int device_expert_layers() const override {
    return glm5next_full_graph_device_expert_layers();
  }
  bool supports_cpu_lane() const override { return true; }

  // Tokens per prefill batch (llama.cpp's n_ubatch): MOESTREAM_GLM5NEXT_PREFILL_CHUNK, else the toml's
  // `prefill_batch_tokens`, else 2048 (every batch restreams all cold experts, so 512 prefills a 1065-token prompt at
  // 90 tok/s against 250 for one batch). 0 = the whole prompt in one graph. Each batch continues the previous one's state
  // (up to the sparse top-k of 2048 tokens of context). Whatever is configured is reduced by the prepare check to what
  // fits in free VRAM / host RAM, so an oversized value cannot request memory the box does not have.
  int max_prefill_tokens(const StreamingEngine &e) const override {
    if (const char *v = std::getenv("MOESTREAM_GLM5NEXT_PREFILL_CHUNK"); v && v[0])
      return std::atoi(v);
    return e.prefill_batch_tokens > 0 ? e.prefill_batch_tokens : 2048;
  }
  int suggested_prefill_batch(const StreamingEngine &) const override {
    return glm5next_full_graph_suggested_batch();
  }

  bool supports_prefill_continuation(const StreamingEngine &) const override { return true; }

  // MTP and multi-token decode are not graph-supported.
  bool accepts(const StreamingEngine &e, const ModelGraphRequest &r) const {
    if (e.mtp_enabled)
      return false;
    return r.phase != ModelGraphPhase::Decode ||
           (r.n_tokens == 1 && r.pos_base >= 0);
  }

  bool prefill_prepare(StreamingEngine &e, const ModelGraphRequest &r) const {
    return glm5next_full_graph_prepare(e, r.n_tokens, r.pos_base, r.token_ids);
  }
  bool prefill_run(StreamingEngine &e, const ModelGraphRequest &r) const {
    return glm5next_full_graph_run(e, r.hidden_states, r.n_tokens, r.pos_base,
                                   r.token_ids);
  }
  bool decode_prepare(StreamingEngine &e, const ModelGraphRequest &r) const {
    return glm5next_full_graph_decode_prepare(e, r.pos_base, r.token_ids);
  }
  bool decode_run(StreamingEngine &e, const ModelGraphRequest &r) const {
    return glm5next_full_graph_decode_run(e, r.hidden_states, r.pos_base,
                                          r.token_ids);
  }
};

class DeepseekV4GraphModel final
    : public FullGraphModelBase<DeepseekV4GraphModel> {
 public:
  ModelFamily family() const override { return ModelFamily::DeepSeek_V4; }
  const char *name() const override { return "deepseek_v4"; }

  bool enabled(const StreamingEngine &, ModelGraphPhase phase) const override {
    return phase == ModelGraphPhase::Prefill && deepseek_v4_full_graph_enabled();
  }
  void shutdown() const override { deepseek_v4_full_graph_shutdown(); }
  int device_expert_layers() const override {
    return deepseek_v4_full_graph_device_expert_layers();
  }

  // Prefill-only graph.
  bool accepts(const StreamingEngine &, const ModelGraphRequest &r) const {
    return r.phase == ModelGraphPhase::Prefill;
  }

  bool prefill_prepare(StreamingEngine &e, const ModelGraphRequest &r) const {
    return deepseek_v4_full_graph_prepare(e, r.n_tokens, r.pos_base,
                                          r.token_ids);
  }
  bool prefill_run(StreamingEngine &e, const ModelGraphRequest &r) const {
    return deepseek_v4_full_graph_run(e, r.hidden_states, r.n_tokens,
                                      r.pos_base, r.token_ids);
  }
  // accepts() rejects decode, so these are never reached.
  bool decode_prepare(StreamingEngine &, const ModelGraphRequest &) const {
    return false;
  }
  bool decode_run(StreamingEngine &, const ModelGraphRequest &) const {
    return false;
  }
};

class DeepseekV41GraphModel final
    : public FullGraphModelBase<DeepseekV41GraphModel> {
 public:
  ModelFamily family() const override { return ModelFamily::DeepSeek_V41; }
  const char *name() const override { return "deepseek_v41"; }

  bool enabled(const StreamingEngine &e, ModelGraphPhase phase) const override {
    return !e.mtp_enabled &&
           (phase == ModelGraphPhase::Decode
                ? deepseek_v41_full_graph_decode_enabled()
                : deepseek_v41_full_graph_enabled());
  }
  void shutdown() const override { deepseek_v41_full_graph_shutdown(); }
  int device_expert_layers() const override {
    return deepseek_v41_full_graph_device_expert_layers();
  }
  int max_prefill_tokens(const StreamingEngine &) const override {
    static const int chunk = [] {
      const char *v = std::getenv("MOESTREAM_DEEPSEEK_V41_PREFILL_CHUNK");
      // Native cache-backed attention supports the whole prompt. A 128-token
      // cap sent the rest of long prompts through serial decode.
      return v && v[0] ? std::max(0, std::atoi(v)) : 0;
    }();
    return chunk;
  }

  bool accepts(const StreamingEngine &e, const ModelGraphRequest &r) const {
    return !e.mtp_enabled &&
           (r.phase == ModelGraphPhase::Prefill ? r.pos_base == 0
               : r.n_tokens == 1 && r.pos_base >= 0);
  }
  bool prefill_prepare(StreamingEngine &e,
                       const ModelGraphRequest &r) const {
    return deepseek_v41_full_graph_prepare(e, r.n_tokens, r.pos_base,
                                           r.token_ids);
  }
  bool prefill_run(StreamingEngine &e, const ModelGraphRequest &r) const {
    return deepseek_v41_full_graph_run(e, r.hidden_states, r.n_tokens,
                                       r.pos_base, r.token_ids);
  }
  bool decode_prepare(StreamingEngine &e, const ModelGraphRequest &r) const {
    return deepseek_v41_full_graph_decode_prepare(e, r.pos_base, r.token_ids);
  }
  bool decode_run(StreamingEngine &e, const ModelGraphRequest &r) const {
    return deepseek_v41_full_graph_decode_run(e, r.hidden_states, r.pos_base,
        r.token_ids, r.logits ? r.logits : e.qwen4exp_full_graph_logits_target,
        r.vocab > 0 ? r.vocab : e.qwen4exp_full_graph_logits_target_size);
  }
};

} // namespace

const FullGraphModel *full_graph_model(ModelFamily family) {
  static const Qwen4ExpGraphModel qwen;
  static const Glm5NextGraphModel glm;
  static const DeepseekV4GraphModel dsv4;
  static const DeepseekV41GraphModel dsv41;
  switch (family) {
  case ModelFamily::Qwen4Exp:
    return &qwen;
  case ModelFamily::GLM5NEXT:
    return &glm;
  case ModelFamily::DeepSeek_V4:
    return &dsv4;
  case ModelFamily::DeepSeek_V41:
    return &dsv41;
  default:
    return nullptr;
  }
}
