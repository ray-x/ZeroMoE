// full_graph_models.cpp -- the FullGraphModel of each family and the registry.
//
// These classes are thin: each maps the common request onto the family's
// graph implementation (src/ggml_<family>_graph*.cpp), which keeps its own
// state.  Request validation and prefill/decode dispatch live in
// FullGraphModelBase.
#include "full_graph_model.hpp"

#include "engine.hpp"
#include "ggml_bailingmoe3_graph.hpp"
#include "ggml_deepseek_v41_graph.hpp"
#include "ggml_glm5next_graph.hpp"
#include "ggml_qwen4exp_graph.hpp"
#include "ggml_mimo2_graph.hpp"
#include "graph_config.hpp"
#include "graph_settings_global.hpp"
#include "quant_table.hpp"
#include <map>
#include <string>
#include <vector>

#include <cstdlib>
#include <algorithm>

namespace {

class Qwen4ExpGraphModel final : public FullGraphModelBase<Qwen4ExpGraphModel> {
 public:
  GraphRuntimeTuning runtime_tuning(const StreamingEngine &) const override { return {}; }
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
  bool device_expert_trailing() const override {
    return qwen4exp_full_graph_device_expert_trailing();
  }

  // Tokens per prefill graph batch: ZEROMOE_QWEN4EXP_PREFILL_CHUNK, else the toml's `prefill_batch_tokens`, else the
  // whole prompt (0). Batches continue each other (recurrent state, dense KV rows and the PLE window are carried through
  // the engine), and the memory pre-check lowers a batch that does not fit to the one that does (suggested batch).
  int max_prefill_tokens(const StreamingEngine &e) const override {
    if (const char *v = std::getenv("ZEROMOE_QWEN4EXP_PREFILL_CHUNK"); v && v[0])
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
  GraphRuntimeTuning runtime_tuning(const StreamingEngine &) const override { return {}; }
  bool device_expert_trailing() const override { return false; }
  ModelFamily family() const override { return ModelFamily::GLM5NEXT; }
  GraphEngineCaps engine_caps(const StreamingEngine &) const override {
    const bool on = glm5next_full_graph_enabled();
    return {on, on, on};
  }
  void bind_vram_cache(GpuTopology *t) const override { glm5next_full_graph_set_vram_cache(t); }
  const char *name() const override { return "glm5next"; }

  bool enabled(const StreamingEngine &engine,
               ModelGraphPhase phase) const override {
    if (engine.mtp_enabled || !glm5next_full_graph_enabled())
      return false;
    if (phase == ModelGraphPhase::Prefill) {
      // The batched sparse hot/cold MUL_MAT_ID split reaches a CUDA MMQ
      // route-compaction limitation for inactive branch slots, so graph
      // prefill stays opt-in until that is fixed upstream.
      const char *prefill = std::getenv("ZEROMOE_GLM5NEXT_FULL_GRAPH_PREFILL");
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

  // Tokens per prefill batch (llama.cpp's n_ubatch): ZEROMOE_GLM5NEXT_PREFILL_CHUNK, else the toml's
  // `prefill_batch_tokens`, else 2048 (every batch restreams all cold experts, so 512 prefills a 1065-token prompt at
  // 90 tok/s against 250 for one batch). 0 = the whole prompt in one graph. Each batch continues the previous one's state
  // (up to the sparse top-k of 2048 tokens of context). Whatever is configured is reduced by the prepare check to what
  // fits in free VRAM / host RAM, so an oversized value cannot request memory the box does not have.
  int max_prefill_tokens(const StreamingEngine &e) const override {
    if (const char *v = std::getenv("ZEROMOE_GLM5NEXT_PREFILL_CHUNK"); v && v[0])
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

// GLM-5.x "glm-dsa" (absorbed MLA with RoPE, plain residual, routed MoE).  Shares the glm5next graph implementation; the
// DSA indexer is not part of it (dense attention, as the layered path).  Opt-in: ZEROMOE_GLM_DSA_FULL_GRAPH=1 enables the
// decode graph, ZEROMOE_GLM_DSA_FULL_GRAPH_PREFILL=1 the batched prefill graph.
class GlmDsaGraphModel final : public FullGraphModelBase<GlmDsaGraphModel> {
 public:
  GraphRuntimeTuning runtime_tuning(const StreamingEngine &) const override { return {}; }
  bool device_expert_trailing() const override { return false; }
  ModelFamily family() const override { return ModelFamily::GLM_DSA; }
  GraphEngineCaps engine_caps(const StreamingEngine &e) const override {
    const bool on = enabled(e, ModelGraphPhase::Prefill) || enabled(e, ModelGraphPhase::Decode);
    return {on, on, on};
  }
  void bind_vram_cache(GpuTopology *t) const override { glm5next_full_graph_set_vram_cache(t); }
  const char *name() const override { return "glm_dsa"; }

  bool enabled(const StreamingEngine &engine, ModelGraphPhase phase) const override {
    if (engine.mtp_enabled)
      return false;
    const char *v = std::getenv(phase == ModelGraphPhase::Prefill ? "ZEROMOE_GLM_DSA_FULL_GRAPH_PREFILL"
                                                                  : "ZEROMOE_GLM_DSA_FULL_GRAPH");
    return v && v[0] == '1';
  }
  bool bootstrap(StreamingEngine &engine) const override { return glm5next_full_graph_bootstrap(engine); }
  void shutdown() const override { glm5next_full_graph_shutdown(); }
  int device_expert_layers() const override { return glm5next_full_graph_device_expert_layers(); }
  bool supports_cpu_lane() const override {
    const char *v = std::getenv("ZEROMOE_GLM_DSA_CPU_LANE");
    return v && v[0] == '1';
  }

  int max_prefill_tokens(const StreamingEngine &e) const override {
    if (const char *v = std::getenv("ZEROMOE_GLM5NEXT_PREFILL_CHUNK"); v && v[0])
      return std::atoi(v);
    return e.prefill_batch_tokens > 0 ? e.prefill_batch_tokens : 2048;
  }
  int suggested_prefill_batch(const StreamingEngine &) const override { return glm5next_full_graph_suggested_batch(); }
  bool supports_prefill_continuation(const StreamingEngine &) const override { return true; }

  bool accepts(const StreamingEngine &e, const ModelGraphRequest &r) const {
    if (e.mtp_enabled)
      return false;
    return r.phase != ModelGraphPhase::Decode || (r.n_tokens == 1 && r.pos_base >= 0);
  }
  bool prefill_prepare(StreamingEngine &e, const ModelGraphRequest &r) const {
    return glm5next_full_graph_prepare(e, r.n_tokens, r.pos_base, r.token_ids);
  }
  bool prefill_run(StreamingEngine &e, const ModelGraphRequest &r) const {
    return glm5next_full_graph_run(e, r.hidden_states, r.n_tokens, r.pos_base, r.token_ids);
  }
  bool decode_prepare(StreamingEngine &e, const ModelGraphRequest &r) const {
    return glm5next_full_graph_decode_prepare(e, r.pos_base, r.token_ids);
  }
  bool decode_run(StreamingEngine &e, const ModelGraphRequest &r) const {
    return glm5next_full_graph_decode_run(e, r.hidden_states, r.pos_base, r.token_ids);
  }
};

class Ling31FlashGraphModel final : public FullGraphModelBase<Ling31FlashGraphModel> {
  static const zeromoe::graphsettings::ResolvedSettings &settings() {
    return zeromoe::graphsettings::settings_for(zeromoe::graphsettings::Family::Ling);
  }
 public:
  GraphRuntimeTuning runtime_tuning(const StreamingEngine &) const override { return {}; }
  int device_expert_layers() const override { return 0; }
  bool device_expert_trailing() const override { return false; }
  ModelFamily family() const override { return ModelFamily::BailingMoE3; }
  GraphEngineCaps engine_caps(const StreamingEngine &e) const override {
    const bool on = enabled(e, ModelGraphPhase::Prefill) || enabled(e, ModelGraphPhase::Decode);
    return {on, on, on};
  }
  void bind_vram_cache(GpuTopology *t) const override { glm5next_full_graph_set_vram_cache(t); }
  // The opt-in native-cache Ling graph dispatches MoE through the engine's caches like DeepSeek-V4.1.
  bool uses_native_moe_cache() const override { return bailingmoe3_native_graph_enabled(); }
  const char *name() const override { return "ling-3.1-flash"; }
  bool enabled(const StreamingEngine &engine,
               ModelGraphPhase phase) const override {
    return !engine.mtp_enabled && ling_full_graph_enabled() &&
           (phase == ModelGraphPhase::Prefill ||
            phase == ModelGraphPhase::Decode);
  }
  bool bootstrap(StreamingEngine &engine) const override {
    if (bailingmoe3_hybrid_cuda_graph_enabled())
      return glm5next_full_graph_bootstrap(engine);
    return bailingmoe3_native_graph_enabled()
               ? true
               : glm5next_full_graph_bootstrap(engine);
  }
  void shutdown() const override {
    if (bailingmoe3_hybrid_cuda_graph_enabled())
      glm5next_full_graph_shutdown();
    else if (bailingmoe3_native_graph_enabled())
      bailingmoe3_native_graph_shutdown();
    else
      glm5next_full_graph_shutdown();
  }
  bool supports_cpu_lane() const override {
    return !bailingmoe3_native_graph_enabled();
  }
  int max_prefill_tokens(const StreamingEngine &e) const override {
    const auto &chunk = settings().tun.prefill_chunk;
    return int(zeromoe::graphsettings::prefill_chunk_value(chunk, e.prefill_batch_tokens, 2048));
  }
  int min_prefill_batch() const override { return int(settings().tun.prefill_min_batch.value); }
  bool supports_prefill_continuation(const StreamingEngine &) const override {
    return true;
  }
  bool prefill_prepare(StreamingEngine &e, const ModelGraphRequest &r) const {
    if (bailingmoe3_hybrid_cuda_graph_enabled())
      return glm5next_full_graph_prepare(e, r.n_tokens, r.pos_base,
                                         r.token_ids);
    return bailingmoe3_native_graph_enabled()
               ? bailingmoe3_native_graph_prepare(e, r.n_tokens, r.pos_base,
                                                 r.token_ids)
               : glm5next_full_graph_prepare(e, r.n_tokens, r.pos_base,
                                             r.token_ids);
  }
  bool prefill_run(StreamingEngine &e, const ModelGraphRequest &r) const {
    if (bailingmoe3_hybrid_cuda_graph_enabled())
      return glm5next_full_graph_run(e, r.hidden_states, r.n_tokens,
                                     r.pos_base, r.token_ids);
    return bailingmoe3_native_graph_enabled()
               ? bailingmoe3_native_graph_run(e, r.hidden_states, r.n_tokens,
                                              r.pos_base, r.token_ids)
               : glm5next_full_graph_run(e, r.hidden_states, r.n_tokens,
                                         r.pos_base, r.token_ids);
  }
  bool decode_prepare(StreamingEngine &e, const ModelGraphRequest &r) const {
    if (bailingmoe3_hybrid_cuda_graph_enabled())
      return glm5next_full_graph_decode_prepare(e, r.pos_base, r.token_ids);
    return bailingmoe3_native_graph_enabled()
               ? bailingmoe3_native_graph_prepare(e, 1, r.pos_base,
                                                 r.token_ids)
               : glm5next_full_graph_decode_prepare(e, r.pos_base,
                                                    r.token_ids);
  }
  bool decode_run(StreamingEngine &e, const ModelGraphRequest &r) const {
    if (bailingmoe3_hybrid_cuda_graph_enabled())
      return glm5next_full_graph_decode_run(e, r.hidden_states, r.pos_base,
                                           r.token_ids);
    return bailingmoe3_native_graph_enabled()
               ? bailingmoe3_native_graph_run(e, r.hidden_states, 1,
                                              r.pos_base, r.token_ids)
               : glm5next_full_graph_decode_run(e, r.hidden_states,
                                                r.pos_base, r.token_ids);
  }
};

class DeepseekV41GraphModel final
    : public FullGraphModelBase<DeepseekV41GraphModel> {
 public:
  GraphRuntimeTuning runtime_tuning(const StreamingEngine &) const override { return {}; }
  bool device_expert_trailing() const override { return false; }
  bool supports_cpu_lane() const override { return false; }
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
      const char *v = std::getenv("ZEROMOE_DEEPSEEK_V41_PREFILL_CHUNK");
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

class MiMo2GraphModel final : public FullGraphModelBase<MiMo2GraphModel> {
  static const zeromoe::graphsettings::ResolvedSettings &settings() {
    return zeromoe::graphsettings::settings_for(zeromoe::graphsettings::Family::MiMo2);
  }
 public:
  GraphRuntimeTuning runtime_tuning(const StreamingEngine &) const override { return {}; }
  int device_expert_layers() const override { return 0; }
  bool device_expert_trailing() const override { return false; }
  ModelFamily family() const override { return ModelFamily::MiMo2; }
  // MiMo aliases the cache (row-aligned slots) and drops its active-layer pool, which its in-graph path never
  // uses (measured: +0.17 GiB free per GPU, speed and tokens unchanged). It uses no RAM cache.
  GraphEngineCaps engine_caps(const StreamingEngine &) const override {
    return {mimo2_ingraph_moe_enabled(), false, mimo2_ingraph_moe_enabled()};
  }
  void bind_vram_cache(GpuTopology *t) const override { mimo2_full_graph_set_vram_cache(t); }
  const char *name() const override { return "mimo2"; }
  bool enabled(const StreamingEngine &e, ModelGraphPhase) const override {
    return !e.mtp_enabled && mimo2_full_graph_enabled();
  }
  bool uses_native_moe_cache() const override { return true; }
  // In-graph hot/cold MoE with the shared early-ready CPU lane (decode).
  bool supports_cpu_lane() const override { return true; }
  // Later prompt chunks must run as batches too; the default minimum (64) made every chunk
  // after the first fall back to one-token steps.
  int min_prefill_batch() const override { return int(settings().tun.prefill_min_batch.value); }
  bool graph_only() const override { return true; }
  void bind_runtime(StreamingEngine &e, RAMSafetyManager &r,
                      LockFreePrefetcher &p, LayerScratch &s) const override {
    mimo2_bind_runtime(e, r, p, s);
  }
  void shutdown() const override { mimo2_full_graph_shutdown(); }
  int max_prefill_tokens(const StreamingEngine &e) const override {
    // Compatibility env > canonical graph key > engine chunk > family default.
    const auto &chunk = settings().tun.prefill_chunk;
    const long v = zeromoe::graphsettings::prefill_chunk_value(chunk, e.prefill_batch_tokens, 512);
    return int(std::clamp<long>(v, 1, 512));
  }
  bool supports_prefill_continuation(const StreamingEngine &) const override { return true; }
  bool prefill_prepare(StreamingEngine &e, const ModelGraphRequest &r) const {
    return mimo2_full_graph_prepare(e, r.n_tokens, r.pos_base, r.token_ids);
  }
  bool prefill_run(StreamingEngine &e, const ModelGraphRequest &r) const {
    return mimo2_full_graph_run(e, r.hidden_states, r.n_tokens, r.pos_base, r.token_ids);
  }
  bool decode_prepare(StreamingEngine &e, const ModelGraphRequest &r) const {
    return r.n_tokens == 1 && mimo2_full_graph_prepare(e, 1, r.pos_base, r.token_ids);
  }
  bool decode_run(StreamingEngine &e, const ModelGraphRequest &r) const {
    return mimo2_full_graph_run(e, r.hidden_states, 1, r.pos_base, r.token_ids,
        r.logits ? r.logits : e.qwen4exp_full_graph_logits_target,
        r.vocab > 0 ? r.vocab : e.qwen4exp_full_graph_logits_target_size);
  }
};
} // namespace

namespace {
// Tensor types outside the shared quant table are sized one byte per element, which
// silently truncates reads: report them (and abort in strict mode).
std::vector<std::string> graph_preflight(const FullGraphModel &m, const StreamingEngine &e) {
  std::vector<std::string> out;
  std::map<uint32_t, size_t> unknown;
  for (const auto &kv : e.model_file.tensors)
    if (!zeromoe::quant::find(static_cast<uint32_t>(kv.second.type)))
      ++unknown[static_cast<uint32_t>(kv.second.type)];
  for (const auto &u : unknown)
    out.push_back("tensor type " + std::to_string(u.first) + " (" + std::to_string(u.second) +
                  " tensors) is not in memory/quant_table.hpp: it is sized one byte per element");
  const int max_chunk = m.max_prefill_tokens(e);
  if (max_chunk > 0 && max_chunk < m.min_prefill_batch())
    out.push_back("max_prefill_tokens=" + std::to_string(max_chunk) + " is below min_prefill_batch=" +
                  std::to_string(m.min_prefill_batch()) + ": prefill would run as one-token steps");
  return out;
}
} // namespace

void log_graph_capabilities(const FullGraphModel &m, const StreamingEngine &e) {
  for (const std::string &f : graph_preflight(m, e))
    LOG_WARN("[PREFLIGHT] %s: %s\n", m.name(), f.c_str());
  if (zeromoe::graphcfg::flag_on("ZEROMOE_PREFLIGHT_STRICT") && !graph_preflight(m, e).empty()) {
    LOG_ERROR("[PREFLIGHT] strict mode: refusing to run with the findings above\n");
    std::abort();
  }
  const GraphEngineCaps c = m.engine_caps(e);
  LOG_INFO("[GRAPH-CAPS] %s: prefill=%d decode=%d cpu_lane=%d prefill_continuation=%d native_moe_cache=%d "
           "graph_only=%d device_expert_layers=%d trailing=%d aliases_vram_cache=%d "
           "ram_cache_with_preload=%d replaces_active_layer_area=%d min_prefill_batch=%d max_prefill_tokens=%d\n",
           m.name(), m.enabled(e, ModelGraphPhase::Prefill), m.enabled(e, ModelGraphPhase::Decode),
           m.supports_cpu_lane(), m.supports_prefill_continuation(e), m.uses_native_moe_cache(), m.graph_only(),
           m.device_expert_layers(e), m.device_expert_trailing(), c.aliases_vram_cache,
           c.ram_cache_with_preload, c.replaces_active_layer_area, m.min_prefill_batch(),
           m.max_prefill_tokens(e));
  const GraphRuntimeTuning tuning = m.runtime_tuning(e);
  if (tuning.requires_preloaded_ddr_experts ||
      tuning.default_ram_pin_fraction >= 0.0 ||
      tuning.default_ram_pinned_budget_gb >= 0.0) {
    LOG_INFO("[GRAPH-TUNING] %s policy: requires_ddr_source=%d ram_preload_all=%d "
             "default_pin_fraction=%.2f default_pin_budget=%.1f GiB\n",
             m.name(), tuning.requires_preloaded_ddr_experts,
             e.qwen_ram_preload_all_experts, tuning.default_ram_pin_fraction,
             tuning.default_ram_pinned_budget_gb);
  }
}

const FullGraphModel *full_graph_model(ModelFamily family) {
  static const Qwen4ExpGraphModel qwen;
  static const Glm5NextGraphModel glm;
  static const GlmDsaGraphModel glm_dsa;
  static const Ling31FlashGraphModel ling;
  static const DeepseekV41GraphModel dsv41;
  static const MiMo2GraphModel mimo;
  switch (family) {
  case ModelFamily::MiMo2:
    return &mimo;
  case ModelFamily::Qwen4Exp:
    return &qwen;
  case ModelFamily::GLM5NEXT:
    return &glm;
  case ModelFamily::GLM_DSA:
    return &glm_dsa;
  case ModelFamily::BailingMoE3:
    return &ling;
  case ModelFamily::DeepSeek_V41:
    return &dsv41;
  default:
    return nullptr;
  }
}
