// full_graph_models.cpp -- the FullGraphModel of each family and the registry.
//
// These classes are thin: each maps the common request onto the family's
// graph implementation (src/ggml_<family>_graph*.cpp), which keeps its own
// state.  Request validation and prefill/decode dispatch live in
// FullGraphModelBase.
#include "full_graph_model.hpp"

#include "engine.hpp"
#include "dense_dispatch.hpp"
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
#include <cstring>
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#if defined(ZEROMOE_USE_GGML_CUDA)
#include <cuda_runtime.h>
#endif

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

  // Tokens per prefill graph batch use llama.cpp's -ub/--ubatch-size (through
  // `prefill_batch_tokens`) or the matching TOML setting; 0 keeps whole-prompt
  // prefill. Batches continue each other through recurrent state, dense KV rows,
  // and the PLE window. The memory pre-check lowers a batch to one that fits.
  int max_prefill_tokens(const StreamingEngine &e) const override {
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

class Qwen35MoEGraphModel final : public FullGraphModelBase<Qwen35MoEGraphModel> {
public:
  static const zeromoe::graphsettings::ResolvedSettings &settings() {
    return zeromoe::graphsettings::settings_for(
        zeromoe::graphsettings::Family::Qwen);
  }
  int requested_device_expert_layers(const StreamingEngine &engine) const {
    const auto &setting = settings().tun.device_expert_layers;
    const int n_layers =
        std::max(0, engine.model_file.architecture_config.n_layers);
    // A usage-ranked expert byte budget is an alternative to a complete-layer
    // block. Do not combine it with the model-default 22-layer request unless
    // the caller explicitly asks for a hybrid of both placement strategies.
    if (engine.qwen4exp_graph_expert_vram_budget_gb > 0.0 &&
        setting.src == zeromoe::graphsettings::Source::Default)
      return 0;
    // The generic Qwen default (2) predates full-graph offload tuning and
    // leaves nearly all 122B experts on the CPU. Start from a measured split
    // request; the memory planner below fits it to currently free VRAM.
    if (setting.src == zeromoe::graphsettings::Source::Default)
      return std::min(22, n_layers);
    // Zero deliberately keeps all routed experts in DDR. Positive counts are
    // upper bounds; the memory planner reduces them to the largest block that
    // fits. Full-fit recipes set the target layer count explicitly.
    return std::clamp(static_cast<int>(setting.value), 0, n_layers);
  }
  bool requested_device_expert_trailing() const {
    const auto &setting = settings().tun.device_expert_trailing;
    // The 122B smoke used the trailing block. Honor explicit TOML, env, or
    // CLI choices; use that measured direction only when no value was given.
    return setting.src == zeromoe::graphsettings::Source::Default
               ? true
               : setting.value != 0;
  }

  bool supports_architecture(const StreamingEngine &engine) const {
    const auto it = engine.model_file.kv_string.find("general.architecture");
    return it != engine.model_file.kv_string.end() &&
           it->second == "qwen35moe";
  }
  bool tensor_shape(const StreamingEngine &engine, const std::string &name,
                    std::initializer_list<uint64_t> expected) const {
    const auto it = engine.model_file.tensors.find(name);
    if (it == engine.model_file.tensors.end()) {
      LOG_ERROR("[QWEN35MOE-TOPOLOGY] required tensor is missing: %s\n",
                name.c_str());
      return false;
    }
    if (!zeromoe::quant::find(static_cast<uint32_t>(it->second.type))) {
      LOG_ERROR("[QWEN35MOE-TOPOLOGY] tensor %s has unsupported GGML type=%u\n",
                name.c_str(), static_cast<unsigned>(it->second.type));
      return false;
    }
    // GGML tensors always have four extents; llama.cpp compares an expected
    // shape after padding omitted trailing dimensions with 1. Match that
    // contract so a GGUF vector encoded as [D, 1] is accepted as [D], while
    // still rejecting any non-singleton topology difference.
    const size_t rank = std::max(it->second.shape.size(), expected.size());
    for (size_t axis = 0; axis < rank; ++axis) {
      const uint64_t dim = axis < it->second.shape.size()
                               ? it->second.shape[axis]
                               : 1;
      const uint64_t expected_dim = axis < expected.size()
                                        ? *(expected.begin() + axis)
                                        : 1;
      if (dim != expected_dim) {
        LOG_ERROR("[QWEN35MOE-TOPOLOGY] tensor %s shape mismatch axis=%zu "
                  "got=%llu expected=%llu\n",
                  name.c_str(), axis, static_cast<unsigned long long>(dim),
                  static_cast<unsigned long long>(expected_dim));
        return false;
      }
    }
    return true;
  }
  bool supports_graph_topology(const StreamingEngine &engine) const {
    const auto &c = engine.model_file.architecture_config;
    if (!supports_architecture(engine) || c.n_layers <= 0 || c.hidden <= 0 ||
        c.n_heads <= 0 || c.head_dim <= 0 || c.n_head_kv <= 0 ||
        c.n_experts <= 0 || c.topk <= 0 || c.moe_inter <= 0 ||
        c.n_expert_shared <= 0 || c.ssm_group_count <= 0 ||
        c.ssm_state_size <= 0 || c.ssm_inner_size <= 0 ||
        c.ssm_conv_kernel <= 0 || c.ssm_time_step_rank <= 0) {
      LOG_ERROR("[QWEN35MOE-TOPOLOGY] invalid architecture dimensions: "
                "arch=%s layers=%d hidden=%d heads=%d head_dim=%d kv_heads=%d "
                "experts=%d topk=%d moe_inter=%d shared=%d ssm={groups=%d "
                "state=%d inner=%d conv=%d rank=%d}\n",
                "qwen35moe",
                c.n_layers, c.hidden, c.n_heads, c.head_dim, c.n_head_kv,
                c.n_experts, c.topk, c.moe_inter, c.n_expert_shared,
                c.ssm_group_count, c.ssm_state_size, c.ssm_inner_size,
                c.ssm_conv_kernel, c.ssm_time_step_rank);
      return false;
    }

    const uint64_t D = static_cast<uint64_t>(c.hidden);
    const uint64_t H = static_cast<uint64_t>(c.n_heads);
    const uint64_t Hkv = static_cast<uint64_t>(c.n_head_kv);
    const uint64_t hd = static_cast<uint64_t>(c.head_dim);
    const uint64_t E = static_cast<uint64_t>(c.n_experts);
    const uint64_t F = static_cast<uint64_t>(c.moe_inter);
    const uint64_t shared_width = F * static_cast<uint64_t>(c.n_expert_shared);
    const uint64_t conv =
        2ull * static_cast<uint64_t>(c.ssm_group_count) *
            static_cast<uint64_t>(c.ssm_state_size) +
        static_cast<uint64_t>(c.ssm_inner_size);
    const auto has = [&](int layer, const char *stem,
                         std::initializer_list<uint64_t> shape) {
      return tensor_shape(engine,
                          "blk." + std::to_string(layer) + "." + stem,
                          shape);
    };

    for (int layer = 0; layer < c.n_layers; ++layer) {
      if (!has(layer, "attn_norm.weight", {D}) ||
          !has(layer, "post_attention_norm.weight", {D}) ||
          !has(layer, "ffn_gate_inp.weight", {D, E}) ||
          !has(layer, "ffn_gate_exps.weight", {D, F, E}) ||
          !has(layer, "ffn_up_exps.weight", {D, F, E}) ||
          !has(layer, "ffn_down_exps.weight", {F, D, E}) ||
          !has(layer, "ffn_gate_shexp.weight", {D, shared_width}) ||
          !has(layer, "ffn_up_shexp.weight", {D, shared_width}) ||
          !has(layer, "ffn_down_shexp.weight", {shared_width, D}) ||
          !has(layer, "ffn_gate_inp_shexp.weight", {D}))
        return false;

      if (qwen35moe_is_gdn_layer(c, layer)) {
        if (!has(layer, "attn_qkv.weight", {D, conv}) ||
            !has(layer, "attn_gate.weight",
                 {D, static_cast<uint64_t>(c.ssm_inner_size)}) ||
            !has(layer, "ssm_alpha.weight",
                 {D, static_cast<uint64_t>(c.ssm_time_step_rank)}) ||
            !has(layer, "ssm_beta.weight",
                 {D, static_cast<uint64_t>(c.ssm_time_step_rank)}) ||
            !has(layer, "ssm_dt.bias",
                 {static_cast<uint64_t>(c.ssm_time_step_rank)}) ||
            !has(layer, "ssm_a",
                 {static_cast<uint64_t>(c.ssm_time_step_rank)}) ||
            !has(layer, "ssm_conv1d.weight",
                 {static_cast<uint64_t>(c.ssm_conv_kernel), conv}) ||
            !has(layer, "ssm_norm.weight",
                 {static_cast<uint64_t>(c.ssm_state_size)}) ||
            !has(layer, "ssm_out.weight",
                 {static_cast<uint64_t>(c.ssm_inner_size), D}))
          return false;
      } else {
        if (!has(layer, "attn_q.weight", {D, 2 * H * hd}) ||
            !has(layer, "attn_k.weight", {D, Hkv * hd}) ||
            !has(layer, "attn_v.weight", {D, Hkv * hd}) ||
            !has(layer, "attn_q_norm.weight", {hd}) ||
            !has(layer, "attn_k_norm.weight", {hd}) ||
            !has(layer, "attn_output.weight", {H * hd, D}))
          return false;
      }
    }
    const auto output = engine.model_file.tensors.find("output.weight");
    return tensor_shape(engine, "output_norm.weight", {D}) &&
           output != engine.model_file.tensors.end() &&
           zeromoe::quant::find(static_cast<uint32_t>(output->second.type)) &&
           output->second.shape.size() == 2 && output->second.shape[0] == D &&
           output->second.shape[1] > 0;
  }
  ModelFamily family() const override { return ModelFamily::Qwen; }
  const char *name() const override { return "qwen35moe-full-graph"; }
  GraphRuntimeTuning runtime_tuning(const StreamingEngine &engine) const override {
    const auto &c = engine.model_file.architecture_config;
    const bool plan_ready = engine.cuda_moe_graph_memory_plan != "not-applicable";
    const bool plan_insufficient =
        engine.cuda_moe_graph_memory_plan == "insufficient";
    bool has_ddr_experts =
        plan_ready && !plan_insufficient &&
        engine.cuda_moe_graph_device_expert_layers < c.n_layers;
    if (plan_ready && !plan_insufficient) {
      for (int layer = 0; layer < c.n_layers; ++layer) {
        if (is_cpu_affinity_layer(layer)) {
          has_ddr_experts = true;
          break;
        }
      }
    }
    if (!has_ddr_experts)
      return {};
    return {.requires_preloaded_ddr_experts = true,
            .default_ram_pin_fraction = 0.90,
            .default_ram_pinned_budget_gb = 45.0};
  }
  void shutdown() const override { qwen4exp_full_graph_shutdown(); }
  bool enabled(const StreamingEngine &engine,
               ModelGraphPhase phase) const override {
#if defined(ZEROMOE_USE_GGML_CUDA)
    if (engine.cuda_moe_graph_memory_plan == "insufficient")
      return false;
    return supports_architecture(engine) && !engine.mtp_enabled &&
           engine.model_file.architecture_config.n_experts > 0 &&
       (phase == ModelGraphPhase::Decode ||
        (phase == ModelGraphPhase::Prefill &&
         engine.cuda_moe_full_graph_decode_required));
#else
    (void)engine;
    (void)phase;
    return false;
#endif
  }
  GraphMemoryPlan plan_decode_memory(const StreamingEngine &engine) const override {
    GraphMemoryPlan plan;
    const auto &c = engine.model_file.architecture_config;
    if (!supports_graph_topology(engine)) {
      plan.fit = GraphMemoryFit::Unsupported;
      plan.reason =
          "qwen35moe architecture or required tensor topology is unsupported";
      return plan;
    }
    if (engine.mtp_enabled) {
      plan.fit = GraphMemoryFit::Unsupported;
      plan.reason = "Qwen3.5/3.6 graph decode does not support MTP";
      return plan;
    }
    if (engine.max_t <= 0) {
      plan.reason = "Qwen full-graph planner cannot size KV state without max_t";
      return plan;
    }

    auto add_saturating = [](size_t a, size_t b) {
      return b > std::numeric_limits<size_t>::max() - a
                 ? std::numeric_limits<size_t>::max()
                 : a + b;
    };
    auto mul_saturating = [](size_t a, size_t b) {
      return a != 0 && b > std::numeric_limits<size_t>::max() / a
                 ? std::numeric_limits<size_t>::max()
                 : a * b;
    };
    auto tensor_layer = [](const std::string &name) {
      if (name.compare(0, 4, "blk.") != 0)
        return -1;
      char *end = nullptr;
      const long layer = std::strtol(name.c_str() + 4, &end, 10);
      if (end == name.c_str() + 4 || !end || *end != '.' || layer < 0 ||
          layer > std::numeric_limits<int>::max())
        return -1;
      return static_cast<int>(layer);
    };
    auto routed_expert_projection = [](const std::string &name) {
      for (const char *suffix : {"ffn_gate_exps.weight",
                                 "ffn_up_exps.weight",
                                 "ffn_down_exps.weight"}) {
        const size_t length = std::strlen(suffix);
        if (name.size() >= length &&
            name.compare(name.size() - length, length, suffix) == 0)
          return true;
      }
      return false;
    };

    // Match init_device_dense_arena() exactly instead of treating the GGUF
    // file size as VRAM demand. The helper counts only loaded dense tensors
    // actually consumed by this graph; routed expert weights are planned
    // separately below and NextN/CPU embedding data stay out of VRAM.
    const size_t dense_and_aux_bytes =
        qwen35moe_full_graph_dense_weight_bytes(engine);
    if (dense_and_aux_bytes == 0) {
      plan.fit = GraphMemoryFit::Unsupported;
      plan.reason = "qwen35moe dense graph-weight manifest is empty";
      return plan;
    }

    // Routed experts remain available from complete preloaded DDR arenas.
    // Only count the block selected for this run as device-resident.
    std::vector<size_t> expert_bytes_per_layer(
        static_cast<size_t>(c.n_layers), 0);
    for (const auto &entry : engine.model_file.tensors) {
      const int layer = tensor_layer(entry.first);
      if (layer < 0 || layer >= c.n_layers ||
          !routed_expert_projection(entry.first))
        continue;
      auto &layer_bytes = expert_bytes_per_layer[static_cast<size_t>(layer)];
      layer_bytes = add_saturating(
          layer_bytes, static_cast<size_t>(entry.second.size_bytes));
    }
    if (std::any_of(expert_bytes_per_layer.begin(),
                    expert_bytes_per_layer.end(),
                    [](size_t bytes) { return bytes == 0; })) {
      plan.fit = GraphMemoryFit::Unsupported;
      plan.reason =
          "qwen35moe target graph has an incomplete routed-expert tensor set";
      return plan;
    }

    const double usage_budget_gb =
        engine.qwen4exp_graph_expert_vram_budget_gb;
    size_t usage_budget_bytes = 0;
    if (usage_budget_gb > 0.0) {
      if (engine.qwen4exp_option_b_top_n > 0 ||
          engine.qwen4exp_option_b_global_top_k > 0) {
        plan.fit = GraphMemoryFit::Unsupported;
        plan.reason =
            "qwen35moe usage VRAM budget cannot be combined with Option-B "
            "top-N/global-top-K placement";
        return plan;
      }
      const long double byte_count = static_cast<long double>(usage_budget_gb) *
                                     1024.0L * 1024.0L * 1024.0L;
      if (!std::isfinite(usage_budget_gb) || byte_count <= 0.0L ||
          byte_count > static_cast<long double>(
                           std::numeric_limits<size_t>::max())) {
        plan.fit = GraphMemoryFit::Unsupported;
        plan.reason = "qwen35moe usage VRAM budget is not a valid byte size";
        return plan;
      }
      usage_budget_bytes = static_cast<size_t>(byte_count);
      const std::string usage_path = engine.qwen4exp_graph_pin_file.empty()
                                         ? std::string(".zeromoe_usage")
                                         : engine.qwen4exp_graph_pin_file;
      const auto [usage, max_count] = StreamingEngine::load_usage(usage_path);
      size_t eligible_usage = 0;
      size_t zero_usage = 0;
      size_t out_of_range_usage = 0;
      size_t cpu_affinity_usage = 0;
      for (const auto &entry : usage) {
        if (entry.count == 0) {
          ++zero_usage;
        } else if (entry.layer < 0 || entry.layer >= c.n_layers ||
                   entry.expert < 0 || entry.expert >= c.n_experts) {
          ++out_of_range_usage;
        } else if (is_cpu_affinity_layer(entry.layer)) {
          ++cpu_affinity_usage;
        } else {
          ++eligible_usage;
        }
      }
      LOG_INFO("[QWEN35MOE-PLAN] usage histogram path=%s parsed=%zu "
               "eligible=%zu zero=%zu out_of_range=%zu cpu_affinity=%zu "
               "max_count=%llu layers=%d experts=%d\n",
               usage_path.c_str(), usage.size(), eligible_usage, zero_usage,
               out_of_range_usage, cpu_affinity_usage,
               static_cast<unsigned long long>(max_count), c.n_layers,
               c.n_experts);
      if (eligible_usage == 0) {
        plan.fit = GraphMemoryFit::Unsupported;
        plan.reason = "qwen35moe usage VRAM budget requires a non-empty, "
                      "model-compatible usage file: " + usage_path;
        return plan;
      }
    }

    const int kv_heads = std::max(1, c.n_head_kv);
    const int head_dim = c.head_dim > 0
                             ? c.head_dim
                             : c.hidden / std::max(1, c.n_heads);
    const size_t kv_capacity = static_cast<size_t>(engine.max_t);
    size_t full_attention_layers = 0;
    size_t gdn_layers = 0;
    for (int layer = 0; layer < c.n_layers; ++layer) {
      if (qwen35moe_is_gdn_layer(c, layer))
        ++gdn_layers;
      else
        ++full_attention_layers;
    }
    const size_t kv_bytes = mul_saturating(
        mul_saturating(mul_saturating(mul_saturating(
                           full_attention_layers, 2),
                       static_cast<size_t>(head_dim)),
                   static_cast<size_t>(kv_heads)),
        mul_saturating(kv_capacity, sizeof(float)));
    const size_t conv_width = static_cast<size_t>(
        2 * c.ssm_group_count * c.ssm_state_size + c.ssm_inner_size);
    const size_t gdn_state_per_layer = add_saturating(
        mul_saturating(static_cast<size_t>(std::max(0, c.ssm_conv_kernel - 1)),
                       conv_width),
        mul_saturating(
            mul_saturating(static_cast<size_t>(c.ssm_state_size),
                           static_cast<size_t>(c.ssm_state_size)),
            static_cast<size_t>(std::max(1, c.ssm_time_step_rank))));
    const size_t gdn_bytes = mul_saturating(
        mul_saturating(gdn_layers, gdn_state_per_layer), sizeof(float));
  // This is the decode-graph workspace reserve. The builder checks prefill
  // workspace separately for each requested batch; a decode Fits plan does
  // not prove that full-graph prefill also fits. Require that path separately
  // in onboarding profiles that claim graph prefill support.
  constexpr size_t kDecodeWorkspaceBytes =
      (1ull * 1024ull * 1024ull * 1024ull) +
      (256ull * 1024ull * 1024ull);
    const size_t fixed_device_bytes = add_saturating(
        add_saturating(dense_and_aux_bytes, add_saturating(kv_bytes, gdn_bytes)),
        kDecodeWorkspaceBytes);

    const auto &layer_setting = settings().tun.device_expert_layers;
    const int requested_layers = requested_device_expert_layers(engine);
    const bool trailing = requested_device_expert_trailing();

#if defined(ZEROMOE_USE_GGML_CUDA)
    int active_device = -1;
    int target_device = engine.gpu_device_id;
    if (cudaGetDevice(&active_device) != cudaSuccess)
      active_device = -1;
    if (target_device < 0)
      target_device = active_device;
    if (target_device < 0 || cudaSetDevice(target_device) != cudaSuccess) {
      plan.reason = "CUDA could not select the Qwen graph device for memory planning";
      return plan;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t mem_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (active_device >= 0 && active_device != target_device)
      cudaSetDevice(active_device);
    if (mem_status != cudaSuccess || free_bytes == 0) {
      plan.reason = "CUDA could not report free VRAM for Qwen full-graph planning";
      return plan;
    }

    int selected_layers = -1;
    size_t selected_required_bytes = 0;
    for (int count = requested_layers; count >= 0; --count) {
      const int first_layer = trailing ? c.n_layers - count : 0;
      size_t resident_expert_bytes = 0;
      for (int layer = first_layer; layer < first_layer + count; ++layer) {
        if (is_cpu_affinity_layer(layer))
          continue;
        resident_expert_bytes = add_saturating(
            resident_expert_bytes,
            expert_bytes_per_layer[static_cast<size_t>(layer)]);
      }
      // The shared graph's usage-ranked byte-budget selector places experts
      // from the remaining layers. Count its full ceiling here; its runtime
      // placement may spend less, but must never exceed this estimate.
      if (usage_budget_bytes > 0) {
        size_t split_capacity_bytes = 0;
        for (int layer = 0; layer < c.n_layers; ++layer) {
          if (is_cpu_affinity_layer(layer))
            continue;
          const bool in_device_block =
              count > 0 &&
              (trailing ? layer >= c.n_layers - count : layer < count);
          if (in_device_block)
            continue;
          split_capacity_bytes = add_saturating(
              split_capacity_bytes,
              expert_bytes_per_layer[static_cast<size_t>(layer)]);
        }
        resident_expert_bytes = add_saturating(
            resident_expert_bytes,
            std::min(usage_budget_bytes, split_capacity_bytes));
      }
      const size_t device_weights =
          add_saturating(dense_and_aux_bytes, resident_expert_bytes);
      const size_t alignment_bytes = device_weights / 100;
      const size_t required = add_saturating(
          add_saturating(device_weights, alignment_bytes),
          add_saturating(add_saturating(kv_bytes, gdn_bytes),
                         kDecodeWorkspaceBytes));
      if (required > 0 && required <= free_bytes) {
        selected_layers = count;
        selected_required_bytes = required;
        break;
      }
    }

    plan.available_device_bytes = free_bytes;
    plan.device_expert_layers = selected_layers >= 0 ? selected_layers : 0;
    plan.device_expert_trailing = trailing;
    if (selected_layers >= 0) {
      plan.fit = GraphMemoryFit::Fits;
      plan.required_device_bytes = selected_required_bytes;
      const bool usage_hotset_only =
          usage_budget_gb > 0.0 && requested_layers == 0;
      const bool automatic_request =
          layer_setting.src != zeromoe::graphsettings::Source::Default &&
          layer_setting.value == 0 && !usage_hotset_only;
      const std::string request_label =
          usage_hotset_only
              ? "usage-hotset-only"
              : layer_setting.src == zeromoe::graphsettings::Source::Default
              ? "22(model-default)"
              : automatic_request
                    ? "auto-fit"
                    : std::to_string(layer_setting.value);
      LOG_INFO("[QWEN35MOE-PLAN] placement=VRAM+DDR requested_layers=%s "
               "resident_layers=%d/%d trailing=%d required_device_gib=%.2f "
               "available_device_gib=%.2f usage_hotset_budget_gib=%.2f "
               "host_experts=complete-preloaded\n",
               request_label.c_str(), selected_layers, c.n_layers, trailing,
               static_cast<double>(selected_required_bytes) / (1024.0 * 1024.0 * 1024.0),
               static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0),
               usage_budget_gb);
      if (selected_layers == 0 && usage_budget_gb > 0.0) {
        LOG_INFO("[QWEN35MOE-PLAN] no complete expert layer is resident; "
                 "up to %.2f GiB of usage-ranked experts is planned for VRAM\n",
                 usage_budget_gb);
      } else if (selected_layers == 0) {
        LOG_WARN("[QWEN35MOE-PLAN] graph fits with all routed experts in DDR; "
                 "decode will be CPU/DDR-bound\n");
      }
      plan.reason =
          "qwen35moe full graph with " + std::to_string(selected_layers) +
          " complete expert layers and " + std::to_string(usage_budget_gb) +
          " GiB of usage-ranked expert hotset in VRAM; remaining expert "
          "projections use the complete DDR source";
    } else {
      const size_t minimum_weights =
          add_saturating(dense_and_aux_bytes, usage_budget_bytes);
      const size_t minimum_required = add_saturating(
          add_saturating(fixed_device_bytes, usage_budget_bytes),
          minimum_weights / 100);
      plan.fit = GraphMemoryFit::Insufficient;
      plan.required_device_bytes = minimum_required;
      LOG_WARN("[QWEN35MOE-PLAN] no requested VRAM placement fits: minimum "
               "graph requires %.2f GiB (usage-hotset budget %.2f GiB); "
               "free %.2f GiB\n",
               static_cast<double>(minimum_required) / (1024.0 * 1024.0 * 1024.0),
               usage_budget_gb,
               static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0));
      plan.reason =
          "qwen35moe dense weights, F32 KV/GDN state, decode workspace, and "
          "requested expert placement exceed free VRAM";
    }
    (void)total_bytes;
    return plan;
#else
    plan.reason = "Qwen full-graph CUDA memory planning is unavailable in this build";
    return plan;
#endif
  }
  GraphEngineCaps engine_caps(const StreamingEngine &) const override {
    // Nonresident routed experts are read from the complete preloaded DDR
    // graph arenas; the graph owns only its selected VRAM expert block.
    return {false, true, false, true};
  }
  int device_expert_layers() const override { return 0; }
  int device_expert_layers(const StreamingEngine &engine) const override {
    return engine.cuda_moe_graph_memory_plan == "not-applicable"
               ? requested_device_expert_layers(engine)
               : engine.cuda_moe_graph_device_expert_layers;
  }
  bool device_expert_trailing() const override {
    return requested_device_expert_trailing();
  }
  bool device_expert_trailing(const StreamingEngine &engine) const override {
    return engine.cuda_moe_graph_memory_plan == "not-applicable"
               ? requested_device_expert_trailing()
               : engine.cuda_moe_graph_device_expert_trailing;
  }
  moe_routed::RouterSpec router_spec(const ArchitectureConfig &c,
                                      int n_tokens) const override {
    // Match CPUQwenMoEBackend: softmax over all experts, top-k, then
    // renormalize the selected weights. That path does not use group routing
    // or a GGUF selection-bias tensor.
    auto spec = moe_routed::router_spec(c, n_tokens,
                                        /*honor_gating_func=*/false);
    spec.weights_norm = true;
    spec.n_groups = 0;
    spec.n_groups_used = 0;
    return spec;
  }
  bool supports_cpu_lane() const override { return true; }
  int max_prefill_tokens(const StreamingEngine &engine) const override {
    if (const char *value =
            zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_PREFILL_CHUNK");
        value && value[0])
      return std::atoi(value);
    return engine.prefill_batch_tokens > 0 ? engine.prefill_batch_tokens : 16;
  }
  int suggested_prefill_batch(const StreamingEngine &) const override {
    return qwen4exp_full_graph_suggested_batch();
  }
  bool supports_prefill_continuation(const StreamingEngine &) const override {
    return true;
  }
  int min_prefill_batch() const override { return 2; }
  bool requires_full_graph_prefill(
      const StreamingEngine &engine) const override {
    return supports_architecture(engine) &&
           engine.cuda_moe_full_graph_decode_required;
  }
  bool prefill_prepare(StreamingEngine &engine,
                       const ModelGraphRequest &request) const {
    return qwen4exp_full_graph_prepare(engine, request.n_tokens,
                                       request.pos_base, request.token_ids);
}
  bool prefill_run(StreamingEngine &engine,
                   const ModelGraphRequest &request) const {
    return qwen4exp_full_graph_run(engine, request.hidden_states,
                                   request.n_tokens, request.pos_base,
                                   request.token_ids);
  }
  bool decode_prepare(StreamingEngine &engine,
                      const ModelGraphRequest &request) const {
    return qwen4exp_full_graph_decode_prepare(
        engine, request.pos_base, request.token_ids);
  }
  bool decode_run(StreamingEngine &engine,
                  const ModelGraphRequest &request) const {
    float *logits = request.logits ? request.logits
                                   : engine.qwen4exp_full_graph_logits_target;
    const int vocab = request.vocab > 0
                          ? request.vocab
                          : engine.qwen4exp_full_graph_logits_target_size;
    return qwen4exp_full_graph_decode_run(
        engine, request.hidden_states, request.pos_base, request.token_ids,
        logits, vocab);
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

  bool enabled(const StreamingEngine &engine, ModelGraphPhase) const override {
    // Prefill and decode both run through the graph, like every other family
    // (glm-dsa, Ling, MiMo, Qwen). MTP keeps the layered path: its speculative
    // loop is not integrated with this graph. Graph prefill used to be opt-in
    // (ZEROMOE_GLM5NEXT_FULL_GRAPH_PREFILL) because the batched hot/cold
    // MUL_MAT_ID split hit a CUDA MMQ route-compaction limitation for inactive
    // branch slots; ggml patch 0006 (mmq-id-inactive-slots) fixes it and is part
    // of the required patch series.
    return !engine.mtp_enabled;
  }
  bool bootstrap(StreamingEngine &engine) const override {
    return glm5next_full_graph_bootstrap(engine);
  }
  void shutdown() const override { glm5next_full_graph_shutdown(); }
  int device_expert_layers() const override {
    return glm5next_full_graph_device_expert_layers();
  }
  bool supports_cpu_lane() const override { return true; }

  // Tokens per prefill batch use llama.cpp's -ub/--ubatch-size (through
  // `prefill_batch_tokens`) or the matching TOML setting, then default to 2048.
  // Every batch restreams all cold experts, so 512 prefills a 1065-token prompt
  // at 90 tok/s against 250 for one batch. Each batch continues the previous
  // one's state. The prepare check reduces an oversized value to what fits in
  // free VRAM and host RAM.
  int max_prefill_tokens(const StreamingEngine &e) const override {
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
// DSA indexer is not part of it (dense attention, as the layered path).  Full-graph prefill and decode are the defaults;
// MTP is the only supported opt-out because its speculative loop is not integrated with this graph.
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

  bool enabled(const StreamingEngine &engine, ModelGraphPhase) const override {
    // Prefill and decode both default to the graph.  MTP keeps the layered
    // path because this adapter does not implement speculative graph decode.
    return !engine.mtp_enabled;
  }
  bool bootstrap(StreamingEngine &engine) const override { return glm5next_full_graph_bootstrap(engine); }
  void shutdown() const override { glm5next_full_graph_shutdown(); }
  int device_expert_layers() const override { return glm5next_full_graph_device_expert_layers(); }
  bool supports_cpu_lane() const override {
    return zeromoe::graphsettings::active_settings().tun.dsa_cpu_lane.value != 0;
  }

  int max_prefill_tokens(const StreamingEngine &e) const override {
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

  bool supports_architecture(const StreamingEngine &e) const {
    const auto it = e.model_file.kv_string.find("general.architecture");
    return it != e.model_file.kv_string.end() && it->second == "deepseek41";
  }

  bool plan_decode_memory_after_placement() const override { return true; }

  GraphMemoryPlan plan_decode_memory(const StreamingEngine &e) const override {
    GraphMemoryPlan plan;
    if (!supports_architecture(e)) {
      plan.fit = GraphMemoryFit::Unsupported;
      plan.reason = "DeepSeek-V4.1 adapter only supports general.architecture=deepseek41";
      return plan;
    }
    const auto &c = e.model_file.architecture_config;
    if (c.n_layers != 40 || c.hidden != 5120 || c.n_experts != 384 ||
        c.topk != 6 || c.hc_mult != 4 || c.n_expert_shared != 1 ||
        c.dsv41_engram_layer_ids.empty()) {
      plan.fit = GraphMemoryFit::Unsupported;
      plan.reason = "DeepSeek-V4.1 Engram tensor topology does not match the registered 1017 graph adapter";
      return plan;
    }
    // Dense weights and the engine-owned DDR/VRAM expert caches are already
    // allocated when this final plan runs. Their bytes therefore reduce the
    // live free-VRAM estimates below. Account the graph's scheduler workspace
    // plus its explicit 1 GiB construction safety floor on every participating
    // device; historical decode graphs used ~1 MiB workspace, while the
    // larger prompt graph is checked separately by its builder.
    if (e.moe_placement.model_slots == 0 || !e.gpu) {
      plan.reason = "DeepSeek native-cache placement is not ready; final planning runs after cache allocation";
      return plan;
    }
#if defined(ZEROMOE_USE_GGML_CUDA)
    constexpr size_t kGraphWorkspaceEstimate = 64ull * 1024ull * 1024ull;
    constexpr size_t kGraphConstructionReserve = 1ull * 1024ull * 1024ull * 1024ull;
    plan.required_device_bytes =
        kGraphWorkspaceEstimate + kGraphConstructionReserve;
    plan.device_expert_layers = 0;
    plan.device_expert_trailing = false;

    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess)
      active_device = -1;
    size_t min_free_bytes = std::numeric_limits<size_t>::max();
    int checked_devices = 0;
    int limiting_device = -1;
    for (const auto &device : e.gpu->topology.devices) {
      if (device.role == GpuRole::Unused)
        continue;
      if (cudaSetDevice(device.id) != cudaSuccess) {
        if (active_device >= 0)
          cudaSetDevice(active_device);
        plan.required_device_bytes = 0;
        plan.reason = "CUDA could not select every participating DeepSeek graph device";
        return plan;
      }
      size_t free_bytes = 0, total_bytes = 0;
      const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
      (void)total_bytes;
      if (status != cudaSuccess || free_bytes == 0) {
        if (active_device >= 0)
          cudaSetDevice(active_device);
        plan.required_device_bytes = 0;
        plan.reason = "CUDA could not report free VRAM for DeepSeek graph planning";
        return plan;
      }
      if (free_bytes < min_free_bytes) {
        min_free_bytes = free_bytes;
        limiting_device = device.id;
      }
      ++checked_devices;
    }
    if (active_device >= 0)
      cudaSetDevice(active_device);
    if (checked_devices == 0 || min_free_bytes == std::numeric_limits<size_t>::max()) {
      plan.required_device_bytes = 0;
      plan.reason = "DeepSeek graph has no active CUDA device for final memory planning";
      return plan;
    }
    plan.available_device_bytes = min_free_bytes;
    if (plan.required_device_bytes <= plan.available_device_bytes) {
      plan.fit = GraphMemoryFit::Fits;
      plan.reason = "engine-owned dense/MoE allocations are committed; graph workspace plus 1 GiB safety floor fits on all active GPUs";
      LOG_INFO("[DSV41-PLAN] memory_plan=fits devices=%d limiting_gpu=%d "
               "required_gib=%.3f available_gib=%.3f DDR_gib=%.2f VRAM_gib=%.2f\n",
               checked_devices, limiting_device,
               static_cast<double>(plan.required_device_bytes) / (1024.0 * 1024.0 * 1024.0),
               static_cast<double>(plan.available_device_bytes) / (1024.0 * 1024.0 * 1024.0),
               static_cast<double>(e.moe_placement.ddr_bytes) / (1024.0 * 1024.0 * 1024.0),
               static_cast<double>(e.moe_placement.vram_bytes) / (1024.0 * 1024.0 * 1024.0));
    } else {
      plan.fit = GraphMemoryFit::Insufficient;
      plan.reason = "DeepSeek decode graph workspace plus 1 GiB safety floor exceeds free VRAM after engine-owned dense/MoE allocations";
      LOG_WARN("[DSV41-PLAN] memory_plan=insufficient limiting_gpu=%d "
               "required_gib=%.3f available_gib=%.3f\n",
               limiting_device,
               static_cast<double>(plan.required_device_bytes) / (1024.0 * 1024.0 * 1024.0),
               static_cast<double>(plan.available_device_bytes) / (1024.0 * 1024.0 * 1024.0));
    }
#else
    plan.required_device_bytes = 0;
    plan.reason = "DeepSeek CUDA graph memory planning is unavailable in this build";
#endif
    return plan;
  }
  GraphEngineCaps engine_caps(const StreamingEngine &) const override {
    return {.aliases_vram_cache = true};
  }
  bool uses_native_moe_cache() const override { return true; }

  bool enabled(const StreamingEngine &e, ModelGraphPhase phase) const override {
    if (phase == ModelGraphPhase::Decode &&
        e.cuda_moe_graph_memory_plan == "insufficient")
      return false;
    return supports_architecture(e) && !e.mtp_enabled &&
           (phase == ModelGraphPhase::Decode
                ? deepseek_v41_full_graph_decode_enabled()
                : deepseek_v41_full_graph_enabled());
  }
  void shutdown() const override { deepseek_v41_full_graph_shutdown(); }
  int device_expert_layers() const override {
    return deepseek_v41_full_graph_device_expert_layers();
  }
  int max_prefill_tokens(const StreamingEngine &e) const override {
    // Native cache-backed attention supports whole-prompt prefill by default.
    // The shared TOML/CLI batch limit remains available for explicit runs.
    return e.prefill_batch_tokens > 0 ? e.prefill_batch_tokens : 0;
  }

  bool accepts(const StreamingEngine &e, const ModelGraphRequest &r) const {
    return supports_architecture(e) && !e.mtp_enabled &&
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
  if (zeromoe::graphsettings::active_settings().diag.preflight_strict.value &&
      !graph_preflight(m, e).empty()) {
    LOG_ERROR("[PREFLIGHT] strict mode: refusing to run with the findings above\n");
    std::abort();
  }
  const GraphEngineCaps c = m.engine_caps(e);
  LOG_INFO("[GRAPH-CAPS] %s: prefill=%d decode=%d cpu_lane=%d prefill_continuation=%d native_moe_cache=%d "
           "graph_only=%d device_expert_layers=%d trailing=%d aliases_vram_cache=%d owns_expert_storage=%d "
           "ram_cache_with_preload=%d replaces_active_layer_area=%d min_prefill_batch=%d max_prefill_tokens=%d\n",
           m.name(), m.enabled(e, ModelGraphPhase::Prefill), m.enabled(e, ModelGraphPhase::Decode),
           m.supports_cpu_lane(), m.supports_prefill_continuation(e), m.uses_native_moe_cache(), m.graph_only(),
           m.device_expert_layers(e), m.device_expert_trailing(e), c.aliases_vram_cache,
           c.owns_expert_storage, c.ram_cache_with_preload, c.replaces_active_layer_area, m.min_prefill_batch(),
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
static const Qwen35MoEGraphModel qwen35moe;
  static const Glm5NextGraphModel glm;
  static const GlmDsaGraphModel glm_dsa;
  static const Ling31FlashGraphModel ling;
  static const DeepseekV41GraphModel dsv41;
  static const MiMo2GraphModel mimo;
  switch (family) {
case ModelFamily::Qwen:
return &qwen35moe;
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
