// family_ops.hpp — the single, live per-family dispatch table for the GPU
// forward pass.
//
// Context (see CLAUDE.md's "DeepSeek-V4/V4-Flash" plan for the full
// writeup): the live GPU loop (layers_forward_gguf, execution_layer.cpp)
// used to dispatch attention on one bool (ArchitectureConfig::is_mla),
// residual-add was inlined identically twice with no per-family hook at
// all, and MoE gating was a second bool (use_sigmoid_gate) -- each set by
// raw `arch0 == "literal-string"` comparisons scattered through
// graph_initializer.cpp, independent of and redundant with this project's
// OWN ModelFamily enum / detect_model_family() (model_family.hpp). Adding a
// family whose residual isn't a plain per-layer add (DeepSeek-V4's
// Hyper-Connections) had no boolean flag that could express it.
//
// FamilyOps replaces all of that with one struct, resolved ONCE per model
// load from ModelFamily via resolve_family_ops() below, and read from
// StreamingEngine::family_ops everywhere the old is_mla/use_sigmoid_gate
// checks used to live. ModelFamily/detect_model_family() (model_family.hpp)
// stays the single source of truth for "which family is this file" --
// FamilyOps only answers "what does that family DO", so a new family is one
// new detect_model_family() arch-string branch + one new switch case here,
// not N scattered edits.
//
// Deliberate split in what gets a function pointer vs. a plain enum field:
//   - STRUCTURALLY different code (a whole different attention algorithm)
//     -> a function pointer. mla_attention_absorbed_gguf and
//     dense_attention_gguf (attention_layer.cpp) already share an identical
//     signature, which is what makes this hook trivial to add without
//     touching either existing function.
//   - PARAMETRIC variation (which gating formula, which cache shape) -> an
//     enum, switched at the one shared call site that needs it. Turning
//     every such variation into virtual dispatch would be churn without
//     payoff; leaving it as scattered raw string comparisons (the thing
//     this header exists to fix) is the opposite mistake.
//
// Deliberately free of engine.hpp / CUDA headers (StreamingEngine is
// forward-declared only), matching model_family.hpp's own discipline, so
// this header alone lint-checks cleanly on a machine with no CUDA Toolkit
// (see CLAUDE.md's macOS/non-CUDA dev-iteration workflow). family_ops.cpp,
// which actually calls into attention_layer.cpp's real functions, hits the
// usual cuda_runtime.h wall same as execution_layer.cpp/graph_initializer.cpp
// already do -- that's expected, not a regression.
#pragma once
#include "model_family.hpp"
#include <cstdint>

struct StreamingEngine;
struct ArchitectureConfig;

// CacheShape/GatingFunc are defined in gguf_loader.hpp (which model_family.hpp
// -- included above -- already pulls in transitively), not here: gguf_loader.
// hpp's own ArchitectureConfig needs them as native field types, and
// family_ops.hpp -> model_family.hpp -> gguf_loader.hpp is already the
// dependency direction, so they can't be defined on this side without a
// circular include. See gguf_loader.hpp's own comment on these two enums.
//
// NOT the same axis as AttentionType (model_family.hpp) -- AttentionType is
// a coarse documentation-only label three dormant CPU/GPU backend classes
// each declare for themselves (see model_base.hpp) and nothing in the live
// path reads; CacheShape is the thing graph_initializer.cpp's cache-
// allocation code actually switches on. GatingFunc's GGUF-file-level
// override (expert_gating_func) still wins over FamilyOps::default_gating
// when the file sets one -- this enum only supplies the DEFAULT, exactly
// mirroring the precedence the old use_sigmoid_gate fallback already had.

// Per-layer attention compute. mla_attention_absorbed_gguf and
// dense_attention_gguf (attention_layer.cpp) already match this signature;
// dsv4_attention_gguf (Project B) will too.
using AttentionFn = void (*)(StreamingEngine &engine, int layer_idx,
                              const float *hidden_states, int S,
                              int pos_base, float *out_context);

// Allocates whatever engine-level cache fields this family's CacheShape
// needs (Lc/Rc, Kc/Vc, or -- Project B -- KVc), sized from max_ctx and the
// resolved ArchitectureConfig. Called once at graph-init time.
using AllocCacheFn = void (*)(StreamingEngine &engine,
                               const ArchitectureConfig &c, int max_ctx);

// Hyper-connection bracket hooks, called immediately before/after EVERY
// sub-block (attention and FFN/MoE each get their own call, distinguished
// by sub_block: 0 = attention, 1 = FFN/MoE). For every family that doesn't
// use hyper-connections (uses_hyper_connections == false, hc_mult == 1),
// these are the two shared no-op/plain-add functions below -- NOT a special
// case, but the natural hc_mult==1 degenerate case of the same two hooks
// DeepSeek-V4 uses. Every existing attention_fn/dense_ffn_gguf/
// moe_forward_gguf keeps reading and writing plain [S,D] hidden_states
// either way; only these two hooks (and, for DeepSeek-V4, the model-level
// expand/collapse hooks in generation_loop.hpp) ever see the multi-stream
// shape.
using HcPreFn = void (*)(StreamingEngine &engine, int layer_idx,
                          int sub_block, float *hidden_states, int S, int D);
using HcPostFn = void (*)(StreamingEngine &engine, int layer_idx,
                           int sub_block, float *hidden_states,
                           const float *sub_block_out, int S, int D);

// Shared, family-agnostic hook implementations (family_ops.cpp). Every
// existing family (and any new family that doesn't need hyper-connections)
// points its hc_pre_fn/hc_post_fn at these.
void hc_pre_noop(StreamingEngine &engine, int layer_idx, int sub_block,
                  float *hidden_states, int S, int D);
// hc_post_plain_add is the plain residual add (hidden_states += sub_block_out)
// -- a lift, not a rewrite, of the AVX-512 loop layers_forward_gguf used to
// inline twice. Bit-for-bit identical output for every family that uses it.
void hc_post_plain_add(StreamingEngine &engine, int layer_idx, int sub_block,
                        float *hidden_states, const float *sub_block_out,
                        int S, int D);

// Optional per-layer INPUT hook, called at the very top of each layer's
// iteration in layers_forward_gguf, BEFORE hc_pre_fn -- for families that
// inject something into the (possibly multi-stream) residual ahead of the
// attention sub-block. Added for Qwen4Exp's PLE n-gram hash embedding,
// which modifies the HC stream buffer at PLE layers (qwen4exp.cpp's
// build_ple call sits between the HC-stream carry-over and the attention
// hc_mix). token_ids/pos_base are the same values layers_forward_gguf
// already receives. nullptr (the default) = no hook, zero cost.
using LayerInputFn = void (*)(StreamingEngine &engine, int layer_idx, int S,
                              int pos_base, const int *token_ids);

// ---------------------------------------------------------------------------
// Whole-model graph dispatch
// ---------------------------------------------------------------------------
//
// The layer loop and the whole-model GGML graph are two execution strategies
// for the same model family.  Keep the lifecycle common even though the graph
// topology remains family-specific.  This is the MoEStream equivalent of
// llama.cpp's model.build_graph()/set_inputs()/run_graph() boundary: the
// caller owns the request, while the selected family owns graph construction,
// state binding, and its scheduler-backed execution.
enum class ModelGraphPhase : uint8_t {
  Prefill,
  Decode,
};

struct ModelGraphRequest {
  ModelGraphPhase phase = ModelGraphPhase::Prefill;
  int n_tokens = 0;
  int pos_base = 0;
  const int *token_ids = nullptr;
  float *hidden_states = nullptr;
  float *logits = nullptr;
  int vocab = 0;
};

using ModelGraphEnabledFn = bool (*)(const StreamingEngine &engine,
                                     ModelGraphPhase phase);
using ModelGraphBootstrapFn = bool (*)(StreamingEngine &engine);
using ModelGraphBuildFn = bool (*)(StreamingEngine &engine,
                                   const ModelGraphRequest &request);
using ModelGraphRunFn = bool (*)(StreamingEngine &engine,
                                 const ModelGraphRequest &request);
using ModelGraphDeviceExpertLayersFn = int (*)();
using ModelGraphShutdownFn = void (*)();

struct ModelGraphOps {
  ModelGraphEnabledFn enabled = nullptr;
  ModelGraphBootstrapFn bootstrap = nullptr;
  ModelGraphBuildFn build = nullptr;
  ModelGraphRunFn run = nullptr;
  ModelGraphDeviceExpertLayersFn device_expert_layers = nullptr;
  ModelGraphShutdownFn shutdown = nullptr;
};

// Resolved once with the rest of FamilyOps.  An empty table means that the
// family uses the established layer-by-layer path.
ModelGraphOps model_graph_ops_for(ModelFamily fam);

struct FamilyOps {
  ModelFamily family = ModelFamily::Unknown;
  AttentionType attn_type = AttentionType::DenseQKV;
  CacheShape cache_shape = CacheShape::DenseKV;
  GatingFunc default_gating = GatingFunc::Softmax;
  bool uses_hyper_connections = false;
  int hc_mult = 1; // 1 = hyper-connections inactive

  AttentionFn attention_fn = nullptr;    // required, non-null once resolved
  AllocCacheFn alloc_cache_fn = nullptr; // required, non-null once resolved
  HcPreFn hc_pre_fn = &hc_pre_noop;
  HcPostFn hc_post_fn = &hc_post_plain_add;
  LayerInputFn layer_input_fn = nullptr; // optional; nullptr = no hook
  ModelGraphOps graph_ops;               // optional whole-model graph path
};

// The single place ModelFamily -> "what does this family actually do" gets
// resolved. hc_mult is a GGUF hyperparameter (hyper_connection.count), not
// derivable from ModelFamily alone, so the already-populated
// ArchitectureConfig is passed in -- call this AFTER the rest of c has been
// read from the GGUF file (graph_initializer.cpp), not before.
FamilyOps resolve_family_ops(ModelFamily fam, const ArchitectureConfig &c);
