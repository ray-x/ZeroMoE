#pragma once

// The per-layer interfaces a family implements (rollout step 5, prototype).
//
//   AttentionBlock  attention/state of one layer: its host inputs (positions-derived
//                   masks, KV row tables) and the subgraph.
//   FfnBlock        the feed-forward of one layer: dense SwiGLU, the shared MoE block,
//                   or the legacy cache-backed callback. A model may mix them per layer.
//   LayerPlan       the residual structure around them. The usual pre-norm structure is
//                   PreNormResidualPlan; a family with another one (DeepSeek-V4.1
//                   hyper-connections keep several residual streams) provides its own
//                   plan instead of bending the blocks.
//
// Everything graph-wide (layer loop, MoE block, placement, offload tier, early-ready lane,
// prefill chunking, preflight) stays in the shared skeleton. Blocks are built once per
// graph, so nothing here is on the per-token path.
//
// KV state is described by KvLayerSpec and allocated by allocate_kv_cache; recurrent state is StateSpec/allocate_state_set.

#include "graph_core.hpp"

#include <functional>
#include <string>
#include <vector>

namespace zeromoe::graph {

class AttentionBlock {
 public:
  virtual ~AttentionBlock() = default;
  // Creates the host inputs a graph of g.S tokens needs beyond the positions (masks, KV row tables).
  // false = the graph cannot be built.
  virtual bool declare_inputs() = 0;
  // Fills those inputs for tokens starting at `position`.
  virtual void set_inputs(int position) = 0;
  // x: [D, S] residual stream. Returns the attention output [D, S] (before the residual add).
  virtual ggml_tensor *build(ggml_tensor *x, int layer) = 0;
};

class FfnBlock {
 public:
  virtual ~FfnBlock() = default;
  // cur: [D, S] residual stream after attention. Returns the FFN output [D, S]; the block
  // applies its own pre-norm.
  virtual ggml_tensor *build(ggml_tensor *cur, int layer) = 0;
};

class LayerPlan {
 public:
  virtual ~LayerPlan() = default;
  // One layer: returns the residual stream after it.
  virtual ggml_tensor *build_layer(ggml_tensor *cur, int layer) = 0;
};

// Blocks are built per graph and hold what they need (the family's graph object, its weights), so the
// interfaces carry only the layer and the tensors; nothing downcasts GraphCore.

// cur += attention(cur); cur += ffn(cur). `place` lets a family pin each result to a backend
// (identity when empty); `ffn_for_layer` picks the FFN block, so dense and MoE layers mix.
class PreNormResidualPlan final : public LayerPlan {
 public:
  using Place = std::function<ggml_tensor *(ggml_tensor *)>;
  using FfnFor = std::function<FfnBlock &(int layer)>;
  PreNormResidualPlan(GraphCore &g, AttentionBlock &attention, FfnFor ffn_for_layer, Place place = {})
      : g_(g), attention_(attention), ffn_for_layer_(std::move(ffn_for_layer)), place_(std::move(place)) {}

  ggml_tensor *build_layer(ggml_tensor *cur, int layer) override {
    cur = placed(ggml_add(g_.ctx, cur, attention_.build(cur, layer)));
    return placed(ggml_add(g_.ctx, cur, ffn_for_layer_(layer).build(cur, layer)));
  }

 private:
  ggml_tensor *placed(ggml_tensor *t) { return place_ ? place_(t) : t; }
  GraphCore &g_;
  AttentionBlock &attention_;
  FfnFor ffn_for_layer_;
  Place place_;
};

// The layer loop: input -> layers [0, n_layers) -> residual stream.
inline ggml_tensor *build_layer_stack(ggml_tensor *input, LayerPlan &plan, int n_layers) {
  ggml_tensor *cur = input;
  for (int l = 0; l < n_layers; ++l) cur = plan.build_layer(cur, l);
  return cur;
}

// ---- skeleton helpers every family repeated ---------------------------------------------------

// The I32 token-position input of a graph of g.S tokens.
inline ggml_tensor *declare_positions(GraphCore &g, const char *name) {
  g.positions = host_input(g, g.ctx, GGML_TYPE_I32, {g.S}, size_t(g.S) * 4, name);
  return g.positions;
}

// Final norm + LM head for the decode graph's last token; marks logits as an output.
inline ggml_tensor *build_output_head(GraphCore &g, ggml_tensor *cur,
                                      const std::function<ggml_tensor *(ggml_tensor *)> &final_norm,
                                      ggml_tensor *head_weight,
                                      const std::function<ggml_tensor *(ggml_tensor *)> &place = {}) {
  ggml_tensor *logits = ggml_mul_mat(g.ctx, head_weight, final_norm(cur));
  if (place) logits = place(logits);
  g.logits = logits;
  ggml_set_output(g.logits);
  ggml_build_forward_expand(g.graph, g.logits);
  return g.logits;
}

// Per-layer KV state described as data, so a family states its layout instead of allocating it.
struct KvLayerSpec { int heads = 0; int k_dim = 0; int v_dim = 0; };
struct KvCache {
  std::vector<ggml_tensor *> keys, values;  // [dim, capacity*heads] F16, zeroed
  ggml_backend_buffer_t buffer = nullptr;
};

// Allocates the cache on `backend`. The tensor metadata is duplicated into `meta_ctx` so it outlives the
// temporary context used for the allocation (which also keeps it from duplicating borrowed weights).
inline bool allocate_kv_cache(ggml_context *meta_ctx, size_t temp_ctx_bytes, ggml_backend_t backend,
                              const std::vector<KvLayerSpec> &layers, int capacity, KvCache *out) {
  ggml_context *tmp = ggml_init({temp_ctx_bytes, nullptr, true});
  if (!tmp) return false;
  for (const KvLayerSpec &l : layers) {
    out->keys.push_back(ggml_new_tensor_2d(tmp, GGML_TYPE_F16, l.k_dim, int64_t(capacity) * l.heads));
    out->values.push_back(ggml_new_tensor_2d(tmp, GGML_TYPE_F16, l.v_dim, int64_t(capacity) * l.heads));
  }
  out->buffer = ggml_backend_alloc_ctx_tensors(tmp, backend);
  if (!out->buffer) { ggml_free(tmp); return false; }
  ggml_backend_buffer_clear(out->buffer, 0);
  for (auto *vec : {&out->keys, &out->values})
    for (auto &t : *vec) {
      auto *copy = ggml_dup_tensor(meta_ctx, t);
      copy->data = t->data;
      copy->buffer = t->buffer;
      t = copy;
    }
  ggml_free(tmp);
  return true;
}

// Per-layer recurrent / auxiliary state (GDN or KDA conv history and matrix state, DSA indexer caches) described
// as data, like the KV cache. A family lists one spec per tensor; the helper allocates them zeroed in one buffer
// on `backend`, with metadata duplicated into `meta_ctx`. `next` is the in-graph update the decode graph copies
// back (state_out -> state), kept by the caller.
struct StateSpec {
  std::string name;
  ggml_type type = GGML_TYPE_F32;
  std::vector<int64_t> shape;  // ne[0..3]
  int layer = -1;
};
struct StateSet {
  std::vector<ggml_tensor *> tensors;  // same order as the specs
  ggml_backend_buffer_t buffer = nullptr;
};

inline bool allocate_state_set(ggml_context *meta_ctx, size_t temp_ctx_bytes, ggml_backend_t backend,
                               const std::vector<StateSpec> &specs, StateSet *out) {
  ggml_context *tmp = ggml_init({temp_ctx_bytes, nullptr, true});
  if (!tmp) return false;
  std::vector<ggml_tensor *> made;
  for (const StateSpec &sp : specs) {
    int64_t ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < sp.shape.size() && i < 4; ++i) ne[i] = sp.shape[i];
    ggml_tensor *t = ggml_new_tensor(tmp, sp.type, int(sp.shape.size() < 4 ? sp.shape.size() : 4), ne);
    if (!t) { ggml_free(tmp); return false; }
    ggml_set_name(t, sp.name.c_str());
    made.push_back(t);
  }
  out->buffer = ggml_backend_alloc_ctx_tensors(tmp, backend);
  if (!out->buffer) { ggml_free(tmp); return false; }
  ggml_backend_buffer_clear(out->buffer, 0);
  for (ggml_tensor *t : made) {
    ggml_tensor *copy = ggml_dup_tensor(meta_ctx, t);
    ggml_set_name(copy, t->name);
    copy->data = t->data;
    copy->buffer = t->buffer;
    out->tensors.push_back(copy);
  }
  ggml_free(tmp);
  return true;
}

} // namespace zeromoe::graph
