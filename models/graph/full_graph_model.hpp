// full_graph_model.hpp -- the whole-model-graph interface of a model family.
//
// One FullGraphModel per family replaces three things that used to be kept in
// sync by hand: the per-family adapter functions in family_ops.cpp, the
// ModelGraphOps function-pointer table and the FamilySpec policy object.
//
//   - Lifecycle: enabled / bootstrap / build / run / shutdown.  A family's
//     graph implementation (ggml_<family>_graph*.cpp) is reached only through
//     these.
//   - Policy: the router description and whether cold experts may run on a
//     CPU lane.  Graph builders ask the model instead of testing ModelFamily.
//
// To add a family: derive from FullGraphModelBase<Derived>, implement
// prefill_prepare / prefill_run (and decode_* if it has a decode graph),
// override the policy that differs, and add it to full_graph_model() in
// full_graph_models.cpp.
#pragma once

#include "family_ops.hpp"
#include "model_family.hpp"
#include "moe_routed_config.hpp"
#include "moe_execution_strategy.hpp"

struct StreamingEngine;
class RAMSafetyManager;
class LockFreePrefetcher;
struct LayerScratch;

class FullGraphModel {
 public:
  virtual ~FullGraphModel() = default;

  virtual ModelFamily family() const = 0;
  virtual const char *name() const = 0;

  // ---- lifecycle ---------------------------------------------------------
  // Whether this phase may use the graph at all (environment gate, MTP, ...).
  virtual bool enabled(const StreamingEngine &engine,
                       ModelGraphPhase phase) const = 0;
  // Optional work done once at engine start (expert source preload, ...).
  // Returns false only when startup must fail.
  virtual bool bootstrap(StreamingEngine &) const { return true; }
  // Builds or reuses the graph for `request`; false = caller falls back.
  virtual bool build(StreamingEngine &engine,
                     const ModelGraphRequest &request) const = 0;
  // Executes the built graph and writes the request's outputs.
  virtual bool run(StreamingEngine &engine,
                   const ModelGraphRequest &request) const = 0;
  // Releases scheduler buffers before the shared CUDA backend goes away.
  virtual void shutdown() const {}
  // Native-cache graph sites borrow the engine runtime rather than owning a
  // second expert store. New families bind it through this shared lifecycle hook.
  virtual bool uses_native_moe_cache() const { return false; }
  virtual bool graph_only() const { return false; }
  virtual void bind_runtime(StreamingEngine &, RAMSafetyManager &,
                             LockFreePrefetcher &, LayerScratch &) const {}

  // Largest prompt batch the prefill graph should be built for (0 = no limit). Its compute
  // buffers grow with the batch, so a longer prompt runs this many tokens through the graph and
  // the rest as S=1 steps (see generate_text).
  virtual int max_prefill_tokens(const StreamingEngine &) const { return 0; }
  // Smallest prefill batch worth running through the graph (a smaller one is stepped / not planned).
  virtual int min_prefill_batch() const { return 64; }
  // After a prefill batch was declined for memory: the batch size the family worked out would fit (0 = no suggestion).
  virtual int suggested_prefill_batch(const StreamingEngine &) const { return 0; }

  // Whether a prefill chunk that starts after position 0 can run through the graph, continuing the state the earlier
  // chunk left in the engine (build() is then asked with pos_base > 0 and may still decline a particular chunk).
  virtual bool supports_prefill_continuation(const StreamingEngine &) const { return false; }

  // Number of MoE layers whose routed experts the graph places in VRAM.
  virtual int device_expert_layers() const { return 0; }

  // ---- policy ---------------------------------------------------------------
  virtual moe_routed::RouterSpec router_spec(const ArchitectureConfig &c,
                                             int n_tokens) const {
    return moe_routed::router_spec(c, n_tokens, /*honor_gating_func=*/true);
  }
  // Whether a layer's cold experts may run on a private CPU lane concurrently
  // with the layer's GPU work.
  virtual bool supports_cpu_lane() const { return false; }
  virtual moestream::graph::MoEExecutionCapabilities
  moe_execution_capabilities() const {
    moestream::graph::MoEExecutionCapabilities caps;
    caps.early_ready = supports_cpu_lane();
    caps.gpu_assist = caps.early_ready;
    return caps;
  }
};

// CRTP base: owns the request validation and the prefill/decode dispatch that
// every family used to repeat.  Derived provides
//
//   bool prefill_prepare(StreamingEngine &, const ModelGraphRequest &) const;
//   bool prefill_run    (StreamingEngine &, const ModelGraphRequest &) const;
//   bool decode_prepare (StreamingEngine &, const ModelGraphRequest &) const;
//   bool decode_run     (StreamingEngine &, const ModelGraphRequest &) const;
//
// and may hide `accepts()` to refuse request shapes up front.  Dispatch is
// resolved at compile time; only the registry boundary is virtual.
template <class Derived>
class FullGraphModelBase : public FullGraphModel {
 public:
  bool build(StreamingEngine &engine,
             const ModelGraphRequest &request) const final {
    if (!request.token_ids || request.n_tokens <= 0 ||
        !self().accepts(engine, request))
      return false;
    return request.phase == ModelGraphPhase::Decode
               ? self().decode_prepare(engine, request)
               : self().prefill_prepare(engine, request);
  }

  bool run(StreamingEngine &engine,
           const ModelGraphRequest &request) const final {
    if (!request.hidden_states || !request.token_ids || request.n_tokens <= 0 ||
        !self().accepts(engine, request))
      return false;
    return request.phase == ModelGraphPhase::Decode
               ? self().decode_run(engine, request)
               : self().prefill_run(engine, request);
  }

  // Default: every shape is acceptable.  Derived hides this to narrow it.
  bool accepts(const StreamingEngine &, const ModelGraphRequest &) const {
    return true;
  }

 private:
  const Derived &self() const { return static_cast<const Derived &>(*this); }
};

// The model for `family`, or nullptr when the family has no whole-model graph
// (it then uses the layer-by-layer path).
const FullGraphModel *full_graph_model(ModelFamily family);

// Router description for `family`; families without a graph model use the
// default softmax/gating-function router.
inline moe_routed::RouterSpec family_router_spec(ModelFamily family,
                                                 const ArchitectureConfig &c,
                                                 int n_tokens) {
  if (const FullGraphModel *m = full_graph_model(family))
    return m->router_spec(c, n_tokens);
  return moe_routed::router_spec(c, n_tokens, /*honor_gating_func=*/true);
}

inline bool family_supports_cpu_lane(ModelFamily family) {
  const FullGraphModel *m = full_graph_model(family);
  return m && m->supports_cpu_lane();
}
