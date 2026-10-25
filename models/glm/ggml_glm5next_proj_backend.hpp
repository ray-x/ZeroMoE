#pragma once
// ---------------------------------------------------------------------------
// GLM5NEXT KDA projection fusion (opt-in, GLM5NEXT only).
//
// The KDA block has eight dense projections in its prefill path. Six of them
// read the same normalized token matrix and the other two are the second
// projections of the f_a/g_a branches. Keeping the complete projection DAG
// in one GGML CUDA graph removes seven host dispatch/synchronization points
// and one repeated activation upload per layer. The recurrent KDA update and
// its state remain in the existing CUDA path; this class only owns the dense
// projection subgraph.
//
// This header deliberately has no ggml dependency. Without
// ZEROMOE_USE_GGML_CUDA every method returns false, so callers retain the
// existing dense_proj implementation unchanged.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>

struct GgmlGlm5nextProjWeightRef {
  const void *host_data = nullptr;
  uint32_t type = 0;
  int64_t ne0 = 0;  // input dimension
  int64_t ne1 = 0;  // output dimension
  size_t nbytes = 0;
};

class GgmlGlm5nextKdaProjRunner {
 public:
  static bool compiled_in();
  static GgmlGlm5nextKdaProjRunner &instance();

  // Configure the shared CUDA backend and the fixed prefill shape. S is the
  // current token count; a later call with a different S rebuilds only the
  // graph/workspace, not the backend.
  bool ensure(int device, int D, int d_inner, int hd, int H, int S);

  // Upload one layer's KDA projection weights. Only one layer's set is kept
  // in this runner because the existing dense pool already owns the model's
  // persistent copies and this path is a dispatch-fusion optimization, not a
  // second whole-model weight tier.
  bool ensure_layer(int layer, const GgmlGlm5nextProjWeightRef &q,
                   const GgmlGlm5nextProjWeightRef &k,
                   const GgmlGlm5nextProjWeightRef &v,
                   const GgmlGlm5nextProjWeightRef &f_a,
                   const GgmlGlm5nextProjWeightRef &f_b,
                   const GgmlGlm5nextProjWeightRef &beta,
                   const GgmlGlm5nextProjWeightRef &g_a,
                   const GgmlGlm5nextProjWeightRef &g_b);

  // Run the complete projection DAG. Outputs use the same flat [S, O]
  // layout as dense_proj: q/k/v/f_b/g_b are [S,d_inner], beta is [S,H].
  // f_b_out is still the raw projection; the caller applies GLM's exact
  // dt-bias/decay transform, and beta's sigmoid, exactly as before.
  bool run(int layer, const float *hidden_states, int S, float *q_out,
           float *k_out, float *v_out, float *f_b_out, float *beta_out,
           float *g_b_out);

  void shutdown();
  ~GgmlGlm5nextKdaProjRunner();

 private:
  GgmlGlm5nextKdaProjRunner() = default;
  GgmlGlm5nextKdaProjRunner(const GgmlGlm5nextKdaProjRunner &) = delete;
  GgmlGlm5nextKdaProjRunner &operator=(const GgmlGlm5nextKdaProjRunner &) = delete;

  static constexpr int kWeightCount = 8;

  void destroy_graph();
  bool build_graph();
  bool weight_signature_matches(const GgmlGlm5nextProjWeightRef refs[kWeightCount]) const;
  bool allocate_weights(const GgmlGlm5nextProjWeightRef refs[kWeightCount]);

  void *backend_ = nullptr;       // shared, not owned
  void *galloc_ = nullptr;        // ggml_gallocr_t
  void *weight_ctx_ = nullptr;    // ggml_context*
  void *weight_buf_ = nullptr;    // ggml_backend_buffer_t
  void *weights_[kWeightCount]{}; // ggml_tensor*

  void *graph_ctx_ = nullptr;     // ggml_context*
  void *graph_ = nullptr;         // ggml_cgraph*
  void *x_ = nullptr;             // ggml_tensor*
  void *outputs_[6]{};            // q, k, v, f_b, beta, g_b

  uint32_t weight_types_[kWeightCount]{};
  int64_t weight_ne0_[kWeightCount]{};
  int64_t weight_ne1_[kWeightCount]{};
  int device_ = -1;
  int D_ = 0;
  int d_inner_ = 0;
  int hd_ = 0;
  int H_ = 0;
  int S_ = 0;
  int current_layer_ = -1;
  bool weights_ready_ = false;
  bool graph_ready_ = false;
};

