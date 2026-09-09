#pragma once
// ---------------------------------------------------------------------------
// Qwen4Exp GDN's four shared-input dense_proj calls, fused into ONE ggml
// graph (EXPERIMENTAL). Sibling of ggml_hc_backend.hpp/ggml_moe_backend.hpp
// -- same shared-backend, same per-layer weight-bracket pattern.
//
// WHY: attn_qkv, attn_gate, ssm_alpha and ssm_beta (src/attention_layer.cpp's
// qwen4exp_gdn_gguf) all read the EXACT SAME `hidden_states` input, but each
// goes through dense_proj as an independent call -- its own H2D upload of
// that SAME activation, its own GPU dispatch, its own D2H download. At a
// 512-token prefill this project's own [PREFILL] report shows `dense_gpu`'s
// up+dn (H2D+D2H) running several times larger than cmp (real compute) for
// exactly this population of calls -- the fixed per-call transfer cost, not
// the matmul, dominates. Fusing the four into one graph turns 4 redundant
// uploads of the same ~10-40 KB/token activation into 1, and moves the
// decay-log (`softplus(alpha+dt_bias)*ssm_a`) and beta-sigmoid postprocessing
// -- currently two more serial CPU loops -- onto the GPU as graph ops, so
// nothing downstream of the matmuls needs a second host round trip either.
//
// SCOPE: originally deliberately narrow (just the 4 projections), extended
// 2026-09-05 to OPTIONALLY chain the causal conv1d+SiLU directly onto the
// qkv tensor this graph already produces, via run_with_conv() -- see that
// method's own comment for why. The per-head L2 norm, the chunkwise
// delta-rule recurrence, and the gated output norm are still untouched --
// those already have their own real, previously-verified fast paths (an
// OMP-parallel L2 norm, a dedicated GPU chunkwise kernel for the recurrence,
// an OMP-parallel output norm) and this is not trying to replace them.
//
// This header is ALWAYS compilable. Without MOESTREAM_USE_GGML_CUDA every
// entry point returns false and the caller falls back to its own dispatch.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>

// One projection weight, described by the caller so this header needs
// nothing from engine.hpp. `type` is a GGUFType numeric (ggml's own
// numbering, per include/gguf_loader.hpp's enum).
struct GgmlGdnProjWeightRef {
  const void *host_data = nullptr;
  // Optional device copy owned by the engine's dense pool.  The runner uses
  // it directly and only allocates a private copy when this is absent.
  const void *device_data = nullptr;
  size_t device_bytes = 0;
  uint32_t type = 0;
  int64_t ne0 = 0;  // input dim (D)
  int64_t ne1 = 0;  // output dim
  size_t nbytes = 0;
};

class GgmlGdnProjRunner {
 public:
  static bool compiled_in();
  static GgmlGdnProjRunner &instance();

  // Shape/config for the whole model. Idempotent; false if unusable.
  bool ensure(int device, int D, int conv_dim, int value_dim, int H_v);

  // Upload one layer's four weights plus the small (H_v-element) dt_bias/
  // ssm_a decay-gate constants, kept resident for the process (36 layers'
  // worth is a small fraction of what HC/MoE already keep resident).
  // dt_bias_host/ssm_a_host are F32 host arrays of H_v elements each.
  bool ensure_weights(int layer, const GgmlGdnProjWeightRef &qkv,
                      const GgmlGdnProjWeightRef &gate,
                      const GgmlGdnProjWeightRef &alpha,
                      const GgmlGdnProjWeightRef &beta,
                      const float *dt_bias_host, const float *ssm_a_host);
  bool weights_ready(int layer) const;

  // Add the two weights needed by the full device GDN path. The four
  // projection weights and decay vectors are shared with ensure_weights(),
  // so this allocates only ssm_norm and ssm_out for the layer.
  bool ensure_full_weights(int layer, const GgmlGdnProjWeightRef &norm,
                           const GgmlGdnProjWeightRef &out);
  bool full_weights_ready(int layer) const;

  // Computes qkv[S,conv_dim], gate[S,value_dim], g_log[S,H_v] (already
  // decay-log-processed: softplus(alpha+dt_bias)*ssm_a) and beta[S,H_v]
  // (already sigmoid'd) from ONE shared upload of hidden_states[S,D].
  bool run(int layer, const float *hidden_states, int S, float *qkv_out,
          float *gate_out, float *g_log_out, float *beta_out);

  // Upload one layer's conv1d weight (EngineConfig::qwen4exp_gdn_conv_gpu),
  // shape [d_conv, conv_dim] -- the on-disk GGUF layout already matches this
  // ne order exactly (see run_with_conv's own comment), so host_data is
  // uploaded byte-for-byte, no transform. Separate from ensure_weights/
  // weights_ready since conv chaining is an independent opt-in on top of
  // the base 4-projection fusion.
  bool ensure_conv_weight(int layer, const GgmlGdnProjWeightRef &conv_w);
  bool conv_weight_ready(int layer) const;

  // Like run(), but ALSO chains GDN's causal conv1d + SiLU directly onto the
  // qkv tensor this graph already computes on-device -- qkv_out becomes
  // conv_out (post-conv+SiLU), never leaving the GPU in between. Removes the
  // wasteful "download qkv, then re-upload the identical bytes for
  // qwen4exp_conv1d_silu_gpu_host" round trip a real BeeLlama profile showed
  // costing ~65-70x more than necessary at this exact op (see CLAUDE.md's
  // 2026-09-05 "conv1d round-trip elimination" entry).
  //
  // history is a [conv_dim, d_conv-1] host array, read AND updated in place
  // (same in/out contract as attn_causal_conv1d_silu / qwen4exp_conv1d_silu_
  // gpu_host) -- the update itself is a small (channels*(d_conv-1) floats)
  // device->host download of the graph's own concatenated conv-input tail,
  // not a full-size round trip.
  //
  // Requires ensure_conv_weight(layer, ...) to have already succeeded for
  // this layer; returns false otherwise (caller falls back to run() + a
  // separate conv step).
  bool run_with_conv(int layer, const float *hidden_states, int S,
                     float *conv_out, float *gate_out, float *g_log_out,
                     float *beta_out, float *history, int d_conv);

  // Complete Qwen4Exp GDN attention on the CUDA backend. The graph includes
  // projection, short convolution, L2-normalized Q/K, gated delta net,
  // gated RMSNorm and ssm_out. history and state are the same host-side
  // persistent buffers used by the existing path and are updated in place.
  bool run_full(int layer, const float *hidden_states, int S,
                float *out_context, float *history, int d_conv,
                float *state, float eps);

  // Device handoff form of run_full. device_hidden and device_output are raw
  // F32 [D,S] buffers on this runner's CUDA device. The output is retained in
  // a small persistent buffer until the matching HC post-combine consumes it;
  // history and recurrent state keep their existing host-side contracts.
  bool run_full_device(int layer, const void *device_hidden, int S,
                       const void **device_output, float *history, int d_conv,
                       float *state, float eps);

  // Shared-backend tensor form of run_full_device(). The input is an F32
  // [D,S] GGML tensor and is copied into the graph input asynchronously on
  // the same GGML CUDA stream used by HC.
  bool run_full_device_tensor(int layer, const void *device_hidden_tensor,
                              int S, const void **device_output,
                              float *history, int d_conv, float *state,
                              float eps);

  // Download recurrent state and convolution history produced by all full
  // device GDN layers. This is called once at the end of a forward pass.
  void finish_full_device_updates();
  bool copy_full_output_to_host(const void *device_output, float *host, int S);

  void shutdown();
  ~GgmlGdnProjRunner();

 private:
  GgmlGdnProjRunner() = default;
  GgmlGdnProjRunner(const GgmlGdnProjRunner &) = delete;
  GgmlGdnProjRunner &operator=(const GgmlGdnProjRunner &) = delete;

  struct Bracket {
    void *ctx = nullptr;    // ggml_context*
    void *buf = nullptr;    // ggml_backend_buffer_t
    void *w_qkv = nullptr;  // ggml_tensor*
    void *w_gate = nullptr;
    void *w_alpha = nullptr;
    void *w_beta = nullptr;
    void *dt_bias = nullptr;
    void *ssm_a = nullptr;
    bool ready = false;
    // Separate allocation/readiness from the block above: uploaded only
    // when qwen4exp_gdn_conv_gpu is also on, via its own small ctx/buf pair
    // (kept apart from ready's own ctx/buf so ensure_conv_weight can be
    // added or retried independently of the base 4-projection weights).
    void *conv_ctx = nullptr;
    void *conv_buf = nullptr;
    void *conv_w = nullptr;
    bool conv_ready = false;
    void *full_ctx = nullptr;
    void *full_buf = nullptr;
    void *ssm_norm = nullptr;
    void *ssm_out = nullptr;
    bool full_ready = false;
  };

  static constexpr int kMaxLayers = 256;

  struct FullState {
    void *ctx = nullptr;       // ggml_context*
    void *buf = nullptr;       // ggml_backend_buffer_t
    void *history = nullptr;   // ggml_tensor*
    void *state = nullptr;     // ggml_tensor*
    // Full-GDN graphs are submitted asynchronously.  These resources are
    // per layer so the next layer cannot overwrite an output or reset the
    // temporary allocator while HC is still consuming the previous one.
    void *output_ctx = nullptr; // ggml_context*
    void *output_buf = nullptr; // ggml_backend_buffer_t
    void *output = nullptr;     // ggml_tensor* [D, output_cap]
    int output_cap = 0;
    size_t workspace_bytes = 0;
    float *host_history = nullptr;
    float *host_state = nullptr;
    size_t history_bytes = 0;
    size_t state_bytes = 0;
    bool pending = false;
  };

  struct FullDeviceSlot {
    void *galloc = nullptr;       // ggml_gallocr_t
    void *done_event = nullptr;   // ggml_backend_event_t
    bool in_flight = false;
  };

  void *backend_ = nullptr;  // shared, not owned (ggml_shared_backend)
  void *external_buf_ = nullptr; // tiny non-owning-pointer GGML source buffer
  void *galloc_ = nullptr;   // ggml_gallocr_t, private to this runner
  // GDN's fused delta-rule op can use auxiliary CUDA streams. A single
  // allocator reset across queued layer graphs can therefore reuse temporary
  // addresses too early even though the main backend stream is ordered. Eight
  // slots keep the pipeline asynchronous while bounding the extra workspace.
  // Two slots keep the extra allocation near 190 MiB on the 24 GiB card;
  // larger rings compete with the per-layer GDN/HC state and expert cache.
  static constexpr int kFullDeviceSlots = 2;
  FullDeviceSlot full_device_slots_[kFullDeviceSlots];
  size_t full_device_workspace_bytes_ = 0;
  int device_ = -1;
  int D_ = 0, conv_dim_ = 0, value_dim_ = 0, H_v_ = 0;
  size_t graph_workspace_bytes_ = 0;
  // Host-output full-GDN fallback keeps its original shared persistent output
  // because that path is synchronous.  Device handoff uses FullState::output
  // above, one allocation per layer.
  void *full_output_ctx_ = nullptr;
  void *full_output_buf_ = nullptr;
  void *t_full_output_ = nullptr;  // [D, full_output_cap_], F32
  int full_output_cap_ = 0;

  Bracket brackets_[kMaxLayers];
  FullState full_states_[kMaxLayers];

  bool ensure_full_output(int layer, int S);
  bool ensure_full_state(int layer, int d_conv, int hd);
  // Called only after a device weight allocation fails.  Completed layers'
  // projection/conv/full weight buffers are no longer needed by the current
  // prefill after the backend barrier.  Recurrent state is deliberately kept
  // because decode consumes it after prefill.
  void release_completed_weights(int before_layer);
  bool run_full_device_impl(int layer, const void *device_hidden,
                            const void *device_hidden_tensor, int S,
                            const void **device_output, float *history,
                            int d_conv, float *state, float eps);
};
