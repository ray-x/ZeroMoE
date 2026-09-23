#pragma once
// ---------------------------------------------------------------------------
// Qwen4Exp hyper-connections on the GPU, as ggml graphs (EXPERIMENTAL).
//
// WHY: measured 2026-09-05 on a 441-token qwen4exp prefill, HC is 61% of the
// whole pass, in two separate ways that the report line keeps apart:
//   * 3.29 s of CPU compute (the `hyper-connections` per-layer-hook counter),
//     scalar/OMP work over ~4.5M elements per call, 96 calls per prefill; and
//   * ~2.3 s of the 2.67 s `dense_gpu` bucket, because its three dense_proj
//     calls move `xn`/`gate` -- [S, hc_dim] = 441 x 10240 x 4 = 18.1 MB each --
//     across PCIe six times per layer, 5.3 GB per prefill, at ~2.4 GB/s.
// Both disappear together only if the intermediates never leave the device,
// which is what this runner is for: `hc_stream_buf`, `xn`, `lo`, `gate` and
// `inject` all stay in VRAM, and the only per-call traffic is the [S, D]
// hidden state (4.5 MB) that the host genuinely needs between sub-blocks.
//
// MATH (must stay byte-faithful to src/hyper_connections.cpp's
// qwen4exp_hc_mix_impl / qwen4exp_hc_post_combine -- that file is the spec):
//   pre_mix, per token s:
//     xn     = rms_norm(streams[D, HC, S])        // over ne0 = D, no gain
//     xn     = xn * norm_w[hc_dim]                // gain, flattened layout
//     lo     = silu(down(xn) / HC)                // [hc_lr, S]
//     gate   = sigmoid(up(lo))                    // [hc_dim, S]
//     hidden = mean over the HC streams of (xn * gate)   // [D, S]
//     inject = inject_w(xn)                       // [HC, S], kept on device
//   post_combine:
//     streams[.,h,s] += sub_block_out[.,s] * 2*sigmoid(inject[h,s] / HC)
//
// Engine layout of hc_stream_buf is (s*HC + h)*D + d, i.e. stream-major per
// token, which is exactly ggml's [ne0=D, ne1=HC, ne2=S] -- so the buffer maps
// over with no shuffle, and ggml_rms_norm's own "normalise over ne0" is the
// per-(stream, token) norm the CPU code writes by hand.
//
// Always compilable: without MOESTREAM_USE_GGML_CUDA every entry point returns
// false and the caller runs the existing CPU path unchanged.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <vector>

// The HC/GDN/MoE runners share one GGML CUDA backend stream. When enabled,
// graph temporaries are reused by stream order instead of waiting on a host
// event before every bracket. The default remains conservative; callers must
// opt in after validating the complete device-resident pipeline.
void set_ggml_hc_async_graph_reuse(bool enabled);

// Qwen-only opt-in for the pinned host staging/readback path used when a
// device-HC result must feed a CPU-routed MoE layer. The default is false;
// callers must enable it only after comparing the complete decode path.
void set_qwen_hc_async_readback(bool enabled);

// One HC weight, described by the caller so this header needs nothing from
// engine.hpp. `type` is a GGUFType numeric, which is ggml's own numbering.
struct GgmlHcWeightRef {
  const void *host_data = nullptr;
  // Optional device copy owned by the engine's dense pool.  When present the
  // runner wraps this pointer as a GGML source tensor instead of uploading a
  // second per-layer copy into its own buffer.
  const void *device_data = nullptr;
  size_t device_bytes = 0;
  uint32_t type = 0;
  int64_t ne0 = 0;  // input dim  (ggml src0 ne0 -- the contracted axis)
  int64_t ne1 = 0;  // output dim
  size_t nbytes = 0;
  // Qwen4Exp device-HC mode opts into an exact F32-activation matmul for
  // Q8_0 down/up projections.  Other GGML HC users retain the normal GGML
  // quantized graph path.
  bool exact_f32_q8 = false;
  // Qwen-only fused implementation: keep Q8_0 weights compressed and use an
  // exact F32-activation CUDA dot product.  This is separate from the F32
  // expansion reference path so the latter remains available for validation.
  bool custom_q8 = false;
};

class GgmlHcRunner {
 public:
  static bool compiled_in();
  static GgmlHcRunner &instance();

  // Shape/config for the whole model. Idempotent; false if unusable.
  bool ensure(int device, int d_model, int hc_mult, int hc_low_rank,
              float eps);

  // Upload one (layer, sub_block) bracket's four weights. Done once each and
  // kept resident for the process -- all 96 brackets are ~0.7 GB total, and
  // re-uploading per call would reintroduce exactly the PCIe traffic this
  // exists to remove. Returns false if the tensors are unusable (missing,
  // wrong shape, unsupported quant type).
  bool ensure_weights(int layer, int sub_block, const GgmlHcWeightRef &norm,
                      const GgmlHcWeightRef &down, const GgmlHcWeightRef &up,
                      const GgmlHcWeightRef &inject);

  bool weights_ready(int layer, int sub_block) const;

  // hc_stream_buf lives on the device between these two. Host and device are
  // synchronised ONLY at the model-level expand/collapse brackets, not per
  // layer -- at [D, HC, S] = 18 MB a per-call round trip would cost more than
  // the CPU path it replaces.
  bool upload_streams(const float *host, int S);
  bool download_streams(float *host, int S);
  bool download_inject(float *host, int S);

  // out_hidden receives the mean-collapsed [S, D]; inject is stashed on the
  // device for the matching post_combine.
  // If device_output is non-null, it receives a stable raw device pointer to
  // the mixed [D,S] result.  The host result is optional in that form; this
  // is used by the Qwen4Exp GDN path to keep the HC -> GDN boundary on CUDA.
  bool pre_mix(int layer, int sub_block, int S, float *out_hidden,
               const void **device_output = nullptr);
  bool post_combine(int layer, int sub_block, int S,
                    const float *sub_block_out);

  // Device-resident variant for the Qwen4Exp GGML path. The source is the
  // [D,S] F32 output of GgmlMoeRunner on the same shared backend; it is
  // copied device-to-device into the post graph's [D,1,S] input. The normal
  // host variant above remains the fallback and is used by every other
  // model family.
  bool post_combine_device(int layer, int sub_block, int S,
                           const void *device_sub_block_out);

  // Qwen-only fused post-combine for the CPU-routed/shared-GPU path. The
  // producer fence is an opaque ggml_backend_event_t owned by
  // GgmlQwenSharedRunner; the implementation bridges it to CUDA stream 0,
  // launches one fused HC update kernel, and publishes a fence back to the
  // shared GGML stream.
  bool post_combine_device_fused(int layer, int sub_block, int S,
                                 const void *device_sub_block_out,
                                 const void *producer_event);

  // Same operation as post_combine_device, for a raw F32 [D,S] device
  // buffer whose producer is another runner and therefore has no ggml tensor
  // metadata to validate.
  bool post_combine_device_raw(int layer, int sub_block, int S,
                               const void *device_data, size_t bytes);

  // Copy the persistent pre_mix result back only when a device GDN attempt
  // has to fall back to the older host attention path.
  bool copy_pre_mix_to_host(const void *device_data, float *host, int S);

  // Opaque GGML tensor for the persistent [D, HC, S] HC state. This is used
  // only by Qwen's device-side router pilot; callers must not retain it past
  // the current forward pass or treat it as a host pointer.
  const void *device_streams_tensor() const { return t_streams_; }

  // Opaque GGML tensor for the persistent [D,S] mixed activation produced by
  // pre_mix().  Keep the producer handle authoritative at this boundary;
  // callers must not substitute its raw CUDA data pointer because downstream
  // GGML graph builders need the tensor shape/stride metadata as well.
  const void *device_mixed_tensor() const { return t_mixed_; }

  // Queue the completion dependency for the most recent pre_mix on the
  // shared GGML stream and on CUDA's legacy stream. This is device-side
  // ordering only; it does not block the host thread.
  void queue_pre_mix_dependency();

  // Return the raw F32 CUDA pointer behind a device_output returned by
  // pre_mix(). The handle itself is intentionally opaque to callers; this
  // bridge is used by Qwen4Exp's dense-attention CUDA projections.
  const void *device_data(const void *device_output) const;

  void shutdown();
  ~GgmlHcRunner();

 private:
  GgmlHcRunner() = default;
  GgmlHcRunner(const GgmlHcRunner &) = delete;
  GgmlHcRunner &operator=(const GgmlHcRunner &) = delete;

  struct Bracket {
    void *ctx = nullptr;   // ggml_context*
    void *buf = nullptr;   // ggml_backend_buffer_t
    void *norm = nullptr;  // ggml_tensor*
    void *down = nullptr;
    void *up = nullptr;
    void *inject = nullptr;
    // GGML's CUDA Q8_0 matmul quantizes the F32 activation to Q8_1 before
    // the dot product.  That approximation is fine for ordinary model
    // projections, but HC's low-rank gate is also used as the next-layer
    // router input: the resulting small error changes expert IDs.  Keep an
    // exact F32 dequantization for Q8_0 HC down/up weights, owned by the
    // bracket for the lifetime of the graph metadata.
    std::vector<float> down_f32;
    std::vector<float> up_f32;
    // Whether this bracket was built for exact Q8 decode mode.  Prefill keeps
    // the original quantized graph; decode can rebuild a bracket with exact
    // F32 projections without confusing the persistent graph's source types.
    bool exact_f32_q8 = false;
    bool custom_q8 = false;
    bool ready = false;
  };

  bool ensure_state(int S);
  bool ensure_custom_buffers(int S);
  bool ensure_readback_buffer(size_t bytes);
  bool ensure_pre_graph(int S, const Bracket &b, int layer, int sub_block);
  bool ensure_post_graph(int S);
  // Called only after a device weight allocation fails.  The completed
  // layers no longer need their HC weights during this forward pass; reclaim
  // them after one backend barrier and retry the allocation.  Keep the
  // bracket used as the persistent pre-graph's source metadata alive.
  void release_completed_weights(int before_layer);
  // Opportunistic version used before an exact decode bracket is allocated.
  // CUDA event polling reclaims completed weights without a host-side backend
  // synchronization on the normal device-HC path.
  void release_completed_weights_nonblocking(int before_layer);
  void free_pre_graph();
  bool post_combine_impl(int layer, int sub_block, int S,
                         const float *host_sub_block_out,
                         const void *device_sub_block_out,
                         size_t device_bytes = 0);

  void *backend_ = nullptr;  // shared, not owned
  void *external_buf_ = nullptr; // tiny non-owning-pointer GGML source buffer
  // One allocator PER GRAPH SHAPE, not one for the runner. ggml_gallocr
  // caches an allocation plan for the last graph it saw; pre_mix and
  // post_combine are different graphs, so alternating them through a single
  // allocator makes it re-plan and re-allocate its compute buffer on every
  // call -- 96 cudaMalloc/free pairs per prefill.
  void *galloc_pre_ = nullptr;   // ggml_gallocr_t
  void *galloc_post_ = nullptr;  // ggml_gallocr_t
  int device_ = -1;
  int d_model_ = 0, hc_ = 0, hc_lr_ = 0, hc_dim_ = 0;
  float eps_ = 1e-6f;

  // Persistent device state: the HC streams and the last pre_mix's inject.
  void *state_ctx_ = nullptr;
  void *state_buf_ = nullptr;
  void *t_streams_ = nullptr;  // [D, HC, S_cap]
  void *t_inject_ = nullptr;   // [HC, S_cap]
  void *t_mixed_ = nullptr;    // [D, S_cap], stable last pre_mix result

  // A CUDA graph captures the addresses of its tensor sources. Retargeting
  // one captured graph's weight tensor data pointers for every layer is not
  // sufficient: the CUDA graph executable can retain the old kernel
  // arguments. Keep one fixed graph per (sub-block, layer) for the active
  // batch shape instead. The temporary workspace is still shared through
  // galloc_pre_; graph launches are ordered on the same backend stream.
  struct PreGraphSlot {
    void *ctx = nullptr;    // ggml_context*
    void *graph = nullptr;  // ggml_cgraph*
    void *norm = nullptr;   // ggml_tensor* (fixed weight source)
    void *down = nullptr;
    void *up = nullptr;
    void *inject = nullptr;
    int S = 0;
    uint32_t down_type = 0;
    uint32_t up_type = 0;
    uint32_t inject_type = 0;
  };
  static constexpr int kPreGraphLayerCount = 256;
  PreGraphSlot pre_graphs_[2][kPreGraphLayerCount] = {};
  int pre_graph_cache_S_ = 0;
  PreGraphSlot *active_pre_graph_ = nullptr;
  // Weight storage can switch between prefill's quantized graph and decode's
  // exact-Q8 graph. A transition needs one backend drain before old tensor
  // buffers are replaced, not one drain per layer/sub-block bracket.
  bool weight_mode_valid_ = false;
  bool weight_mode_exact_ = false;
  bool weight_mode_custom_ = false;

  // Small persistent work buffers used only by the explicit custom Q8 decode
  // mode. They avoid a large F32 copy of every Q8 HC bracket.
  void *custom_xn_ = nullptr;
  void *custom_lo_ = nullptr;
  void *custom_gate_ = nullptr;
  size_t custom_capacity_ = 0;

  // Optional Qwen decode readback path.  HC produces a small [D,S] F32
  // result which the CPU-routed MoE must consume.  Keep one reusable pinned
  // staging buffer and a nonblocking copy stream so the host does not use
  // cudaMemcpy() on pageable memory for every CPU-affinity layer.
  void *readback_host_ = nullptr;
  size_t readback_capacity_ = 0;
  void *readback_stream_ = nullptr;     // cudaStream_t
  void *readback_done_event_ = nullptr; // cudaEvent_t

  // The post-combine graph is built once for the active request shape and
  // reused for every layer/sub-block.  Rebuilding it at the handoff point
  // allowed GGML's CUDA allocator to recycle temporary memory while the
  // preceding MoE graph still had auxiliary CUDA work in flight.
  void *post_ctx_ = nullptr;    // ggml_context*
  void *post_graph_ = nullptr;  // ggml_cgraph*
  void *post_input_ = nullptr;  // ggml_tensor* [D,S]
  void *post_source_view_ = nullptr; // ggml_tensor* [D,S], retargeted per producer
  int post_graph_S_ = 0;
  int s_cap_ = 0;
  int inject_valid_S_ = 0;
  size_t pre_graph_workspace_bytes_ = 0;
  size_t post_graph_workspace_bytes_ = 0;

  // The pre graph publishes to two consumers: the shared GGML stream and the
  // legacy CUDA stream used by Qwen's attention/GDN code.  A single event
  // re-recorded for every bracket is unsafe while the legacy stream still has
  // an earlier cudaStreamWaitEvent queued; use a stable event per layer and
  // bracket instead.  The events are reused only after the forward-level
  // HC download has drained the backend at the next request boundary.
  static constexpr int kPreDoneEventCount = 2 * 256;
  void *pre_done_events_[kPreDoneEventCount] = {}; // ggml_backend_event_t
  void *active_pre_done_event_ = nullptr;          // current bracket's event
  void *post_done_event_ = nullptr;  // ggml_backend_event_t
  // Producer fence for a raw CUDA attention output copied into the shared
  // GGML post graph. The event is recorded on CUDA stream 0 and waited on by
  // the GGML backend stream, so the D2D handoff need not block the host.
  void *raw_input_event_ = nullptr;  // ggml_backend_event_t
  bool pre_in_flight_ = false;
  bool post_in_flight_ = false;
  bool post_fused_in_flight_ = false;

  Bracket brackets_[2][256];  // [sub_block][layer]
};
