#pragma once
// ---------------------------------------------------------------------------
// ggml MUL_MAT_ID prefill backend for the routed-expert FFN (EXPERIMENTAL).
//
// WHY THIS EXISTS
// ---------------
// Phase 3b (src/execution_layer.cpp, moe_prefill_batched) already groups a
// prefill batch's (token, expert) pairs BY EXPERT, which removed the largest
// algorithmic problem -- the legacy per-token loop re-read every expert's
// weight slab once per token. What it did NOT remove is the per-expert HOST
// orchestration that still sits between buckets: a residency decision, an
// optional stage + H2D, an activation gather, a dispatch, a synchronisation
// and accumulation bookkeeping, once per expert per layer.
//
// llama.cpp/beellama pay none of that. Their whole routed FFN is three
// GGML_OP_MUL_MAT_ID nodes in one backend graph; ggml-cuda sorts token rows by
// expert ON THE GPU (get_rows into a sorted scratch buffer), derives expert
// bounds, and dispatches one grouped matmul per expert without ever returning
// to C++. This adapter builds that same graph against MoEStream's own streamed
// weights so the two execution models can be compared directly. When a routed
// expert already has a persistent page-locked RAM-tier slab, its three
// projection planes are copied directly on the copy stream; otherwise the
// existing file-backed prefill-stream fallback supplies the GPU graph. When
// Qwen's ram_preload_all_experts contract is active, that fallback is refused
// and the already-populated DDR tier is the only permitted expert source. The
// Qwen decode CPU-affinity path is separate: it binds MUL_MAT_ID directly to
// the complete, preloaded DDR arena and never uses a file-backed source.
//
// SHAPE CONTRACT (matches llama.cpp's build_moe_ffn exactly)
//   weights   gate/up : [d_model, d_ff,    n_experts]   (quantised, 3-D)
//             down    : [d_ff,    d_model, n_experts]
//   x                 : [d_model, 1, n_tokens]          F32
//   ids               : [topk, n_tokens]                I32, values are
//                       indices into the COMPACT expert set (0..n_experts-1),
//                       not global expert ids
//   router weights    : [1, topk, n_tokens]             F32
//   result            : [d_model, n_tokens]             F32
//
// The compact set is "the experts this layer actually routed to", which is why
// begin_layer() is called per layer: n_experts and the per-layer quant types
// both change. A layer whose gate/up/down types are not uniform across its
// experts cannot be expressed as a single 3-D tensor at all and is rejected
// here rather than silently mis-read (GLM-5.3's UD-IQ2_M layers are genuinely
// mixed; qwen4exp's are uniform -- see the [QUANT] startup report).
//
// The graph still waits for the whole layer's compact weight tensor to land
// before computing: chunked H2D overlaps host staging, but a single graph
// cannot safely consume a chunk while later expert planes are being filled.
// Removing that final transfer/compute serialization requires a chunk-level
// graph, which is deliberately left out of this correctness-first change.
//
// This header is ALWAYS compilable. Without MOESTREAM_USE_GGML_CUDA every
// entry point returns false/zero and the caller falls back to its own
// dispatch, so execution_layer.cpp needs no #ifdef at the call sites.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <memory>
#include <thread>
#include <vector>

#include "ggml_moe_transfer_manager.hpp"

// Enable stream-ordered reuse of the persistent MoE graph workspace. This is
// separate from the copy-stream experiment and is disabled by default.
void set_ggml_moe_async_graph_reuse(bool enabled);

// Shared ggml CUDA backend, created on first use and reused by EVERY ggml
// sub-graph runner in this project (MoE FFN, hyper-connections, ...). One
// backend means one CUDA stream and one memory pool for all of them, which is
// both cheaper than a backend per subsystem and the precondition for ever
// fusing these graphs into a single per-layer graph. Returns a ggml_backend_t
// as void* so this header stays includable without ggml's own headers.
// Returns nullptr when built without MOESTREAM_USE_GGML_CUDA.
void *ggml_shared_backend(int device);

// GPU router for Qwen4Exp's device-resident HC output.  The runner mirrors
// the routing part of llama.cpp's build_moe_ffn: GEMM -> gating activation ->
// optional selection bias -> top-k IDs, then gathers the unbiased weights and
// applies the model's top-k normalization/scale.  The caller receives the
// compact [topk, tokens] IDs and weights on the host; the large [hidden,
// tokens] activation never crosses the PCIe boundary for routing.
struct GgmlRouterWeightRef {
  const void *host_data = nullptr;
  uint32_t type = 0;
  int64_t ne0 = 0;
  int64_t ne1 = 0;
  size_t nbytes = 0;
};

class GgmlRouterRunner {
 public:
  static bool compiled_in();
  static GgmlRouterRunner &instance();

  bool ensure(int device, int d_model, int n_experts, int topk, int n_tokens,
              bool normalize, float weight_scale);
  bool ensure_layer(int layer, const GgmlRouterWeightRef &router,
                   const float *selection_bias);

  bool run(int layer, const void *device_x, int32_t *ids_ks, float *weights_ks);

  // Same as run(), but the input is a GGML tensor owned by the shared backend.
  // Queue the HC -> router input copy on that backend's stream so the router
  // cannot race the preceding HC graph. The host receives only the small
  // route table after the router graph completes.
  bool run_tensor(int layer, const void *device_tensor, int32_t *ids_ks,
                  float *weights_ks);

  // Same as run_tensor(), but consumes stream 0 of one token from a GGML
  // [D, HC, S] tensor. Qwen's advisory router pilot uses this to project
  // the device-authoritative HC state without downloading the full HC slab.
  // The returned route table is still the small host-visible [topk,1]
  // result required by the cache/prefetch policy.
  bool run_tensor_stream0(int layer, const void *device_tensor,
                          int token_index, int32_t *ids_ks,
                          float *weights_ks);
  bool run_tensor_stream0_device(int layer, const void *device_tensor,
                                 int token_index, const void **device_ids,
                                 const void **device_weights);

  // Queue the same router graph but leave its compact route outputs on the
  // shared GGML CUDA backend.  This is used only when the caller has already
  // established that every expert in the layer is resident on the MoE GPU,
  // so no host-side cache lookup is needed before the expert graph runs.
  // `device_ids` is [topk,tokens] I32 and `device_weights` is
  // [1,topk,tokens] F32.  Neither pointer is owned by the caller.
  bool run_tensor_device(int layer, const void *device_tensor,
                         const void **device_ids,
                         const void **device_weights);

  // Diagnostic-only export of a device route table. This synchronizes the
  // router graph and is called only by the explicit MoE set trace, so normal
  // device routing remains host-free.
  bool copy_route_to_host(int layer, int32_t *ids_ks, float *weights_ks);

  // Queue the compact route-table readback after run_tensor_stream0_device().
  // The IDs/weights are copied into a small pinned host buffer and fenced by
  // a backend event, so the next layer can poll the result instead of
  // unconditionally synchronizing the router backend at the layer boundary.
  bool queue_route_to_host(int layer);

  // Diagnostic-only export of the pre-softmax router logits.  This is used
  // to distinguish a GEMM/input mismatch from a top-k/layout mismatch.
  bool copy_logits_to_host(int layer, float *logits);

  void shutdown();
  ~GgmlRouterRunner();

 private:
  GgmlRouterRunner() = default;
  GgmlRouterRunner(const GgmlRouterRunner &) = delete;
  GgmlRouterRunner &operator=(const GgmlRouterRunner &) = delete;

  struct Layer {
    void *weight_ctx = nullptr; // ggml_context*
    void *weight_buf = nullptr; // ggml_backend_buffer_t
    void *weight = nullptr;     // ggml_tensor*
    void *bias = nullptr;       // ggml_tensor*, optional
    void *graph_ctx = nullptr;  // ggml_context*
    void *galloc = nullptr;     // ggml_gallocr_t
    void *graph = nullptr;      // ggml_cgraph*
    void *x = nullptr;          // ggml_tensor*
    void *ids = nullptr;        // ggml_tensor* [topk,tokens]
    void *weights = nullptr;    // ggml_tensor* [1,topk,tokens]
    void *logits = nullptr;     // ggml_tensor* [experts,tokens]
    void *route_done_event = nullptr; // ggml_backend_event_t
    int32_t *route_ids_host = nullptr;
    float *route_weights_host = nullptr;
    bool route_readback_pending = false;
    bool graph_in_flight = false;
    bool ready = false;
  };

  bool build_graph(int layer);
  void free_layer(Layer &layer);
  bool run_tensor_view(int layer, const void *src_view,
                       const void **device_ids,
                       const void **device_weights);

  void *backend_ = nullptr; // shared, not owned
  int device_ = -1;
  int d_model_ = 0;
  int n_experts_ = 0;
  int topk_ = 0;
  int n_tokens_ = 0;
  bool normalize_ = false;
  float weight_scale_ = 0.0f;
  size_t graph_workspace_bytes_ = 0;
  Layer layers_[256];
};

// Qwen4Exp shared-expert FFN runner.  It consumes the HC mixed activation and
// the routed-expert result on the shared GGML CUDA stream, computes the
// always-on gate/up/SwiGLU/down path, and returns routed+shared as one device
// tensor for HC post-combine.  The host fallback remains available when a
// model's shared tensors do not satisfy this narrow Qwen shape contract.
struct GgmlSharedWeightRef {
  const void *host_data = nullptr;
  // Optional persistent device copy owned by the engine's dense pool.  When
  // present, the shared runner binds its GGML weight tensor directly to this
  // pointer instead of uploading a second per-layer copy.
  const void *device_data = nullptr;
  size_t device_bytes = 0;
  uint32_t type = 0;
  int64_t ne0 = 0;
  int64_t ne1 = 0;
  size_t nbytes = 0;
};

class GgmlQwenSharedRunner {
 public:
  static bool compiled_in();
  static GgmlQwenSharedRunner &instance();
  // Qwen's shared SwiGLU clamp is per-layer. Keeping one runner per layer
  // avoids rebuilding/synchronizing the persistent CUDA graph whenever the
  // caller advances to a layer with a different clamp value.
  static GgmlQwenSharedRunner &instance_for_layer(int layer);

  bool ensure(int device, int d_model, int shared_inter, int n_tokens,
              bool has_gate, float clamp);
  bool ensure_layer(int layer, const GgmlSharedWeightRef &gate,
                   const GgmlSharedWeightRef &up,
                   const GgmlSharedWeightRef &down,
                   const GgmlSharedWeightRef *gate_inp);
  bool run(int layer, const void *device_x, const void *routed_output,
           const void **combined_output);

  // Qwen CPU-affinity variant: upload a DDR-backed CPU routed result into
  // the persistent routed input and keep the shared expert plus the routed+
  // shared combine on the CUDA graph.
  bool run_host_routed(int layer, const void *device_x,
                       const float *host_routed,
                       const void **combined_output);

  // Qwen-only split-path overlap. Queue the shared expert without a routed
  // input, allowing the caller to run the DDR-backed CPU routed graph at the
  // same time. The follow-up uploads the completed CPU result and adds it to
  // the already-queued shared output on the same GGML stream.
  bool run_shared_only(int layer, const void *device_x,
                       const void **shared_output);
  bool run_host_routed_add(int layer, const float *host_routed,
                           const void **combined_output);
  // Add a device-resident routed vector to an already-combined device result.
  // This is the second leg of CPUOffload Milestone B: the caller has already
  // queued the shared+hot graph, and the device stream has been fenced on the
  // CPU worker's H2D completion event before invoking this graph.
  bool run_device_routed_add(int layer, const void *base_output_tensor,
                             const void *device_routed,
                             const void **combined_output);

  // Opaque CUDA stream used by this runner's shared GGML backend.  The
  // CPUOffload ring uses it only for cudaStreamWaitEvent; keeping the public
  // type opaque preserves the no-CUDA header contract.
  void *backend_stream() const;

  // Copy the result of the most recent run() to a row-major host buffer.
  // This is the deliberate boundary for the host-HC fallback: the routed and
  // shared FFNs remain one device graph, then HC post-combine consumes the
  // combined result on the host. The method synchronizes only as required by
  // ggml_backend_tensor_get().
  bool copy_output_to_host(const void *combined_output, float *host_out);

  // Completion fence for the most recent shared graph submission. This is an
  // opaque ggml_backend_event_t used only by Qwen's fused HC post-combine
  // bridge; callers must not free or retain it past the runner lifetime.
  const void *output_done_event() const { return output_done_event_; }

  void shutdown();
  ~GgmlQwenSharedRunner();

 private:
  GgmlQwenSharedRunner() = default;
  GgmlQwenSharedRunner(const GgmlQwenSharedRunner &) = delete;
  GgmlQwenSharedRunner &operator=(const GgmlQwenSharedRunner &) = delete;

  bool build_graph();
  void free_graph();
  bool record_output_event();

  void *backend_ = nullptr; // borrowed from ggml_shared_backend()
  void *external_buf_ = nullptr; // non-owning-pointer GGML source marker
  int device_ = -1;
  int d_model_ = 0;
  int shared_inter_ = 0;
  int n_tokens_ = 0;
  bool has_gate_ = false;
  float clamp_ = 0.0f;

  void *weight_ctx_ = nullptr; // ggml_context*
  void *weight_buf_ = nullptr; // ggml_backend_buffer_t
  void *gate_ = nullptr;       // ggml_tensor*
  void *up_ = nullptr;
  void *down_ = nullptr;
  void *gate_inp_ = nullptr;   // optional ggml_tensor*
  void *owned_gate_data_ = nullptr;
  void *owned_up_data_ = nullptr;
  void *owned_down_data_ = nullptr;
  void *owned_gate_inp_data_ = nullptr;
  uint32_t gate_type_ = 0;
  uint32_t up_type_ = 0;
  uint32_t down_type_ = 0;
  uint32_t gate_inp_type_ = 0;

  void *graph_ctx_ = nullptr;  // ggml_context*
  void *galloc_ = nullptr;     // ggml_gallocr_t
  void *graph_ = nullptr;      // ggml_cgraph*
  void *overlap_shared_galloc_ = nullptr;  // ggml_gallocr_t
  void *overlap_combine_galloc_ = nullptr; // ggml_gallocr_t
  void *overlap_shared_graph_ = nullptr;   // ggml_cgraph*
  void *overlap_combine_graph_ = nullptr;  // ggml_cgraph*
  void *x_ = nullptr;           // ggml_tensor* [D,S]
  void *routed_ = nullptr;     // ggml_tensor* [D,S]
  void *output_ = nullptr;     // ggml_tensor* [D,S]
  // Graph-owned shared-expert nodes retained for opt-in parity diagnostics.
  void *shared_raw_ = nullptr; // ggml_tensor* [D,S], before scalar gate
  void *shared_ = nullptr;     // ggml_tensor* [D,S], after scalar gate
  void *overlap_x_ = nullptr;   // ggml_tensor* [D,S]
  void *overlap_routed_ = nullptr; // ggml_tensor* [D,S]
  void *overlap_shared_input_ = nullptr; // ggml_tensor* [D,S]
  void *overlap_shared_output_ = nullptr; // ggml_tensor* [D,S]
  void *overlap_combined_ = nullptr; // ggml_tensor* [D,S]
  void *zero_routed_data_ = nullptr; // device F32 zeros for shared-only run
  // Direct-input mode temporarily retargets graph input data pointers to
  // producer tensors. Restore these allocations before a host upload.
  void *x_owned_data_ = nullptr;
  void *routed_owned_data_ = nullptr;
  void *overlap_x_owned_data_ = nullptr;
  void *overlap_routed_owned_data_ = nullptr;
  bool graph_ready_ = false;
  bool overlap_graph_ready_ = false;
  void *output_done_event_ = nullptr; // ggml_backend_event_t

};

// Test-only helper (2026-09-10, added to isolate a real GgmlQwenSharedRunner
// NaN bug -- see main_test.cpp's own test). Wraps host_data as a device-
// resident [ne0,ne1] F32 tensor matching exactly what GgmlQwenSharedRunner::
// run()'s device_x/routed_output parameters expect (type/ne/nb/data/buffer
// all checked there), without main_test.cpp needing its own raw ggml.h
// includes. Returns false/no-op without MOESTREAM_USE_GGML_CUDA. Caller
// owns *out_ctx/*out_buf and must free them via
// ggml_moe_test_free_device_tensor once done with *out_tensor.
bool ggml_moe_test_make_device_tensor(int device, int64_t ne0, int64_t ne1,
                                      const float *host_data,
                                      void **out_tensor, void **out_ctx,
                                      void **out_buf);
void ggml_moe_test_free_device_tensor(void *ctx, void *buf);
// Synchronizes the given device before reading, so a caller doesn't need to
// separately track/wait on whatever stream produced `tensor`.
bool ggml_moe_test_download_tensor(int device, const void *tensor,
                                   float *host_out, size_t n_floats);

struct GgmlMoeLayerPlan {
  int device_id = -1;   // CUDA device the weights and graph live on
  int n_experts = 0;    // size of the compact routed set for this layer
  // Requested size of the expert AXIS of the weight tensors. The runner
  // rounds this to a grow-only compact-slot high-water mark, because letting
  // ne2 follow n_experts (275, 211, 217, ... per layer) makes ggml re-plan
  // and reallocate its compute buffer on every single layer. Slots beyond
  // n_experts are never referenced by `ids`.
  int n_expert_slots = 0;
  int n_tokens = 0;     // S
  int topk = 0;         // experts per token
  int d_model = 0;      // D
  int d_ff = 0;         // moe_inter
  // In the chunk pipeline, n_tokens is the padded routed-pair count and
  // source_tokens is the original hidden-state batch. Normal graphs leave
  // source_tokens at zero and use n_tokens for both dimensions.
  int source_tokens = 0;
  bool chunked = false;
  // GGUFType numeric values. MoEStream's GGUFType is ggml's own numbering
  // (see include/gguf_loader.hpp's enum, which documents the 2026 renumbering
  // it was corrected against), so these are cast straight to ggml_type.
  uint32_t gate_type = 0;
  uint32_t up_type = 0;
  uint32_t down_type = 0;
  int chunk_experts = 0;  // experts per staging chunk (pinned buffer size)

  // Optional zero-copy view of a complete resident layer.  The three
  // projections are stored interleaved in the topology's fixed-size VRAM
  // slabs as [gate|up|down] for each expert.  GGML's MUL_MAT_ID kernels
  // already consume nb[2], so a regular tensor descriptor can view those
  // slabs without repacking them into a second planar allocation.
  bool external_weights = false;
  const void * external_gate = nullptr;
  const void * external_up = nullptr;
  const void * external_down = nullptr;
  size_t external_expert_stride = 0;
};

// Timing for the actual CUDA copy-stream service interval.  The byte counts
// are source-specific; `d2d_seconds` and `h2d_seconds` are CUDA-event
// service intervals, not bytes divided by request wall time. The optional
// `h2d_dma_seconds` measures only the actual H2D DMA calls, excluding packed
// device-unpack kernels between them.
struct GgmlMoeTransferStats {
  double d2d_seconds = 0.0;
  double h2d_seconds = 0.0;
  double h2d_dma_seconds = 0.0;
  size_t d2d_bytes = 0;
  size_t h2d_bytes = 0;
};

// Select the reference planar H2D path without using a process-wide
// environment variable. The engine sets this from EngineConfig so a Qwen
// tuning choice cannot silently alter GLM/DeepSeek runners.
void set_moe_ggml_planar_copies(bool enabled);

// One expert's three destination pointers inside a pinned staging buffer.
struct GgmlMoeStageSlot {
  uint8_t *gate = nullptr;
  uint8_t *up = nullptr;
  uint8_t *down = nullptr;
};

class GgmlMoeRunner : private moe::GgmlMoeTransferManager {
 public:
  // Pinned staging buffers (memcpy || H2D).
  //
  // 2026-09-09: TRIED raising this to 4, reverted -- real, measured, net
  // loss, not a win. The hypothesis was sound and half-confirmed: the
  // "Prefill expert stream:" shutdown line showed 1.593s of a 488-token
  // prefill's 4.93s expert= bucket was CPU blocked waiting for a staging
  // buffer to free up (t_prefill_stream_wait), on top of 2.518s of real H2D
  // transfer already close to this box's PCIe ceiling (10.07 of ~12.3
  // GB/s). Raising NBUF to 4 DID cut that wait time as predicted (3 real
  // runs: 0.172s / 0.834s / 0.956s, vs 1.593s baseline) -- but overall
  // prefill throughput came out WORSE across all 3 runs (59.09 / 61.18 /
  // 60.49 tok/s, mean 60.25) than the NBUF=2 baseline (62.48 / 62.77 /
  // 63.20, mean 62.82). The first run's `disk=` bucket (which folds in the
  // RAM-tier->pinned-staging memcpy) jumped 0.28s->2.21s -- more buffers in
  // flight apparently increases contention/overhead on that memcpy path
  // (or per-event bookkeeping cost) by more than the wait-time saved.
  // Net: the buffer-wait time was real but was NOT the dominant lever it
  // looked like in isolation -- don't re-raise this without re-measuring
  // BOTH the wait time AND the disk= bucket together, not just the metric
  // that motivated the change.
  //
  // 2026-09-10: TRIED NBUF=3, REVERTED. Decisive negative result via a
  // precise per-layer overhead measurement (inter-layer wall-clock gap
  // minus that layer's own reported graph= compute time, summed across all
  // 48 layers, from the [GGML-MOE] layer=N timestamps): NBUF=2 -> 138.32
  // ms/layer overhead, NBUF=3 -> 139.89 ms/layer -- essentially identical,
  // despite NBUF=3 cutting the "CPU blocked on double-buffer handoff" bucket
  // from 1.6s to 1.25s. The wait-time drop was real but got absorbed
  // elsewhere (graph_sum itself grew 688ms->865ms), so total per-layer cost
  // didn't move. Conclusion: NBUF is NOT the real lever at 2, 3, OR 4 --
  // all three land in the same ~138-140 ms/layer range. The real ~140ms/
  // layer floor (measured against full-graph's ~54ms/layer-equivalent for
  // the identical work) comes from something else inherent to the
  // per-layer host-orchestrated dispatch design itself, not buffer depth.
  // Don't re-try NBUF tuning without a new hypothesis for what THAT is.
  //
  // 2026-09-10 (later same day): RE-VERIFIED against the post-fix baseline
  // (RAM/VRAM tier independence fix + vram_preload_prefix_layers adopted
  // since the test above; also after the real GgmlQwenSharedRunner
  // device_x bug was fixed, see ggml_qwen_shared_runner.cpp/Test 35). Same 574-
  // token real-text prompt, prefill= from the shutdown report's Rates
  // line: NBUF=2 baseline 79.84 / 86.37 / 85.66 tok/s (mean 83.96), NBUF=3
  // 83.57 tok/s -- squarely inside the NBUF=2 spread, not a separate
  // cluster. Conclusion unchanged: NBUF is still not a real lever, now
  // confirmed against the current, materially faster baseline too.
  static constexpr int NBUF = moe::GgmlMoeTransferManager::kTransferBuffers;
  // Keep one compact weight buffer. The graph stores one mutable set of GGML
  // weight descriptors; alternating allocations without a separate graph
  // descriptor set would let a queued layer observe the next layer's data
  // pointer. The backend-stream path can still overlap host staging and
  // transfer work, but true weight double-buffering requires per-buffer graph
  // state and is intentionally not enabled by this runner.
  static constexpr int NWEIGHT_BUFS =
      moe::GgmlMoeTransferManager::kWeightBuffers;
  // A prefill can queue all model layers before the final synchronization.
  // Keep one completion event per possible layer so an event still being
  // waited on by the copy backend is never re-recorded.
  static constexpr int NWEIGHT_EVENTS =
      moe::GgmlMoeTransferManager::kWeightEvents;

  // False when built without MOESTREAM_USE_GGML_CUDA. Checked before anything
  // else so the flag can be left set in a config that is also used by a build
  // without the backend.
  static bool compiled_in();

  // Process-wide singleton: the CUDA backend, its weight buffer and its graph
  // allocator are all expensive and are reused across layers and requests.
  static GgmlMoeRunner &instance();

  // Qwen4Exp's routed projections use more than one quantization signature
  // across the model. Keep one persistent graph/weight state per signature so
  // switching IQ1/IQ2 layers does not destroy the previous graph while its
  // queued HC consumer is still running. The selected runner becomes the
  // active runner for the current layer.
  static GgmlMoeRunner &instance_for(uint32_t gate_type, uint32_t up_type,
                                      uint32_t down_type);
  static GgmlMoeRunner &active_instance();

  // Request-boundary helpers for the small pool of persistent runners.
  static void finish_all_h2d();
  static void collect_all_h2d(double &seconds, size_t &bytes);
  static void collect_all_transfer_stats(GgmlMoeTransferStats &stats);

  // Queue a device-side dependency before Qwen's persistent HC mixed-input
  // tensor is reused by the next pre-mix graph.  The direct HC->MoE input
  // path intentionally aliases that tensor instead of copying it, so the
  // previous MoE graph must have finished reading it first.  This is an
  // asynchronous backend event wait, not a host synchronize.
  static void queue_input_reuse_dependency();

  // Size/validate storage for one layer's compact expert set. Returns false if
  // the plan is unsupported (bad shape, unsupported quant type, allocation
  // failure) -- the caller must then run its own path for this layer.
  bool begin_layer(const GgmlMoeLayerPlan &plan);

  // Per-expert byte counts for the current layer, i.e. exactly how much the
  // caller must memcpy into each staging slot.
  size_t gate_expert_bytes() const { return gate_exp_bytes_; }
  size_t up_expert_bytes() const { return up_exp_bytes_; }
  size_t down_expert_bytes() const { return down_exp_bytes_; }

  // Block until staging buffer `buf`'s previous H2D has drained, so its bytes
  // can be overwritten. Cheap no-op when nothing is in flight.
  void wait_stage_free(int buf);

  // Destination pointers for expert `slot` (0 .. chunk_experts-1) in `buf`.
  GgmlMoeStageSlot stage_slot(int buf, int slot) const;

  // Issue the H2D for experts [first, first+count) from staging buffer `buf`
  // into the layer's weight planes. `staged[i]` false means expert first+i is
  // NOT in the staging buffer (it was filled device-to-device from the VRAM
  // cache instead) and must be skipped. Consecutive staged experts are merged
  // into one copy per plane, so the all-staged case is still 3 copies total.
  // Consecutive staged experts are copied as one packed [gate|up|down] run.
  // Asynchronous; sync_copies()/run_layer() wait for it.
  bool push_chunk(int buf, int first_expert, int count,
                  const bool *staged = nullptr);

  // Fill expert `expert_index`'s three plane slots straight from an expert
  // slab ALREADY resident in this GPU's VRAM cache, laid out [gate|up|down].
  // ~600 GB/s device-to-device instead of a host memcpy plus a 12 GB/s H2D --
  // the whole point of doing it. Ordered on the same copy stream as the H2D,
  // so one wait covers both. Asynchronous: the source slab must stay held
  // (hold_for_compute) until sync_copies() returns.
  bool copy_expert_d2d(int buf, int expert_index, const void *src_slab);

  // Mark the boundary between the D2D reuse phase and the RAM H2D phase for
  // the current copy batch.  The runner also records this lazily in
  // push_chunk(), so older callers that issue no D2D copies remain valid.
  bool mark_copy_d2d_done(int buf);

  // Wait for every in-flight weight copy (H2D and D2D) on this layer. Safe to
  // call more than once; run_layer() does it internally.
  void sync_copies();

  // Qwen-only A/B: pack all VRAM-cache D2D records for one chunk into the
  // transfer buffer, then issue one device unpack-scatter kernel at the D2D
  // boundary. Other model families leave this disabled; the default is also
  // disabled until the real-model correctness/performance test passes.
  void set_batch_d2d_unpack(bool enabled) { batch_d2d_unpack_ = enabled; }

  // Start the copy batch for staging buffer `buf`. The caller must invoke
  // this after wait_stage_free(buf) and before copy_expert_h2d()/push_chunk().
  // Keeping the start event ahead of every H2D in the batch makes the timing
  // include direct copies from persistent RAM slabs as well as copies from
  // the temporary staging buffer.
  bool begin_copy_batch(int buf);

  // Queue one expert's [gate|up|down] slab directly from a persistently pinned
  // RAM-tier allocation into the three planar GGML weight tensors. This is
  // the RAM-resident form of the fast path: no CPU bounce memcpy and no
  // per-tensor synchronization. The source must remain held until
  // sync_copies() returns.
  bool copy_expert_h2d(int buf, int expert_index, const void *src_slab);

  // Queue a consecutive run of [gate|up|down] slabs from a fixed-stride,
  // persistently pinned RAM arena. The destination slots must also be
  // consecutive in the runner's compact expert axis. This uses one
  // cudaMemcpy2DAsync per projection instead of three cudaMemcpyAsync calls
  // per expert, while retaining the ordinary planar GGML layout. Returns
  // Packed-transfer mode transparently falls back to copy_expert_h2d() for
  // each expert; invalid fixed-stride arguments still return false.
  bool copy_expert_h2d_strided(int buf, int first_expert, int count,
                               const void *src_slab, size_t src_stride);

  // Build and run the graph, then ADD the [n_tokens, d_model] result into
  // `out_add` (row-major, same layout as the caller's `output`).
  //   x   : [n_tokens * d_model] F32, row-major (scratch.normed)
  //   ids : [n_tokens * topk]    I32, compact-set indices
  //   w   : [n_tokens * topk]    F32, router weights
  bool run_layer(const float *x, const int32_t *ids, const float *w,
                 float *out_add);

  // Run the same graph but leave its [D, S] F32 result in the shared CUDA
  // backend. `device_output` is non-owning and is valid until the next
  // run_layer_device()/begin_layer() call; the caller must consume it
  // immediately. This is intentionally an additive API: existing callers
  // keep the host-output behaviour above.
  bool run_layer_device(const float *x, const int32_t *ids, const float *w,
                        const void **device_output);

  // The persistent GGML tensor receiving the host input in the most recent
  // run_layer_device() call. It is valid until the next begin_layer() or
  // run_layer_device*() call and is used by Qwen's shared-expert graph to
  // consume the same activation without another host upload or D2D copy.
  const void *device_input_tensor() const { return graph_x_; }

  // Same as run_layer_device(), but `device_x` is an already resident F32
  // [D,S] activation on this runner's CUDA device.  This avoids downloading
  // a hyper-connection result to the host and uploading it again just before
  // the MoE graph.  The host form above remains the fallback.
  bool run_layer_device_input(const void *device_x, const int32_t *ids,
                              const float *w, const void **device_output);

  // Same as run_layer_device_input(), but the input is a GGML tensor from
  // the shared backend. The HC -> MoE handoff is queued on the GGML CUDA
  // stream, so it preserves producer ordering without a raw synchronous
  // device copy.
  bool run_layer_device_input_tensor(const void *device_tensor,
                                     const int32_t *ids, const float *w,
                                     const void **device_output);

  // Same as run_layer_device_input_tensor(), but route IDs and weights are
  // already GGML tensors on this runner's shared CUDA backend.  The tiny
  // route tables are copied device-to-device, avoiding the router's
  // GPU->host->GPU boundary while preserving the existing graph storage.
  bool run_layer_device_input_tensor_device_route(
      const void *device_tensor, const void *device_ids,
      const void *device_weights, const void **device_output);

  // Chunk-pipeline variant. The source activation is [D,source_tokens]. The
  // three host arrays are padded to plan.n_tokens and describe one routed
  // expert chunk. Token rows are gathered on the GPU, and the chunk result is
  // scatter-added into a persistent device accumulator.
  bool run_chunk_device_input_tensor(const void *device_tensor,
                                     const int32_t *token_ids,
                                     const int32_t *ids,
                                     const float *weights, int copy_buf,
                                     const void **device_output);
  bool finish_chunked_output(const void **device_output);

  // Queue the completion dependency for the most recent device output on the
  // shared GGML backend.  Consumers in another runner (HC) must call this
  // before reading graph_out_; a same-backend tensor copy is not sufficient
  // when graph execution uses an internal CUDA stream.
  void wait_device_output();

  // Force the next run_layer_device*() call to rebuild graph_ctx_ from
  // scratch, even if the layer shape (n_tokens/topk/d_model/...) matches
  // what's already cached. Safe to call any time (drains any in-flight
  // graph first, exactly like ensure_graph()'s own shape-mismatch rebuild
  // path -- this just exposes that same sequence publicly instead of
  // duplicating it).
  //
  // Real crash this exists for (2026-09-09): graph_ctx_'s tensor-object
  // pool (ensure_graph()'s own ggml_tensor_overhead()*128 budget) is sized
  // for roughly one forward pass's worth of per-layer run_layer_device_impl
  // calls (each adds a small, fixed number of tensor objects into the SAME
  // reused context whenever the shape matches -- normal decode/prefill
  // never accumulates enough to matter). Qwen4ExpMtpSpeculator::run()
  // (mtp_speculative.cpp) re-runs the target trunk over the WHOLE prompt a
  // second time, at the SAME shape, immediately after the caller's own
  // real prefill already ran it once -- the two calls' accumulated tensor
  // objects landed in the SAME never-reset context and overflowed it by a
  // few dozen bytes (`ggml_new_object: not enough space in the context's
  // memory pool`, a hard ggml_abort, not a catchable error). Call this
  // between the two passes to force a clean rebuild instead of growing the
  // shared budget for every caller.
  void invalidate_graph();

  // Copy a result returned by run_layer_device() to a row-major host buffer.
  // Used only for the guarded fallback/debug path when the device consumer
  // cannot accept the result.
  bool copy_device_output_to_host(const void *device_output, float *host_out);

  // Drain and reset the accumulated H2D timing/byte counters.
  void collect_h2d(double &seconds, size_t &bytes);

  // Drain and reset source-separated CUDA-event timing.  collect_h2d() is
  // retained for existing callers; new performance reports should use this
  // method so D2D cache reuse is not mislabeled as PCIe traffic.
  void collect_transfer_stats(GgmlMoeTransferStats &stats);

  // Complete outstanding backend-stream staging transfers at the end of a
  // forward pass. Per-layer collection leaves them pending so copies can
  // overlap later host orchestration and graph execution.
  void finish_h2d();

  // Keep a VRAM-cache source protected until this runner's copy stream has
  // finished its queued D2D transfer. This lets the caller return from a
  // layer without synchronizing the copy stream, while retaining the
  // hold_for_compute/release_hold eviction invariant.
  void defer_vram_hold(void *topology, int layer, int expert,
                       const uint8_t *device_ptr);
  void release_completed_vram_holds();

  // Keep an asynchronous host source alive/pinned until the runner's copy
  // completion event has drained.  The callback is type-erased so this
  // backend does not depend on RAMSafetyManager's ownership model.
  void defer_host_release(std::function<void()> release);

  // Release every device/pinned allocation. Safe to call more than once.
  void shutdown();

  // Full-graph submission mode: expert H2D copies are queued on the shared
  // GGML CUDA stream and are ordered directly before the MoE graph. While it
  // is enabled, the caller uses pinned staging sources so RAM-tier blocks do
  // not need to remain held after the CPU staging memcpy.
  bool backend_stream_copies() const { return backend_stream_copies_; }

  ~GgmlMoeRunner();

 private:
  GgmlMoeRunner() = default;
  GgmlMoeRunner(const GgmlMoeRunner &) = delete;
  GgmlMoeRunner &operator=(const GgmlMoeRunner &) = delete;

  static GgmlMoeRunner *runner_pool();
  static constexpr int NH2D_TIMING_EVENTS =
      moe::GgmlMoeTransferManager::kH2dTimingEvents;

  bool ensure_backend(int device);
  bool ensure_weight_buffer(size_t bytes, size_t packed_bytes);
  bool ensure_stage_buffers(size_t bytes_per_buf);
  void account_copy_buffer(int buf);
  bool ensure_h2d_timing_events();
  bool begin_h2d_timing(int buf, int &slot);
  bool end_h2d_timing(int buf, int slot);
  void free_graph();
  bool ensure_graph();
  bool ensure_chunk_state();
  void free_chunk_state();
  void free_weight_tensors();
  bool run_layer_device_impl(const float *host_x, const void *device_x,
                             const int32_t *ids, const float *w,
                             const void **device_output,
                             bool device_tensor = false,
                             const void *device_ids = nullptr,
                             const void *device_weights = nullptr);

  // --- backend ---
  void *backend_ = nullptr;      // ggml_backend_t (shared, not owned)
  bool owns_backend_ = false;
  void *galloc_ = nullptr;       // ggml_gallocr_t
  int device_ = -1;

  // --- per-layer weight storage (device) ---
  void *wbuf_[NWEIGHT_BUFS] = {}; // ggml_backend_buffer_t
  size_t wbuf_bytes_ = 0;        // capacity of each compact weight buffer
  void *wctx_ = nullptr;         // ggml_context* holding the 3 weight tensors
  void *t_gate_ = nullptr;       // ggml_tensor*
  void *t_up_ = nullptr;
  void *t_down_ = nullptr;
  uint8_t *w_gate_base_ = nullptr;   // planar GGML weight regions in wbuf_
  uint8_t *w_up_base_ = nullptr;
  uint8_t *w_down_base_ = nullptr;
  // GGML tensors remain in their validated planar layout. The separate packed
  // device buffer is only a transfer landing zone: [gate|up|down] for expert
  // 0, then expert 1, ... . One H2D/D2D copy lands there and a CUDA unpack
  // kernel scatters it into the three GGML planes.
  int weight_slots_ = 0;         // axis-2 capacity of the persistent tensors
  int slot_capacity_ = 0;        // grow-only high-water mark across layers
  int weight_d_model_ = 0;
  int weight_d_ff_ = 0;
  uint32_t weight_gate_type_ = 0;
  uint32_t weight_up_type_ = 0;
  uint32_t weight_down_type_ = 0;
  bool weight_external_ = false;

  // One-byte marker buffer used to tell GGML-CUDA that external_weights'
  // pointers are device tensors.  The bytes themselves belong to
  // GpuTopology and are never owned or freed here.
  void *external_buf_ = nullptr; // ggml_backend_buffer_t

  // --- persistent graph for the current shape key ---
  void *graph_ctx_ = nullptr;    // ggml_context*
  void *graph_ = nullptr;        // ggml_cgraph*
  void *graph_x_ = nullptr;      // ggml_tensor*
  // `graph_x_` is the flat parent; MUL_MAT_ID consumes the persistent reshape
  // below.  Both data pointers must follow a direct device-input handoff.
  void *graph_x_input_ = nullptr; // ggml_tensor* (reshape/view of graph_x_)
  // Reusable view of a producer-owned [D,S_cap] input when the current graph
  // consumes only its first S rows. Allocating this in run_layer_* per layer
  // exhausts the persistent graph context during decode.
  void *graph_source_view_ = nullptr; // ggml_tensor*
  void *graph_ids_ = nullptr;    // ggml_tensor*
  void *graph_w_ = nullptr;      // ggml_tensor*
  void *graph_token_ids_ = nullptr; // ggml_tensor* (chunk graph only)
  void *graph_gate_out_ = nullptr; // ggml_tensor* (debug only)
  void *graph_up_out_ = nullptr;   // ggml_tensor* (debug only)
  void *graph_act_out_ = nullptr;  // ggml_tensor* (debug only)
  void *graph_down_out_ = nullptr; // ggml_tensor* (debug only)
  void *graph_out_ = nullptr;    // ggml_tensor*
  int graph_tokens_ = 0;
  int graph_source_tokens_ = 0;
  int graph_topk_ = 0;
  int graph_d_model_ = 0;
  int graph_d_ff_ = 0;
  int graph_slots_ = 0;
  uint32_t graph_gate_type_ = 0;
  uint32_t graph_up_type_ = 0;
  uint32_t graph_down_type_ = 0;
  bool graph_chunked_ = false;
  size_t graph_workspace_bytes_ = 0;
  void *graph_done_event_ = nullptr; // ggml_backend_event_t
  bool graph_in_flight_ = false;

  // Device-side chunk accumulation. graph_out_ is one chunk in [D,pairs]
  // layout; the scatter kernel reduces it into [D,source_tokens]. The final
  // GGML tensor lets the existing HC device handoff consume the result.
  void *chunk_accum_ = nullptr;       // raw CUDA float*
  size_t chunk_accum_bytes_ = 0;
  void *chunk_final_ctx_ = nullptr;   // ggml_context*
  void *chunk_final_buf_ = nullptr;   // ggml_backend_buffer_t
  void *chunk_final_ = nullptr;       // ggml_tensor*
  int chunk_final_tokens_ = 0;
  void *chunk_ready_event_ = nullptr; // ggml_backend_event_t
  void *chunk_graph_event_ = nullptr; // ggml_backend_event_t
  void *chunk_graph_events_[NBUF] = {}; // ggml_backend_event_t per slot
  void *chunk_accum_event_ = nullptr; // ggml_backend_event_t
  void *chunk_weight_done_events_[NBUF] = {};
  void *chunk_scatter_done_events_[NBUF] = {}; // raw CUDA events
  void *chunk_token_device_[NBUF] = {};       // raw CUDA int32 buffers
  bool chunk_weight_recorded_[NBUF] = {};
  bool chunk_scatter_recorded_[NBUF] = {};
  size_t chunk_token_bytes_ = 0;
  bool chunk_ready_recorded_ = false;
  bool chunk_accum_recorded_ = false;
  bool chunk_input_published_ = false;
  bool chunk_has_submission_ = false;
  int chunk_stage_index_ = 0;
  std::vector<int32_t> chunk_token_stage_[NBUF];
  std::vector<int32_t> chunk_id_stage_[NBUF];
  std::vector<float> chunk_weight_stage_[NBUF];
  // ggml_backend_tensor_set_async may leave the host source in use until the
  // queued graph reaches the backend stream.  Chunk graphs are deliberately
  // kept in flight, so the two alternating weight buffers are not sufficient
  // as route-table storage.  Retain one immutable route-table triple per
  // submitted chunk until finish_chunked_output() publishes the reduction.
  std::vector<std::vector<int32_t>> chunk_token_keepalive_;
  std::vector<std::vector<int32_t>> chunk_id_keepalive_;
  std::vector<std::vector<float>> chunk_weight_keepalive_;

  // IDs and router weights are supplied by per-layer local vectors in the
  // execution path. In asynchronous graph mode those vectors can disappear
  // before the backend's H2D stream consumes them. Keep a persistent pinned
  // host ring so the route tables can use ggml_backend_tensor_set_async()
  // instead of the synchronous setter (which showed up as one blocking
  // cudaMemcpy pair per MoE layer during decode). A request has at most 48
  // layers here; the extra slots cover a request boundary before the backend
  // is drained. The ring is drained only on wrap or capacity growth.
  static constexpr int NROUTE_STAGE = 64;
  int32_t *route_ids_pinned_ = nullptr;
  float *route_weights_pinned_ = nullptr;
  size_t route_stage_capacity_ = 0;
  int route_stage_index_ = 0;

  bool ensure_route_stage(size_t route_count);

  // --- current plan ---
  GgmlMoeLayerPlan plan_{};
  size_t gate_exp_bytes_ = 0;
  size_t up_exp_bytes_ = 0;
  size_t down_exp_bytes_ = 0;
  bool layer_ready_ = false;

};

// Qwen4Exp CPU routed-expert runner.  This is deliberately separate from the
// generic CPU fallback and from the CUDA GgmlMoeRunner: Qwen's CPU-affinity
// path is the only caller, so changing its graph or quantized activation
// policy cannot alter GLM/DeepSeek/other model families.
//
// The legacy fallback copies selected expert slabs into one compact
// [gate|up|down] group before the graph runs.  The Qwen CPU-affinity path can
// instead use a fixed-stride DDR arena: every expert keeps its existing RAM
// slab, and GGML MUL_MAT_ID indexes the real global expert IDs directly.
struct GgmlQwenCpuExpertRef {
  const void *gate = nullptr;
  const void *up = nullptr;
  const void *down = nullptr;
  size_t gate_bytes = 0;
  size_t up_bytes = 0;
  size_t down_bytes = 0;
  uint32_t gate_type = 0;
  uint32_t up_type = 0;
  uint32_t down_type = 0;
};

// Opt-in Layer-0 host-node diagnostics.  The CUDA graph bridge sets the
// current layer around its callback so the persistent CPU graph can expose
// the SwiGLU and raw down-projection tensors without changing its hot-path
// API or allocating diagnostic state during normal decode.
void qwen_cpu_moe_diag_set_layer(int layer);

struct GgmlQwenDirectWorkerSample {
  int64_t wall_start_ns = 0;
  int64_t wall_end_ns = 0;
  int64_t cpu_start_ns = 0;
  int64_t cpu_end_ns = 0;
  long voluntary_start = 0;
  long voluntary_end = 0;
  long involuntary_start = 0;
  long involuntary_end = 0;
  int cpu_start = -1;
  int cpu_end = -1;
};

class GgmlQwenCpuMoeRunner {
 public:
  // Called by the persistent worker after the CPU graph has finished.  The
  // callback is deliberately a raw function pointer so the decode path does
  // not allocate a std::function per token.  It may enqueue CUDA work, but it
  // must not free or reuse the request buffers.
  using AsyncCompletionFn = void (*)(void *user, bool success);

  static bool compiled_in();
  static GgmlQwenCpuMoeRunner &instance();
  // Qwen's packed DDR arenas use more than one real quantization/stride
  // signature (IQ1_M and IQ2_XXS in the current checkpoint).  Keep one
  // persistent runner per signature so switching CPU-affinity layers does not
  // tear down and rebuild the graph plan on every layer.
  static GgmlQwenCpuMoeRunner &instance_for_strided(
      const GgmlQwenCpuExpertRef &layer, int n_experts, int d_model,
      int d_ff, int topk, size_t expert_stride, bool output_per_expert);
  static GgmlQwenCpuMoeRunner &instance_for_strided_planes(
      const GgmlQwenCpuExpertRef &layer, int n_experts, int d_model,
      int d_ff, int topk, size_t gate_stride, size_t up_stride,
      size_t down_stride, bool output_per_expert);

  bool run(const float *input, float *output,
           const GgmlQwenCpuExpertRef *experts, int n_experts,
           const float *router_weights, int d_model, int d_ff, int topk);

  // BeeLlama-style CPU-offload entry point. The three projection pointers are
  // the first expert's views in a complete host-RAM layer arena. `expert_stride`
  // is the byte distance between expert IDs; route IDs remain global and
  // selected expert weights are not copied per token.
  bool run_strided(const float *input, float *output,
                  const GgmlQwenCpuExpertRef &layer,
                  const void *gate_plane, const void *up_plane,
                  const void *down_plane, const int *expert_ids,
                  int n_experts, const float *router_weights, int d_model,
                  int d_ff, int topk, size_t expert_stride);

  // Projection-compact variant. Each projection has its own expert-axis
  // stride; the three planes are not assumed to be adjacent per expert.
  bool run_strided_planes(
      const float *input, float *output, const GgmlQwenCpuExpertRef &layer,
      const void *gate_plane, const void *up_plane, const void *down_plane,
      const int *expert_ids, int n_experts, const float *router_weights,
      int d_model, int d_ff, int topk, size_t gate_stride, size_t up_stride,
      size_t down_stride);

  // Qwen-only CPU/GPU partition: run the CPU graph through the down
  // projection, returning the unweighted [d_model, topk] expert outputs.
  // The top-k weighting and reduction are performed by
  // GgmlQwenDeviceCombineRunner on the shared CUDA stream.  This keeps the
  // DDR-backed strided arena and the global route IDs unchanged while moving
  // the small, arithmetic-heavy combine off the CPU.
  bool run_strided_down(const float *input, float *expert_output,
                        const GgmlQwenCpuExpertRef &layer,
                        const void *gate_plane, const void *up_plane,
                        const void *down_plane, const int *expert_ids,
                        int n_experts, int d_model, int d_ff, int topk,
                        size_t expert_stride);

  // Queue one strided CPU graph execution on the runner's persistent worker.
  // The caller may use the interval before wait_strided_async() to run the
  // Qwen shared expert on CUDA. Requests are strictly one at a time.
  bool start_strided_async(
      const float *input, float *output, const GgmlQwenCpuExpertRef &layer,
      const void *gate_plane, const void *up_plane, const void *down_plane,
      const int *expert_ids, int n_experts, const float *router_weights,
      int d_model, int d_ff, int topk, size_t expert_stride);
  bool start_strided_async(
      const float *input, float *output, const GgmlQwenCpuExpertRef &layer,
      const void *gate_plane, const void *up_plane, const void *down_plane,
      const int *expert_ids, int n_experts, const float *router_weights,
      int d_model, int d_ff, int topk, size_t expert_stride,
      AsyncCompletionFn completion, void *completion_user);
  bool wait_strided_async();

  // Actual GGML worker count for diagnostics. This may differ from the
  // process-wide OpenMP maximum when MOESTREAM_QWEN_CPU_THREADS is used.
  int thread_count() const { return n_threads_; }

  // Start the pre-sized direct team during graph construction, before graph
  // capture. Replay-time arming only changes an atomic state and never
  // creates threads or allocates memory.
  void prepare_direct_workers();
  void arm_direct_workers();

  // Host-node runners are inactive for most of a decode step. When ggml is
  // built without OpenMP these calls suspend/resume the persistent ggml CPU
  // worker team, so it cannot compete with the ordinary scheduler between
  // callbacks. They are harmless with an OpenMP-backed ggml build because
  // ggml implements them as no-ops there.
  void pause_workers();
  void resume_workers();

  // The full Qwen graph already owns a persistent CPU worker pool for its
  // ordinary CPU splits.  A host-node MoE runner may borrow that pool while
  // its callback is active instead of creating a second execution context.
  // The caller retains ownership and must release it before destroying the
  // pool.
  void bind_external_threadpool(void *threadpool, int n_threads);
  void unbind_external_threadpool(void *threadpool);
  bool uses_external_threadpool() const { return cpu_threadpool_borrowed_; }

  void shutdown();
  ~GgmlQwenCpuMoeRunner();

  // The signature cache owns a small fixed set of persistent runners.  Keep
  // default construction available for that cache; copy remains disabled so
  // runners cannot accidentally share graph/backend state.
  GgmlQwenCpuMoeRunner() = default;

 private:
  GgmlQwenCpuMoeRunner(const GgmlQwenCpuMoeRunner &) = delete;
  GgmlQwenCpuMoeRunner &operator=(const GgmlQwenCpuMoeRunner &) = delete;

  struct AsyncStridedRequest;

  bool ensure_graph(const GgmlQwenCpuExpertRef &ref, int n_experts,
                    int d_model, int d_ff, int topk, bool strided,
                    const void *gate_plane, const void *up_plane,
                    const void *down_plane, size_t expert_stride,
                    bool output_per_expert, bool merged_gate_up,
                    size_t up_expert_stride = 0,
                    size_t down_expert_stride = 0);
  bool ensure_direct_signature(const GgmlQwenCpuExpertRef &ref,
                              int n_experts, int d_model, int d_ff, int topk,
                              size_t gate_stride, size_t up_stride,
                              size_t down_stride);
  bool direct_signature_supported(const GgmlQwenCpuExpertRef &ref,
                                  int d_model, int d_ff, int topk,
                                  size_t gate_stride, size_t up_stride,
                                  size_t down_stride) const;
  bool direct_pool_enabled() const;
  bool submit_direct_request(const AsyncStridedRequest &request);
  void run_direct_caller_phases(const AsyncStridedRequest &request);
  bool run_direct_fused_request(const AsyncStridedRequest &request);
  bool run_direct_request(const AsyncStridedRequest &request);
  void direct_worker_loop(int worker_index);
  void ensure_direct_workers();
  void stop_direct_pool();
  bool has_strided_signature(const GgmlQwenCpuExpertRef &layer,
                             int n_experts, int d_model, int d_ff, int topk,
                             size_t expert_stride,
                             bool output_per_expert) const;
  bool has_strided_planes_signature(
      const GgmlQwenCpuExpertRef &layer, int n_experts, int d_model,
      int d_ff, int topk, size_t gate_stride, size_t up_stride,
      size_t down_stride, bool output_per_expert) const;
  bool compute_graph();
  void free_graph();
  void async_worker_loop();

  struct AsyncStridedRequest {
    const float *input = nullptr;
    float *output = nullptr;
    GgmlQwenCpuExpertRef layer{};
    const void *gate_plane = nullptr;
    const void *up_plane = nullptr;
    const void *down_plane = nullptr;
    const int *expert_ids = nullptr;
    int n_experts = 0;
    const float *router_weights = nullptr;
    int d_model = 0;
    int d_ff = 0;
    int topk = 0;
    size_t expert_stride = 0;
    size_t up_expert_stride = 0;
    size_t down_expert_stride = 0;
    AsyncCompletionFn completion = nullptr;
    void *completion_user = nullptr;
  };

  void *cpu_backend_ = nullptr;       // ggml_backend_t, owned
  // BeeLlama keeps one CPU worker pool attached to its CPU backend for the
  // lifetime of the context.  Keep the Qwen-only runner's pool persistent as
  // well; leaving this null makes ggml_backend_graph_compute() create and
  // destroy a disposable pool for every CPU-affinity layer call.
  void *cpu_threadpool_ = nullptr;    // ggml_threadpool_t, owned
  bool cpu_threadpool_borrowed_ = false;
  void *weight_ctx_ = nullptr;        // ggml_context*, owned
  void *weight_buf_ = nullptr;        // ggml_backend_buffer_t, owned
  std::vector<void *> external_weight_bufs_; // three non-owning CPU pointer buffers
  void *gate_ = nullptr;              // ggml_tensor* (separate gate path)
  void *up_ = nullptr;                // ggml_tensor*
  void *gate_up_ = nullptr;           // ggml_tensor* (merged gate+up path)
  void *down_ = nullptr;              // ggml_tensor*
  void *graph_ctx_ = nullptr;         // ggml_context*, owned
  void *galloc_ = nullptr;            // ggml_gallocr_t, owned
  // CPU backend execution plan, created once for the persistent graph.  The
  // plan owns the reusable GGML work buffer and avoids rerunning
  // ggml_graph_plan() for every CPU-affinity layer/token call.
  void *graph_plan_ = nullptr;        // ggml_backend_graph_plan_t, owned
  void *graph_ = nullptr;             // ggml_cgraph*
  void *x_ = nullptr;                 // ggml_tensor*
  void *ids_ = nullptr;               // ggml_tensor*
  void *weights_ = nullptr;           // ggml_tensor*
  void *output_ = nullptr;            // ggml_tensor*
  void *act_ = nullptr;               // ggml_tensor*, SwiGLU output
  void *down_output_ = nullptr;       // ggml_tensor*, raw down projection

  void *weight_storage_ = nullptr;
  size_t weight_storage_bytes_ = 0;
  int n_experts_ = 0;
  int d_model_ = 0;
  int d_ff_ = 0;
  int topk_ = 0;
  uint32_t gate_type_ = 0;
  uint32_t up_type_ = 0;
  uint32_t down_type_ = 0;
  int n_threads_ = 0;
  bool strided_mode_ = false;
  bool merged_gate_up_ = false;
  bool output_per_expert_ = false;
  size_t expert_stride_ = 0;
  size_t up_expert_stride_ = 0;
  size_t down_expert_stride_ = 0;

  // The compact GGML graph owns one [gate|up|down] slot per selected
  // expert.  Qwen decode commonly selects the same expert in the same slot
  // on adjacent tokens; remember the source slabs so run() does not memcpy
  // an unchanged slot into the compact staging buffer again.
  struct CachedExpertSource {
    const void *gate = nullptr;
    const void *up = nullptr;
    const void *down = nullptr;
    size_t gate_bytes = 0;
    size_t up_bytes = 0;
    size_t down_bytes = 0;
  };
  std::vector<CachedExpertSource> cached_sources_;

  // The worker owns CPU graph execution while the decode thread submits the
  // independent GPU shared-expert path. All request pointers remain valid
  // until wait_strided_async() returns.
  std::mutex async_mutex_;
  std::condition_variable async_cv_;
  std::thread async_worker_;
  AsyncStridedRequest async_request_{};
  bool async_stop_ = false;
  bool async_request_pending_ = false;
  bool async_request_done_ = true;
  bool async_result_ = false;

  // Opt-in direct CPU MoE executor.  It keeps a fixed worker team alive and
  // runs only the dependency phases required by Qwen's routed FFN.  The
  // buffers are allocated when a signature is selected, never in decode.
  struct DirectQ8KBlock {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
  };

  bool direct_ready_ = false;
  int direct_threads_ = 0;
  std::vector<std::thread> direct_workers_;
  // Pack the phase command and its generation into one publication word.
  // Workers must observe a matching (generation, command) pair; publishing
  // them through separate atomics permits a descheduled worker to observe a
  // new generation with the previous command.
  std::atomic<uint64_t> direct_phase_{0};
  std::atomic<bool> direct_armed_{false};
  std::atomic<int> direct_completed_{0};
  std::atomic<bool> direct_finished_{true};
  std::atomic<bool> direct_result_{false};
  std::atomic<bool> direct_stop_{false};
  std::atomic<bool> direct_request_ok_{true};
  std::atomic<int64_t> direct_timing_input_ns_{0};
  std::atomic<int64_t> direct_timing_phase1_start_ns_{0};
  std::atomic<int64_t> direct_timing_phase1_ns_{0};
  std::atomic<int64_t> direct_timing_down_start_ns_{0};
  std::atomic<int64_t> direct_timing_down_ns_{0};
  std::atomic<int64_t> direct_timing_reduce_start_ns_{0};
  std::atomic<int64_t> direct_timing_reduce_ns_{0};
  std::vector<uint8_t> direct_input_q8k_;
  std::vector<uint8_t> direct_down_q8_0_;
  std::vector<float> direct_gate_output_;
  std::vector<float> direct_up_output_;
  std::vector<float> direct_down_output_;
  std::vector<float *> direct_fused_gate_output_ptrs_;
  std::vector<float *> direct_fused_up_output_ptrs_;
  std::vector<float *> direct_fused_down_output_ptrs_;
  std::vector<const uint8_t *> direct_fused_gate_weight_ptrs_;
  std::vector<const uint8_t *> direct_fused_up_weight_ptrs_;
  std::vector<const uint8_t *> direct_fused_down_weight_ptrs_;
  std::vector<const uint8_t *> direct_gate_rows_;
  std::vector<const uint8_t *> direct_up_rows_;
  std::vector<const uint8_t *> direct_down_rows_;
  std::vector<GgmlQwenDirectWorkerSample> direct_gate_samples_;
  std::vector<GgmlQwenDirectWorkerSample> direct_down_samples_;
  std::vector<GgmlQwenDirectWorkerSample> direct_reduce_samples_;
};

// Qwen-only decode graph that mirrors BeeLlama's backend topology for a
// CPU-affinity routed layer.  The graph contains the CUDA router (including
// softmax/top-k) and the host-RAM MUL_MAT_ID expert path in one scheduler
// submission.  The scheduler owns the CUDA->CPU route/input handoff; callers
// do not first run a separate router graph and then a separate CPU graph.
//
// This deliberately consumes the existing packed DDR arena directly.  It is
// not a replacement for GpuTopology's dynamic VRAM cache and is never used
// by non-Qwen families.
struct GgmlQwenDecodeGraphRef {
  const void * router_device = nullptr;
  size_t router_device_bytes = 0;
  const void * router_host = nullptr;
  size_t router_host_bytes = 0;
  uint32_t router_type = 0;
  int64_t router_ne0 = 0;
  int64_t router_ne1 = 0;
  const float * selection_bias = nullptr;
  const void * gate_plane = nullptr;
  const void * up_plane = nullptr;
  const void * down_plane = nullptr;
  uint32_t gate_type = 0;
  uint32_t up_type = 0;
  uint32_t down_type = 0;
  size_t gate_bytes = 0;
  size_t up_bytes = 0;
  size_t down_bytes = 0;
  size_t expert_stride = 0;

  // Optional Qwen shared-expert branch.  When all four weights have a
  // persistent device copy, the scheduler keeps the shared FFN and the
  // routed+shared add in the same graph as the DDR-backed CPU expert path.
  // Leaving this empty preserves the older host-routed/shared-runner
  // fallback for partial or lazy configurations.
  GgmlSharedWeightRef shared_gate;
  GgmlSharedWeightRef shared_up;
  GgmlSharedWeightRef shared_down;
  GgmlSharedWeightRef shared_gate_inp;
  bool shared_has_gate = false;
  float shared_clamp = 0.0f;
};

class GgmlQwenDecodeGraphRunner {
 public:
  static bool compiled_in();
  static GgmlQwenDecodeGraphRunner &instance();
  // Qwen's DDR arenas can use more than one routed-expert quantization
  // signature. Keep a persistent scheduler graph per signature so changing
  // from IQ1_M to IQ2_XXS does not destroy a live scheduler and rebuild it in
  // the middle of decode.
  static GgmlQwenDecodeGraphRunner &instance_for_signature(
      uint32_t gate_type, uint32_t up_type, uint32_t down_type,
      size_t expert_stride);

  bool ensure(int device, int d_model, int n_experts, int topk, int d_ff,
              bool normalize, float weight_scale);
  bool run(const void *device_input, const GgmlQwenDecodeGraphRef &ref,
           float *host_output, const void **device_output = nullptr,
           int32_t *host_ids = nullptr,
           float *host_weights = nullptr);

  void shutdown();
  ~GgmlQwenDecodeGraphRunner();

 private:
  GgmlQwenDecodeGraphRunner() = default;
  GgmlQwenDecodeGraphRunner(const GgmlQwenDecodeGraphRunner &) = delete;
  GgmlQwenDecodeGraphRunner &operator=(const GgmlQwenDecodeGraphRunner &) = delete;

  bool rebuild(const GgmlQwenDecodeGraphRef &ref);
  bool has_signature(uint32_t gate_type, uint32_t up_type,
                    uint32_t down_type, size_t expert_stride) const;
  void free_graph();

  void *cuda_backend_ = nullptr; // borrowed shared CUDA backend
  void *cpu_backend_ = nullptr;  // owned CPU backend
  void *cpu_threadpool_ = nullptr; // ggml_threadpool_t, owned
  int cpu_threads_ = 0;
  void *sched_ = nullptr;        // ggml_backend_sched_t, owned
  void *external_cuda_buf_ = nullptr; // non-owning marker for device pointers
  void *router_ctx_ = nullptr;   // ggml_context*, owned
  void *router_buf_ = nullptr;   // CUDA buffer, owned
  void *router_ = nullptr;       // ggml_tensor*
  void *bias_ = nullptr;         // ggml_tensor*, optional F32 CUDA tensor
  void *expert_ctx_ = nullptr;   // ggml_context*, owned
  void *gate_ = nullptr;         // host tensor view
  void *up_ = nullptr;
  void *down_ = nullptr;
  std::vector<void *> expert_bufs_; // host non-owning buffers
  void *graph_ctx_ = nullptr;    // ggml_context*, owned
  void *graph_ = nullptr;        // ggml_cgraph*
  void *x_ = nullptr;            // CUDA graph input
  void *ids_ = nullptr;           // graph route IDs, host result
  void *weights_ = nullptr;       // graph route weights, host result
  void *output_ = nullptr;       // routed CPU or combined CUDA graph output
  void *shared_gate_ = nullptr;  // optional CUDA shared-expert weights
  void *shared_up_ = nullptr;
  void *shared_down_ = nullptr;
  void *shared_gate_inp_ = nullptr;
  bool shared_has_gate_ = false;
  float shared_clamp_ = 0.0f;
  uint32_t shared_gate_type_ = 0;
  uint32_t shared_up_type_ = 0;
  uint32_t shared_down_type_ = 0;
  uint32_t shared_gate_inp_type_ = 0;
  bool shared_graph_ = false;
  bool router_external_ = false;
  int device_ = -1;
  int d_model_ = 0;
  int n_experts_ = 0;
  int topk_ = 0;
  int d_ff_ = 0;
  bool normalize_ = false;
  float weight_scale_ = 0.0f;
  uint32_t gate_type_ = 0;
  uint32_t up_type_ = 0;
  uint32_t down_type_ = 0;
  size_t expert_stride_ = 0;
  bool ready_ = false;
};

// Qwen-only GPU top-k weighting/reduction for CPU-affinity layers.  The
// runner is intentionally separate from GgmlQwenSharedRunner: its output is
// a normal GGML CUDA tensor, so the existing shared-expert graph can consume
// it directly on the same backend stream without a device-to-host round trip.
// Other model families never call this class.
class GgmlQwenDeviceCombineRunner {
 public:
  static bool compiled_in();
  static GgmlQwenDeviceCombineRunner &instance();

  bool ensure(int device, int d_model, int topk);
  bool run(const float *host_expert_output, const float *host_weights,
           const void **device_output);

  void shutdown();
  ~GgmlQwenDeviceCombineRunner();

 private:
  GgmlQwenDeviceCombineRunner() = default;
  GgmlQwenDeviceCombineRunner(const GgmlQwenDeviceCombineRunner &) = delete;
  GgmlQwenDeviceCombineRunner &operator=(
      const GgmlQwenDeviceCombineRunner &) = delete;

  static constexpr int kStageBuffers = 4;

  void *backend_ = nullptr;  // borrowed from ggml_shared_backend()
  void *graph_ctx_ = nullptr; // ggml_context*
  void *galloc_ = nullptr;    // ggml_gallocr_t
  void *graph_ = nullptr;     // ggml_cgraph*
  void *expert_output_ = nullptr; // ggml_tensor* [D,topk]
  void *weights_ = nullptr;       // ggml_tensor* [1,topk]
  void *output_ = nullptr;        // ggml_tensor* [D,1]
  std::vector<void *> host_stage_;
  std::vector<void *> weight_stage_;
  std::vector<void *> stage_events_; // ggml_backend_event_t
  std::vector<uint8_t> stage_used_;
  int device_ = -1;
  int d_model_ = 0;
  int topk_ = 0;
  int stage_next_ = 0;
  bool graph_ready_ = false;
};
