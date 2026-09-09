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
// existing mmap -> pinned staging fallback supplies the graph.
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
#include <chrono>
#include <functional>
#include <vector>

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
    bool graph_in_flight = false;
    bool ready = false;
  };

  bool build_graph(int layer);
  void free_layer(Layer &layer);

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
  uint32_t type = 0;
  int64_t ne0 = 0;
  int64_t ne1 = 0;
  size_t nbytes = 0;
};

class GgmlQwenSharedRunner {
 public:
  static bool compiled_in();
  static GgmlQwenSharedRunner &instance();

  bool ensure(int device, int d_model, int shared_inter, int n_tokens,
              bool has_gate, float clamp);
  bool ensure_layer(int layer, const GgmlSharedWeightRef &gate,
                   const GgmlSharedWeightRef &up,
                   const GgmlSharedWeightRef &down,
                   const GgmlSharedWeightRef *gate_inp);
  bool run(int layer, const void *device_x, const void *routed_output,
           const void **combined_output);

  void shutdown();
  ~GgmlQwenSharedRunner();

 private:
  GgmlQwenSharedRunner() = default;
  GgmlQwenSharedRunner(const GgmlQwenSharedRunner &) = delete;
  GgmlQwenSharedRunner &operator=(const GgmlQwenSharedRunner &) = delete;

  bool build_graph();
  void free_graph();

  void *backend_ = nullptr; // borrowed from ggml_shared_backend()
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
  uint32_t gate_type_ = 0;
  uint32_t up_type_ = 0;
  uint32_t down_type_ = 0;
  uint32_t gate_inp_type_ = 0;

  void *graph_ctx_ = nullptr;  // ggml_context*
  void *galloc_ = nullptr;     // ggml_gallocr_t
  void *graph_ = nullptr;      // ggml_cgraph*
  void *x_ = nullptr;           // ggml_tensor* [D,S]
  void *routed_ = nullptr;     // ggml_tensor* [D,S]
  void *output_ = nullptr;     // ggml_tensor* [D,S]
  bool graph_ready_ = false;
};

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

// One expert's three destination pointers inside a pinned staging buffer.
struct GgmlMoeStageSlot {
  uint8_t *gate = nullptr;
  uint8_t *up = nullptr;
  uint8_t *down = nullptr;
};

class GgmlMoeRunner {
 public:
  static constexpr int NBUF = 2;  // pinned staging buffers (memcpy || H2D)
  static constexpr int NWEIGHT_BUFS = 1;
  // A prefill can queue all model layers before the final synchronization.
  // Keep one completion event per possible layer so an event still being
  // waited on by the copy backend is never re-recorded.
  static constexpr int NWEIGHT_EVENTS = 64;

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
  bool copy_expert_d2d(int expert_index, const void *src_slab);

  // Wait for every in-flight weight copy (H2D and D2D) on this layer. Safe to
  // call more than once; run_layer() does it internally.
  void sync_copies();

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

  // Copy a result returned by run_layer_device() to a row-major host buffer.
  // Used only for the guarded fallback/debug path when the device consumer
  // cannot accept the result.
  bool copy_device_output_to_host(const void *device_output, float *host_out);

  // Drain and reset the accumulated H2D timing/byte counters.
  void collect_h2d(double &seconds, size_t &bytes);

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

  // Bee-style submission mode: expert H2D copies are queued on the shared
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

  bool ensure_backend(int device);
  bool ensure_weight_buffer(size_t bytes, size_t packed_bytes);
  bool ensure_stage_buffers(size_t bytes_per_buf);
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
  size_t weight_expert_stride_ = 0;
  size_t weight_raw_expert_stride_ = 0;
  size_t weight_tail_padding_ = 0;
  size_t weight_gate_plane_offset_ = 0;
  size_t weight_up_plane_offset_ = 0;
  size_t weight_down_plane_offset_ = 0;
  void *packed_wbuf_[NWEIGHT_BUFS] = {}; // raw CUDA packed transfer buffers
  size_t packed_wbuf_bytes_ = 0;
  uint8_t *packed_base_ = nullptr;
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
  // before the backend's H2D stream consumes them, so retain one stable host
  // copy per possible layer submission. A request has at most 48 layers here;
  // the extra slots also cover a request boundary before the backend is
  // drained, without reintroducing a per-layer device synchronization.
  static constexpr int NROUTE_STAGE = 64;
  std::vector<int32_t> route_ids_stage_[NROUTE_STAGE];
  std::vector<float> route_weights_stage_[NROUTE_STAGE];
  int route_stage_index_ = 0;

  // --- staging (pinned host) ---
  uint8_t *h_stage_[NBUF] = {nullptr, nullptr};
  void *d_stage_slot_map_[NBUF] = {nullptr, nullptr};
  std::vector<int32_t> stage_slot_map_[NBUF];
  size_t stage_bytes_ = 0;       // per buffer
  // cudaMallocHost() staging is real DDR.  Keep its charge separate from
  // stage_bytes_ so a failed/partial setup cannot leave the process-wide host
  // admission ledger thinking these pages are still live (or, worse, allow a
  // later resize to cross host_ram_gb).
  size_t stage_host_reservation_bytes_ = 0;
  int stage_slot_capacity_ = 0;
  void *copy_stream_ = nullptr;  // cudaStream_t
  void *copy_backend_ = nullptr; // ggml_backend_t, backend-stream H2D only
  void *ev_free_[NBUF] = {nullptr, nullptr};
  void *ev_t0_[NBUF] = {nullptr, nullptr};
  // Timing events for backend-stream copies.  The GGML staging event is a
  // completion/fencing event, but wall-clock timing around its host-side
  // wait includes time spent waiting for the buffer to become reusable.  A
  // CUDA event pair measures only the copy-stream interval.
  void *backend_copy_t0_event_[NBUF] = {nullptr, nullptr};
  void *backend_copy_t1_event_[NBUF] = {nullptr, nullptr};
  void *backend_stage_event_[NBUF] = {nullptr, nullptr}; // ggml event
  void *backend_weight_done_event_[NWEIGHT_EVENTS] = {};
  void *backend_weight_last_event_[NWEIGHT_BUFS] = {};
  bool backend_weight_buffer_in_use_[NWEIGHT_BUFS] = {};
  int backend_weight_event_next_ = 0;
  int weight_buffer_index_ = 0;
  bool backend_stream_copies_ = false;
  bool raw_d2d_pending_ = false;
  std::chrono::steady_clock::time_point backend_copy_t0_[NBUF];
  bool pending_[NBUF] = {false, false};
  size_t pending_bytes_[NBUF] = {0, 0};

  struct DeferredVramHold {
    void * topology = nullptr;
    int layer = -1;
    int expert = -1;
    const uint8_t * device_ptr = nullptr;
  };
  std::vector<DeferredVramHold> deferred_vram_holds_;
  std::vector<std::function<void()>> deferred_host_releases_;
  bool copy_batch_started_[NBUF] = {false, false};
  size_t direct_h2d_bytes_[NBUF] = {0, 0};

  // --- current plan ---
  GgmlMoeLayerPlan plan_{};
  size_t gate_exp_bytes_ = 0;
  size_t up_exp_bytes_ = 0;
  size_t down_exp_bytes_ = 0;
  bool layer_ready_ = false;

  double h2d_seconds_ = 0.0;
  size_t h2d_bytes_ = 0;
};
