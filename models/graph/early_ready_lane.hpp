#pragma once

// CPU lane that runs a layer's cold experts concurrently with its GPU work.
//
// The layer forks (activation, routed ids, router weights) to private pinned
// buffers from inside the CUDA stream, then the patched scheduler runs the
// lane's private CPU graph as soon as that fork event fires, instead of after
// the layer's hot GPU matmuls. The cold result is uploaded on a side stream and
// the merge waits on it. Private storage means GPU scratch reuse cannot race
// with the synchronous cold graph.
//
// Owned by the graph builder of whichever model uses it (Qwen4Exp Option-B,
// GLM5NEXT SSD offload). Optional extras (next-layer prediction, VRAM staging)
// are Qwen-only and left null elsewhere.
#include "cuda_raii.hpp"
#include "graph_config.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "memory_guard.hpp"
#include <cuda_runtime.h>
#include "qwen4exp_stage_pool.hpp"
#include "moe_routed_ggml.hpp"
#include "pilot_ridge.hpp"
#include "cold_moe_strategy.hpp"  // the lane's strategy view (adapter-first migration)
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <vector>

extern "C" int ggml_backend_sched_early_cpu_supported() __attribute__((weak));

// Optional learned correction for the next-layer pilot (bench/qwen_ridge_export.py):
// scores = softmax(router x) + (x - mu) R + c. Loaded once from
// MOESTREAM_QWEN_EARLY_READY_RIDGE; a missing/invalid file leaves the plain pilot.
// The table format and loader are shared with GLM5NEXT via PilotRidgeTable.
inline const PilotRidgeTable &qwen4exp_ridge() {
  static const PilotRidgeTable table =
      PilotRidgeTable::load("MOESTREAM_QWEN_EARLY_READY_RIDGE", "QWEN-RIDGE");
  return table;
}

// One second CUDA backend (own stream) and one VRAM ring of `slots` experts per plane, shared by the GPU-assist
// graphs of every layer of a model: layers run one after another, so they can reuse the same slots.
struct GpuAssistShared {
  ggml_backend_t be = nullptr;
  ggml_backend_buffer_t ring[3] = {nullptr, nullptr, nullptr};
  size_t plane_cap[3] = {0, 0, 0};
  int slots = 0;
  ~GpuAssistShared() {
    for (auto b : ring) if (b) ggml_backend_buffer_free(b);
    if (be) ggml_backend_free(be);
  }
};

// MOESTREAM_GLM5NEXT_ROUTE_TRACE=<file>: "P <pos>" at the start of each decode token, then "R <layer> <ids...>" per layer.
inline FILE *route_trace_file() {
  static FILE *f = [] {
    const char *path = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_ROUTE_TRACE");
    return path && path[0] ? std::fopen(path, "a") : nullptr;
  }();
  return f;
}

struct EarlyReadyLane {
  static constexpr uint64_t kMagic = 0x514541524c593133ull;
  uint64_t magic = kMagic;
  ggml_context *ctx = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_backend_t cpu = nullptr;
  ggml_backend_buffer_t scratch = nullptr, pinned_buffer = nullptr;
  ggml_backend_buffer_t device_buffer = nullptr;
  ggml_backend_buffer_t output_device_buffer = nullptr;
  ggml_backend_buffer_t tables_device_buffer = nullptr;
  ggml_backend_buffer_t prediction_buffer = nullptr;
  ggml_backend_buffer_t ridge_buffer = nullptr;
  ggml_backend_buffer_t prediction_marker_buffer = nullptr;
  ggml_tensor *output = nullptr, *gpu_input = nullptr;
  ggml_tensor *gpu_output = nullptr, *cpu_marker = nullptr;
  ggml_tensor *gpu_tables[4] = {};
  ggml_tensor *prediction = nullptr;
  ggml_tensor *prediction_marker = nullptr;
  int prediction_layer = -1;
  size_t expert_bytes = 0;
  // RAII owners (released after the destructor body, i.e. after the ggml buffers
  // that wrap `pinned`, on the device they were created on).
  moestream::cuda::PinnedBuffer<uint8_t> pinned;
  float *input = nullptr, *weights = nullptr;
  moestream::cuda::PinnedBuffer<uint8_t> prediction_pinned;
  moestream::cuda::PinnedBuffer<uint8_t> stage_pinned;
  int32_t *prediction_ids = nullptr, *stage_ids = nullptr;
  unsigned long long *prediction_epoch = nullptr;
  moestream::cuda::CudaEvent prediction_ready;
  Qwen4ExpStagePool *stage_source = nullptr, *stage_target = nullptr;
  ggml_backend_buffer_t stage_routes_buffer = nullptr;
  // stage_gpu_ids contains valid dispatch IDs for every slot (inactive slots
  // use expert 0 and are zeroed by stage_gpu_mask). stage_gpu_slots retains
  // the selected staging slot or -1 for host bookkeeping and verification.
  ggml_tensor *stage_gpu_ids = nullptr, *stage_gpu_slots = nullptr,
              *stage_gpu_mask = nullptr;
  int32_t *cpu_remap = nullptr;
  float *cpu_mask = nullptr;
  std::vector<int32_t> original_remap;
  std::vector<float> original_mask;
  uint64_t staged_routes = 0;
  int32_t *ids = nullptr, *position = nullptr;
  moestream::cuda::CudaEvent ready;
  cudaStream_t producer_stream = nullptr;
  moestream::cuda::CudaStream output_stream;
  moestream::cuda::CudaEvent output_done;
  int device = -1, d = 0, k = 0, n_experts = 0, expected_pos = -1;
  int trace_layer = -1;  // layer index for MOESTREAM_GLM5NEXT_ROUTE_TRACE lines
  bool completed = false;
  bool residency_ready = false;  // residency complete (owned experts resident, safe to compute) —
                                 // distinct from `completed`, which marks run() finished (§3.8)
  bool skip_all_hot = false;
  bool async_output_copy = false;
  bool resident_tables = false;
  std::vector<uint8_t> cold_mask;
  // Buffers owned on behalf of the builder (e.g. host copies of routing tables).
  std::vector<ggml_backend_buffer_t> extra_buffers;
  size_t reservation = 0;
  uint64_t calls = 0, early_calls = 0, gpu_pending = 0;  // `calls` is the per-token generation (§3.8)
  uint64_t all_hot_visits = 0, all_hot_skips = 0;
  uint64_t output_uploads = 0;
  int64_t wait_us = 0, compute_us = 0;
  int64_t begin_us = 0, p1_us = 0, p2_us = 0, plain_us = 0; // lane breakdown: begin_load, phase-1 graph, phase-2 graph, unsplit graph

  ~EarlyReadyLane() {
    if (device >= 0) cudaSetDevice(device);
    if (output_device_buffer) ggml_backend_buffer_free(output_device_buffer);
    if (tables_device_buffer) ggml_backend_buffer_free(tables_device_buffer);
    if (prediction_buffer) ggml_backend_buffer_free(prediction_buffer);
    if (ridge_buffer) ggml_backend_buffer_free(ridge_buffer);
    if (prediction_marker_buffer) ggml_backend_buffer_free(prediction_marker_buffer);
    if (stage_routes_buffer) ggml_backend_buffer_free(stage_routes_buffer);
    if (device_buffer) ggml_backend_buffer_free(device_buffer);
    if (scratch) ggml_backend_buffer_free(scratch);
    if (pinned_buffer) ggml_backend_buffer_free(pinned_buffer);
    if (ctx) ggml_free(ctx);
    for (auto *b : extra_buffers) ggml_backend_buffer_free(b);
    if (reservation) MemoryGuard::release_host_bytes(reservation);
  }

  static ggml_tensor *alias(ggml_context *ctx, ggml_tensor *src) {
    auto *t = ggml_dup_tensor(ctx, src);
    std::memcpy(t->nb, src->nb, sizeof(t->nb));
    t->data = src->data;
    t->buffer = src->buffer;
    return t;
  }

  // Separate gate/up/down expert tensors (Qwen4Exp).
  bool prepare(ggml_backend_t backend, int dev, int dim, int topk,
               ggml_tensor *gate, ggml_tensor *up, ggml_tensor *down,
               ggml_tensor *remap, ggml_tensor *mask) {
    moe_routed::ExpertWeights w;
    w.gate = gate;
    w.up = up;
    w.down = down;
    return prepare(backend, dev, dim, topk, w, remap, mask);
  }

  // Called on the CPU lane right before its graph runs with the routed experts
  // this lane owns (mask != 0, deduplicated, in route order). Return false to
  // fail the token. Used by the SSD-offload path to make them resident.
  std::function<bool(const int32_t *experts, int n)> before_compute;
  // Overlapped variant of before_compute for the SSD-offload path. begin_load
  // starts making the owned experts resident (in the background) and returns the
  // ones that are not ready yet in `pending`; finish_load blocks until they are.
  // When some owned experts are already resident the lane computes those first,
  // while the missing ones are still being read, and the rest afterwards. Each
  // route is independent, so the two partial outputs add up exactly.
  std::function<bool(const int32_t *experts, int n, std::vector<int32_t> &pending)> begin_load;
  std::function<bool()> finish_load;
  // GPU assist (decode only): up to `assist_slots` of the layer's resident cold experts are copied to a
  // private VRAM ring and computed on a second CUDA backend (own stream) while the CPU graph runs the
  // rest. The copy reads the expert straight from the host arena; `expert_planes` returns its three plane
  // pointers. Results differ from the CPU kernels only by float rounding.
  std::function<bool(int expert, const void *planes[3])> expert_planes;
  // The strategy view of this lane (adapter-first migration). EarlyReadyLaneStrategy forwards the
  // adapter's callbacks to the members above and re-expresses `residency_ready` (residency complete)
  // as the strategy state machine. It owns no CUDA resources and is destroyed (trivially) with the lane.
  std::unique_ptr<moestream::graph::ColdMoEStrategy> strategy;
  uint64_t assist_runs = 0, assist_experts = 0;
  int64_t assist_stall_us = 0;
  uint64_t split_runs = 0;
  int64_t split_stall_us = 0;  // time finish_load still blocked after phase 1
  std::vector<float> phase1_output;
  struct Assist {
    std::shared_ptr<GpuAssistShared> keep;  // keeps the shared backend and ring alive
    ggml_backend_t be = nullptr;
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *x = nullptr, *ids = nullptr, *w = nullptr, *out = nullptr;
    ggml_tensor *ring[3] = {nullptr, nullptr, nullptr};
    ggml_cgraph *graph = nullptr;
    int slots = 0;
    size_t plane[3] = {0, 0, 0};
    std::vector<float> result;
    ~Assist() {
      if (buf) ggml_backend_buffer_free(buf);
      if (ctx) ggml_free(ctx);
    }
  } assists[2];  // [0] primary assist device, [1] optional second device (own ring and link)
  Assist &assist = assists[0];
  // SwiGLU clamp applied by the private graph (0 = none); set before prepare().
  float swiglu_clamp = 0.0f;
  bool swiglu_clamp_fused = false;

  bool prepare(ggml_backend_t backend, int dev, int dim, int topk,
               const moe_routed::ExpertWeights &w,
               ggml_tensor *remap, ggml_tensor *mask) {
    ggml_tensor *gate = w.gate_up ? w.gate_up : w.gate;
    ggml_tensor *up = w.gate_up ? nullptr : w.up;
    ggml_tensor *down = w.down;
    cpu = backend; device = dev; d = dim; k = topk;
    n_experts = gate->ne[2];
    cpu_remap = static_cast<int32_t *>(remap->data);
    cpu_mask = static_cast<float *>(mask->data);
    original_remap.assign(cpu_remap, cpu_remap + n_experts);
    original_mask.assign(cpu_mask, cpu_mask + n_experts);
    expert_bytes = ggml_row_size(gate->type, gate->ne[0]) * gate->ne[1] +
                   (up ? ggml_row_size(up->type, up->ne[0]) * up->ne[1] : 0) +
                   ggml_row_size(down->type, down->ne[0]) * down->ne[1];
    const char *skip = moestream::graphcfg::env("MOESTREAM_QWEN_EARLY_READY_SKIP_ALL_HOT");
    skip_all_hot = skip && skip[0] == '1';
    const char *copy = moestream::graphcfg::env("MOESTREAM_QWEN_EARLY_READY_ASYNC_OUTPUT");
    // Accepted production profile (early-ready + immutable resident route tables +
    // asynchronous cold-output upload): on unless explicitly disabled with =0.
    async_output_copy = !(copy && copy[0] == '0');
    const char *tables = moestream::graphcfg::env("MOESTREAM_QWEN_EARLY_READY_RESIDENT_TABLES");
    resident_tables = !(tables && tables[0] == '0');
    cold_mask.resize(n_experts);
    for (int e = 0; e < n_experts; ++e)
      cold_mask[e] = static_cast<const float *>(mask->data)[e] != 0.0f;
    // Conservative admission for metadata, pins and small FFN intermediates.
    const size_t reserve = 2 * 1024 * 1024;
    if (!MemoryGuard::try_reserve_host_bytes(reserve)) return false;
    reservation = reserve;
    const size_t meta = 256 * ggml_tensor_overhead() +
                        ggml_graph_overhead_custom(256, false);
    ctx = ggml_init({meta, nullptr, true});
    if (!ctx) return false;
    const auto align = [](size_t n) { return (n + 63) & ~size_t(63); };
    const size_t id_off = align(d * sizeof(float));
    const size_t weight_off = id_off + align(k * sizeof(int32_t));
    const size_t pos_off = weight_off + align(k * sizeof(float));
    const size_t out_off = pos_off + 64;
    const size_t pin_bytes = out_off + align(d * k * sizeof(float));
    cudaSetDevice(device);
    if (pinned.allocate(pin_bytes, device) != cudaSuccess ||
        ready.create(cudaEventDisableTiming | cudaEventBlockingSync, device) !=
            cudaSuccess)
      return false;
    std::memset(pinned.get(), 0, pin_bytes);
    auto *base = pinned.get();
    input = reinterpret_cast<float *>(base);
    ids = reinterpret_cast<int32_t *>(base + id_off);
    weights = reinterpret_cast<float *>(base + weight_off);
    position = reinterpret_cast<int32_t *>(base + pos_off);
    pinned_buffer = ggml_backend_cpu_buffer_from_ptr(pinned.get(), pin_bytes);
    if (!pinned_buffer) return false;
    auto *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, 1, 1);
    auto *route = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, 1);
    auto *prob = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, k, 1);
    x->data = input; x->buffer = pinned_buffer;
    route->data = ids; route->buffer = pinned_buffer;
    prob->data = weights; prob->buffer = pinned_buffer;
    auto *cold_ids = ggml_reshape_2d(ctx,
        ggml_get_rows(ctx, alias(ctx, remap), route), k, 1);
    auto *cold_weights = ggml_mul(ctx, prob,
        ggml_get_rows(ctx, alias(ctx, mask), route));
    moe_routed::ExpertWeights private_w;
    if (w.gate_up) {
      private_w.gate_up = alias(ctx, w.gate_up);
    } else {
      private_w.gate = alias(ctx, w.gate);
      private_w.up = alias(ctx, w.up);
    }
    private_w.down = alias(ctx, down);
    const moe_routed::FfnNodes ffn =
        moe_routed::expert_ffn(ctx, private_w, x, cold_ids, swiglu_clamp,
                                swiglu_clamp_fused);
    output = ggml_mul(ctx, ffn.down, cold_weights);
    output->data = base + out_off; output->buffer = pinned_buffer;
    graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, output);
    scratch = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    return scratch && meta + pin_bytes + ggml_backend_buffer_get_size(scratch)
                                     <= reservation;
  }

  // Builds the private GPU graph for up to shared->slots cold experts. Separate gate/up/down tensors only.
  bool prepare_gpu_assist(const moe_routed::ExpertWeights &w, std::shared_ptr<GpuAssistShared> shared, int which = 0) {
    if (!shared || !shared->be || w.gate_up || !w.gate || !w.up || !w.down || which < 0 || which > 1) return false;
    Assist &assist = assists[which];
    assist.keep = shared;
    assist.be = shared->be;
    assist.slots = shared->slots;
    const size_t meta = 128 * ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false);
    assist.ctx = ggml_init({meta, nullptr, true});
    if (!assist.ctx) return false;
    ggml_tensor *src[3] = {w.gate, w.up, w.down};
    moe_routed::ExpertWeights rw;
    for (int i = 0; i < 3; ++i) {
      assist.ring[i] = ggml_new_tensor_3d(assist.ctx, src[i]->type, src[i]->ne[0], src[i]->ne[1], assist.slots);
      if (!assist.ring[i] || assist.ring[i]->nb[2] != src[i]->nb[2] ||
          ggml_nbytes(assist.ring[i]) > shared->plane_cap[i] * static_cast<size_t>(assist.slots) ||
          ggml_backend_tensor_alloc(shared->ring[i], assist.ring[i],
                                    ggml_backend_buffer_get_base(shared->ring[i])) != GGML_STATUS_SUCCESS)
        return false;
      assist.plane[i] = assist.ring[i]->nb[2];
    }
    rw.gate = assist.ring[0]; rw.up = assist.ring[1]; rw.down = assist.ring[2];
    assist.x = ggml_new_tensor_3d(assist.ctx, GGML_TYPE_F32, d, 1, 1);
    assist.ids = ggml_new_tensor_2d(assist.ctx, GGML_TYPE_I32, k, 1);
    assist.w = ggml_new_tensor_3d(assist.ctx, GGML_TYPE_F32, 1, k, 1);
    const moe_routed::FfnNodes ffn = moe_routed::expert_ffn(assist.ctx, rw, assist.x, assist.ids,
                                                           swiglu_clamp, swiglu_clamp_fused);
    assist.out = ggml_mul(assist.ctx, ffn.down, assist.w);
    assist.graph = ggml_new_graph_custom(assist.ctx, 128, false);
    ggml_build_forward_expand(assist.graph, assist.out);
    assist.buf = ggml_backend_alloc_ctx_tensors(assist.ctx, assist.be);
    assist.result.assign(static_cast<size_t>(d) * k, 0.0f);
    return assist.buf != nullptr;
  }

  bool prepare_gpu_output(ggml_backend_t backend, ggml_context *outer_ctx) {
    gpu_output = ggml_dup_tensor(outer_ctx, output);
    output_device_buffer = ggml_backend_alloc_buffer(backend, ggml_nbytes(gpu_output));
    return output_device_buffer &&
        ggml_backend_tensor_alloc(output_device_buffer, gpu_output,
            ggml_backend_buffer_get_base(output_device_buffer)) == GGML_STATUS_SUCCESS &&
        output_stream.create(cudaStreamNonBlocking, device) == cudaSuccess &&
        output_done.create(cudaEventDisableTiming, device) == cudaSuccess;
  }

  bool prepare_gpu_tables(ggml_backend_t backend, ggml_context *outer_ctx,
                          ggml_tensor *remap_gpu, ggml_tensor *remap_cpu,
                          ggml_tensor *mask_gpu, ggml_tensor *mask_cpu) {
    ggml_tensor *sources[] = {remap_gpu, remap_cpu, mask_gpu, mask_cpu};
    const size_t alignment = ggml_backend_buft_get_alignment(
        ggml_backend_get_default_buffer_type(backend));
    size_t offsets[4], bytes = 0;
    for (int i = 0; i < 4; ++i) {
      offsets[i] = bytes;
      gpu_tables[i] = ggml_dup_tensor(outer_ctx, sources[i]);
      bytes += (ggml_nbytes(gpu_tables[i]) + alignment - 1) / alignment * alignment;
    }
    tables_device_buffer = ggml_backend_alloc_buffer(backend, bytes);
    if (!tables_device_buffer) return false;
    auto *base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(tables_device_buffer));
    for (int i = 0; i < 4; ++i) {
      if (ggml_backend_tensor_alloc(tables_device_buffer, gpu_tables[i],
                                   base + offsets[i]) != GGML_STATUS_SUCCESS)
        return false;
      // Immutable lookup tables are uploaded once, rather than copied as
      // user inputs before every backend split on every decoded token.
      ggml_backend_tensor_set(gpu_tables[i], sources[i]->data, 0, ggml_nbytes(sources[i]));
    }
    return true;
  }

  bool prepare_prediction(ggml_backend_t backend, ggml_context *outer_ctx,
                          ggml_tensor *router, ggml_tensor *bias, int target) {
    auto *probs = ggml_soft_max(outer_ctx, ggml_mul_mat(outer_ctx, router, gpu_input));
    ggml_tensor *selection = probs;
    const auto &ridge = qwen4exp_ridge();
    if (const auto *rec = ridge.find(target); rec && ridge.d == int(gpu_input->ne[0]) &&
                                                ridge.e == int(router->ne[1])) {
      auto *mu_t = ggml_new_tensor_2d(outer_ctx, GGML_TYPE_F32, ridge.d, 1);
      auto *r_t = ggml_new_tensor_2d(outer_ctx, GGML_TYPE_F16, ridge.d, ridge.e);
      auto *c_t = ggml_new_tensor_2d(outer_ctx, GGML_TYPE_F32, ridge.e, 1);
      ggml_tensor *parts[3] = {mu_t, r_t, c_t};
      size_t offsets[3], total = 0;
      for (int i = 0; i < 3; ++i) {
        offsets[i] = total;
        total += (ggml_nbytes(parts[i]) + 255) & ~size_t(255);
      }
      ridge_buffer = ggml_backend_alloc_buffer(backend, total);
      if (!ridge_buffer) return false;
      auto *base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(ridge_buffer));
      for (int i = 0; i < 3; ++i)
        if (ggml_backend_tensor_alloc(ridge_buffer, parts[i], base + offsets[i]) != GGML_STATUS_SUCCESS)
          return false;
      ggml_backend_tensor_set(mu_t, rec->mu.data(), 0, ggml_nbytes(mu_t));
      ggml_backend_tensor_set(r_t, rec->r.data(), 0, ggml_nbytes(r_t));
      ggml_backend_tensor_set(c_t, rec->c.data(), 0, ggml_nbytes(c_t));
      auto *corr = ggml_mul_mat(outer_ctx, r_t, ggml_sub(outer_ctx, gpu_input, mu_t));
      selection = ggml_add(outer_ctx, ggml_add(outer_ctx, probs, corr), c_t);
    }
    if (bias) selection = ggml_add(outer_ctx, selection, bias);
    prediction = ggml_cont(outer_ctx, ggml_argsort_top_k(outer_ctx, selection, k));
    prediction_layer = target;
    prediction_buffer = ggml_backend_alloc_buffer(backend, ggml_nbytes(prediction));
    return prediction_buffer &&
        ggml_backend_tensor_alloc(prediction_buffer, prediction,
            ggml_backend_buffer_get_base(prediction_buffer)) == GGML_STATUS_SUCCESS;
  }

  bool prepare_prediction_readback(ggml_backend_t backend, ggml_context *outer_ctx) {
    if (prediction_pinned.allocate(128 + k * sizeof(int32_t), device) != cudaSuccess ||
        prediction_ready.create(cudaEventDisableTiming | cudaEventBlockingSync,
                                device) != cudaSuccess) return false;
    prediction_ids = reinterpret_cast<int32_t *>(prediction_pinned.get());
    prediction_epoch = reinterpret_cast<unsigned long long *>(
        prediction_pinned.get() + 64);
    std::memset(prediction_pinned.get(), 0, 128 + k * sizeof(int32_t));
    auto *destination = ggml_dup_tensor(outer_ctx, prediction);
    prediction_marker_buffer = ggml_backend_alloc_buffer(backend, ggml_nbytes(destination));
    if (!prediction_marker_buffer || ggml_backend_tensor_alloc(prediction_marker_buffer,
        destination, ggml_backend_buffer_get_base(prediction_marker_buffer)) != GGML_STATUS_SUCCESS)
      return false;
    prediction_marker = ggml_cpy(outer_ctx, prediction, destination);
    prediction_marker->extra = this;
    return prediction_marker != nullptr;
  }

  bool prepare_stage_routes(ggml_backend_t backend, ggml_context *outer_ctx) {
    stage_gpu_ids = ggml_new_tensor_2d(outer_ctx, GGML_TYPE_I32, k, 1);
    stage_gpu_slots = ggml_new_tensor_2d(outer_ctx, GGML_TYPE_I32, k, 1);
    stage_gpu_mask = ggml_new_tensor_3d(outer_ctx, GGML_TYPE_F32, 1, k, 1);
    stage_routes_buffer = ggml_backend_alloc_buffer(backend, 512);
    if (!stage_routes_buffer) return false;
    auto *base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(stage_routes_buffer));
    if (ggml_backend_tensor_alloc(stage_routes_buffer, stage_gpu_ids, base) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(stage_routes_buffer, stage_gpu_slots, base + 128) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(stage_routes_buffer, stage_gpu_mask, base + 256) != GGML_STATUS_SUCCESS)
      return false;
    // Prediction storage is otherwise unused on the target; keep the mapped
    // IDs in a separate cache line from producer prediction/epoch snapshots.
    if (stage_pinned.allocate(k * sizeof(int32_t), device) != cudaSuccess) return false;
    stage_ids = reinterpret_cast<int32_t *>(stage_pinned.get());
    return true;
  }

  static cudaError_t record_external(cudaEvent_t event, cudaStream_t stream) {
    cudaStreamCaptureStatus status;
    const auto result = cudaStreamIsCapturing(stream, &status);
    if (result != cudaSuccess) return result;
    return status == cudaStreamCaptureStatusNone ? cudaEventRecord(event, stream) :
        cudaEventRecordWithFlags(event, stream, cudaEventRecordExternal);
  }

  bool enqueue(const ggml_tensor *node, cudaStream_t stream) {
    if (node->type == GGML_TYPE_I32 && stage_source) {
      GGML_ASSERT(cudaMemcpyAsync(prediction_ids, node->src[0]->data, k * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream) == cudaSuccess);
      GGML_ASSERT(cudaMemcpyAsync(prediction_epoch, stage_source->epoch, sizeof(unsigned long long),
                    cudaMemcpyDeviceToHost, stream) == cudaSuccess);
      GGML_ASSERT(record_external(prediction_ready, stream) == cudaSuccess);
      return true;
    }
    // src[4] (token position) is optional: lanes whose graph has no positions
    // tensor skip the staleness check.
    GGML_ASSERT(node->src[0] && node->src[2] && node->src[3]);
    const auto checked = [](cudaError_t status, const char *operation) {
      if (status != cudaSuccess)
        GGML_ABORT("early-ready %s: %s", operation, cudaGetErrorString(status));
    };
    if (stage_source)
      checked(moestream_qwen_stage_begin(stage_source->epoch, stream), "stage epoch");
    if (stage_target) {
      checked(moestream_qwen_stage_map(static_cast<const int *>(node->src[2]->data),
          stage_target->epoch, stage_target->slot_ids, stage_target->slot_epochs,
          static_cast<int *>(stage_gpu_ids->data),
          static_cast<int *>(stage_gpu_slots->data),
          static_cast<float *>(stage_gpu_mask->data),
          k, stage_target->slots, stream), "stage route map");
      checked(cudaMemcpyAsync(stage_ids, stage_gpu_slots->data, k * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, stream), "stage route readback");
    }
    checked(cudaMemcpyAsync(node->data, node->src[0]->data, d * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream), "input snapshot");
    checked(cudaMemcpyAsync(input, node->data, d * sizeof(float),
                            cudaMemcpyDeviceToHost, stream), "input readback");
    checked(cudaMemcpyAsync(ids, node->src[2]->data, k * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream), "route readback");
    checked(cudaMemcpyAsync(weights, node->src[3]->data, k * sizeof(float),
                            cudaMemcpyDeviceToHost, stream), "weight readback");
    if (node->src[4])
      checked(cudaMemcpyAsync(position, node->src[4]->data, sizeof(int32_t),
                              cudaMemcpyDeviceToHost, stream), "position readback");
    cudaStreamCaptureStatus capture;
    checked(cudaStreamIsCapturing(stream, &capture), "capture query");
    // External records survive capture as host-visible graph event nodes.
    checked(capture == cudaStreamCaptureStatusNone
                ? cudaEventRecord(ready, stream)
                : cudaEventRecordWithFlags(ready, stream, cudaEventRecordExternal),
            "input event record");
    producer_stream = stream;
    return true;
  }

  // §3.8 rule 1: before compute reads an owned expert's planes, the strategy must report Ready.
  // The split-load path is the one exception — phase 1 computes only the already-resident subset
  // with the pending experts masked out — so it asserts Ready after its second finish_load().
  bool strategy_ready() const {
    if (!strategy) return true;
    moestream::graph::ColdMoEInvocation inv;
    inv.layer = trace_layer;
    inv.generation = calls + 1;
    return strategy->state(inv) == moestream::graph::ColdMoEStrategy::State::Ready;
  }

  void run() {
    GGML_ASSERT(!completed);
    const int64_t start = ggml_time_us();
    const cudaError_t status = cudaEventSynchronize(ready);
    if (status != cudaSuccess)
      GGML_ABORT("early-ready input event: %s", cudaGetErrorString(status));
    GGML_ASSERT(expected_pos < 0 || *position == expected_pos);
    // MOESTREAM_GLM5NEXT_ROUTE_TRACE=<file>: one line per (decode token, layer): "pos layer id id ..." (all k routed ids).
    if (FILE *route_trace = route_trace_file()) {
      std::fprintf(route_trace, "R %d", trace_layer);
      for (int i = 0; i < k; ++i) std::fprintf(route_trace, " %d", ids[i]);
      std::fputc('\n', route_trace);
      std::fflush(route_trace);
    }
    const char *stage_no_request_env =
        moestream::graphcfg::env("MOESTREAM_QWEN_EARLY_READY_STAGE_NO_REQUEST");
    const bool stage_no_request =
        stage_no_request_env && stage_no_request_env[0] == '1';
    if (stage_source && !stage_no_request)
      stage_source->request(calls + 1, prediction_ids, prediction_epoch, prediction_ready);
    if (stage_target) {
      std::copy(original_remap.begin(), original_remap.end(), cpu_remap);
      std::copy(original_mask.begin(), original_mask.end(), cpu_mask);
      bool staged_ready = false;
      for (int i = 0; i < k; ++i) {
        GGML_ASSERT(ids[i] >= 0 && ids[i] < n_experts);
        GGML_ASSERT(stage_ids[i] >= -1 && stage_ids[i] < stage_target->slots);
        if (stage_ids[i] >= 0) {
          GGML_ASSERT(cold_mask[ids[i]]);
          cpu_remap[ids[i]] = -1;
          cpu_mask[ids[i]] = 0.0f;
          staged_ready = true;
          ++staged_routes;
        }
      }
      if (staged_ready)
        GGML_ASSERT(stage_target->wait_for_epoch(calls + 1, producer_stream));
    }
    int cold_count = 0;
    for (int i = 0; i < k; ++i) {
      GGML_ASSERT(ids[i] >= 0 && ids[i] < n_experts);
      GGML_ASSERT(std::isfinite(weights[i]));
      cold_count += cpu_mask[ids[i]] != 0.0f;
    }
    const cudaError_t tail = cudaStreamQuery(producer_stream);
    GGML_ASSERT(tail == cudaSuccess || tail == cudaErrorNotReady);
    gpu_pending += tail == cudaErrorNotReady;
    int32_t owned[64];
    int n_owned = 0;
    if ((before_compute || begin_load) && cold_count > 0) {
      for (int i = 0; i < k && n_owned < 64; ++i) {
        if (cpu_mask[ids[i]] == 0.0f) continue;
        bool seen = false;
        for (int j = 0; j < n_owned; ++j) seen = seen || owned[j] == ids[i];
        if (!seen) owned[n_owned++] = ids[i];
      }
    }
    bool two_phase = false;
    std::vector<int32_t> pending;
    if (begin_load && n_owned > 0) {
      const int64_t b0 = ggml_time_us();
      GGML_ASSERT(begin_load(owned, n_owned, pending));
      begin_us += ggml_time_us() - b0;
      two_phase = !pending.empty() && static_cast<int>(pending.size()) < n_owned;
      if (!two_phase) GGML_ASSERT(finish_load());
    } else if (before_compute && n_owned > 0) {
      GGML_ASSERT(before_compute(owned, n_owned));
    }
    // Residency is complete (every owned expert resident) unless the split-load path still has
    // experts pending; the two-phase branch flips `residency_ready` after its second finish_load().
    if (!two_phase) {
      residency_ready = true;
      GGML_ASSERT(strategy_ready());
    }
    // GPU assist: hand some of the experts that are ready now to the assist backend(s). Device 0 takes up to
    // min(slots, ready/2); a second device (own ring and link) takes more, as long as one expert stays on the CPU.
    int32_t assisted[2][8];
    int n_assisted[2] = {0, 0};
    if (expert_planes && n_owned > 1 && (assists[0].graph || assists[1].graph)) {
      const auto is_pending = [&](int32_t e) {
        for (int32_t p : pending) if (p == e) return true;
        return false;
      };
      int n_ready = 0;
      for (int j = 0; j < n_owned; ++j) n_ready += !is_pending(owned[j]);
      int cursor = n_owned - 1;
      int taken = 0;
      for (int a = 0; a < 2; ++a) {
        Assist &A = assists[a];
        if (!A.graph) continue;
        // The second device's link is slower, so it only helps while the CPU still has about as much to do as that copy
        // takes (MOESTREAM_GLM5NEXT_ASSIST2_KEEP experts stay on the CPU, default 1).
        static const int keep = std::max(1, moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_ASSIST2_KEEP", 1));
        const int want = a == 0 ? std::min({A.slots, n_ready / 2, 8})
                                : std::min({A.slots, std::max(0, n_ready - taken - keep), 8});
        for (; cursor >= 0 && n_assisted[a] < want; --cursor) {
          if (is_pending(owned[cursor])) continue;
          const void *planes[3];
          if (!expert_planes(owned[cursor], planes)) continue;
          for (int pl = 0; pl < 3; ++pl)
            ggml_backend_tensor_set_async(A.be, A.ring[pl], planes[pl],
                                          static_cast<size_t>(n_assisted[a]) * A.plane[pl], A.plane[pl]);
          assisted[a][n_assisted[a]++] = owned[cursor];
          ++taken;
        }
        if (n_assisted[a] > 0) {
          int32_t slot_ids[64];
          float slot_w[64];
          for (int i = 0; i < k; ++i) {
            slot_ids[i] = -1;
            slot_w[i] = 0.0f;
            for (int j = 0; j < n_assisted[a]; ++j)
              if (assisted[a][j] == ids[i]) {
                slot_ids[i] = j;
                slot_w[i] = weights[i] * original_mask[ids[i]];
              }
          }
          ggml_backend_tensor_set_async(A.be, A.x, input, 0, static_cast<size_t>(d) * sizeof(float));
          ggml_backend_tensor_set_async(A.be, A.ids, slot_ids, 0, k * sizeof(int32_t));
          ggml_backend_tensor_set_async(A.be, A.w, slot_w, 0, k * sizeof(float));
          GGML_ASSERT(ggml_backend_graph_compute_async(A.be, A.graph) == GGML_STATUS_SUCCESS);
          // The CPU graphs below must not compute these experts.
          for (int j = 0; j < n_assisted[a]; ++j) {
            cpu_remap[assisted[a][j]] = -1;
            cpu_mask[assisted[a][j]] = 0.0f;
          }
        }
      }
    }
    const int64_t copied = ggml_time_us();
    all_hot_visits += cold_count == 0;
    if (skip_all_hot && cold_count == 0) {
      // Negative IDs produce positive zero in the original cold branch.
      // Keep the existing H2D copy and merge consuming the same output shape.
      std::memset(output->data, 0, ggml_nbytes(output));
      ++all_hot_skips;
    } else if (two_phase) {
      // Phase 1: owned experts that are already resident, while the rest is read.
      const auto is_pending = [&](int32_t e) {
        for (int32_t p : pending) if (p == e) return true;
        return false;
      };
      for (int j = 0; j < n_owned; ++j)
        if (is_pending(owned[j])) { cpu_remap[owned[j]] = -1; cpu_mask[owned[j]] = 0.0f; }
      const int64_t g1 = ggml_time_us();
      GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
      p1_us += ggml_time_us() - g1;
      phase1_output.assign(static_cast<float *>(output->data),
                           static_cast<float *>(output->data) + static_cast<size_t>(d) * k);
      const int64_t wait0 = ggml_time_us();
      GGML_ASSERT(finish_load());
      residency_ready = true;  // all owned experts resident after the second-phase load
      GGML_ASSERT(strategy_ready());
      split_stall_us += ggml_time_us() - wait0;
      // Phase 2: the experts that had to be loaded.
      for (int j = 0; j < n_owned; ++j) {
        const int32_t e = owned[j];
        if (is_pending(e)) {
          cpu_remap[e] = original_remap[e]; cpu_mask[e] = original_mask[e];
        } else {
          cpu_remap[e] = -1; cpu_mask[e] = 0.0f;
        }
      }
      const int64_t g2 = ggml_time_us();
      GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
      p2_us += ggml_time_us() - g2;
      float *out = static_cast<float *>(output->data);
      for (size_t i = 0; i < phase1_output.size(); ++i) out[i] += phase1_output[i];
      for (int j = 0; j < n_owned; ++j) {
        cpu_remap[owned[j]] = original_remap[owned[j]];
        cpu_mask[owned[j]] = original_mask[owned[j]];
      }
      ++split_runs;
    } else {
      const int64_t g0 = ggml_time_us();
      GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
      plain_us += ggml_time_us() - g0;
    }
    {
      int total_assisted = 0;
      for (int a = 0; a < 2; ++a) {
        if (n_assisted[a] == 0) continue;
        Assist &A = assists[a];
        const int64_t a0 = ggml_time_us();
        ggml_backend_synchronize(A.be);
        ggml_backend_tensor_get(A.out, A.result.data(), 0, A.result.size() * sizeof(float));
        assist_stall_us += ggml_time_us() - a0;
        float *out = static_cast<float *>(output->data);
        for (int i = 0; i < k; ++i)
          for (int j = 0; j < n_assisted[a]; ++j)
            if (assisted[a][j] == ids[i])
              std::memcpy(out + static_cast<size_t>(i) * d, A.result.data() + static_cast<size_t>(i) * d,
                          static_cast<size_t>(d) * sizeof(float));
        for (int j = 0; j < n_assisted[a]; ++j) {
          cpu_remap[assisted[a][j]] = original_remap[assisted[a][j]];
          cpu_mask[assisted[a][j]] = original_mask[assisted[a][j]];
        }
        total_assisted += n_assisted[a];
      }
      if (total_assisted > 0) {
        ++assist_runs;
        assist_experts += static_cast<uint64_t>(total_assisted);
      }
    }
    const int64_t done = ggml_time_us();
    for (int i = 0; i < d * k; ++i)
      GGML_ASSERT(std::isfinite(static_cast<float *>(output->data)[i]));
    if (async_output_copy) {
      GGML_ASSERT(cudaMemcpyAsync(gpu_output->data, output->data, ggml_nbytes(output),
          cudaMemcpyHostToDevice, output_stream) == cudaSuccess);
      GGML_ASSERT(cudaEventRecord(output_done, output_stream) == cudaSuccess);
      GGML_ASSERT(cudaStreamWaitEvent(producer_stream, output_done, 0) == cudaSuccess);
      ++output_uploads;
    }
    wait_us += copied - start; compute_us += done - copied;
    ++calls; completed = true;
  }

  static void fallback(ggml_tensor *, const ggml_tensor *, int ith, int,
                       void *userdata) {
    // Never run a nested GGML graph inside the CPU worker team.
    GGML_ASSERT(ith == 0);
    GGML_ABORT("early-ready marker requires the scheduler bridge");
    (void)userdata;
  }

  // Host-node variant: D2H copies are already complete when this is called,
  // so it must only execute the prepared CPU graph. It performs no CUDA calls
  // and leaves the result in the owned pinned `output` buffer.
  bool compute_host_callback() {
    if (!cpu || !graph || !output ||
        ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS)
      return false;
    const size_t n = static_cast<size_t>(ggml_nelements(output));
    const auto *values = static_cast<const float *>(output->data);
    for (size_t i = 0; i < n; ++i)
      if (!std::isfinite(values[i])) return false;
    return true;
  }

  bool compute_host_callback_reduced(float *destination, int width) {
    if (!destination || !compute_host_callback() || output->type != GGML_TYPE_F32 ||
        output->ne[0] != width || output->ne[1] <= 0)
      return false;
    const auto *values = static_cast<const float *>(output->data);
    const size_t row_stride = output->nb[1] / sizeof(float);
    for (int i = 0; i < width; ++i) {
      float sum = 0.0f;
      for (int row = 0; row < output->ne[1]; ++row)
        sum += values[static_cast<size_t>(row) * row_stride + i];
      destination[i] = sum;
    }
    return true;
  }
};

// Scheduler hook (weak symbol in the patched ggml). Defined once, in
// ggml_qwen4exp_graph.cpp: an `inline` definition is not emitted unless the
// TU uses it, which silently disables early-ready.
extern "C" int moestream_ggml_sched_early_cpu(ggml_backend_t backend, ggml_cgraph *graph,
                                              int64_t *wait_us, int64_t *compute_us);
