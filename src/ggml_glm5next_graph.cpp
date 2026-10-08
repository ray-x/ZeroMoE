// GLM-5.3-Flash (glm5next) full prefill graph.
//
// The legacy engine executes one layer at a time and returns the hidden state
// to C++ between layers.  This bridge keeps the complete GLM trunk in one
// GGML graph.  Host backed expert tensors remain valid GGML weight nodes, so
// the CUDA scheduler owns the required copies and dependency edges.

#include "ggml_glm5next_graph.hpp"
#include "ggml_deepseek_v4_graph.hpp"

#include "dense_dispatch.hpp"
#include "direct_io.hpp"
#include "engine.hpp"
#include "expert_cache.hpp"
#include "offload_store.hpp"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "ggml_moe_backend.hpp"
#include "gpu_topology.hpp"
#include "kquant_dequant.hpp"
#include "logger.hpp"
#include "memory_guard.hpp"
#include "mmid_route_skip.hpp"
#include "moe_partial_route.hpp"
#include "moe_partial_route_ggml.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <shared_mutex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(MOESTREAM_USE_GGML_CUDA)

#include <cuda_runtime.h>

namespace {

constexpr int kGraphSize = 60000;
constexpr size_t kTensorContextBytes = 192ull * 1024ull * 1024ull;

struct HostAllocation {
  void *ptr = nullptr;
  size_t bytes = 0;
  bool pinned = false;
  size_t reservation_bytes = 0;
};

struct DynamicUploadCopy {
  const void *source = nullptr;
  void *destination = nullptr;
  size_t bytes = 0;
  size_t staging_offset = 0;
};

struct DynamicUploadJob {
  int layer = -1;
  int hot_slot = -1;
  int incoming_expert = -1;
  int evicted_expert = -1;
  std::vector<DynamicUploadCopy> copies;
};

struct DynamicUploadResult {
  int layer = -1;
  int hot_slot = -1;
  int incoming_expert = -1;
  bool success = false;
  size_t bytes = 0;
  double elapsed_ms = 0.0;
};

struct DynamicUploadState {
  std::mutex mutex;
  std::condition_variable cv;
  std::deque<DynamicUploadJob> queue;
  std::vector<DynamicUploadResult> completed;
  std::thread worker;
  bool stopping = false;
  bool active = false;
  cudaStream_t stream = nullptr;
  void *staging = nullptr;
  size_t staging_capacity = 0;
  size_t host_reservation_bytes = 0;
};

// A graph can keep historically hot experts in a compact CUDA tensor while
// presenting the remaining experts as a compact host tensor.  Device maps
// translate the model's original expert IDs, so routing stays inside GGML.
struct PartialExpertLayer {
  std::vector<int> hot;
  std::vector<int> cold;
  // Physical cache-slot lookup used by the graph's no-copy VRAM hot branch.
  std::vector<int32_t> vram_slot_by_expert;
  uint8_t *vram_layer_base = nullptr;
  size_t vram_layer_slots = 0;
  size_t vram_slot_stride = 0;
  std::vector<int> slot_last_use;
  std::vector<uint32_t> miss_count;
  std::vector<uint8_t> slot_in_flight;
  std::vector<uint8_t> expert_in_flight;
  uint64_t route_count = 0;
  uint64_t hot_route_count = 0;
  ggml_tensor *hot_gate_up = nullptr;
  ggml_tensor *hot_gate = nullptr;
  ggml_tensor *hot_up = nullptr;
  ggml_tensor *hot_down = nullptr;
  ggml_tensor *cold_gate_up = nullptr;
  ggml_tensor *cold_gate = nullptr;
  ggml_tensor *cold_up = nullptr;
  ggml_tensor *cold_down = nullptr;
  ggml_tensor *cold_map = nullptr;
  ggml_tensor *hot_map = nullptr;
  ggml_tensor *cold_mask = nullptr;
  ggml_tensor *hot_mask = nullptr;
  // Second-GPU hot set (MOESTREAM_GLM5NEXT_GPU1_HOT_GIB): experts moved out of `cold` that decode serves from the
  // 5060 Ti. Prefill / no-GPU1 builds fold them back into the cold branch (identity-indexed full arena).
  std::vector<int> hot1;
  ggml_tensor *hot1_gate = nullptr;
  ggml_tensor *hot1_up = nullptr;
  ggml_tensor *hot1_down = nullptr;
  ggml_tensor *hot1_map = nullptr;
  ggml_tensor *hot1_mask = nullptr;
  // -1 = not yet decided for this build_graph() pass. Caches
  // engine->graph_ram_layer_ready(layer) the first time it is consulted for
  // this layer during a single graph build. Background prefetch/promotion
  // worker threads mutate the live RAM-tier state concurrently with graph
  // construction, so partial_map()'s cold-branch indexing scheme (full
  // arena vs. compacted load_cold_weight) and expert_weight()'s/
  // expert_gate_up_weight()'s tensor-shape choice must agree on ONE
  // snapshot of readiness, not two independent live reads taken moments
  // apart -- a disagreement bakes a map that indexes an expert id past the
  // end of a smaller compacted weight tensor, which crashes CPU
  // ggml_compute_forward_mul_mat_id (assert(i02 < n_as) is compiled out in
  // release builds).
  int ram_ready_cached = -1;
};

struct Glm5nextIndexerSeed {
  int layer = -1;
  int token_count = 0;
  int pool_count = 0;
  std::vector<float> key_gate;
  std::vector<float> pooled_keys;
};

struct Glm5nextIndexerLayer {
  int layer = -1;
  ggml_tensor *key_gate_out = nullptr; // prefill: [2*key_length, S]
  ggml_tensor *pooled_out = nullptr;   // prefill: [key_length, S/kpool]
  ggml_tensor *key_gate_cache = nullptr; // decode: [2*key_length, capacity]
  ggml_tensor *pooled_cache = nullptr;   // decode: [key_length, capacity/kpool]
};

struct FullGraph {
  StreamingEngine *engine = nullptr;
  GpuTopology *topology = nullptr;
  int device = -1;
  int S = 0;
  bool decode_mode = false;
  int kv_capacity = 0;
  int decode_last_pos = -1;
  int D = 0;
  int HC = 0;
  int n_layers = 0;
  int H = 0;
  int hd = 0;
  int d_inner = 0;
  int d_conv = 0;
  int qk_nope = 0;
  int qk_rope = 0;
  int kv_lora = 0;
  int v_head = 0;
  int o_group_count = 0;
  int o_lora_rank = 0;
  int sliding_window = 0;
  int first_dense = 0;
  int device_expert_layers = 0;
  ggml_backend_t gpu1 = nullptr; // optional second CUDA device for expert layers
  int gpu1_first = -1, gpu1_last = -2;
  std::vector<ggml_backend_buffer_t> gpu1_buffers;
  std::unordered_map<int, ggml_tensor *> stage_pred; // layer -> pilot ids predicted for it (built one layer earlier)
  size_t gpu1_bytes = 0;
  int indexer_heads = 0;
  int indexer_key_length = 0;
  int indexer_top_k = 0;
  int indexer_kpool = 0;
  bool indexer_select_tail = false;
  bool sparse_decode = false;
  bool partial_expert_residency = false;
  bool vram_reference_experts = false;
  bool dynamic_expert_cache = false;
  bool dynamic_upload_disabled = false;
  bool partial_sources_compacted = false;
  size_t partial_hot_bytes = 0;
  size_t partial_cold_bytes = 0;
  size_t partial_hot_records = 0;
  size_t vram_reference_bytes = 0;
  std::string partial_usage_file;
  std::vector<PartialExpertLayer> partial_experts;
  uint64_t dynamic_clock = 0;
  uint64_t dynamic_promotions = 0;
  double dynamic_compute_ms_ema = 0.0;
  double dynamic_upload_ms_per_job = 0.0;
  std::unique_ptr<DynamicUploadState> dynamic_upload;

  ggml_backend_t cuda = nullptr;
  ggml_backend_t cpu = nullptr;
  ggml_backend_sched_t sched = nullptr;
  ggml_backend_buffer_t dense_buffer = nullptr;
  void *dense_base = nullptr;
  size_t dense_capacity = 0;
  size_t dense_used = 0;
  int dense_tensors = 0;
  ggml_backend_buffer_t expert_buffer = nullptr;
  void *expert_base = nullptr;
  size_t expert_capacity = 0;
  size_t expert_used = 0;
  ggml_backend_buffer_t device_state_buffer = nullptr;
  void *device_state_base = nullptr;
  size_t device_state_capacity = 0;
  size_t device_state_used = 0;
  ggml_context *ctx = nullptr;
  ggml_cgraph *graph = nullptr;

  ggml_tensor *input = nullptr;  // [D, HC, S], host input
  ggml_tensor *logits = nullptr; // decode only: LM-head output [V]
  ggml_tensor *output = nullptr; // [D, S], after HC mean; LM head norms
  ggml_tensor *final_streams = nullptr;
  ggml_tensor *attn_mask = nullptr; // [S, S, 1, 1], F16 causal mask
  ggml_tensor *decode_indices = nullptr; // [1], I32 DSA KV-cache row
  ggml_tensor *indexer_pool_member_ids = nullptr; // [kpool, n_pools], I32
  ggml_tensor *indexer_pool_mask = nullptr; // [n_pools], F32, per-token
  ggml_tensor *indexer_pool_update_rows = nullptr; // [kpool], I32
  ggml_tensor *indexer_pool_update_index = nullptr; // [1], I32
  ggml_tensor *indexer_tail_indices = nullptr; // [kpool-1], I32
  ggml_tensor *indexer_tail_mask = nullptr; // [kpool-1], F16
  ggml_tensor *positions = nullptr; // [S], I32 token positions
  ggml_tensor *positions_mod4 = nullptr;
  ggml_tensor *positions_mod128 = nullptr;
  ggml_tensor *block_positions4 = nullptr;
  ggml_tensor *block_positions128 = nullptr;
  ggml_tensor *csa_zero_values = nullptr;
  ggml_tensor *csa_zero_scores = nullptr;

  struct Dsv4LayerTensors {
    int layer = -1;
    int ratio = 0;
    ggml_tensor *raw_kv = nullptr;             // [qh, S], rotated K==V
    ggml_tensor *comp_kv = nullptr;            // [width, S]
    ggml_tensor *comp_score = nullptr;         // [width, S], incl. APE
    ggml_tensor *indexer_comp_kv = nullptr;    // CSA only
    ggml_tensor *indexer_comp_score = nullptr; // CSA only
    ggml_tensor *keys = nullptr;               // [qh, floor(S/ratio)]
    ggml_tensor *indexer_keys = nullptr;       // CSA only
    ggml_tensor *mask = nullptr;               // [SWA + compressed + sink, S]
  };
  std::vector<Dsv4LayerTensors> dsv4_layers;
  std::vector<ggml_tensor *> dsv4_layer_outputs;
  // Decode-only per-layer timing marks (MOESTREAM_GLM5NEXT_GRAPH_LAYER_TIMING=1):
  // node -> {layer, 0=after attention, 1=after FFN}.
  std::unordered_map<const ggml_tensor *, std::pair<int, int>> layer_marks;
  std::vector<std::pair<std::string, ggml_tensor *>> dsv4_stage_outputs;

  std::vector<ggml_tensor *> kda_state_out;
  std::vector<ggml_tensor *> kda_hist_out;
  std::vector<int> kda_layers;
  std::vector<float *> kda_state_inputs;
  std::vector<float *> kda_hist_inputs;
  std::vector<ggml_tensor *> dsa_kv_out;
  std::vector<ggml_tensor *> dsa_kv_storage;
  std::vector<ggml_tensor *> kda_state_storage;
  std::vector<ggml_tensor *> kda_hist_storage;
  std::vector<ggml_tensor *> kda_state_updates;
  std::vector<ggml_tensor *> kda_hist_updates;
  std::vector<int> dsa_layers;
  std::vector<std::pair<int, ggml_tensor *>> moe_route_ids;
  struct VramExpertHold {
    int layer = -1;
    int expert = -1;
    const uint8_t *device_ptr = nullptr;
  };
  std::vector<VramExpertHold> vram_expert_holds;
  std::vector<Glm5nextIndexerLayer> glm5next_indexer_layers;
  std::vector<Glm5nextIndexerSeed> glm5next_indexer_seeds;

  std::vector<HostAllocation> owned;
  std::vector<ggml_backend_buffer_t> host_buffers;
  std::vector<void *> host_inputs;

  size_t host_weight_bytes = 0;
  size_t host_reference_bytes = 0;
  size_t host_pinned_bytes = 0;
  size_t host_pin_failures = 0;
  // Any routed projections that lack a stable engine-owned DDR arena are
  // kept in one graph-owned DDR source arena. Keeping the large projections
  // as individual malloc/cudaHostRegister allocations made graph preparation
  // spend minutes in allocator/page registration. Existing graph-RAM arenas
  // and VRAM cache slots are referenced directly instead of copied here.
  // GLM's dynamic cache can instead keep read-only views into GGUF shard
  // mappings; those pages remain reclaimable by the OS and avoid duplicating
  // the full expert set in a second anonymous allocation.
  bool expert_source_mmap = false;
  void *expert_source_base = nullptr;
  size_t expert_source_capacity = 0;
  size_t expert_source_used = 0;
  size_t expert_source_tensors = 0;
  std::unordered_map<std::string, void *> expert_source_views;
  std::unordered_map<std::string, size_t> expert_source_view_bytes;
  // Partial compaction is in-place, but hot uploads still need the original
  // full-source addresses after the cold views have been rewritten.
  std::unordered_map<std::string, void *> expert_source_full_views;
  std::unordered_map<std::string, size_t> expert_source_full_view_bytes;
  // Expert source slices are registered individually rather than registering
  // the complete 67--78 GiB arena.  The latter can exceed this host's CUDA
  // host-registration ceiling and would leave the whole graph on pageable
  // memory.  Each entry here is a non-owning range inside expert_source_base;
  // destroy() unregisters it before the arena is freed.
  std::vector<HostAllocation> expert_source_pinned_ranges;
  size_t host_reservation_bytes = 0;
  // ggml_init(no_alloc=true) still owns a host-side context arena.  Keep it
  // in the same admission ledger as the graph's explicit host buffers; the
  // old watchdog only saw the latter and could therefore under-report the
  // graph's host peak during construction.
  size_t context_reservation_bytes = 0;
  size_t graph_workspace_bytes = 0;
  bool ready = false;

  bool has_expert_source() const {
    if (expert_source_mmap)
      return !expert_source_views.empty();
    return expert_source_base != nullptr && expert_source_capacity != 0;
  }

  // Transfer only the source-side state to a graph with a different prompt
  // shape.  The graph context/scheduler and its prompt-sized tensors must be
  // rebuilt, while the large DDR expert arena must remain alive.  Partial
  // residency's tensor pointers belong to the old GGML context, so preserve
  // only its layer/expert lists and clear those pointers before the old graph
  // is destroyed.
  bool move_expert_source_to(FullGraph &dst) {
    if (!has_expert_source() || dst.has_expert_source())
      return false;
    if (!expert_source_mmap) {
      const auto owner = std::find_if(
          owned.begin(), owned.end(), [this](const HostAllocation &allocation) {
            return allocation.ptr == expert_source_base;
          });
      if (owner == owned.end()) {
        LOG_ERROR("[GGML-GLM5NEXT-FULL] expert source arena has no tracked "
                  "owner; refusing graph handoff\n");
        return false;
      }
      // The source views are non-owning slices into this allocation. Transfer
      // the allocation itself before the old prompt graph is destroyed, or
      // the reusable decode graph will retain dangling expert pointers.
      dst.owned.push_back(std::move(*owner));
      owned.erase(owner);
    }

    dst.expert_source_mmap = expert_source_mmap;
    dst.expert_source_base = expert_source_base;
    dst.expert_source_capacity = expert_source_capacity;
    dst.expert_source_used = expert_source_used;
    dst.expert_source_tensors = expert_source_tensors;
    dst.expert_source_views = std::move(expert_source_views);
    dst.expert_source_view_bytes = std::move(expert_source_view_bytes);
    dst.expert_source_full_views = std::move(expert_source_full_views);
    dst.expert_source_full_view_bytes =
        std::move(expert_source_full_view_bytes);
    dst.expert_source_pinned_ranges = std::move(expert_source_pinned_ranges);
    dst.host_reservation_bytes = host_reservation_bytes;

    dst.partial_expert_residency = partial_expert_residency;
    dst.vram_reference_experts = vram_reference_experts;
    dst.topology = topology;
    dst.dynamic_expert_cache = dynamic_expert_cache;
    dst.dynamic_upload_disabled = dynamic_upload_disabled;
    dst.partial_sources_compacted = partial_sources_compacted;
    dst.partial_hot_bytes = partial_hot_bytes;
    dst.partial_cold_bytes = partial_cold_bytes;
    dst.partial_hot_records = partial_hot_records;
    dst.vram_reference_bytes = vram_reference_bytes;
    dst.partial_usage_file = std::move(partial_usage_file);
    dst.partial_experts = std::move(partial_experts);
    dst.vram_expert_holds = std::move(vram_expert_holds);
    dst.host_reference_bytes = host_reference_bytes;
    dst.dynamic_clock = dynamic_clock;
    dst.dynamic_promotions = dynamic_promotions;
    dst.dynamic_compute_ms_ema = dynamic_compute_ms_ema;
    dst.dynamic_upload_ms_per_job = dynamic_upload_ms_per_job;
    dst.dynamic_upload = std::move(dynamic_upload);
    for (PartialExpertLayer &p : dst.partial_experts) {
      p.hot_gate_up = nullptr;
      p.hot_gate = nullptr;
      p.hot_up = nullptr;
      p.hot_down = nullptr;
      p.cold_gate_up = nullptr;
      p.cold_gate = nullptr;
      p.cold_up = nullptr;
      p.cold_down = nullptr;
      p.cold_map = nullptr;
      p.hot_map = nullptr;
      p.cold_mask = nullptr;
      p.hot_mask = nullptr;
      p.hot1_gate = nullptr;
      p.hot1_up = nullptr;
      p.hot1_down = nullptr;
      p.hot1_map = nullptr;
      p.hot1_mask = nullptr;
    }

    expert_source_mmap = false;
    expert_source_base = nullptr;
    expert_source_capacity = 0;
    expert_source_used = 0;
    expert_source_tensors = 0;
    expert_source_pinned_ranges.clear();
    host_reservation_bytes = 0;
    partial_expert_residency = false;
    vram_reference_experts = false;
    dynamic_expert_cache = false;
    dynamic_upload_disabled = false;
    partial_sources_compacted = false;
    partial_hot_bytes = 0;
    partial_cold_bytes = 0;
    partial_hot_records = 0;
    vram_reference_bytes = 0;
    partial_usage_file.clear();
    partial_experts.clear();
    vram_expert_holds.clear();
    dynamic_clock = 0;
    dynamic_promotions = 0;
    dynamic_compute_ms_ema = 0.0;
    dynamic_upload_ms_per_job = 0.0;
    return true;
  }

  ~FullGraph() { destroy(); }

  void destroy() {
    if (dynamic_upload) {
      {
        std::lock_guard<std::mutex> lock(dynamic_upload->mutex);
        dynamic_upload->stopping = true;
      }
      dynamic_upload->cv.notify_all();
      if (dynamic_upload->worker.joinable())
        dynamic_upload->worker.join();
      if (device >= 0)
        cudaSetDevice(device);
      if (dynamic_upload->stream) {
        cudaStreamSynchronize(dynamic_upload->stream);
        cudaStreamDestroy(dynamic_upload->stream);
      }
      if (dynamic_upload->staging) {
        cudaHostUnregister(dynamic_upload->staging);
        std::free(dynamic_upload->staging);
      }
      MemoryGuard::release_host_bytes(
          dynamic_upload->host_reservation_bytes);
      dynamic_upload.reset();
    }
    if (sched) {
      ggml_backend_sched_free(sched);
      sched = nullptr;
    }
    if (cpu) {
      ggml_backend_free(cpu);
      cpu = nullptr;
    }
    for (ggml_backend_buffer_t b : gpu1_buffers)
      if (b)
        ggml_backend_buffer_free(b);
    gpu1_buffers.clear();
    if (expert_buffer) {
      ggml_backend_buffer_free(expert_buffer);
      expert_buffer = nullptr;
    }
    if (device_state_buffer) {
      ggml_backend_buffer_free(device_state_buffer);
      device_state_buffer = nullptr;
    }
    if (dense_buffer) {
      ggml_backend_buffer_free(dense_buffer);
      dense_buffer = nullptr;
    }
    partial_expert_residency = false;
    dynamic_expert_cache = false;
    dynamic_upload_disabled = false;
    partial_sources_compacted = false;
    partial_hot_bytes = 0;
    partial_cold_bytes = 0;
    partial_hot_records = 0;
    vram_reference_bytes = 0;
    partial_usage_file.clear();
    partial_experts.clear();
    for (ggml_backend_buffer_t b : host_buffers)
      if (b)
        ggml_backend_buffer_free(b);
    host_buffers.clear();
    if (ctx) {
      ggml_free(ctx);
      ctx = nullptr;
    }
    for (HostAllocation &a : owned) {
      if (a.pinned)
        cudaHostUnregister(a.ptr);
      std::free(a.ptr);
      MemoryGuard::release_host_bytes(a.reservation_bytes);
    }
    owned.clear();
    host_inputs.clear();
    expert_source_mmap = false;
    expert_source_base = nullptr;
    expert_source_capacity = 0;
    expert_source_used = 0;
    expert_source_tensors = 0;
    expert_source_views.clear();
    expert_source_view_bytes.clear();
    expert_source_full_views.clear();
    expert_source_full_view_bytes.clear();
    for (const HostAllocation &a : expert_source_pinned_ranges)
      if (a.pinned)
        cudaHostUnregister(a.ptr);
    expert_source_pinned_ranges.clear();
    cuda = nullptr;
    graph = nullptr;
    input = output = final_streams = attn_mask = decode_indices = positions =
        nullptr;
    indexer_pool_member_ids = indexer_pool_mask =
        indexer_pool_update_rows = indexer_pool_update_index =
            indexer_tail_indices = indexer_tail_mask = nullptr;
    positions_mod4 = positions_mod128 = block_positions4 = block_positions128 =
        nullptr;
    csa_zero_values = csa_zero_scores = nullptr;
    kda_state_out.clear();
    kda_hist_out.clear();
    kda_layers.clear();
    kda_state_inputs.clear();
    kda_hist_inputs.clear();
    dsa_kv_out.clear();
    dsa_kv_storage.clear();
    kda_state_storage.clear();
    kda_hist_storage.clear();
    kda_state_updates.clear();
    kda_hist_updates.clear();
    dsa_layers.clear();
    moe_route_ids.clear();
    glm5next_indexer_layers.clear();
    glm5next_indexer_seeds.clear();
    dsv4_layers.clear();
    dsv4_stage_outputs.clear();
    MemoryGuard::release_host_bytes(host_reservation_bytes);
    host_reservation_bytes = 0;
    MemoryGuard::release_host_bytes(context_reservation_bytes);
    context_reservation_bytes = 0;
    engine = nullptr;
    topology = nullptr;
    ready = false;
  }
};

std::unique_ptr<FullGraph> g_graph;
// The large host expert source is independent of prompt length.  Keep it in
// a source-only container while the prompt-shaped GGML graph is rebuilt or
// before the first prompt arrives.
std::unique_ptr<FullGraph> g_glm5next_source;
GpuTopology *g_glm5next_vram_cache = nullptr;
StreamingEngine *g_glm5next_decode_disabled_engine = nullptr;

bool env_enabled() {
  static const bool enabled = [] {
    // Full graph is the GLM5NEXT execution path; MOESTREAM_GLM5NEXT_FULL_GRAPH=0 opts out (legacy layered path).
    const char *v = std::getenv("MOESTREAM_GLM5NEXT_FULL_GRAPH");
    return !v || !v[0] || v[0] != '0';
  }();
  return enabled;
}


// ---------------------------------------------------------------------------
// SSD-offload routed experts (opt-in, MOESTREAM_GLM5NEXT_OFFLOAD_FIRST_LAYER).
// Layers >= that index do not get a fully resident DDR source.  Their experts
// live in a virtual per-projection arena (slot == expert id, so MUL_MAT_ID needs
// no remapping) that is filled on demand from the offload store / model by a
// host-side LRU cache.  A custom CPU node in front of every MUL_MAT_ID makes the
// routed experts resident before the matmul runs.
// ---------------------------------------------------------------------------
const GGUFTensor *model_tensor(const StreamingEngine &engine,
                               const std::string &name);
int pilot_stage_k();
struct OffloadState;
struct OffloadSite {
  OffloadState *state = nullptr;
  int layer = 0;
  bool prefill = false;
};
struct OffloadState {
  int first = 0, n_layers = 0, n_experts = 0;
  std::unique_ptr<moestream::offload::ExpertCache> cache;
  std::unique_ptr<moestream::offload::VirtualArena> arena;
  moestream::offload::OffloadStore store;
  moestream::offload::ModelReader model;
  std::unique_ptr<moestream::offload::ExpertResidency> residency;
  std::deque<OffloadSite> sites;
  // Pinned (cudaHostRegister) expert slots, so the pilot-staging DMA reads DDR directly instead of through a CPU bounce copy.
  bool pin_slots = false;
  std::vector<uint8_t> pinned; // [(layer - first) * n_experts + expert]
  std::atomic<uint64_t> pin_calls{0}, pin_fail{0};
  void pin_expert(int layer, int expert) {
    uint8_t &f = pinned[(size_t)(layer - first) * (size_t)n_experts + (size_t)expert];
    if (f)
      return;
    int ok = 0;
    for (int k = 0; k < 3; ++k)
      if (cudaHostRegister(arena->ptr(layer, expert, k), (size_t)arena->plane_bytes(layer, k), cudaHostRegisterPortable) == cudaSuccess)
        ++ok;
      else
        cudaGetLastError();
    if (ok == 3) {
      f = 1;
      pin_calls.fetch_add(1, std::memory_order_relaxed);
    } else {
      pin_fail.fetch_add(1, std::memory_order_relaxed);
      for (int k = 0; k < ok; ++k)
        cudaHostUnregister(arena->ptr(layer, expert, k)); // partially registered: undo (the unregistered plane is the failed one)
    }
  }
  void unpin_expert(int layer, int expert) {
    uint8_t &f = pinned[(size_t)(layer - first) * (size_t)n_experts + (size_t)expert];
    if (!f)
      return;
    for (int k = 0; k < 3; ++k)
      cudaHostUnregister(arena->ptr(layer, expert, k));
    f = 0;
  }
  bool is_pinned(int layer, int expert) const {
    return pin_slots && pinned[(size_t)(layer - first) * (size_t)n_experts + (size_t)expert] != 0;
  }
  ~OffloadState() {
    if (pin_slots && arena)
      for (int l = first; l < first + n_layers; ++l)
        for (int e = 0; e < n_experts; ++e)
          unpin_expert(l, e);
  }
  std::atomic<uint64_t> calls{0}, routes{0}, misses{0}, loads{0}, evictions{0},
      errors{0}, load_us{0}, transient_loads{0}, cb_us{0}, cold_routes{0};
  std::atomic<uint64_t> last_report_calls{0};
};
std::unique_ptr<OffloadState> g_offload;
std::mutex g_offload_mutex;
std::shared_mutex g_arena_mu; // exclusive: cache eviction/loads (miss nodes); shared: the pilot-staging worker reading arena slots

int offload_first_layer() {
  static const int first = [] {
    const char *v = std::getenv("MOESTREAM_GLM5NEXT_OFFLOAD_FIRST_LAYER");
    return v && v[0] ? std::atoi(v) : -1;
  }();
  return first;
}
bool offload_layer(int layer) {
  const int f = offload_first_layer();
  return f >= 0 && layer >= f;
}
std::string offload_env(const char *name, const char *dflt = "") {
  const char *v = std::getenv(name);
  return v && v[0] ? v : dflt;
}

void offload_report(OffloadState &o, const char *why) {
  LOG_INFO("[GGML-GLM5NEXT-OFFLOAD] %s: calls=%llu routes=%llu misses=%llu "
           "loads=%llu transient=%llu evictions=%llu errors=%llu load=%.1f ms "
           "callback=%.1f ms cold_routes=%llu\n",
           why, (unsigned long long)o.calls.load(),
           (unsigned long long)o.routes.load(),
           (unsigned long long)o.misses.load(),
           (unsigned long long)o.loads.load(),
           (unsigned long long)o.transient_loads.load(),
           (unsigned long long)o.evictions.load(),
           (unsigned long long)o.errors.load(), o.load_us.load() / 1000.0,
           o.cb_us.load() / 1000.0, (unsigned long long)o.cold_routes.load());
}

// Created on first use and shared by the prompt and decode graphs.
OffloadState *get_offload(const StreamingEngine &engine,
                          const std::vector<PartialExpertLayer> *partial = nullptr) {
  std::lock_guard<std::mutex> lock(g_offload_mutex);
  if (g_offload)
    return g_offload.get();
  const auto &c = engine.model_file.architecture_config;
  auto o = std::make_unique<OffloadState>();
  o->first = offload_first_layer();
  o->n_layers = c.n_layers - o->first;
  o->n_experts = c.n_experts;
  if (o->first < c.first_dense || o->n_layers <= 0 || o->n_experts <= 0) {
    LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] bad first layer %d\n", o->first);
    return nullptr;
  }
  std::vector<std::array<uint64_t, 3>> planes;
  uint64_t expert_bytes_sum = 0;
  for (int l = o->first; l < c.n_layers; ++l) {
    std::array<uint64_t, 3> pb{};
    int k = 0;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(l) + "." + stem);
      if (!m || m->size_bytes % (uint64_t)o->n_experts != 0) {
        LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] layer %d %s missing/uneven\n", l,
                  stem);
        return nullptr;
      }
      pb[(size_t)k] = m->size_bytes / (uint64_t)o->n_experts;
      if (pb[(size_t)k] % 4096 != 0) {
        LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] layer %d %s per-expert %llu is not "
                  "a multiple of 4096\n",
                  l, stem, (unsigned long long)pb[(size_t)k]);
        return nullptr;
      }
      expert_bytes_sum += pb[(size_t)k];
      ++k;
    }
    planes.push_back(pb);
  }
  o->arena = std::make_unique<moestream::offload::VirtualArena>(
      o->first, o->n_layers, o->n_experts, planes,
      /*no_huge_pages=*/offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_THP", "0") != "1");
  if (!o->arena->ok()) {
    LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] virtual arena mmap failed\n");
    return nullptr;
  }

  std::string err;
  std::vector<moestream::offload::ExpertSource *> sources;
  const std::string store_path = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_STORE");
  if (!store_path.empty()) {
    if (o->store.open(store_path, &err))
      sources.push_back(&o->store);
    else
      LOG_WARN("[GGML-GLM5NEXT-OFFLOAD] store %s unusable (%s); model only\n",
               store_path.c_str(), err.c_str());
  }
  std::vector<std::string> shards;
  {
    const std::string first_shard = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_MODEL");
    if (first_shard.empty()) {
      LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] MOESTREAM_GLM5NEXT_OFFLOAD_MODEL "
                "(first shard) is required\n");
      return nullptr;
    }
    const size_t pos = first_shard.rfind("-00001-of-");
    if (pos == std::string::npos) {
      shards.push_back(first_shard);
    } else {
      const std::string head = first_shard.substr(0, pos);
      const int total = std::atoi(first_shard.c_str() + pos + 10);
      char buf[64];
      for (int i = 1; i <= std::max(1, total); ++i) {
        std::snprintf(buf, sizeof buf, "-%05d-of-%05d.gguf", i, total);
        shards.push_back(head + buf);
      }
    }
  }
  if (!o->model.open(shards, &err)) {
    LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] model reader: %s\n", err.c_str());
    return nullptr;
  }
  sources.push_back(&o->model);

  moestream::offload::ExpertCache::Config cc;
  cc.first_layer = o->first;
  cc.n_layers = o->n_layers;
  cc.n_experts = o->n_experts;
  const double ddr_gib =
      std::atof(offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB", "40").c_str());
  cc.capacity = (size_t)(ddr_gib * 1024.0 * 1024.0 * 1024.0 /
                         ((double)expert_bytes_sum / (double)o->n_layers));
  cc.policy = moestream::offload::ExpertCache::Policy::Lru;
  cc.pinned_fraction =
      std::atof(offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_PINNED", "0.5").c_str());
  o->cache = std::make_unique<moestream::offload::ExpertCache>(cc);
  o->residency = std::make_unique<moestream::offload::ExpertResidency>(
      *o->cache, *o->arena, sources);

  if (pilot_stage_k() > 0 && offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_PIN", "0") == "1") {
    o->pin_slots = true;
    o->pinned.assign((size_t)o->n_layers * (size_t)o->n_experts, 0);
    OffloadState *op = o.get();
    o->arena->after_populate = [op](int l, int e) { op->pin_expert(l, e); };
    o->arena->before_evict = [op](int l, int e) { op->unpin_expert(l, e); };
  }
  const std::string usage = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_USAGE");
  if (!usage.empty()) {
    const auto t0 = std::chrono::steady_clock::now();
    auto ranked = moestream::offload::rank_hottest_first(
        usage, o->first, o->n_layers, o->n_experts);
    if (partial) {
      // Experts that live in VRAM are served by the GPU branch; do not spend DDR on them.
      std::vector<std::vector<uint8_t>> hot((size_t)c.n_layers);
      size_t n_hot = 0;
      for (size_t l = 0; l < partial->size() && l < hot.size(); ++l) {
        hot[l].assign((size_t)o->n_experts, 0);
        for (const std::vector<int> *set : {&(*partial)[l].hot, &(*partial)[l].hot1})
          for (int e : *set)
            if (e >= 0 && e < o->n_experts) {
              hot[l][(size_t)e] = 1;
              ++n_hot;
            }
      }
      ranked.erase(std::remove_if(ranked.begin(), ranked.end(),
                                  [&](const moestream::offload::ExpertKey &k) {
                                    return (size_t)k.layer < hot.size() &&
                                           !hot[(size_t)k.layer].empty() &&
                                           hot[(size_t)k.layer][(size_t)k.expert];
                                  }),
                   ranked.end());
      LOG_INFO("[GGML-GLM5NEXT-OFFLOAD] excluded %zu VRAM-hot experts from the "
               "DDR seed\n", n_hot);
    }
    o->cache->seed(ranked);
    const size_t failed = o->residency->populate_resident();
    if (o->pin_slots)
      LOG_INFO("[GGML-GLM5NEXT-OFFLOAD] pinned %llu experts (%llu failed) for direct DMA\n",
               (unsigned long long)o->pin_calls.load(), (unsigned long long)o->pin_fail.load());
    LOG_INFO("[GGML-GLM5NEXT-OFFLOAD] seeded %zu/%zu experts (%zu failed) in "
             "%.1f s\n",
             o->cache->size(), cc.capacity, failed,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                 .count());
  }
  LOG_INFO("[GGML-GLM5NEXT-OFFLOAD] layers %d..%d, capacity %zu experts "
           "(%.1f GiB), pinned %.2f, store=%s\n",
           o->first, c.n_layers - 1, cc.capacity, ddr_gib, cc.pinned_fraction,
           store_path.empty() ? "<none>" : store_path.c_str());
  g_offload = std::move(o);
  return g_offload.get();
}

void offload_miss_cb(ggml_tensor *dst, const ggml_tensor *a, int ith, int,
                     void *ud) {
  if (ith != 0)
    return;
  OffloadSite *site = static_cast<OffloadSite *>(ud);
  OffloadState &o = *site->state;
  const auto cb_t0 = std::chrono::steady_clock::now();
  std::vector<int32_t> ids;
  ids.reserve((size_t)ggml_nelements(a));
  for (int64_t i1 = 0; i1 < a->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < a->ne[0]; ++i0)
      ids.push_back(*(const int32_t *)((const char *)a->data + i1 * a->nb[1] +
                                       i0 * a->nb[0]));
  static const bool dbg = std::getenv("MOESTREAM_GLM5NEXT_OFFLOAD_DEBUG") != nullptr;
  static std::atomic<int> dbg_lines{0};
  if (dbg && !site->prefill)
    for (size_t k = 0; k < ids.size(); ++k)
      if (ids[k] >= 0 && !o.cache->resident({site->layer, ids[k]}) && dbg_lines.fetch_add(1) < 40)
        LOG_INFO("[OFFLOAD-DEBUG] miss layer=%d expert=%d S=%lld k=%zu\n", site->layer, ids[k],
                 (long long)a->ne[1], k);
  std::unique_lock<std::shared_mutex> arena_lock(g_arena_mu);
  const auto r = site->prefill
                     ? o.residency->ensure_transient(site->layer, ids.data(),
                                                     (int)ids.size())
                     : o.residency->ensure(site->layer, ids.data(),
                                           (int)ids.size());
  o.calls.fetch_add(1, std::memory_order_relaxed);
  o.routes.fetch_add(ids.size(), std::memory_order_relaxed);
  o.cold_routes.fetch_add((uint64_t)std::count_if(ids.begin(), ids.end(), [](int32_t v) { return v >= 0; }),
                          std::memory_order_relaxed);
  o.misses.fetch_add((uint64_t)r.misses, std::memory_order_relaxed);
  o.loads.fetch_add((uint64_t)(site->prefill ? 0 : r.loads),
                    std::memory_order_relaxed);
  o.transient_loads.fetch_add((uint64_t)(site->prefill ? r.loads : 0),
                              std::memory_order_relaxed);
  o.evictions.fetch_add((uint64_t)r.evictions, std::memory_order_relaxed);
  o.load_us.fetch_add((uint64_t)(r.load_ms * 1000.0), std::memory_order_relaxed);
  if (r.errors > 0) {
    o.errors.fetch_add((uint64_t)r.errors, std::memory_order_relaxed);
    LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] layer %d: %d expert load error(s); "
              "aborting rather than computing with missing experts\n",
              site->layer, r.errors);
    offload_report(o, "fatal");
    std::abort();
  }
  // dst has the shape of `a` but is contiguous.
  std::memcpy(dst->data, ids.data(), ids.size() * sizeof(int32_t));
  o.cb_us.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - cb_t0).count(),
                    std::memory_order_relaxed);
  if (!site->prefill && o.calls.load() - o.last_report_calls.load() >= 544) {
    o.last_report_calls.store(o.calls.load());
    offload_report(o, "progress");
  }
}


// ---------------------------------------------------------------------------
// Pilot staging (MOESTREAM_GLM5NEXT_PILOT_STAGE=K): in decode, layer l's graph predicts the K most likely COLD experts of layer l+1
// (layer l+1's router applied to layer l's FFN input, hot experts masked out), a worker copies them DDR -> pinned -> VRAM ring while
// layer l's FFN and layer l+1's attention run, and layer l+1 computes them on the GPU; the CPU cold branch skips whatever was staged.
// A staged expert that was not ready or not routed simply stays on the CPU branch, so correctness never depends on timing.
// ---------------------------------------------------------------------------
int pilot_stage_k() {
  static const int k = [] {
    const char *v = std::getenv("MOESTREAM_GLM5NEXT_PILOT_STAGE");
    return v && v[0] ? std::max(0, std::min(8, std::atoi(v))) : 0;
  }();
  return k;
}
struct StageSite {
  int layer = 0;
  int set = 0;
};
struct StageMissSite;
struct StageResult {
  std::atomic<int> state{0}; // 0 none, 1 pending, 2 ready
  int n = 0;
  int32_t ids[8] = {};
};
struct StageRuntime {
  int K = 0;
  static constexpr int kSets = 2;
  uint64_t plane_bytes[3] = {};
  ggml_backend_buffer_t dev_buf[kSets][3] = {};
  char *dev_base[kSets][3] = {};
  void *pinned[kSets][3] = {};
  cudaStream_t stream = nullptr;
  cudaEvent_t done[kSets] = {};
  std::thread worker;
  std::mutex m;
  std::condition_variable cv, done_cv;
  struct Job {
    int layer = 0, set = 0, n = 0;
    int32_t ids[8] = {};
  };
  std::deque<Job> q;
  bool stop = false;
  // Small copy pool: the DDR -> pinned memcpy of a few multi-MB planes is the long pole of a staging job, so the worker thread and
  // kCopyHelpers helpers split the (expert, plane) pieces.
  static constexpr int kCopyHelpers = 3;
  std::vector<std::thread> helpers;
  std::mutex pm;
  std::condition_variable pcv, pdone;
  std::function<void(int)> pfn;
  int pn = 0;
  std::atomic<int> pnext{0}, pfinished{0};
  uint64_t pgen = 0;
  void run_pieces(int n, std::function<void(int)> fn) {
    {
      std::lock_guard<std::mutex> lock(pm);
      pfn = std::move(fn);
      pn = n;
      pnext.store(0);
      pfinished.store(0);
      ++pgen;
    }
    pcv.notify_all();
    for (;;) {
      const int i = pnext.fetch_add(1);
      if (i >= n)
        break;
      pfn(i);
      pfinished.fetch_add(1);
    }
    std::unique_lock<std::mutex> lock(pm);
    pdone.wait(lock, [&] { return pfinished.load() >= n; });
  }
  void helper_main() {
    uint64_t seen = 0;
    for (;;) {
      std::function<void(int)> fn;
      int n = 0;
      {
        std::unique_lock<std::mutex> lock(pm);
        pcv.wait(lock, [&] { return stop || pgen != seen; });
        if (stop)
          return;
        seen = pgen;
        fn = pfn;
        n = pn;
      }
      for (;;) {
        const int i = pnext.fetch_add(1);
        if (i >= n)
          break;
        fn(i);
        if (pfinished.fetch_add(1) + 1 >= n) {
          std::lock_guard<std::mutex> lock(pm);
          pdone.notify_all();
        }
      }
    }
  }
  StageResult result[128];
  std::deque<StageSite> sites;
  std::deque<StageMissSite> miss_sites;
  std::atomic<uint64_t> direct{0}, jobs{0}, copied{0}, resolves{0}, staged_routes{0}, routes_seen{0}, wait_us{0}, skipped_not_resident{0};
  ~StageRuntime() {
    {
      std::lock_guard<std::mutex> lock(m);
      stop = true;
    }
    cv.notify_all();
    pcv.notify_all();
    if (worker.joinable())
      worker.join();
    for (auto &h : helpers)
      if (h.joinable())
        h.join();
    for (int s = 0; s < kSets; ++s)
      for (int k = 0; k < 3; ++k) {
        if (pinned[s][k])
          cudaFreeHost(pinned[s][k]);
        if (dev_buf[s][k])
          ggml_backend_buffer_free(dev_buf[s][k]);
      }
    for (int s = 0; s < kSets; ++s)
      if (done[s])
        cudaEventDestroy(done[s]);
    if (stream)
      cudaStreamDestroy(stream);
  }
};
std::unique_ptr<StageRuntime> g_stage;
bool g_stage_failed = false;

void stage_worker_main(StageRuntime *st) {
  for (;;) {
    StageRuntime::Job job;
    {
      std::unique_lock<std::mutex> lock(st->m);
      st->cv.wait(lock, [&] { return st->stop || !st->q.empty(); });
      if (st->stop && st->q.empty())
        return;
      job = st->q.front();
      st->q.pop_front();
    }
    OffloadState *o = g_offload.get();
    StageResult &res = st->result[job.layer];
    int staged = 0;
    int32_t staged_ids[8] = {};
    if (o) {
      std::shared_lock<std::shared_mutex> arena_lock(g_arena_mu);
      int chosen[8] = {};
      for (int j = 0; j < job.n && staged < st->K; ++j) {
        const int id = job.ids[j];
        if (id < 0 || id >= o->n_experts || !o->cache->resident({job.layer, id})) {
          st->skipped_not_resident.fetch_add(1, std::memory_order_relaxed);
          continue;
        }
        chosen[staged] = id;
        staged_ids[staged++] = id;
      }
      if (staged > 0) {
        size_t pb[3];
        for (int k = 0; k < 3; ++k)
          pb[k] = (size_t)o->arena->plane_bytes(job.layer, k);
        st->run_pieces(staged * 3, [&](int piece) {
          const int slot = piece / 3, k = piece % 3;
          char *src = static_cast<char *>(o->arena->ptr(job.layer, chosen[slot], k));
          if (o->is_pinned(job.layer, chosen[slot])) {
            // The arena slot is page-locked: the DMA engine reads DDR directly, no CPU copy.
            cudaMemcpyAsync(st->dev_base[job.set][k] + (size_t)slot * pb[k], src, pb[k], cudaMemcpyHostToDevice, st->stream);
            st->direct.fetch_add(1, std::memory_order_relaxed);
            return;
          }
          char *host = static_cast<char *>(st->pinned[job.set][k]) + (size_t)slot * pb[k];
          std::memcpy(host, src, pb[k]);
          cudaMemcpyAsync(st->dev_base[job.set][k] + (size_t)slot * pb[k], host, pb[k], cudaMemcpyHostToDevice, st->stream);
        });
      }
    }
    cudaEventRecord(st->done[job.set], st->stream);
    st->copied.fetch_add((uint64_t)staged, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(st->m);
      res.n = staged;
      std::memcpy(res.ids, staged_ids, sizeof(staged_ids));
      res.state.store(2, std::memory_order_release);
    }
    st->done_cv.notify_all();
  }
}

StageRuntime *get_stage_impl(FullGraph &g);
StageRuntime *get_stage(FullGraph &g) {
  if (g_stage)
    return g_stage.get();
  if (g_stage_failed)
    return nullptr;
  StageRuntime *r = get_stage_impl(g);
  if (!r)
    g_stage_failed = true;
  return r;
}
StageRuntime *get_stage_impl(FullGraph &g) {
  OffloadState *o = g_offload.get();
  const int K = pilot_stage_k();
  if (!o || K <= 0 || !g.cuda)
    return nullptr;
  auto st = std::make_unique<StageRuntime>();
  st->K = K;
  // Per-layer quant types differ in this GGUF (e.g. the down projection), so the ring is sized for the largest plane and each layer
  // addresses it with its own expert stride.
  for (int k = 0; k < 3; ++k) {
    st->plane_bytes[k] = 0;
    for (int l = o->first; l < o->first + o->n_layers; ++l)
      st->plane_bytes[k] = std::max<uint64_t>(st->plane_bytes[k], o->arena->plane_bytes(l, k));
  }
  if (cudaStreamCreateWithFlags(&st->stream, cudaStreamNonBlocking) != cudaSuccess)
    return nullptr;
  for (int s = 0; s < StageRuntime::kSets; ++s) {
    if (cudaEventCreateWithFlags(&st->done[s], cudaEventDisableTiming) != cudaSuccess)
      return nullptr;
    for (int k = 0; k < 3; ++k) {
      const size_t bytes = (size_t)K * st->plane_bytes[k];
      st->dev_buf[s][k] = ggml_backend_alloc_buffer(g.cuda, bytes + 4096);
      if (!st->dev_buf[s][k]) {
        LOG_WARN("[GGML-GLM5NEXT-STAGE] VRAM ring allocation failed (%zu bytes); pilot staging disabled\n", bytes);
        return nullptr;
      }
      ggml_backend_buffer_set_usage(st->dev_buf[s][k], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
      st->dev_base[s][k] = static_cast<char *>(ggml_backend_buffer_get_base(st->dev_buf[s][k]));
      if (cudaMallocHost(&st->pinned[s][k], bytes) != cudaSuccess) {
        LOG_WARN("[GGML-GLM5NEXT-STAGE] pinned bounce allocation failed\n");
        return nullptr;
      }
    }
  }
  st->worker = std::thread(stage_worker_main, st.get());
  for (int h = 0; h < StageRuntime::kCopyHelpers; ++h)
    st->helpers.emplace_back([p = st.get()] { p->helper_main(); });
  LOG_INFO("[GGML-GLM5NEXT-STAGE] ring ready: K=%d, %d sets, expert planes %llu/%llu/%llu bytes\n", K, StageRuntime::kSets,
           (unsigned long long)st->plane_bytes[0], (unsigned long long)st->plane_bytes[1], (unsigned long long)st->plane_bytes[2]);
  g_stage = std::move(st);
  return g_stage.get();
}

struct StageMissSite {
  OffloadSite *off = nullptr; // this layer's offload state/site
  int layer = 0, set = 0;
  int next_layer = -1, next_set = 0; // layer whose experts the pilot input predicts (-1: last layer)
};

void stage_post_job(StageRuntime *st, int layer, int set, const ggml_tensor *pred) {
  StageRuntime::Job job;
  job.layer = layer;
  job.set = set;
  job.n = (int)std::min<int64_t>(pred->ne[0], 8);
  for (int j = 0; j < job.n; ++j)
    job.ids[j] = *(const int32_t *)((const char *)pred->data + j * pred->nb[0]);
  {
    std::lock_guard<std::mutex> lock(st->m);
    st->result[layer].state.store(1, std::memory_order_release);
    st->q.push_back(job);
  }
  st->jobs.fetch_add(1, std::memory_order_relaxed);
  st->cv.notify_one();
}

// One CPU node per layer: (a) wait for the staged copy of this layer, (b) emit per route either the cold id for the CPU branch, or
// -(slot+2) for a route served from the VRAM ring, (c) make the remaining cold experts resident, (d) start the copy for the next layer.
// a: static cold ids [topk,1]; b: real routed ids [topk,1]; c: pilot ids of the next layer [K,1].
void stage_miss_cb(ggml_tensor *dst, const ggml_tensor *a, const ggml_tensor *b, const ggml_tensor *c, int ith, int, void *ud) {
  if (ith != 0)
    return;
  StageMissSite *site = static_cast<StageMissSite *>(ud);
  StageRuntime *st = g_stage.get();
  OffloadState &o = *site->off->state;
  const auto cb_t0 = std::chrono::steady_clock::now();
  const int n = (int)a->ne[0];
  int staged = 0;
  int32_t staged_ids[8] = {};
  StageResult &res = st->result[site->layer];
  if (res.state.load(std::memory_order_acquire) != 0) {
    const auto t0 = std::chrono::steady_clock::now();
    {
      std::unique_lock<std::mutex> lock(st->m);
      st->done_cv.wait(lock, [&] { return res.state.load(std::memory_order_acquire) != 1; });
      staged = res.n;
      std::memcpy(staged_ids, res.ids, sizeof(staged_ids));
      res.state.store(0, std::memory_order_release);
    }
    if (staged > 0)
      cudaEventSynchronize(st->done[site->set]);
    st->wait_us.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count(),
                          std::memory_order_relaxed);
  }
  std::vector<int32_t> ids((size_t)n);
  int32_t *out = static_cast<int32_t *>(dst->data);
  uint64_t hit = 0;
  for (int i = 0; i < n; ++i) {
    const int32_t cold = *(const int32_t *)((const char *)a->data + i * a->nb[0]);
    const int32_t real = *(const int32_t *)((const char *)b->data + i * b->nb[0]);
    int slot = -1;
    for (int j = 0; j < staged; ++j)
      if (staged_ids[j] == real) {
        slot = j;
        break;
      }
    if (slot >= 0 && cold >= 0) {
      out[i] = -(slot + 2);
      ids[(size_t)i] = -1;
      ++hit;
    } else {
      out[i] = cold;
      ids[(size_t)i] = cold;
    }
  }
  st->staged_routes.fetch_add(hit, std::memory_order_relaxed);
  st->routes_seen.fetch_add((uint64_t)n, std::memory_order_relaxed);
  moestream::offload::ExpertResidency::Ensure r;
  {
    std::unique_lock<std::shared_mutex> arena_lock(g_arena_mu);
    r = o.residency->ensure(site->layer, ids.data(), n);
  }
  o.calls.fetch_add(1, std::memory_order_relaxed);
  o.routes.fetch_add((uint64_t)n, std::memory_order_relaxed);
  o.cold_routes.fetch_add((uint64_t)std::count_if(ids.begin(), ids.end(), [](int32_t v) { return v >= 0; }), std::memory_order_relaxed);
  o.misses.fetch_add((uint64_t)r.misses, std::memory_order_relaxed);
  o.loads.fetch_add((uint64_t)r.loads, std::memory_order_relaxed);
  o.evictions.fetch_add((uint64_t)r.evictions, std::memory_order_relaxed);
  o.load_us.fetch_add((uint64_t)(r.load_ms * 1000.0), std::memory_order_relaxed);
  if (r.errors > 0) {
    o.errors.fetch_add((uint64_t)r.errors, std::memory_order_relaxed);
    LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] layer %d: %d expert load error(s); aborting\n", site->layer, r.errors);
    offload_report(o, "fatal");
    std::abort();
  }
  if (site->next_layer >= 0)
    stage_post_job(st, site->next_layer, site->next_set, c);
  o.cb_us.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - cb_t0).count(),
                    std::memory_order_relaxed);
  const uint64_t rs = st->resolves.fetch_add(1, std::memory_order_relaxed) + 1;
  if (rs % 4704 == 0)
    LOG_INFO("[GGML-GLM5NEXT-STAGE] progress: jobs=%llu experts_copied=%llu resolves=%llu staged_routes=%llu/%llu not_resident=%llu "
             "wait=%.1f ms direct_pieces=%llu\n",
             (unsigned long long)st->jobs.load(), (unsigned long long)st->copied.load(), (unsigned long long)rs,
             (unsigned long long)st->staged_routes.load(), (unsigned long long)st->routes_seen.load(),
             (unsigned long long)st->skipped_not_resident.load(), st->wait_us.load() / 1000.0, (unsigned long long)st->direct.load());
  if (o.calls.load() - o.last_report_calls.load() >= 544) {
    o.last_report_calls.store(o.calls.load());
    offload_report(o, "progress");
  }
}

bool dynamic_cache_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOESTREAM_GLM5NEXT_DYNAMIC_CACHE");
    return v && v[0] == '1';
  }();
  return enabled;
}

bool route_stats_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOESTREAM_GLM5NEXT_CACHE_STATS");
    return (v && v[0] == '1') || dynamic_cache_enabled();
  }();
  return enabled;
}

int env_device_expert_layers() {
  const char *v = std::getenv("MOESTREAM_GLM5NEXT_GRAPH_PRELOAD_LAYERS");
  if (!v || !v[0])
    return 0;
  char *end = nullptr;
  long n = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || n <= 0)
    return 0;
  return static_cast<int>(std::min<long>(n, 45));
}

bool type_ok(GGUFType type) {
  return static_cast<uint32_t>(type) < static_cast<uint32_t>(GGML_TYPE_COUNT);
}

ggml_tensor *new_tensor(ggml_context *ctx, ggml_type type,
                        const std::vector<int64_t> &shape) {
  switch (shape.size()) {
  case 1:
    return ggml_new_tensor_1d(ctx, type, shape[0]);
  case 2:
    return ggml_new_tensor_2d(ctx, type, shape[0], shape[1]);
  case 3:
    return ggml_new_tensor_3d(ctx, type, shape[0], shape[1], shape[2]);
  case 4:
    return ggml_new_tensor_4d(ctx, type, shape[0], shape[1], shape[2],
                              shape[3]);
  default:
    return nullptr;
  }
}

const GGUFTensor *model_tensor(const StreamingEngine &engine,
                               const std::string &name) {
  auto it = engine.model_file.tensors.find(name);
  return it == engine.model_file.tensors.end() ? nullptr : &it->second;
}

const QuantizedTensor *resident(const StreamingEngine &engine,
                                const std::string &name) {
  auto it = engine.resident_layers.find(name);
  if (it == engine.resident_layers.end() || !it->second.host_ptr)
    return nullptr;
  return &it->second;
}

ggml_backend_buffer_t bind_host(FullGraph &g, ggml_tensor *t, void *ptr,
                                size_t bytes, ggml_backend_buffer_usage usage) {
  if (!t || !ptr || bytes < ggml_nbytes(t))
    return nullptr;
  ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr(ptr, bytes);
  if (!b)
    return nullptr;
  ggml_backend_buffer_set_usage(b, usage);
  if (ggml_backend_tensor_alloc(b, t, ptr) != GGML_STATUS_SUCCESS) {
    ggml_backend_buffer_free(b);
    return nullptr;
  }
  g.host_buffers.push_back(b);
  return b;
}

void *alloc_host(FullGraph &g, size_t bytes) {
  void *p = nullptr;
  if (bytes == 0 || !MemoryGuard::try_reserve_host_bytes(bytes)) {
    if (bytes != 0)
      LOG_WARN(
          "[GGML-GLM5NEXT-FULL-MEM] host graph allocation refused: %.2f MB "
          "(reserved=%.2f GB, budget=%.2f GB)\n",
          bytes / 1e6, MemoryGuard::reserved_host_bytes() / 1e9,
          MemoryGuard::host_budget_bytes() / 1e9);
    return nullptr;
  }
  const int err = posix_memalign(&p, 4096, bytes);
  if (err != 0 || !p) {
    MemoryGuard::release_host_bytes(bytes);
    return nullptr;
  }
  g.owned.push_back({p, bytes, false, bytes});
  return p;
}

void *alloc_expert_source(FullGraph &g, size_t bytes) {
  if (!g.expert_source_base || bytes == 0)
    return nullptr;
  constexpr size_t kAlignment = 4096;
  const size_t aligned =
      (g.expert_source_used + kAlignment - 1) & ~(kAlignment - 1);
  if (aligned > g.expert_source_capacity ||
      bytes > g.expert_source_capacity - aligned)
    return nullptr;
  void *p = static_cast<uint8_t *>(g.expert_source_base) + aligned;
  g.expert_source_used = aligned + bytes;
  ++g.expert_source_tensors;
  return p;
}

ggml_tensor *host_input(FullGraph &g, ggml_context *ctx, ggml_type type,
                        const std::vector<int64_t> &shape, size_t bytes,
                        const char *name) {
  ggml_tensor *t = new_tensor(ctx, type, shape);
  if (!t)
    return nullptr;
  const size_t required = std::max(bytes, ggml_nbytes(t));
  void *p = alloc_host(g, required);
  if (!p || !bind_host(g, t, p, required, GGML_BACKEND_BUFFER_USAGE_COMPUTE))
    return nullptr;
  ggml_set_input(t);
  ggml_set_name(t, name);
  g.host_inputs.push_back(p);
  return t;
}

bool graph_dense_weight_name(const std::string &name) {
  if (name.find("_exps.weight") != std::string::npos ||
      name.find("nextn") != std::string::npos)
    return false;
  if (name.find("indexer.") != std::string::npos ||
      name.find("indexer_compressor_") != std::string::npos)
    return true;
  return name == "output.weight" || name.find("attn") != std::string::npos ||
         name.find("hc_") != std::string::npos ||
         name.find("ssm_") != std::string::npos ||
         name.find("norm") != std::string::npos ||
         name.find("gate_inp") != std::string::npos ||
         name.find("shexp") != std::string::npos ||
         name.find("ffn_gate.weight") != std::string::npos ||
         name.find("ffn_up.weight") != std::string::npos ||
         name.find("ffn_down.weight") != std::string::npos ||
         name.find("exp_probs_b.bias") != std::string::npos;
}

bool arena_alloc(ggml_backend_buffer_t buffer, void *base, size_t capacity,
                 size_t &used, ggml_tensor *t, const void *src,
                 size_t src_bytes) {
  if (!buffer || !base || !t || !src || src_bytes < ggml_nbytes(t))
    return false;
  const auto buft = ggml_backend_buffer_get_type(buffer);
  const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
  const size_t align =
      std::max<size_t>(256, ggml_backend_buft_get_alignment(buft));
  const size_t off = (used + align - 1) & ~(align - 1);
  if (off > capacity || alloc_size > capacity - off)
    return false;
  void *dst = static_cast<uint8_t *>(base) + off;
  if (ggml_backend_tensor_alloc(buffer, t, dst) != GGML_STATUS_SUCCESS)
    return false;
  if (cudaMemcpy(dst, src, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  used = off + alloc_size;
  return true;
}

// The V4 graph builder is kept beside the shared arena helpers, while the
// existing GLM builder's tensor/FFN helpers are defined below it.
ggml_tensor *resident_tensor(FullGraph &g, ggml_context *ctx,
                             const std::string &name);
static bool layer_timing_env() {
  static const bool on = [] {
    const char *v = std::getenv("MOESTREAM_GLM5NEXT_GRAPH_LAYER_TIMING");
    return v && v[0] == '1';
  }();
  return on;
}

ggml_tensor *build_ffn(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *forced_ids);
ggml_tensor *hc_pre(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                    ggml_tensor *fn, ggml_tensor *scale, ggml_tensor *base,
                    ggml_tensor **post, ggml_tensor **comb);
ggml_tensor *hc_post(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                     ggml_tensor *residual, ggml_tensor *post,
                     ggml_tensor *comb);

// ---------------------------------------------------------------------------
// DeepSeek-V4 full prefill graph.
//
// V4 cannot reuse GLM5NEXT's KDA/absorbed-MLA builder: its real Flash file
// has raw SWA attention plus ratio-4 CSA and ratio-128 HCA layers.  The
// builder below keeps the same graph-owned dense/expert arenas and HC/MoE
// nodes, but expresses V4's compressed keys and attention sinks as GGML
// nodes.  For the normal Flash prompt sizes (well below CSA top-k * 4), all
// completed compressed blocks are visible, so the indexer only needs to be
// carried forward for the subsequent decode path; no host top-k loop is
// inserted into the prefill graph.
// ---------------------------------------------------------------------------

bool dsv4_env_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOESTREAM_DEEPSEEK_V4_FULL_GRAPH");
    return v && v[0] == '1';
  }();
  return enabled;
}

bool dsv4_graph_diagnostic_enabled() {
  const char *v = std::getenv("MOESTREAM_DEEPSEEK_V4_GRAPH_DIAGNOSTIC");
  return v && v[0] == '1';
}

int dsv4_env_device_expert_layers() {
  const char *v = std::getenv("MOESTREAM_DEEPSEEK_V4_GRAPH_PRELOAD_LAYERS");
  if (!v || !v[0])
    return 0;
  char *end = nullptr;
  long n = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || n <= 0)
    return 0;
  return static_cast<int>(std::min<long>(n, 43));
}

ggml_tensor *dsv4_host_i32(FullGraph &g, ggml_context *ctx, int n,
                           const char *name, int32_t **out) {
  ggml_tensor *t =
      host_input(g, ctx, GGML_TYPE_I32, {n}, (size_t)n * sizeof(int32_t), name);
  if (t && out)
    *out = static_cast<int32_t *>(t->data);
  return t;
}

ggml_tensor *dsv4_host_i32_2d(FullGraph &g, ggml_context *ctx, int ne0, int ne1,
                              const char *name, int32_t **out) {
  ggml_tensor *t = host_input(g, ctx, GGML_TYPE_I32, {ne0, ne1},
                              (size_t)ne0 * ne1 * sizeof(int32_t), name);
  if (t && out)
    *out = static_cast<int32_t *>(t->data);
  return t;
}

ggml_tensor *dsv4_host_f32(FullGraph &g, ggml_context *ctx,
                           const std::vector<int64_t> &shape, const char *name,
                           float **out) {
  size_t n = 1;
  for (int64_t d : shape)
    n *= (size_t)d;
  ggml_tensor *t =
      host_input(g, ctx, GGML_TYPE_F32, shape, n * sizeof(float), name);
  if (t && out)
    *out = static_cast<float *>(t->data);
  return t;
}

struct Dsv4RopeParams {
  float theta = 10000.0f;
  float freq_scale = 1.0f;
  float ext_factor = 0.0f;
  float attn_factor = 1.0f;
  int n_ctx_orig = 0;
};

Dsv4RopeParams dsv4_rope_params(const FullGraph &g, int ratio) {
  const auto &c = g.engine->model_file.architecture_config;
  Dsv4RopeParams p;
  p.theta = ratio == 0 ? g.engine->model_file.get_config_float("rope.freq_base",
                                                               10000.0f)
                       : c.dsv4_compress_rope_freq_base;
  if (ratio == 0)
    return p;
  p.freq_scale = c.rope_freq_scale;
  p.ext_factor = c.rope_ext_factor;
  p.n_ctx_orig = c.rope_n_ctx_orig;
  // ggml's YaRN kernel multiplies the supplied attention factor by
  // (1+0.1*log(1/freq_scale)) when ext_factor is enabled.  DeepSeek-V4's
  // dsv4_mscale intentionally divides by that term instead, so pre-adjust
  // the argument to make the GGML operation produce the V4 value.
  if (p.ext_factor != 0.0f && p.freq_scale > 0.0f) {
    const float f = 1.0f + 0.1f * std::log(1.0f / p.freq_scale);
    p.attn_factor = c.rope_attn_factor / (f * f);
  } else {
    p.attn_factor = c.rope_attn_factor;
  }
  return p;
}

ggml_tensor *dsv4_rope_tail(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                            int nope, int rope, ggml_tensor *positions,
                            int ratio, bool inverse) {
  if (rope <= 0)
    return x;
  const Dsv4RopeParams p = dsv4_rope_params(g, ratio);
  ggml_tensor *head =
      ggml_view_3d(ctx, x, nope, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
  ggml_tensor *tail = ggml_view_3d(ctx, x, rope, x->ne[1], x->ne[2], x->nb[1],
                                   x->nb[2], (size_t)nope * x->nb[0]);
  ggml_tensor *rot =
      inverse ? ggml_rope_ext_back(ctx, tail, positions, nullptr, rope,
                                   GGML_ROPE_TYPE_NORMAL, p.n_ctx_orig, p.theta,
                                   p.freq_scale, p.ext_factor, p.attn_factor,
                                   32.0f, 1.0f)
              : ggml_rope_ext(ctx, tail, positions, nullptr, rope,
                              GGML_ROPE_TYPE_NORMAL, p.n_ctx_orig, p.theta,
                              p.freq_scale, p.ext_factor, p.attn_factor, 32.0f,
                              1.0f);
  return ggml_concat(ctx, head, rot, 0);
}

ggml_tensor *dsv4_compressed_keys(FullGraph &g, ggml_context *ctx,
                                  ggml_tensor *comp_kv, ggml_tensor *comp_score,
                                  ggml_tensor *comp_ape, ggml_tensor *comp_norm,
                                  int ratio, int key_dim, bool csa,
                                  ggml_tensor *mod_positions,
                                  ggml_tensor *block_positions) {
  const int blocks = g.S / ratio;
  if (blocks <= 0)
    return nullptr;
  const int width = csa ? 2 * key_dim : key_dim;
  const int n_read = blocks * ratio;

  // Match Llama.cpp's state compressor.  The IDs are block-major: for HCA
  // they select every prompt row once; for CSA they select the previous
  // four-row window for every block followed by the current four-row window.
  // This makes the following softmax operate over positions (the first
  // dimension after permute), rather than over key channels.
  const int n_indices = csa ? 2 * n_read : n_read;
  int32_t *index_data = nullptr;
  ggml_tensor *state_read_idxs = dsv4_host_i32(g, ctx, n_indices,
                                               csa ? "dsv4_graph_csa_read_idxs"
                                                   : "dsv4_graph_hca_read_idxs",
                                               &index_data);
  if (!state_read_idxs)
    return nullptr;
  if (!csa) {
    for (int i = 0; i < n_read; ++i)
      index_data[i] = i;
  } else {
    for (int b = 0; b < blocks; ++b) {
      for (int i = 0; i < ratio; ++i) {
        const int j = b * ratio + i;
        // The appended row at index S is the zero/-inf history used for the
        // first CSA block, exactly as dsv4_append_zero_row() in llama.cpp.
        index_data[j] = b == 0 ? g.S : (b - 1) * ratio + i;
        index_data[n_read + j] = j;
      }
    }
  }

  ggml_tensor *kv_source = comp_kv;
  ggml_tensor *score_source = comp_score;
  if (csa) {
    // Use a graph-derived row so this works for both the normal qh-wide CSA
    // compressor and the narrower indexer compressor.  It avoids a separate
    // fixed-width host zero buffer and is the same value-wise operation as
    // llama.cpp's dsv4_append_zero_row().
    ggml_tensor *kv_zero = ggml_reshape_2d(
        ctx, ggml_scale(ctx, ggml_view_1d(ctx, comp_kv, width, 0), 0.0f), width,
        1);
    ggml_tensor *score_zero = ggml_reshape_2d(
        ctx,
        ggml_scale_bias(ctx, ggml_view_1d(ctx, comp_score, width, 0), 0.0f,
                        -INFINITY),
        width, 1);
    kv_source = ggml_concat(ctx, comp_kv, kv_zero, 1);
    score_source = ggml_concat(ctx, comp_score, score_zero, 1);
  }

  ggml_tensor *kv_rows = ggml_get_rows(ctx, kv_source, state_read_idxs);
  ggml_tensor *score_rows = ggml_get_rows(ctx, score_source, state_read_idxs);
  ggml_tensor *values = nullptr;
  ggml_tensor *scores = nullptr;
  if (!csa) {
    ggml_tensor *kv = ggml_reshape_3d(ctx, kv_rows, key_dim, ratio, blocks);
    ggml_tensor *score =
        ggml_reshape_3d(ctx, score_rows, key_dim, ratio, blocks);
    values = ggml_cont(ctx, ggml_permute(ctx, kv, 1, 0, 2, 3));
    scores = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3));
  } else {
    // Each row has [previous/current] channels.  Split the gathered rows
    // before concatenating along the position dimension, as in llama.cpp's
    // build_overlap_compressed_kv_from_state().
    ggml_tensor *kv_prev = ggml_cont(
        ctx, ggml_view_2d(ctx, kv_rows, key_dim, n_read, kv_rows->nb[1], 0));
    ggml_tensor *score_prev =
        ggml_cont(ctx, ggml_view_2d(ctx, score_rows, key_dim, n_read,
                                    score_rows->nb[1], 0));
    const size_t second_half =
        (size_t)n_read * kv_rows->nb[1] + ggml_row_size(kv_rows->type, key_dim);
    const size_t second_score_half = (size_t)n_read * score_rows->nb[1] +
                                     ggml_row_size(score_rows->type, key_dim);
    ggml_tensor *kv_cur =
        ggml_cont(ctx, ggml_view_2d(ctx, kv_rows, key_dim, n_read,
                                    kv_rows->nb[1], second_half));
    ggml_tensor *score_cur =
        ggml_cont(ctx, ggml_view_2d(ctx, score_rows, key_dim, n_read,
                                    score_rows->nb[1], second_score_half));
    ggml_tensor *kv =
        ggml_concat(ctx, ggml_reshape_3d(ctx, kv_prev, key_dim, ratio, blocks),
                    ggml_reshape_3d(ctx, kv_cur, key_dim, ratio, blocks), 1);
    ggml_tensor *score = ggml_concat(
        ctx, ggml_reshape_3d(ctx, score_prev, key_dim, ratio, blocks),
        ggml_reshape_3d(ctx, score_cur, key_dim, ratio, blocks), 1);
    values = ggml_cont(ctx, ggml_permute(ctx, kv, 1, 0, 2, 3));
    scores = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3));
  }

  ggml_tensor *weights = ggml_soft_max(ctx, scores);
  ggml_tensor *comp = ggml_mul(ctx, values, weights);
  comp = ggml_sum_rows(ctx, comp);
  comp = ggml_cont(ctx, ggml_permute(ctx, comp, 1, 0, 2, 3));
  comp = ggml_mul(
      ctx,
      ggml_rms_norm(ctx, comp, g.engine->model_file.architecture_config.eps),
      comp_norm);
  comp = dsv4_rope_tail(g, ctx, comp, key_dim - g.qk_rope, g.qk_rope,
                        block_positions, ratio, false);
  (void)mod_positions;
  (void)comp_ape;
  return ggml_cont_3d(ctx, comp, key_dim, 1, blocks);
}

ggml_tensor *build_dsv4_attention(FullGraph &g, ggml_context *ctx,
                                  ggml_tensor *cur, int layer) {
  const auto &c = g.engine->model_file.architecture_config;
  const std::string p = "blk." + std::to_string(layer) + ".";
  const int qh = g.qk_nope + g.qk_rope;
  const int ratio =
      c.dsv4_compress_ratios.empty()
          ? 0
          : static_cast<int>(c.dsv4_compress_ratios[(size_t)layer]);
  const bool csa = ratio == 4;
  FullGraph::Dsv4LayerTensors lt;
  lt.layer = layer;
  lt.ratio = ratio;

  ggml_tensor *wqa = resident_tensor(g, ctx, p + "attn_q_a.weight");
  ggml_tensor *wqan = resident_tensor(g, ctx, p + "attn_q_a_norm.weight");
  ggml_tensor *wqb = resident_tensor(g, ctx, p + "attn_q_b.weight");
  ggml_tensor *wkv = resident_tensor(g, ctx, p + "attn_kv.weight");
  ggml_tensor *wkvn = resident_tensor(g, ctx, p + "attn_kv_a_norm.weight");
  ggml_tensor *woa = resident_tensor(g, ctx, p + "attn_output_a.weight");
  ggml_tensor *wob = resident_tensor(g, ctx, p + "attn_output_b.weight");
  if (!wqa || !wqan || !wqb || !wkv || !wkvn || !woa || !wob)
    return nullptr;

  ggml_tensor *q_lora = ggml_mul_mat(ctx, wqa, cur);
  q_lora = ggml_mul(ctx, ggml_rms_norm(ctx, q_lora, c.eps), wqan);
  ggml_tensor *q =
      ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wqb, q_lora), qh, g.H, g.S);
  q = ggml_rms_norm(ctx, q, c.eps); // V4's weightless per-head Q norm
  q = dsv4_rope_tail(g, ctx, q, g.qk_nope, g.qk_rope, g.positions, ratio,
                     false);

  ggml_tensor *kv =
      ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wkv, cur), qh, 1, g.S);
  kv = ggml_mul(ctx, ggml_rms_norm(ctx, kv, c.eps), wkvn);
  kv = dsv4_rope_tail(g, ctx, kv, g.qk_nope, g.qk_rope, g.positions, ratio,
                      false);
  lt.raw_kv = kv;

  ggml_tensor *compressed = nullptr;
  int ncompressed = 0;
  if (ratio != 0) {
    ggml_tensor *ckv = resident_tensor(g, ctx, p + "attn_compressor_kv.weight");
    ggml_tensor *cgate =
        resident_tensor(g, ctx, p + "attn_compressor_gate.weight");
    ggml_tensor *cape =
        resident_tensor(g, ctx, p + "attn_compressor_ape.weight");
    ggml_tensor *cnorm =
        resident_tensor(g, ctx, p + "attn_compressor_norm.weight");
    if (!ckv || !cgate || !cape || !cnorm)
      return nullptr;
    const int width = csa ? 2 * qh : qh;
    lt.comp_kv = ggml_mul_mat(ctx, ckv, cur);
    ggml_tensor *gate = ggml_mul_mat(ctx, cgate, cur);
    ggml_tensor *ape_rows =
        ggml_get_rows(ctx, cape, csa ? g.positions_mod4 : g.positions_mod128);
    lt.comp_score = ggml_add(ctx, gate, ape_rows);
    compressed = dsv4_compressed_keys(
        g, ctx, lt.comp_kv, lt.comp_score, cape, cnorm, ratio, qh, csa,
        csa ? g.positions_mod4 : g.positions_mod128,
        csa ? g.block_positions4 : g.block_positions128);
    ncompressed = g.S / ratio;
    lt.keys = compressed;

    if (csa) {
      ggml_tensor *ikv =
          resident_tensor(g, ctx, p + "indexer_compressor_kv.weight");
      ggml_tensor *igate =
          resident_tensor(g, ctx, p + "indexer_compressor_gate.weight");
      ggml_tensor *iape =
          resident_tensor(g, ctx, p + "indexer_compressor_ape.weight");
      ggml_tensor *inorm =
          resident_tensor(g, ctx, p + "indexer_compressor_norm.weight");
      if (!ikv || !igate || !iape || !inorm)
        return nullptr;
      const int idim = c.dsv4_indexer_key_length;
      lt.indexer_comp_kv = ggml_mul_mat(ctx, ikv, cur);
      lt.indexer_comp_score =
          ggml_add(ctx, ggml_mul_mat(ctx, igate, cur),
                   ggml_get_rows(ctx, iape, g.positions_mod4));
      lt.indexer_keys = dsv4_compressed_keys(
          g, ctx, lt.indexer_comp_kv, lt.indexer_comp_score, iape, inorm, 4,
          idim, true, g.positions_mod4, g.block_positions4);
    }
  }

  ggml_tensor *seq = kv;
  if (compressed) {
    seq = ggml_concat(ctx, seq,
                      ggml_reshape_3d(ctx, compressed, qh, 1, ncompressed), 2);
  }
  const int key_count = g.S + ncompressed;

  // Use GGML's native attention-sink input.  Appending a synthetic K/V
  // dimension would change V4's qh=512 head width to 513, which disables the
  // CUDA FATTN kernel on RTX 3090 and sends every attention layer to the CPU.
  ggml_tensor *sink_tensor = resident_tensor(g, ctx, p + "attn_sinks.weight");
  if (!sink_tensor || sink_tensor->type != GGML_TYPE_F32 ||
      sink_tensor->ne[0] != (int64_t)g.H)
    return nullptr;
  // FATTN's CUDA GQA path requires K->ne[1] to be a multiple of 256 for
  // qh=512.  Pad with zero K/V rows and mask those rows out.  The padding is
  // a graph-only workspace shape choice; it never enters the cache state.
  const int padded_key_count = ((key_count + 255) / 256) * 256;
  ggml_tensor *seq_padded = seq;
  if (padded_key_count > key_count) {
    float *pad_data = nullptr;
    ggml_tensor *pad =
        dsv4_host_f32(g, ctx, {qh, 1, padded_key_count - key_count},
                      "dsv4_graph_attention_padding", &pad_data);
    if (!pad)
      return nullptr;
    std::fill(pad_data, pad_data + (size_t)qh * (padded_key_count - key_count),
              0.0f);
    seq_padded = ggml_concat(ctx, seq, pad, 2);
  }
  ggml_tensor *real =
      ggml_repeat_4d(ctx, seq_padded, qh, g.H, padded_key_count, 1);

  float *mask_data = nullptr;
  ggml_tensor *mask =
      dsv4_host_f32(g, ctx, {padded_key_count, g.S, 1, 1},
                    "dsv4_graph_attention_mask_f32", &mask_data);
  if (!mask)
    return nullptr;
  const int window = g.sliding_window > 0 ? g.sliding_window : g.S;
  for (int qpos = 0; qpos < g.S; ++qpos) {
    for (int kpos = 0; kpos < padded_key_count; ++kpos) {
      bool visible = false;
      if (kpos < key_count && kpos < g.S)
        visible = kpos <= qpos && kpos >= std::max(0, qpos - window + 1);
      else if (kpos < key_count)
        visible = (kpos - g.S) < (qpos + 1) / std::max(1, ratio);
      mask_data[(size_t)qpos * padded_key_count + kpos] =
          visible ? 0.0f : -INFINITY;
    }
  }
  // FATTN's CUDA path expects an F16 additive mask.  Keep the host source as
  // F32 while constructing it, then cast once as part of the graph.
  ggml_tensor *mask_f16 = ggml_cast(ctx, mask, GGML_TYPE_F16);
  lt.mask = mask_f16;

  ggml_tensor *q_attn =
      ggml_permute(ctx, ggml_reshape_4d(ctx, q, qh, g.H, g.S, 1), 0, 2, 1, 3);
  ggml_tensor *k_attn = ggml_permute(ctx, real, 0, 2, 1, 3);
  ggml_tensor *v_attn = ggml_permute(ctx, real, 0, 2, 1, 3);
  if (k_attn->type == GGML_TYPE_F32)
    k_attn = ggml_cast(ctx, k_attn, GGML_TYPE_F16);
  if (v_attn->type == GGML_TYPE_F32)
    v_attn = ggml_cast(ctx, v_attn, GGML_TYPE_F16);
  ggml_tensor *attn =
      ggml_flash_attn_ext(ctx, q_attn, k_attn, v_attn, mask_f16,
                          1.0f / std::sqrt((float)qh), 0.0f, 0.0f);
  ggml_flash_attn_ext_add_sinks(attn, sink_tensor);
  ggml_tensor *value = ggml_permute(ctx, attn, 0, 2, 1, 3);
  ggml_tensor *context =
      ggml_cont(ctx, ggml_view_3d(ctx, value, qh, g.H, g.S, value->nb[1],
                                  value->nb[2], 0));
  context = dsv4_rope_tail(g, ctx, context, g.qk_nope, g.qk_rope, g.positions,
                           ratio, true);

  const int groups = std::max(1, c.o_group_count);
  const int rank = c.o_lora_rank;
  const int group_in = (qh * g.H) / groups;
  if (rank <= 0 || groups <= 0 || group_in * groups != qh * g.H ||
      woa->ne[0] != group_in || woa->ne[1] != (int64_t)(groups * rank))
    return nullptr;
  // Keep the grouped output projection as one batched GGML matmul.  This is
  // Llama.cpp's layout: [group_in, groups, tokens] is permuted to
  // [group_in, tokens, groups], projected by [group_in, rank, groups], then
  // restored and made contiguous for wo_b.  The previous loop emitted one
  // matmul and concat per group, creating eight extra graph boundaries.
  ggml_tensor *woa3 = ggml_reshape_3d(ctx, woa, group_in, rank, groups);
  // context is already llama.cpp's [head_dim, n_head, tokens] tensor.  Reshape it
  // directly so the group and token strides are derived from that layout;
  // flattening to 2D first loses the explicit head axis around rope_back.
  ggml_tensor *grouped_input =
      ggml_reshape_3d(ctx, context, group_in, groups, g.S);
  grouped_input = ggml_permute(ctx, grouped_input, 0, 2, 1, 3);
  ggml_tensor *grouped = ggml_mul_mat(ctx, woa3, grouped_input);
  grouped = ggml_permute(ctx, grouped, 0, 2, 1, 3);
  grouped = ggml_cont_2d(ctx, grouped, rank * groups, g.S);
  g.dsv4_layers.push_back(lt);
  return ggml_mul_mat(ctx, wob, grouped);
}

ggml_tensor *dsv4_hash_ids(FullGraph &g, ggml_context *ctx, int layer,
                           const int *token_ids) {
  const auto &c = g.engine->model_file.architecture_config;
  const std::string name =
      "blk." + std::to_string(layer) + ".ffn_gate_tid2eid.weight";
  const QuantizedTensor *table = resident(*g.engine, name);
  if (!table || table->meta.type != GGUFType::I32 ||
      table->meta.shape.size() < 2 ||
      table->meta.shape[0] != (uint64_t)c.topk || !token_ids)
    return nullptr;
  const int64_t n_vocab = (int64_t)table->meta.shape[1];
  const int32_t *src = reinterpret_cast<const int32_t *>(table->host_ptr);
  int32_t *dst = nullptr;
  // ggml_get_rows preserves the ID tensor's shape after its row dimension.
  // Keep IDs as [topk,S], matching the normal argsort_top_k output.  A flat
  // [topk*S] tensor is numerically equivalent for a host loop but changes the
  // routed expert tensor layout and can make MUL_MAT_ID select the wrong
  // token grouping on the CUDA backend.
  ggml_tensor *ids =
      dsv4_host_i32_2d(g, ctx, c.topk, g.S, "dsv4_graph_hash_ids", &dst);
  if (!ids)
    return nullptr;
  for (int s = 0; s < g.S; ++s) {
    const int tok = token_ids[s];
    if (tok < 0 || (int64_t)tok >= n_vocab)
      return nullptr;
    for (int k = 0; k < c.topk; ++k) {
      const int id = src[(size_t)tok * c.topk + k];
      if (id < 0 || id >= c.n_experts)
        return nullptr;
      dst[(size_t)s * c.topk + k] = id;
    }
  }
  return ids;
}

// Real, measured incident (2026-09-09, qwen4exp side, same ggml scheduler
// code path this file also uses): ggml_backend_sched_alloc_graph's internal
// ggml_gallocr_reserve_n_impl does not fail gracefully on a real cudaMalloc
// OOM -- it logs "cudaMalloc failed: out of memory" and then segfaults the
// whole process instead of returning a checkable failure. That happened when
// main.cpp's fallback VRAM cache reservation (a fixed 4GB, unaware of
// device_expert_layers' own VRAM use) left the scheduler's own reserve()
// short by ~1.8GB. main.cpp now sizes that fallback cache against real free
// VRAM and this family's own device_expert_layers estimate, but this check
// stays as a second, independent line of defense: refuse to even attempt the
// allocator call when real free VRAM on the graph's device is below a
// generous floor, and decline to the legacy path instead of risking the same
// crash again.
bool graph_alloc_vram_headroom_ok(const FullGraph &g, const char *log_tag) {
  size_t free_b = 0, total_b = 0;
  int prev_device = -1;
  cudaGetDevice(&prev_device);
  cudaSetDevice(g.device);
  const cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
  if (prev_device >= 0)
    cudaSetDevice(prev_device);
  size_t kMinGraphAllocHeadroomBytes =
      2ull * 1024ull * 1024ull * 1024ull; // 2 GiB
  if (const char *e = std::getenv("MOESTREAM_GLM5NEXT_ALLOC_FLOOR_MIB"))
    kMinGraphAllocHeadroomBytes = (size_t)std::max(0, std::atoi(e)) << 20;
  if (err != cudaSuccess || free_b < kMinGraphAllocHeadroomBytes) {
    LOG_ERROR("%s refusing graph allocator reserve: device %d free=%.2f GiB "
              "< %.2f GiB safety floor (expert arena + dense arena already "
              "committed %.2f GiB) -- falling back to the legacy path\n",
              log_tag, g.device,
              err == cudaSuccess
                  ? static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0)
                  : -1.0,
              static_cast<double>(kMinGraphAllocHeadroomBytes) /
                  (1024.0 * 1024.0 * 1024.0),
              static_cast<double>(g.expert_capacity + g.dense_capacity) /
                  (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  return true;
}

bool build_dsv4_graph(FullGraph &g, const int *token_ids) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_init_params ip = {};
  ip.mem_size = kTensorContextBytes;
  ip.no_alloc = true;
  if (!MemoryGuard::try_reserve_host_bytes(kTensorContextBytes)) {
    LOG_WARN("[GGML-DSV4-FULL-MEM] host graph context refused: %.2f MiB "
             "(reserved=%.2f GiB, budget=%.2f GB)\n",
             (double)kTensorContextBytes / (1024.0 * 1024.0),
             (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
             (double)MemoryGuard::host_budget_bytes() / 1.0e9);
    return false;
  }
  g.context_reservation_bytes = kTensorContextBytes;
  g.ctx = ggml_init(ip);
  if (!g.ctx) {
    MemoryGuard::release_host_bytes(g.context_reservation_bytes);
    g.context_reservation_bytes = 0;
    return false;
  }
  g.graph = ggml_new_graph_custom(g.ctx, kGraphSize, false);
  if (!g.graph)
    return false;

  g.input =
      host_input(g, g.ctx, GGML_TYPE_F32, {g.D, g.HC, g.S},
                 (size_t)g.D * g.HC * g.S * sizeof(float), "dsv4_full_input");
  if (!g.input)
    return false;
  int32_t *pos_data = nullptr;
  g.positions = dsv4_host_i32(g, g.ctx, g.S, "dsv4_full_positions", &pos_data);
  if (!g.positions)
    return false;
  for (int s = 0; s < g.S; ++s)
    pos_data[s] = s;

  bool need4 = false, need128 = false, need_csa = false;
  for (int l = 0; l < g.n_layers; ++l) {
    const int ratio = c.dsv4_compress_ratios.empty()
                          ? 0
                          : (int)c.dsv4_compress_ratios[(size_t)l];
    need4 |= ratio == 4;
    need128 |= ratio == 128;
    need_csa |= ratio == 4;
  }
  auto make_mod = [&](int ratio, const char *name, ggml_tensor **out_t) {
    int32_t *data = nullptr;
    *out_t = dsv4_host_i32(g, g.ctx, g.S, name, &data);
    if (!*out_t)
      return false;
    for (int s = 0; s < g.S; ++s)
      data[s] = s % ratio;
    return true;
  };
  if (need4 && !make_mod(4, "dsv4_full_positions_mod4", &g.positions_mod4))
    return false;
  if (need128 &&
      !make_mod(128, "dsv4_full_positions_mod128", &g.positions_mod128))
    return false;
  auto make_blocks = [&](int ratio, const char *name, ggml_tensor **out_t) {
    const int blocks = g.S / ratio;
    if (blocks <= 0)
      return true;
    int32_t *data = nullptr;
    *out_t = dsv4_host_i32(g, g.ctx, blocks, name, &data);
    if (!*out_t)
      return false;
    for (int b = 0; b < blocks; ++b)
      data[b] = b * ratio;
    return true;
  };
  if (need4 &&
      !make_blocks(4, "dsv4_full_block_positions4", &g.block_positions4))
    return false;
  if (need128 &&
      !make_blocks(128, "dsv4_full_block_positions128", &g.block_positions128))
    return false;
  if (need_csa) {
    float *zv = nullptr;
    float *zs = nullptr;
    g.csa_zero_values = dsv4_host_f32(g, g.ctx, {g.qk_nope + g.qk_rope, 4},
                                      "dsv4_csa_zero_values", &zv);
    g.csa_zero_scores = dsv4_host_f32(g, g.ctx, {g.qk_nope + g.qk_rope, 4},
                                      "dsv4_csa_zero_scores", &zs);
    if (!g.csa_zero_values || !g.csa_zero_scores)
      return false;
    const size_t zero_n = (size_t)(g.qk_nope + g.qk_rope) * 4;
    std::fill(zv, zv + zero_n, 0.0f);
    std::fill(zs, zs + zero_n, -INFINITY);
  }

  ggml_tensor *res = g.input;
  for (int l = 0; l < g.n_layers; ++l) {
    if (MemoryGuard::allocation_blocked())
      return false;
    const std::string p = "blk." + std::to_string(l) + ".";
    ggml_tensor *post = nullptr;
    ggml_tensor *comb = nullptr;
    ggml_tensor *afn = resident_tensor(g, g.ctx, p + "hc_attn_fn.weight");
    ggml_tensor *abase = resident_tensor(g, g.ctx, p + "hc_attn_base.weight");
    ggml_tensor *ascale = resident_tensor(g, g.ctx, p + "hc_attn_scale.weight");
    if (!afn || !abase || !ascale)
      return false;
    ggml_tensor *residual = res;
    ggml_tensor *cur = hc_pre(g, g.ctx, res, afn, ascale, abase, &post, &comb);
    ggml_tensor *an = resident_tensor(g, g.ctx, p + "attn_norm.weight");
    if (!an)
      return false;
    cur = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, cur, c.eps), an);
    ggml_tensor *attn = build_dsv4_attention(g, g.ctx, cur, l);
    if (!attn || g.dsv4_layers.empty() || g.dsv4_layers.back().layer != l)
      return false;
    if (dsv4_graph_diagnostic_enabled())
      g.dsv4_stage_outputs.emplace_back(
          "layer=" + std::to_string(l) + " attention", attn);
    res = hc_post(g, g.ctx, attn, residual, post, comb);

    ggml_tensor *ffn_fn = resident_tensor(g, g.ctx, p + "hc_ffn_fn.weight");
    ggml_tensor *ffn_base = resident_tensor(g, g.ctx, p + "hc_ffn_base.weight");
    ggml_tensor *ffn_scale =
        resident_tensor(g, g.ctx, p + "hc_ffn_scale.weight");
    ggml_tensor *fn = resident_tensor(g, g.ctx, p + "ffn_norm.weight");
    if (!ffn_fn || !ffn_base || !ffn_scale || !fn)
      return false;
    post = comb = nullptr;
    cur = hc_pre(g, g.ctx, res, ffn_fn, ffn_scale, ffn_base, &post, &comb);
    cur = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, cur, c.eps), fn);
    ggml_tensor *forced_ids = l < c.hash_layer_count
                                  ? dsv4_hash_ids(g, g.ctx, l, token_ids)
                                  : nullptr;
    if (l < c.hash_layer_count && !forced_ids)
      return false;
    ggml_tensor *ffn = build_ffn(g, g.ctx, cur, l, forced_ids);
    if (!ffn)
      return false;
    if (dsv4_graph_diagnostic_enabled())
      g.dsv4_stage_outputs.emplace_back("layer=" + std::to_string(l) + " ffn",
                                        ffn);
    res = hc_post(g, g.ctx, ffn, res, post, comb);
    if (dsv4_graph_diagnostic_enabled()) {
      g.dsv4_stage_outputs.emplace_back(
          "layer=" + std::to_string(l) + " residual", res);
      g.dsv4_layer_outputs.push_back(res);
    }
  }

  // DeepSeek-V4's model-level HC head is the learned gated collapse.  The
  // surrounding generation loop still applies output_norm after this graph,
  // exactly as it does after the legacy hc_collapse_output path.
  ggml_tensor *flat = ggml_reshape_2d(g.ctx, res, g.D * g.HC, g.S);
  ggml_tensor *normed = ggml_rms_norm(g.ctx, flat, c.eps);
  ggml_tensor *head_fn = resident_tensor(g, g.ctx, "output_hc_fn.weight");
  ggml_tensor *head_base = resident_tensor(g, g.ctx, "output_hc_base.weight");
  ggml_tensor *head_scale = resident_tensor(g, g.ctx, "output_hc_scale.weight");
  if (!head_fn || !head_base || !head_scale)
    return false;
  ggml_tensor *head_logits = ggml_mul_mat(g.ctx, head_fn, normed);
  ggml_tensor *scale = ggml_view_1d(g.ctx, head_scale, 1, 0);
  ggml_tensor *base = ggml_view_1d(g.ctx, head_base, g.HC, 0);
  ggml_tensor *weights = ggml_scale_bias(
      g.ctx,
      ggml_sigmoid(g.ctx,
                   ggml_add(g.ctx, ggml_mul(g.ctx, head_logits, scale), base)),
      1.0f, c.hc_eps);
  ggml_tensor *out = nullptr;
  for (int h = 0; h < g.HC; ++h) {
    ggml_tensor *stream =
        ggml_view_2d(g.ctx, res, g.D, g.S, res->nb[2], (size_t)h * res->nb[1]);
    ggml_tensor *weight = ggml_view_2d(g.ctx, weights, 1, g.S, weights->nb[1],
                                       (size_t)h * weights->nb[0]);
    ggml_tensor *term = ggml_mul(g.ctx, stream, weight);
    out = out ? ggml_add(g.ctx, out, term) : term;
  }
  g.final_streams = res;
  g.output = out;
  ggml_set_output(g.final_streams);
  ggml_build_forward_expand(g.graph, g.final_streams);
  ggml_set_output(g.output);
  ggml_build_forward_expand(g.graph, g.output);
  for (ggml_tensor *layer_output : g.dsv4_layer_outputs) {
    ggml_set_output(layer_output);
    ggml_build_forward_expand(g.graph, layer_output);
  }
  for (const auto &lt : g.dsv4_layers) {
    ggml_set_output(lt.raw_kv);
    ggml_build_forward_expand(g.graph, lt.raw_kv);
    if (lt.comp_kv) {
      ggml_set_output(lt.comp_kv);
      ggml_build_forward_expand(g.graph, lt.comp_kv);
      ggml_set_output(lt.comp_score);
      ggml_build_forward_expand(g.graph, lt.comp_score);
      if (lt.keys) {
        ggml_set_output(lt.keys);
        ggml_build_forward_expand(g.graph, lt.keys);
      }
    }
    if (lt.indexer_comp_kv) {
      ggml_set_output(lt.indexer_comp_kv);
      ggml_build_forward_expand(g.graph, lt.indexer_comp_kv);
      ggml_set_output(lt.indexer_comp_score);
      ggml_build_forward_expand(g.graph, lt.indexer_comp_score);
      if (lt.indexer_keys) {
        ggml_set_output(lt.indexer_keys);
        ggml_build_forward_expand(g.graph, lt.indexer_keys);
      }
    }
  }
  if (!graph_alloc_vram_headroom_ok(g, "[GGML-DSV4-FULL]"))
    return false;
  if (!ggml_backend_sched_alloc_graph(g.sched, g.graph))
    return false;
  g.graph_workspace_bytes = ggml_backend_sched_get_buffer_size(g.sched, g.cuda);
  int cuda_nodes = 0, cpu_nodes = 0, transitions = 0;
  ggml_backend_t previous = nullptr;
  std::unordered_map<std::string, int> non_cuda;
  for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
    ggml_tensor *node = ggml_graph_node(g.graph, i);
    ggml_backend_t b = ggml_backend_sched_get_tensor_backend(g.sched, node);
    if (b == g.cuda)
      ++cuda_nodes;
    else if (b == g.cpu) {
      ++cpu_nodes;
      ++non_cuda[ggml_op_name(node->op)];
    }
    if (previous && previous != b)
      ++transitions;
    previous = b;
  }
  LOG_INFO("[GGML-DSV4-FULL] scheduler nodes cuda=%d cpu=%d transitions=%d\n",
           cuda_nodes, cpu_nodes, transitions);
  if (!non_cuda.empty()) {
    std::string detail;
    for (const auto &[name, count] : non_cuda) {
      if (!detail.empty())
        detail += ", ";
      detail += name + "=" + std::to_string(count);
    }
    LOG_INFO("[GGML-DSV4-FULL] non-CUDA nodes: %s\n", detail.c_str());
  }
  LOG_INFO("[GGML-DSV4-FULL] graph built layers=%d nodes=%d splits=%d "
           "workspace=%.2f GiB compressed=%zu\n",
           g.n_layers, ggml_graph_n_nodes(g.graph),
           ggml_backend_sched_get_n_splits(g.sched),
           (double)g.graph_workspace_bytes / (1024.0 * 1024.0 * 1024.0),
           std::count_if(g.dsv4_layers.begin(), g.dsv4_layers.end(),
                         [](const auto &x) { return x.ratio != 0; }));
  LOG_INFO(
      "[GGML-DSV4-FULL-MEM] host ledger=%.2f/%.2f GB; graph context=%.2f MiB; "
      "device workspace=%.2f GiB (GPU, outside host_ram_gb)\n",
      (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
      (double)MemoryGuard::host_budget_bytes() / 1.0e9,
      (double)g.context_reservation_bytes / (1024.0 * 1024.0),
      (double)g.graph_workspace_bytes / (1024.0 * 1024.0 * 1024.0));
  g.ready = true;
  return true;
}

std::unique_ptr<FullGraph> g_dsv4_graph;

ggml_tensor *resident_tensor(FullGraph &g, ggml_context *ctx,
                             const std::string &name) {
  const QuantizedTensor *q = resident(*g.engine, name);
  if (!q || !type_ok(q->meta.type))
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : q->meta.shape)
    shape.push_back((int64_t)d);
  ggml_tensor *t = new_tensor(ctx, (ggml_type)q->meta.type, shape);
  if (!t || q->meta.size_bytes < ggml_nbytes(t))
    return nullptr;
  if (graph_dense_weight_name(name)) {
    if (!arena_alloc(g.dense_buffer, g.dense_base, g.dense_capacity,
                     g.dense_used, t, q->host_ptr, q->meta.size_bytes))
      return nullptr;
    ++g.dense_tensors;
  } else if (!bind_host(g, t, q->host_ptr, q->meta.size_bytes,
                        GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
    return nullptr;
  }
  ggml_set_name(t, name.c_str());
  return t;
}

ggml_tensor *load_full_weight(FullGraph &g, ggml_context *ctx,
                              const std::string &name, bool device_expert,
                              bool routed_expert = false) {
  const GGUFTensor *m = model_tensor(*g.engine, name);
  if (!m || !type_ok(m->type) || m->shape.empty())
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : m->shape)
    shape.push_back((int64_t)d);
  ggml_tensor *t = new_tensor(ctx, (ggml_type)m->type, shape);
  if (!t)
    return nullptr;
  const size_t bytes = (size_t)m->size_bytes;
  const size_t required = std::max(bytes, ggml_nbytes(t));
  // Device-resident graph layers use a short-lived aligned read buffer:
  // keeping another DDR copy after the cudaMemcpy would defeat the purpose of
  // the explicit VRAM preload.  All other routed projections use the single
  // graph source arena, while ordinary dense weights retain their existing
  // ownership/lifetime behavior.
  void *p = nullptr;
  bool transient_device_source = false;
  bool transient_host_reserved = false;
  if (device_expert) {
    if (!MemoryGuard::try_reserve_host_bytes(required))
      return nullptr;
    transient_host_reserved = true;
    if (posix_memalign(&p, 4096, required) != 0)
      p = nullptr;
    transient_device_source = true;
  } else if (routed_expert) {
    auto it = g.expert_source_views.find(name);
    p = it == g.expert_source_views.end() ? nullptr : it->second;
  } else {
    p = alloc_host(g, required);
  }
  if (!p) {
    if (transient_host_reserved)
      MemoryGuard::release_host_bytes(required);
    return nullptr;
  }
  // Non-device routed projections are read in one parallel I/O phase before
  // graph construction (see preload_expert_source).  Device-preloaded
  // projections and ordinary static tensors still use the local read path.
  if (device_expert || !routed_expert) {
    const int fd = g.engine->model_file.fd_for_tensor(*m);
    const ssize_t got = direct_io::pread_direct_aware(
        fd, g.engine->model_file.direct_io_active_for(*m),
        g.engine->model_file.direct_io_alignment, p, bytes, m->offset);
    if (got != (ssize_t)bytes) {
      LOG_ERROR("[GGML-GLM5NEXT-FULL] short read for %s: %zd/%zu\n",
                name.c_str(), got, bytes);
      if (transient_device_source)
        std::free(p);
      if (transient_host_reserved)
        MemoryGuard::release_host_bytes(required);
      return nullptr;
    }
  }
  bool ok = false;
  if (device_expert) {
    ok = arena_alloc(g.expert_buffer, g.expert_base, g.expert_capacity,
                     g.expert_used, t, p, bytes);
    if (transient_device_source)
      std::free(p);
    if (transient_host_reserved)
      MemoryGuard::release_host_bytes(required);
  } else if (graph_dense_weight_name(name)) {
    ok = arena_alloc(g.dense_buffer, g.dense_base, g.dense_capacity,
                     g.dense_used, t, p, bytes);
    if (ok)
      ++g.dense_tensors;
  } else {
    ok = bind_host(g, t, p, required, GGML_BACKEND_BUFFER_USAGE_WEIGHTS) !=
         nullptr;
    // Routed projections live in the single pageable DDR source arena.  Do
    // not register each slice separately: cudaHostRegister requires the
    // range to be page aligned and registering 126 slices would both defeat
    // the one-arena design and consume the host pin budget.  Static dense
    // weights keep the existing pinned path.
    if (ok && !routed_expert) {
      const size_t idx = g.owned.size() - 1;
      const cudaError_t err =
          cudaHostRegister(p, required, cudaHostRegisterPortable);
      if (err == cudaSuccess) {
        g.owned[idx].pinned = true;
        g.host_pinned_bytes += required;
      } else {
        ++g.host_pin_failures;
        cudaGetLastError();
      }
    }
  }
  if (!ok)
    return nullptr;
  g.host_weight_bytes += bytes;
  ggml_set_name(t, name.c_str());
  return t;
}

struct PackedGateUpMeta {
  const GGUFTensor *gate = nullptr;
  const GGUFTensor *up = nullptr;
  std::string name;
  size_t bytes = 0;
};

bool packed_gate_up_meta(const StreamingEngine &engine, int layer,
                         PackedGateUpMeta *out) {
  if (!out)
    return false;
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  const std::string gate_name = prefix + "ffn_gate_exps.weight";
  const std::string up_name = prefix + "ffn_up_exps.weight";
  const GGUFTensor *gate = model_tensor(engine, gate_name);
  const GGUFTensor *up = model_tensor(engine, up_name);
  if (!gate || !up || !type_ok(gate->type) || gate->type != up->type ||
      gate->shape.size() != 3 || up->shape != gate->shape ||
      gate->size_bytes == 0 || up->size_bytes == 0 ||
      gate->size_bytes > SIZE_MAX - up->size_bytes)
    return false;

  // Expert tensors are laid out as contiguous expert slices along ne[2].  A
  // native ffn_gate_up_exps tensor interleaves the two projections inside
  // each expert slice: gate[e] followed by up[e].  The source loader below
  // performs that interleave; simply appending the complete gate tensor to
  // the complete up tensor would make expert e read rows belonging to other
  // experts.  Require the packed row layout to be exactly representable by
  // GGML as a sanity check before exposing it to MUL_MAT_ID.
  ggml_init_params ip = {};
  ip.mem_size = 16 * 1024;
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx)
    return false;
  ggml_tensor *packed =
      ggml_new_tensor_3d(ctx, (ggml_type)gate->type, (int64_t)gate->shape[0],
                         (int64_t)gate->shape[1] * 2, (int64_t)gate->shape[2]);
  const size_t expected = packed ? ggml_nbytes(packed) : 0;
  ggml_free(ctx);
  if (!packed || expected != gate->size_bytes + up->size_bytes)
    return false;

  out->gate = gate;
  out->up = up;
  out->name = prefix + "ffn_gate_up_exps.weight";
  out->bytes = expected;
  return true;
}

double full_graph_resident_gb(const char *name) {
  const char *v = std::getenv(name);
  if (!v || !v[0])
    return 0.0;
  char *end = nullptr;
  const double gb = std::strtod(v, &end);
  return end != v && *end == '\0' && gb > 0.0 ? gb : 0.0;
}

size_t graph_host_live_bytes(const StreamingEngine &engine) {
  size_t total = 0;
  const auto add = [&total](size_t bytes) {
    if (bytes > SIZE_MAX - total) {
      total = SIZE_MAX;
      return;
    }
    total += bytes;
  };
  for (const auto &[name, tensor] : engine.resident_layers)
    if (tensor.host_ptr)
      add((size_t)tensor.meta.size_bytes);
  // Pointer-based KV caches do not retain their allocation size.  Their
  // configured sizes are cheap to reconstruct from the architecture and are
  // included explicitly so graph source admission also charges the host KV
  // state already allocated by graph_initializer.
  const auto &c = engine.model_file.architecture_config;
  if (!engine.Lc.empty() && c.kv_lora > 0 && c.n_layers >= 0)
    add((size_t)(c.n_layers + 1) * (size_t)engine.max_t * (size_t)c.kv_lora *
        sizeof(float));
  if (!engine.Rc.empty() && c.qk_rope > 0 && c.n_layers >= 0)
    add((size_t)(c.n_layers + 1) * (size_t)engine.max_t * (size_t)c.qk_rope *
        sizeof(float));
  if (!engine.Kc.empty() && c.n_head_kv > 0 && c.head_dim > 0)
    add((size_t)c.n_layers * (size_t)engine.max_t * (size_t)c.n_head_kv *
        (size_t)c.head_dim * sizeof(float) * 2);
  if (!engine.KVc.empty() && engine.sliding_window > 0 &&
      c.qk_nope + c.qk_rope > 0)
    add((size_t)c.n_layers * (size_t)engine.sliding_window *
        (size_t)(c.qk_nope + c.qk_rope) * sizeof(float));
  for (const auto &v : engine.glm5next_conv_history)
    add(v.size() * sizeof(float));
  for (const auto &v : engine.glm5next_recur_state)
    add(v.size() * sizeof(float));
  for (const auto &state : engine.dsv4_compressed) {
    add(state.keys.size() * sizeof(float));
    add(state.state_kv.size() * sizeof(float));
    add(state.state_score.size() * sizeof(float));
    add(state.indexer_keys.size() * sizeof(float));
    add(state.indexer_state_kv.size() * sizeof(float));
    add(state.indexer_state_score.size() * sizeof(float));
  }
  add(engine.hc_stream_buf.size() * sizeof(float));
  return total;
}

std::string partial_usage_path(const FullGraph &g) {
  const char *env = std::getenv("MOESTREAM_FULL_GRAPH_USAGE_FILE");
  if (env && env[0])
    return env;
  return g.engine->family_ops.family == ModelFamily::DeepSeek_V4
             ? ".moestream_ds4_usage_prefill"
             : ".moestream_usage_prefill_glm53flash";
}

PartialExpertLayer *partial_layer(FullGraph &g, int layer) {
  if (!g.partial_expert_residency || layer < 0 ||
      layer >= (int)g.partial_experts.size())
    return nullptr;
  PartialExpertLayer &p = g.partial_experts[(size_t)layer];
  return p.hot.empty() && p.cold.empty() ? nullptr : &p;
}

const PartialExpertLayer *partial_layer(const FullGraph &g, int layer) {
  if (!g.partial_expert_residency || layer < 0 ||
      layer >= (int)g.partial_experts.size())
    return nullptr;
  const PartialExpertLayer &p = g.partial_experts[(size_t)layer];
  return p.hot.empty() && p.cold.empty() ? nullptr : &p;
}

// Single-snapshot replacement for engine->graph_ram_layer_ready(layer) for
// layers that have a PartialExpertLayer. See PartialExpertLayer::
// ram_ready_cached for why: partial_map(), expert_weight() and
// expert_gate_up_weight() must all see the same answer for one layer within
// one build_graph() call, even though the underlying state is live and
// mutated by background threads. Falls back to the live check for layers
// with no partial residency, where no second tensor's shape depends on the
// same decision.
bool partial_layer_ram_ready(FullGraph &g, int layer) {
  PartialExpertLayer *p = partial_layer(g, layer);
  if (!p)
    return g.engine->graph_ram_layer_ready(layer);
  if (p->ram_ready_cached < 0)
    p->ram_ready_cached = g.engine->graph_ram_layer_ready(layer) ? 1 : 0;
  return p->ram_ready_cached != 0;
}

bool configure_vram_reference_experts(FullGraph &g) {
  if (!g.topology || !g.engine ||
      g.engine->family_ops.family != ModelFamily::GLM5NEXT)
    return false;
  const auto &c = g.engine->model_file.architecture_config;
  if (c.n_experts <= 0 || c.n_layers <= c.first_dense) {
    LOG_WARN("[GGML-GLM5NEXT-CACHE] VRAM reference mode needs the compact, "
             "layer-partitioned expert cache geometry\n");
    return false;
  }
  if (!g.topology->expert_compact_layout)
    LOG_WARN("[GGML-GLM5NEXT-CACHE] cache is not layer-partitioned; using "
             "DDR references only for the graph hot/cold split\n");

  g.partial_experts.assign((size_t)c.n_layers, {});
  size_t aliased_bytes = 0;
  size_t hot_records = 0;
  std::lock_guard<std::recursive_mutex> lock(g.topology->cache_mutex);
  for (int layer = c.first_dense; layer < c.n_layers; ++layer) {
    const size_t raw_stride = g.engine->compute_expert_slab_bytes(layer);
    if (raw_stride == 0)
      continue;
    // Cache slots are padded (4 KiB, or row alignment inside the full-layer
    // prefix), so the physical slot stride is the slot's own size, not the
    // raw slab size.
    size_t stride = raw_stride;
    for (const ExpertLocation &loc : g.topology->expert_vram_cache) {
      if (g.topology->expert_compact_layout && loc.pool_layer == layer &&
          loc.gpu_id == g.device && loc.device_ptr && loc.size >= raw_stride) {
        stride = loc.size;
        break;
      }
    }
    PartialExpertLayer &p = g.partial_experts[(size_t)layer];
    p.vram_slot_by_expert.assign((size_t)c.n_experts, -1);
    p.vram_slot_stride = stride;

    size_t first_offset = SIZE_MAX;
    size_t slot_count = 0;
    for (const ExpertLocation &loc : g.topology->expert_vram_cache) {
      if (g.topology->expert_compact_layout && loc.pool_layer == layer &&
          loc.gpu_id == g.device &&
          loc.device_ptr && loc.size == stride) {
        first_offset = std::min(first_offset, loc.offset);
        ++slot_count;
      }
    }
    if (slot_count > 0) {
      bool region_valid =
          first_offset != SIZE_MAX && slot_count <= SIZE_MAX / stride &&
          first_offset <= SIZE_MAX - slot_count * stride;
      if (region_valid) {
        p.vram_layer_base = nullptr;
        p.vram_layer_slots = slot_count;
        size_t expected_offset = first_offset;
        for (const ExpertLocation &loc : g.topology->expert_vram_cache) {
          if (!g.topology->expert_compact_layout || loc.pool_layer != layer ||
              loc.gpu_id != g.device || !loc.device_ptr || loc.size != stride)
            continue;
          if (loc.offset < first_offset ||
              (loc.offset - first_offset) % stride != 0 ||
              loc.offset >= first_offset + slot_count * stride ||
              loc.offset > SIZE_MAX - stride) {
            region_valid = false;
            break;
          }
          if (!p.vram_layer_base || loc.offset == first_offset) {
            const size_t delta = loc.offset - first_offset;
            const uintptr_t address =
                reinterpret_cast<uintptr_t>(loc.device_ptr);
            if (address < delta) {
              region_valid = false;
              break;
            }
            p.vram_layer_base = reinterpret_cast<uint8_t *>(address - delta);
          }
          expected_offset = std::max(expected_offset, loc.offset + stride);
        }
        region_valid = region_valid && p.vram_layer_base &&
                       expected_offset == first_offset + slot_count * stride;
      }
      if (!region_valid) {
        LOG_WARN("[GGML-GLM5NEXT-CACHE] invalid VRAM cache geometry for "
                 "layer=%d; graph will leave its experts in the cold DDR "
                 "path\n", layer);
        p.vram_layer_base = nullptr;
        p.vram_layer_slots = 0;
        slot_count = 0;
      }
    }

    for (const ExpertLocation &loc : g.topology->expert_vram_cache) {
      if (!g.topology->expert_compact_layout || loc.pool_layer != layer ||
          loc.gpu_id != g.device ||
          loc.layer_idx != layer || loc.expert_id < 0 ||
          loc.expert_id >= c.n_experts || !loc.upload_ready ||
          !loc.device_ptr || loc.size != stride || !p.vram_layer_base)
        continue;
      const size_t delta = loc.offset - first_offset;
      if (delta % stride != 0 || delta / stride >= slot_count) {
        LOG_WARN("[GGML-GLM5NEXT-CACHE] invalid pinned slot map for "
                 "layer=%d expert=%d; leaving it on the cold DDR path\n",
                 layer, loc.expert_id);
        continue;
      }
      const int expert = loc.expert_id;
      if (p.vram_slot_by_expert[(size_t)expert] >= 0)
        continue;
      p.vram_slot_by_expert[(size_t)expert] = (int32_t)(delta / stride);
      p.hot.push_back(expert);
      g.vram_expert_holds.push_back({layer, expert, loc.device_ptr});
      ++hot_records;
      aliased_bytes += stride;
    }

    if (std::getenv("MOESTREAM_GLM5NEXT_DEBUG_PARTIAL") && layer < 8) {
      size_t n_pool = 0, n_ready = 0, n_ptr = 0, n_size = 0, n_gpu = 0;
      for (const ExpertLocation &l2 : g.topology->expert_vram_cache) {
        if (l2.pool_layer != layer) continue;
        ++n_pool; n_ready += l2.upload_ready; n_ptr += l2.device_ptr != nullptr;
        n_size += l2.size == stride; n_gpu += l2.gpu_id == g.device;
      }
      LOG_WARN("[ALIAS-DEBUG] layer=%d compact=%d stride=%zu slot_count=%zu "
               "base=%d pool=%zu ready=%zu ptr=%zu size_ok=%zu gpu_ok=%zu hot=%zu\n",
               layer, (int)g.topology->expert_compact_layout, stride, slot_count,
               p.vram_layer_base != nullptr, n_pool, n_ready, n_ptr, n_size, n_gpu,
               p.hot.size());
    }
    std::vector<uint8_t> is_hot((size_t)c.n_experts, 0);
    for (int expert : p.hot)
      is_hot[(size_t)expert] = 1;
    for (int expert = 0; expert < c.n_experts; ++expert)
      if (!is_hot[(size_t)expert])
        p.cold.push_back(expert);
    p.slot_last_use.assign(p.hot.size(), 0);
    p.slot_in_flight.assign(p.hot.size(), 0);
    p.miss_count.assign((size_t)c.n_experts, 0);
    p.expert_in_flight.assign((size_t)c.n_experts, 0);
  }

  g.partial_expert_residency = true;
  g.vram_reference_experts = true;
  g.dynamic_expert_cache = false;
  g.partial_sources_compacted = false;
  g.partial_hot_bytes = 0; // aliased memory is not a graph allocation
  g.partial_cold_bytes = 0;
  g.partial_hot_records = hot_records;
  g.vram_reference_bytes = aliased_bytes;
  g.partial_usage_file = "<resident-vram-cache>";
  LOG_INFO("[GGML-GLM5NEXT-CACHE] expert storage aliases: %zu resident VRAM "
           "records (%.2f GiB referenced, not copied); DDR cold experts "
           "use existing graph-RAM slabs where available\n",
           hot_records,
           (double)aliased_bytes / (1024.0 * 1024.0 * 1024.0));
  return true;
}

bool configure_partial_experts_base(FullGraph &g) {
  if (configure_vram_reference_experts(g))
    return true;
  const double requested_gb =
      full_graph_resident_gb("MOESTREAM_FULL_GRAPH_RESIDENT_GB");
  if (requested_gb <= 0.0)
    return false;
  const auto &c = g.engine->model_file.architecture_config;
  auto [entries, max_count] =
      StreamingEngine::load_usage(partial_usage_path(g));
  (void)max_count;
  if (entries.empty()) {
    LOG_WARN("[GGML-FULL] partial residency requested but usage file %s is "
             "empty or missing; "
             "keeping all expert weights streamed\n",
             partial_usage_path(g).c_str());
    return false;
  }

  std::vector<std::vector<StreamingEngine::UsageEntry>> by_layer(
      (size_t)std::max(0, c.n_layers));
  for (const auto &e : entries) {
    if (e.layer < c.first_dense || e.layer >= c.n_layers || e.expert < 0 ||
        e.expert >= c.n_experts || e.count == 0)
      continue;
    by_layer[(size_t)e.layer].push_back(e);
  }
  for (auto &v : by_layer) {
    std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) {
      if (a.count != b.count)
        return a.count > b.count;
      return a.expert < b.expert;
    });
    v.erase(std::unique(v.begin(), v.end(),
                        [](const auto &a, const auto &b) {
                          return a.expert == b.expert;
                        }),
            v.end());
  }

  // All GLM5NEXT/DSV4 expert records in a layer are contiguous and have a
  // uniform byte size.  Use the actual GGUF sizes rather than assuming every
  // model has the same per-expert footprint.
  std::vector<size_t> record_bytes((size_t)std::max(0, c.n_layers), 0);
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    PackedGateUpMeta packed;
    const GGUFTensor *down = model_tensor(
        *g.engine, "blk." + std::to_string(l) + ".ffn_down_exps.weight");
    size_t gate_up_bytes = 0;
    if (packed_gate_up_meta(*g.engine, l, &packed)) {
      gate_up_bytes = packed.bytes;
    } else {
      const GGUFTensor *gate = model_tensor(
          *g.engine, "blk." + std::to_string(l) + ".ffn_gate_exps.weight");
      const GGUFTensor *up = model_tensor(
          *g.engine, "blk." + std::to_string(l) + ".ffn_up_exps.weight");
      if (gate && up && gate->size_bytes <= SIZE_MAX - up->size_bytes)
        gate_up_bytes = (size_t)gate->size_bytes + (size_t)up->size_bytes;
    }
    if (!down || c.n_experts <= 0 || gate_up_bytes == 0 ||
        gate_up_bytes % (size_t)c.n_experts != 0 ||
        down->size_bytes % (size_t)c.n_experts != 0 ||
        gate_up_bytes / (size_t)c.n_experts >
            SIZE_MAX - (size_t)down->size_bytes / (size_t)c.n_experts)
      continue;
    record_bytes[(size_t)l] = gate_up_bytes / (size_t)c.n_experts +
                              (size_t)down->size_bytes / (size_t)c.n_experts;
  }

  const size_t budget = requested_gb >= (double)SIZE_MAX
                            ? SIZE_MAX
                            : (size_t)(requested_gb * 1024.0 * 1024.0 * 1024.0);
  g.partial_experts.assign((size_t)std::max(0, c.n_layers), {});
  size_t used = 0;
  size_t hot_records = 0;
  size_t max_rank = 0;
  for (int l = c.first_dense; l < c.n_layers; ++l)
    max_rank = std::max(max_rank, by_layer[(size_t)l].size());

  // Select the same rank from each layer before taking the next rank.  This
  // gives every layer a useful hot set and matches the layer-by-layer graph
  // execution order better than consuming the whole budget in a few layers.
  for (size_t rank = 0; rank < max_rank; ++rank) {
    for (int l = c.first_dense; l < c.n_layers; ++l) {
      if (rank >= by_layer[(size_t)l].size() || record_bytes[(size_t)l] == 0)
        continue;
      const size_t bytes = record_bytes[(size_t)l];
      if (bytes > budget - std::min(used, budget))
        continue;
      g.partial_experts[(size_t)l].hot.push_back(
          by_layer[(size_t)l][rank].expert);
      used += bytes;
      ++hot_records;
    }
  }
  if (hot_records == 0) {
    g.partial_experts.clear();
    LOG_WARN(
        "[GGML-FULL] partial residency budget %.2f GiB was smaller than one "
        "expert record or usage had no compatible layers\n",
        requested_gb);
    return false;
  }

  size_t cold_bytes = 0;
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    PartialExpertLayer &p = g.partial_experts[(size_t)l];
    if (!p.hot.empty()) {
      std::sort(p.hot.begin(), p.hot.end());
      p.hot.erase(std::unique(p.hot.begin(), p.hot.end()), p.hot.end());
    }
    if (record_bytes[(size_t)l] == 0)
      continue;
    std::vector<uint8_t> is_hot((size_t)c.n_experts, 0);
    for (int e : p.hot)
      is_hot[(size_t)e] = 1;
    for (int e = 0; e < c.n_experts; ++e)
      if (!is_hot[(size_t)e])
        p.cold.push_back(e);
    p.slot_last_use.assign(p.hot.size(), 0);
    p.slot_in_flight.assign(p.hot.size(), 0);
    p.miss_count.assign((size_t)c.n_experts, 0);
    p.expert_in_flight.assign((size_t)c.n_experts, 0);
    for (size_t slot = 0; slot < p.hot.size(); ++slot) {
      const int expert = p.hot[slot];
      const auto usage = std::find_if(
          by_layer[(size_t)l].begin(), by_layer[(size_t)l].end(),
          [expert](const StreamingEngine::UsageEntry &entry) {
            return entry.expert == expert;
          });
      if (usage != by_layer[(size_t)l].end()) {
        const size_t rank = (size_t)std::distance(
            by_layer[(size_t)l].begin(), usage);
        const uint64_t initial =
            by_layer[(size_t)l].size() - std::min(
                rank, by_layer[(size_t)l].size());
        p.slot_last_use[slot] = (int)std::min<uint64_t>(initial, INT_MAX);
        g.dynamic_clock = std::max<uint64_t>(
            g.dynamic_clock, by_layer[(size_t)l].size() + 1);
      }
    }
    cold_bytes += record_bytes[(size_t)l] * p.cold.size();
  }
  g.partial_expert_residency = true;
  g.dynamic_expert_cache = dynamic_cache_enabled();
  g.partial_hot_bytes = used;
  g.partial_cold_bytes = cold_bytes;
  g.partial_hot_records = hot_records;
  g.partial_usage_file = partial_usage_path(g);
  LOG_INFO("[GGML-FULL] partial resident set: %.2f GiB hot, %.2f GiB cold, "
           "%zu hot expert records across %d layers, usage=%s%s\n",
           (double)used / (1024.0 * 1024.0 * 1024.0),
           (double)cold_bytes / (1024.0 * 1024.0 * 1024.0), hot_records,
           c.n_layers - c.first_dense, g.partial_usage_file.c_str(),
           g.dynamic_expert_cache ? ", dynamic LRU enabled" : "");
  return true;
}

void assign_gpu1_hot(FullGraph &g, double gib);

// Same hot/cold split for every caller (boot candidate, prompt graph, decode graph) so their layouts compare equal.
bool configure_partial_experts(FullGraph &g) {
  if (!configure_partial_experts_base(g))
    return false;
  if (g.partial_expert_residency && ggml_shared_backend(1)) {
    const char *gh = std::getenv("MOESTREAM_GLM5NEXT_GPU1_HOT_GIB");
    if (gh && gh[0])
      assign_gpu1_hot(g, std::atof(gh));
  }
  return true;
}

bool read_tensor_direct(const StreamingEngine &engine, const GGUFTensor &meta,
                        void *dst) {
  const int fd = engine.model_file.fd_for_tensor(meta);
  const bool is_direct = engine.model_file.direct_io_active_for(meta);
  const ssize_t got = direct_io::pread_direct_aware(
      fd, is_direct, engine.model_file.direct_io_alignment, dst,
      (size_t)meta.size_bytes, meta.offset);
  const bool ok = got == (ssize_t)meta.size_bytes;
#if defined(__linux__)
  // When a GGUF is not 4096-aligned, the fallback pread path populates the
  // kernel page cache in addition to the graph-owned DDR source.  Releasing
  // those file pages after each completed tensor keeps a 78 GiB source load
  // from briefly becoming a second 78 GiB resident copy.  O_DIRECT already
  // bypasses this cache, so the advice is only needed for the default path.
  if (ok && !is_direct)
    (void)posix_fadvise(fd, (off_t)meta.offset, (off_t)meta.size_bytes,
                        POSIX_FADV_DONTNEED);
#endif
  return ok;
}

bool read_pack_gate_up(const StreamingEngine &engine, const GGUFTensor &gate,
                       const GGUFTensor &up, void *dst) {
  if (!dst || gate.shape.size() != 3 || up.shape != gate.shape ||
      gate.shape[2] == 0 || gate.size_bytes % gate.shape[2] != 0 ||
      up.size_bytes % up.shape[2] != 0)
    return false;
  const size_t gate_bytes = (size_t)gate.size_bytes;
  const size_t up_bytes = (size_t)up.size_bytes;
  if (gate_bytes > SIZE_MAX - up_bytes)
    return false;
  const size_t scratch_bytes = gate_bytes + up_bytes;
  if (!MemoryGuard::try_reserve_host_bytes(scratch_bytes))
    return false;
  void *scratch = nullptr;
  if (posix_memalign(&scratch, 4096, scratch_bytes) != 0 || !scratch) {
    MemoryGuard::release_host_bytes(scratch_bytes);
    return false;
  }
  const bool ok_gate = read_tensor_direct(engine, gate, scratch);
  const bool ok_up =
      ok_gate && read_tensor_direct(
                     engine, up, static_cast<uint8_t *>(scratch) + gate_bytes);
  if (!ok_up) {
    std::free(scratch);
    MemoryGuard::release_host_bytes(scratch_bytes);
    return false;
  }

  const size_t n_experts = (size_t)gate.shape[2];
  const size_t gate_expert_bytes = gate_bytes / n_experts;
  const size_t up_expert_bytes = up_bytes / n_experts;
  uint8_t *out = static_cast<uint8_t *>(dst);
  const uint8_t *gate_src = static_cast<const uint8_t *>(scratch);
  const uint8_t *up_src = gate_src + gate_bytes;
  for (size_t e = 0; e < n_experts; ++e) {
    std::memcpy(out, gate_src + e * gate_expert_bytes, gate_expert_bytes);
    out += gate_expert_bytes;
    std::memcpy(out, up_src + e * up_expert_bytes, up_expert_bytes);
    out += up_expert_bytes;
  }
  std::free(scratch);
  MemoryGuard::release_host_bytes(scratch_bytes);
  return true;
}

bool read_tensor_range_direct(const StreamingEngine &engine,
                              const GGUFTensor &meta, size_t relative_offset,
                              void *dst, size_t bytes) {
  if (!dst || relative_offset > (size_t)meta.size_bytes ||
      bytes > (size_t)meta.size_bytes - relative_offset ||
      meta.offset > UINT64_MAX - relative_offset)
    return false;
  const int fd = engine.model_file.fd_for_tensor(meta);
  const bool is_direct = engine.model_file.direct_io_active_for(meta);
  const uint64_t offset = meta.offset + relative_offset;
  const ssize_t got = direct_io::pread_direct_aware(
      fd, is_direct, engine.model_file.direct_io_alignment, dst, bytes, offset);
  const bool ok = got == (ssize_t)bytes;
#if defined(__linux__)
  if (ok && !is_direct)
    (void)posix_fadvise(fd, (off_t)offset, (off_t)bytes,
                        POSIX_FADV_DONTNEED);
#endif
  return ok;
}

bool read_tensor_selected_experts(const StreamingEngine &engine,
                                  const GGUFTensor &meta, int n_experts,
                                  const std::vector<int> &selected,
                                  void *dst) {
  if (selected.empty())
    return true;
  if (!dst || n_experts <= 0 || meta.shape.size() != 3 ||
      meta.shape[2] != (uint64_t)n_experts ||
      meta.size_bytes % (size_t)n_experts != 0)
    return false;
  const size_t record_bytes = (size_t)meta.size_bytes / (size_t)n_experts;
  constexpr size_t kMaxBatchExperts = 8;
  size_t i = 0;
  while (i < selected.size()) {
    if (selected[i] < 0 || selected[i] >= n_experts)
      return false;
    size_t run_end = i + 1;
    while (run_end < selected.size() &&
           selected[run_end] == selected[run_end - 1] + 1)
      ++run_end;
    while (i < run_end) {
      const size_t count = std::min(kMaxBatchExperts, run_end - i);
      const size_t file_offset = (size_t)selected[i] * record_bytes;
      const size_t bytes = count * record_bytes;
      auto *out = static_cast<uint8_t *>(dst) + i * record_bytes;
      if (!read_tensor_range_direct(engine, meta, file_offset, out, bytes))
        return false;
      i += count;
    }
  }
  return true;
}

bool read_pack_gate_up_selected(const StreamingEngine &engine,
                               const GGUFTensor &gate, const GGUFTensor &up,
                               int n_experts,
                               const std::vector<int> &selected, void *dst) {
  if (selected.empty())
    return true;
  if (!dst || gate.shape.size() != 3 || up.shape != gate.shape ||
      n_experts <= 0 || gate.shape[2] != (uint64_t)n_experts ||
      gate.size_bytes % (size_t)n_experts != 0 ||
      up.size_bytes % (size_t)n_experts != 0)
    return false;

  const size_t gate_record = (size_t)gate.size_bytes / (size_t)n_experts;
  const size_t up_record = (size_t)up.size_bytes / (size_t)n_experts;
  constexpr size_t kMaxBatchExperts = 8;
  const size_t max_batch = std::min(kMaxBatchExperts, selected.size());
  if (gate_record > SIZE_MAX / max_batch ||
      up_record > SIZE_MAX / max_batch)
    return false;
  const size_t gate_scratch_bytes = gate_record * max_batch;
  const size_t up_scratch_bytes = up_record * max_batch;
  if (gate_scratch_bytes > SIZE_MAX - up_scratch_bytes)
    return false;
  const size_t scratch_bytes = gate_scratch_bytes + up_scratch_bytes;
  if (!MemoryGuard::try_reserve_host_bytes(scratch_bytes))
    return false;
  void *scratch = nullptr;
  if (posix_memalign(&scratch, 4096, scratch_bytes) != 0 || !scratch) {
    MemoryGuard::release_host_bytes(scratch_bytes);
    return false;
  }

  auto *gate_scratch = static_cast<uint8_t *>(scratch);
  auto *up_scratch = gate_scratch + gate_scratch_bytes;
  auto *out = static_cast<uint8_t *>(dst);
  size_t i = 0;
  bool ok = true;
  while (i < selected.size() && ok) {
    if (selected[i] < 0 || selected[i] >= n_experts) {
      ok = false;
      break;
    }
    size_t run_end = i + 1;
    while (run_end < selected.size() &&
           selected[run_end] == selected[run_end - 1] + 1)
      ++run_end;
    while (i < run_end && ok) {
      const size_t count = std::min(kMaxBatchExperts, run_end - i);
      const size_t file_offset = (size_t)selected[i] * gate_record;
      ok = read_tensor_range_direct(engine, gate, file_offset, gate_scratch,
                                    count * gate_record) &&
           read_tensor_range_direct(engine, up,
                                    (size_t)selected[i] * up_record,
                                    up_scratch, count * up_record);
      if (!ok)
        break;
      for (size_t j = 0; j < count; ++j) {
        uint8_t *record = out + (i + j) * (gate_record + up_record);
        std::memcpy(record, gate_scratch + j * gate_record, gate_record);
        std::memcpy(record + gate_record, up_scratch + j * up_record,
                    up_record);
      }
      i += count;
    }
  }
  std::free(scratch);
  MemoryGuard::release_host_bytes(scratch_bytes);
  return ok;
}

ggml_tensor *load_gate_up_weight(FullGraph &g, ggml_context *ctx, int layer,
                                 bool device_expert) {
  PackedGateUpMeta meta;
  if (!packed_gate_up_meta(*g.engine, layer, &meta))
    return nullptr;

  if (const PartialExpertLayer *p = partial_layer(g, layer)) {
    if (p->cold.empty())
      return nullptr;
    if (g.dynamic_expert_cache && g.expert_source_mmap)
      return nullptr;
    auto ptr_it = g.expert_source_views.find(meta.name);
    auto bytes_it = g.expert_source_view_bytes.find(meta.name);
    if (ptr_it == g.expert_source_views.end() ||
        bytes_it == g.expert_source_view_bytes.end())
      return nullptr;
    const int64_t cold_experts =
        g.dynamic_expert_cache ? (int64_t)meta.gate->shape[2]
                               : (int64_t)p->cold.size();
    ggml_tensor *cold = ggml_new_tensor_3d(
        ctx, (ggml_type)meta.gate->type, (int64_t)meta.gate->shape[0],
        (int64_t)meta.gate->shape[1] * 2, cold_experts);
    if (!cold || bytes_it->second < ggml_nbytes(cold) ||
        !bind_host(g, cold, ptr_it->second, bytes_it->second,
                   GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
      return nullptr;
    g.host_weight_bytes += bytes_it->second;
    ggml_set_name(cold, (meta.name + ".cold").c_str());
    return cold;
  }

  ggml_tensor *t = ggml_new_tensor_3d(
      ctx, (ggml_type)meta.gate->type, (int64_t)meta.gate->shape[0],
      (int64_t)meta.gate->shape[1] * 2, (int64_t)meta.gate->shape[2]);
  if (!t)
    return nullptr;

  void *p = nullptr;
  bool transient = false;
  bool transient_reserved = false;
  if (device_expert) {
    if (!MemoryGuard::try_reserve_host_bytes(meta.bytes))
      return nullptr;
    transient_reserved = true;
    if (posix_memalign(&p, 4096, meta.bytes) != 0)
      p = nullptr;
    transient = true;
  } else {
    auto it = g.expert_source_views.find(meta.name);
    p = it == g.expert_source_views.end() ? nullptr : it->second;
  }
  if (!p) {
    if (transient_reserved)
      MemoryGuard::release_host_bytes(meta.bytes);
    return nullptr;
  }

  if (device_expert) {
    if (!read_pack_gate_up(*g.engine, *meta.gate, *meta.up, p)) {
      LOG_ERROR("[GGML-GLM5NEXT-FULL] short read for packed %s\n",
                meta.name.c_str());
      if (transient)
        std::free(p);
      if (transient_reserved)
        MemoryGuard::release_host_bytes(meta.bytes);
      return nullptr;
    }
  }

  bool ok = false;
  if (device_expert) {
    ok = arena_alloc(g.expert_buffer, g.expert_base, g.expert_capacity,
                     g.expert_used, t, p, meta.bytes);
    if (transient)
      std::free(p);
    if (transient_reserved)
      MemoryGuard::release_host_bytes(meta.bytes);
  } else {
    ok = bind_host(g, t, p, meta.bytes, GGML_BACKEND_BUFFER_USAGE_WEIGHTS) !=
         nullptr;
  }
  if (!ok)
    return nullptr;
  g.host_weight_bytes += meta.bytes;
  ggml_set_name(t, meta.name.c_str());
  return t;
}

ggml_tensor *load_cold_weight(FullGraph &g, ggml_context *ctx, int layer,
                              const char *stem) {
  const PartialExpertLayer *p = partial_layer(g, layer);
  if (!p || p->cold.empty())
    return nullptr;
  const std::string name = "blk." + std::to_string(layer) + "." + stem;
  const GGUFTensor *m = model_tensor(*g.engine, name);
  auto ptr_it = g.expert_source_views.find(name);
  auto bytes_it = g.expert_source_view_bytes.find(name);
  if (!m || ptr_it == g.expert_source_views.end() ||
      bytes_it == g.expert_source_view_bytes.end() || m->shape.size() != 3)
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : m->shape)
    shape.push_back((int64_t)d);
  shape[2] = g.dynamic_expert_cache ? (int64_t)m->shape[2]
                                    : (int64_t)p->cold.size();
  ggml_tensor *t = new_tensor(ctx, (ggml_type)m->type, shape);
  if (!t || bytes_it->second < ggml_nbytes(t) ||
      !bind_host(g, t, ptr_it->second, bytes_it->second,
                 GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
    return nullptr;
  g.host_weight_bytes += bytes_it->second;
  ggml_set_name(t, name.c_str());
  return t;
}

template <typename Loader>
bool arena_load_selected(FullGraph &g, ggml_tensor *t, Loader &&load) {
  if (!t)
    return false;
  const size_t bytes = ggml_nbytes(t);
  if (!MemoryGuard::try_reserve_host_bytes(bytes))
    return false;
  void *scratch = nullptr;
  if (posix_memalign(&scratch, 4096, bytes) != 0 || !scratch) {
    MemoryGuard::release_host_bytes(bytes);
    return false;
  }
  const bool ok = load(scratch) &&
                  arena_alloc(g.expert_buffer, g.expert_base,
                              g.expert_capacity, g.expert_used, t, scratch,
                              bytes);
  std::free(scratch);
  MemoryGuard::release_host_bytes(bytes);
  return ok;
}

ggml_tensor *make_hot_weight(FullGraph &g, ggml_context *ctx, int layer,
                             const char *stem) {
  const PartialExpertLayer *p = partial_layer(g, layer);
  if (!p || p->hot.empty())
    return nullptr;
  const std::string name = "blk." + std::to_string(layer) + "." + stem;
  const GGUFTensor *m = model_tensor(*g.engine, name);
  if (!m || m->shape.size() != 3)
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : m->shape)
    shape.push_back((int64_t)d);
  shape[2] = (int64_t)p->hot.size();
  ggml_tensor *t = new_tensor(ctx, (ggml_type)m->type, shape);
  if (!t || !arena_load_selected(g, t, [&](void *dst) {
        return read_tensor_selected_experts(
            *g.engine, *m, g.engine->model_file.architecture_config.n_experts,
            p->hot, dst);
      }))
    return nullptr;
  ggml_set_name(t, (name + ".hot").c_str());
  return t;
}

ggml_tensor *make_hot_gate_up(FullGraph &g, ggml_context *ctx, int layer) {
  const PartialExpertLayer *p = partial_layer(g, layer);
  if (!p || p->hot.empty())
    return nullptr;
  PackedGateUpMeta meta;
  if (!packed_gate_up_meta(*g.engine, layer, &meta))
    return nullptr;
  ggml_tensor *t = ggml_new_tensor_3d(
      ctx, (ggml_type)meta.gate->type, (int64_t)meta.gate->shape[0],
      (int64_t)meta.gate->shape[1] * 2, (int64_t)p->hot.size());
  if (!t || !arena_load_selected(g, t, [&](void *dst) {
        return read_pack_gate_up_selected(
            *g.engine, *meta.gate, *meta.up,
            g.engine->model_file.architecture_config.n_experts, p->hot, dst);
      }))
    return nullptr;
  ggml_set_name(t, (meta.name + ".hot").c_str());
  return t;
}

ggml_tensor *make_vram_ref_weight(FullGraph &g, ggml_context *ctx, int layer,
                                  const char *stem) {
  const PartialExpertLayer *p = partial_layer(g, layer);
  if (!p || p->hot.empty() || !p->vram_layer_base ||
      p->vram_layer_slots == 0 || p->vram_slot_stride == 0 ||
      !g.expert_buffer)
    return nullptr;
  const std::string name = "blk." + std::to_string(layer) + "." + stem;
  const GGUFTensor *meta = model_tensor(*g.engine, name);
  const size_t n_experts =
      (size_t)g.engine->model_file.architecture_config.n_experts;
  if (!meta || meta->shape.size() != 3 || n_experts == 0 ||
      meta->size_bytes % n_experts != 0)
    return nullptr;
  size_t offset = 0;
  if (std::strcmp(stem, "ffn_up_exps.weight") == 0) {
    const GGUFTensor *gate = model_tensor(
        *g.engine, "blk." + std::to_string(layer) + ".ffn_gate_exps.weight");
    if (!gate || gate->size_bytes % n_experts != 0)
      return nullptr;
    offset = (size_t)gate->size_bytes / n_experts;
  } else if (std::strcmp(stem, "ffn_down_exps.weight") == 0) {
    const GGUFTensor *gate = model_tensor(
        *g.engine, "blk." + std::to_string(layer) + ".ffn_gate_exps.weight");
    const GGUFTensor *up = model_tensor(
        *g.engine, "blk." + std::to_string(layer) + ".ffn_up_exps.weight");
    if (!gate || !up || gate->size_bytes % n_experts != 0 ||
        up->size_bytes % n_experts != 0)
      return nullptr;
    const size_t gate_bytes = (size_t)gate->size_bytes / n_experts;
    const size_t up_bytes = (size_t)up->size_bytes / n_experts;
    if (gate_bytes > SIZE_MAX - up_bytes)
      return nullptr;
    offset = gate_bytes + up_bytes;
  } else if (std::strcmp(stem, "ffn_gate_exps.weight") != 0) {
    return nullptr;
  }
  const size_t per_expert_bytes = (size_t)meta->size_bytes / n_experts;
  if (offset > p->vram_slot_stride ||
      per_expert_bytes > p->vram_slot_stride - offset ||
      p->vram_layer_slots > SIZE_MAX / p->vram_slot_stride)
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t dim : meta->shape)
    shape.push_back((int64_t)dim);
  shape[2] = (int64_t)p->vram_layer_slots;
  ggml_tensor *t = new_tensor(ctx, (ggml_type)meta->type, shape);
  if (!t)
    return nullptr;
  // ggml-cuda mmvq derives the per-expert offset as nb[2]/type_size in block
  // units, so a stride that is not a whole number of rows silently misaddresses
  // every nonzero slot (NaN decode output). Refuse the alias instead.
  // Only nb[2] must be a whole number of rows; the offset just positions the
  // base pointer inside the slot (down starts after gate+up, whose row size
  // differs), so it only needs the block struct's 2-byte alignment.
  if (p->vram_slot_stride % ggml_row_size(t->type, t->ne[0]) != 0 ||
      offset % 2 != 0) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      LOG_STREAM_WARN() << "[GLM5NEXT-GRAPH] (stride=" << p->vram_slot_stride
                        << " offset=" << offset << " row_size="
                        << ggml_row_size(t->type, t->ne[0]) << " type_size="
                        << ggml_type_size(t->type) << " type="
                        << ggml_type_name(t->type) << ")\n";
      LOG_STREAM_WARN() << "[GLM5NEXT-GRAPH] VRAM slot stride "
                        << p->vram_slot_stride << " is not row-aligned for "
                        << name << "; set vram_cache_full_layer_prefix to "
                        << "cover all MoE layers. Not aliasing resident slots.\n";
    }
    return nullptr;
  }
  t->nb[2] = p->vram_slot_stride;
  t->nb[3] = t->nb[2] * (size_t)t->ne[2];
  t->data = p->vram_layer_base + offset;
  // The buffer type tags this tensor for the graph CUDA backend. The actual
  // data pointer remains owned by GpuTopology's expert cache pool.
  t->buffer = g.expert_buffer;
  ggml_set_name(t, (name + ".vram_cache_ref").c_str());
  return t;
}

bool prepare_partial_hot(FullGraph &g, ggml_context *ctx, int layer) {
  PartialExpertLayer *p = partial_layer(g, layer);
  if (!p || p->hot.empty())
    return true;
  if (p->hot_down || p->hot_gate_up || p->hot_gate)
    return true;
  if (g.vram_reference_experts) {
    p->hot_gate = make_vram_ref_weight(g, ctx, layer,
                                       "ffn_gate_exps.weight");
    p->hot_up = make_vram_ref_weight(g, ctx, layer,
                                     "ffn_up_exps.weight");
    p->hot_down = make_vram_ref_weight(g, ctx, layer,
                                       "ffn_down_exps.weight");
    return p->hot_gate && p->hot_up && p->hot_down;
  }
  PackedGateUpMeta packed;
  // The mmap-backed dynamic source intentionally retains gate and up as
  // separate read-only file views.  Keep the hot slots in the same layout so
  // asynchronous promotion can copy each projection directly into its slot;
  // a packed hot tensor cannot be paired with separate cold source tensors.
  if (g.dynamic_expert_cache && g.expert_source_mmap) {
    p->hot_gate = make_hot_weight(g, ctx, layer, "ffn_gate_exps.weight");
    p->hot_up = make_hot_weight(g, ctx, layer, "ffn_up_exps.weight");
    p->hot_down = make_hot_weight(g, ctx, layer, "ffn_down_exps.weight");
    return p->hot_gate && p->hot_up && p->hot_down;
  }
  if (packed_gate_up_meta(*g.engine, layer, &packed)) {
    p->hot_gate_up = make_hot_gate_up(g, ctx, layer);
    p->hot_down = make_hot_weight(g, ctx, layer, "ffn_down_exps.weight");
    return p->hot_gate_up && p->hot_down;
  }
  p->hot_gate = make_hot_weight(g, ctx, layer, "ffn_gate_exps.weight");
  p->hot_up = make_hot_weight(g, ctx, layer, "ffn_up_exps.weight");
  p->hot_down = make_hot_weight(g, ctx, layer, "ffn_down_exps.weight");
  return p->hot_gate && p->hot_up && p->hot_down;
}


ggml_tensor *partial_map_hot1(FullGraph &g, ggml_context *ctx, int layer,
                              bool mask) {
  PartialExpertLayer *p = partial_layer(g, layer);
  if (!p)
    return nullptr;
  ggml_tensor *&slot = mask ? p->hot1_mask : p->hot1_map;
  if (slot)
    return slot;
  const size_t n = (size_t)std::max(0, g.engine->model_file.architecture_config.n_experts);
  ggml_tensor *t = ggml_new_tensor_3d(ctx, mask ? GGML_TYPE_F32 : GGML_TYPE_I32, 1,
                                      (int64_t)n, g.S);
  if (!t)
    return nullptr;
  std::vector<int32_t> index_of_expert(n, -1);
  std::vector<uint8_t> owned(n, 0);
  for (size_t i = 0; i < p->hot1.size(); ++i) {
    const int e = p->hot1[i];
    if (e >= 0 && (size_t)e < n) {
      index_of_expert[(size_t)e] = (int32_t)i;
      owned[(size_t)e] = 1;
    }
  }
  const moe_partial::ExpertBranchMap branch_map = moe_partial::ExpertBranchMap::build(
      index_of_expert, g.S, g.decode_mode && g.S == 1, owned,
      moe_partial::SentinelPolicy::Auto);
  const void *data = mask ? (const void *)branch_map.route_mask().data()
                          : (const void *)branch_map.route_ids().data();
  if (!arena_alloc(g.dense_buffer, g.dense_base, g.dense_capacity, g.dense_used,
                   t, data, ggml_nbytes(t)))
    return nullptr;
  ggml_set_name(t, (std::string("full_graph_hot1_") + (mask ? "mask_" : "map_") +
                    std::to_string(layer)).c_str());
  slot = t;
  return t;
}

// GPU1 hot weights live in buffers that outlive graph rebuilds (key: layer*3+plane).
std::map<int, ggml_backend_buffer_t> g_hot1_buffers;

ggml_tensor *hot1_weight(FullGraph &g, ggml_context *ctx, int layer,
                         const char *stem, int plane) {
  PartialExpertLayer *p = partial_layer(g, layer);
  const GGUFTensor *m = model_tensor(*g.engine, "blk." + std::to_string(layer) + "." + stem);
  const size_t n_experts = (size_t)g.engine->model_file.architecture_config.n_experts;
  if (!p || p->hot1.empty() || !m || !type_ok(m->type) || m->shape.size() != 3 ||
      n_experts == 0 || m->size_bytes % n_experts != 0)
    return nullptr;
  const size_t per_expert = (size_t)m->size_bytes / n_experts;
  std::vector<int64_t> shape;
  for (uint64_t d : m->shape)
    shape.push_back((int64_t)d);
  shape[2] = (int64_t)p->hot1.size();
  ggml_tensor *t = new_tensor(ctx, (ggml_type)m->type, shape);
  if (!t || ggml_nbytes(t) != per_expert * p->hot1.size())
    return nullptr;
  const int key = layer * 3 + plane;
  ggml_backend_buffer_t buf = nullptr;
  auto it = g_hot1_buffers.find(key);
  const bool fresh = it == g_hot1_buffers.end();
  if (fresh) {
    // MOESTREAM_GLM5NEXT_GPU1_HOT_ON_GPU0=1 is a verification aid: same branch, buffers on the main GPU, so its
    // logits can be compared against the GPU1 placement with identical kernels.
    static const bool on_gpu0 = std::getenv("MOESTREAM_GLM5NEXT_GPU1_HOT_ON_GPU0") != nullptr;
    buf = ggml_backend_buft_alloc_buffer(
        ggml_backend_get_default_buffer_type(on_gpu0 ? g.cuda : g.gpu1), ggml_nbytes(t) + 4096);
    if (!buf)
      return nullptr;
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    g_hot1_buffers[key] = buf;
  } else {
    buf = it->second;
  }
  if (ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf)) != GGML_STATUS_SUCCESS)
    return nullptr;
  if (fresh) {
    std::vector<char> stage(per_expert);
    const int fd = g.engine->model_file.fd_for_tensor(*m);
    for (size_t i = 0; i < p->hot1.size(); ++i) {
      const ssize_t got = ::pread(fd, stage.data(), per_expert,
                                  (off_t)(m->offset + (uint64_t)p->hot1[i] * per_expert));
      if (got != (ssize_t)per_expert) {
        LOG_ERROR("[GGML-GLM5NEXT-GPU1-HOT] short read layer %d expert %d\n", layer, p->hot1[i]);
        return nullptr;
      }
      ggml_backend_tensor_set(t, stage.data(), i * per_expert, per_expert);
      ::posix_fadvise(fd, (off_t)(m->offset + (uint64_t)p->hot1[i] * per_expert),
                      (off_t)per_expert, POSIX_FADV_DONTNEED);
    }
    g.gpu1_bytes += ggml_nbytes(t);
  }
  ggml_set_name(t, (std::string("blk.") + std::to_string(layer) + "." + stem + ".gpu1_hot").c_str());
  return t;
}

bool prepare_partial_hot1(FullGraph &g, ggml_context *ctx, int layer) {
  PartialExpertLayer *p = partial_layer(g, layer);
  if (!p || p->hot1.empty() || !(g.decode_mode && g.gpu1) || p->hot1_down)
    return true;
  p->hot1_gate = hot1_weight(g, ctx, layer, "ffn_gate_exps.weight", 0);
  p->hot1_up = hot1_weight(g, ctx, layer, "ffn_up_exps.weight", 1);
  p->hot1_down = hot1_weight(g, ctx, layer, "ffn_down_exps.weight", 2);
  return p->hot1_gate && p->hot1_up && p->hot1_down;
}

// Move the next-hottest cold experts (by usage) into the GPU1 hot set until `gib` of expert bytes is reached.
void assign_gpu1_hot(FullGraph &g, double gib) {
  if (gib <= 0.0 || g.partial_experts.empty())
    return;
  const auto &c = g.engine->model_file.architecture_config;
  const char *uenv = std::getenv("MOESTREAM_GLM5NEXT_GPU1_HOT_USAGE");
  const std::string usage_path = uenv && uenv[0]
                                     ? std::string(uenv)
                                     : offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_USAGE");
  auto [entries, max_count] = StreamingEngine::load_usage(usage_path);
  (void)max_count;
  std::vector<std::vector<uint8_t>> is_cold((size_t)c.n_layers);
  std::vector<size_t> bytes((size_t)c.n_layers, 0);
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    PartialExpertLayer &p = g.partial_experts[(size_t)l];
    is_cold[(size_t)l].assign((size_t)c.n_experts, 0);
    for (int e : p.cold)
      if (e >= 0 && e < c.n_experts)
        is_cold[(size_t)l][(size_t)e] = 1;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
      const GGUFTensor *m = model_tensor(*g.engine, "blk." + std::to_string(l) + "." + stem);
      if (m && c.n_experts > 0)
        bytes[(size_t)l] += (size_t)m->size_bytes / (size_t)c.n_experts;
    }
  }
  std::vector<StreamingEngine::UsageEntry> cand;
  for (const auto &e : entries)
    if (e.layer >= c.first_dense && e.layer < c.n_layers && e.expert >= 0 &&
        e.expert < c.n_experts && e.count > 0 && is_cold[(size_t)e.layer][(size_t)e.expert])
      cand.push_back(e);
  std::sort(cand.begin(), cand.end(), [](const auto &a, const auto &b) {
    if (a.count != b.count)
      return a.count > b.count;
    if (a.layer != b.layer)
      return a.layer < b.layer;
    return a.expert < b.expert;
  });
  const size_t budget = (size_t)(gib * 1024.0 * 1024.0 * 1024.0);
  size_t used = 0, taken = 0;
  for (const auto &e : cand) {
    if (used + bytes[(size_t)e.layer] > budget)
      continue;
    PartialExpertLayer &p = g.partial_experts[(size_t)e.layer];
    p.hot1.push_back(e.expert);
    used += bytes[(size_t)e.layer];
    ++taken;
    is_cold[(size_t)e.layer][(size_t)e.expert] = 0;
  }
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    PartialExpertLayer &p = g.partial_experts[(size_t)l];
    if (p.hot1.empty())
      continue;
    std::sort(p.hot1.begin(), p.hot1.end());
    p.cold.erase(std::remove_if(p.cold.begin(), p.cold.end(),
                                [&](int e) { return !is_cold[(size_t)l][(size_t)e]; }),
                 p.cold.end());
  }
  if (std::getenv("MOESTREAM_GLM5NEXT_OFFLOAD_DEBUG")) {
    for (int l : {3, 4}) {
      std::string a, b;
      for (int e : g.partial_experts[(size_t)l].hot1) a += std::to_string(e) + ",";
      for (int e : g.partial_experts[(size_t)l].hot) b += std::to_string(e) + ",";
      LOG_INFO("[HOT1-DEBUG] layer %d hot1=%s\n[HOT1-DEBUG] layer %d hot=%s cold=%zu\n", l, a.c_str(), l, b.c_str(),
               g.partial_experts[(size_t)l].cold.size());
    }
  }
  LOG_INFO("[GGML-GLM5NEXT-GPU1-HOT] %zu experts (%.2f GiB) assigned to GPU1\n", taken,
           (double)used / (1024.0 * 1024.0 * 1024.0));
}

ggml_tensor *partial_map(FullGraph &g, ggml_context *ctx, int layer, bool hot,
                         bool mask) {
  PartialExpertLayer *p = partial_layer(g, layer);
  if (!p)
    return nullptr;
  ggml_tensor *&slot = mask ? (hot ? p->hot_mask : p->cold_mask)
                            : (hot ? p->hot_map : p->cold_map);
  if (slot)
    return slot;
  const auto &c = g.engine->model_file.architecture_config;
  const size_t n = (size_t)std::max(0, c.n_experts);
  const ggml_type type = mask ? GGML_TYPE_F32 : GGML_TYPE_I32;
  // GET_ROWS requires src0->ne[2] == ids->ne[1].  Repeat the small map for
  // every prompt token so the map is a regular GGML input to the CUDA graph.
  ggml_tensor *t = ggml_new_tensor_3d(ctx, type, 1, (int64_t)n, g.S);
  if (!t)
    return nullptr;
  std::vector<uint8_t> is_hot(n, 0);
  for (int e : p->hot)
    if (e >= 0 && (size_t)e < n)
      is_hot[(size_t)e] = 1;

  // How each expert is addressed inside THIS branch's weight tensor (-1 = the
  // branch does not hold it).  The sentinel for routes it does not own, the
  // 0/1 ownership mask and the per-token repetition all come from
  // moe_partial::ExpertBranchMap, so every model inherits the same default
  // (negative-id skip for single-token decode on a capable ggml, valid dummy id
  // otherwise) -- see include/moe_partial_route.hpp.
  std::vector<int32_t> index_of_expert(n, -1);
  if (g.dynamic_expert_cache && !hot) {
    for (size_t expert = 0; expert < n; ++expert)
      if (!is_hot[expert])
        index_of_expert[expert] = (int32_t)expert;
  } else if (g.vram_reference_experts) {
    if (hot) {
      for (int expert : p->hot) {
        if (expert < 0 || (size_t)expert >= n ||
            (size_t)expert >= p->vram_slot_by_expert.size())
          continue;
        const int32_t physical_slot = p->vram_slot_by_expert[(size_t)expert];
        if (physical_slot < 0)
          return nullptr;
        index_of_expert[(size_t)expert] = physical_slot;
      }
    } else {
      int32_t cold_slot = 0;
      const bool full_ddr_view =
          offload_layer(layer) || partial_layer_ram_ready(g, layer);
      for (int expert : p->cold) {
        if (expert < 0 || (size_t)expert >= n)
          return nullptr;
        index_of_expert[(size_t)expert] = full_ddr_view ? expert : cold_slot++;
      }
    }
  } else {
    const auto &ids = hot ? p->hot : p->cold;
    for (size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] < 0 || (size_t)ids[i] >= n)
        continue;
      index_of_expert[(size_t)ids[i]] = (int32_t)i;
    }
  }
  std::vector<uint8_t> owned(n, 0);
  for (size_t e = 0; e < n; ++e)
    owned[e] = (is_hot[e] != 0) == hot ? 1 : 0;
  if (!hot && !p->hot1.empty()) {
    // hot1 experts left `cold`; decode serves them from GPU1, every other build keeps them here.
    const bool on_gpu1 = g.decode_mode && g.gpu1;
    for (int e : p->hot1) {
      if (e < 0 || (size_t)e >= n)
        continue;
      if (on_gpu1) {
        owned[(size_t)e] = 0;
      } else {
        index_of_expert[(size_t)e] = e; // full-arena identity view
        owned[(size_t)e] = 1;
      }
    }
  }

  // MOESTREAM_GLM5NEXT_PARTIAL_ZERO_SENTINEL=1 forces the valid dummy id (A/B tests).
  static const bool force_dummy =
      std::getenv("MOESTREAM_GLM5NEXT_PARTIAL_ZERO_SENTINEL") != nullptr;
  const moe_partial::ExpertBranchMap branch_map =
      moe_partial::ExpertBranchMap::build(
          index_of_expert, g.S, g.decode_mode && g.S == 1, owned,
          force_dummy ? moe_partial::SentinelPolicy::ForceDummy
                      : moe_partial::SentinelPolicy::Auto);
  static bool logged_inactive = false;
  if (!logged_inactive) {
    logged_inactive = true;
    LOG_INFO("[GGML-GLM5NEXT-FULL] partial-layer inactive route id=%d "
             "(mmid negative-skip %s%s)\n",
             (int)branch_map.inactive_id(),
             mmid_negative_skip_available() ? "available" : "unavailable",
             force_dummy ? ", forced dummy by env" : "");
  }
  const void *data = mask ? (const void *)branch_map.route_mask().data()
                          : (const void *)branch_map.route_ids().data();
  if (!arena_alloc(g.dense_buffer, g.dense_base, g.dense_capacity, g.dense_used,
                   t, data, ggml_nbytes(t)))
    return nullptr;
  ggml_set_name(t, (std::string("full_graph_") + (hot ? "hot_" : "cold_") +
                    (mask ? "mask_" : "map_") + std::to_string(layer))
                       .c_str());
  slot = t;
  return slot;
}

bool upload_partial_maps(FullGraph &g, PartialExpertLayer &p) {
  const auto &c = g.engine->model_file.architecture_config;
  const int layer = (int)(&p - g.partial_experts.data());
  if (c.n_experts <= 0 || g.S <= 0 || !p.hot_map || !p.hot_mask ||
      !p.cold_map || !p.cold_mask)
    return false;
  const size_t n = (size_t)c.n_experts;
  const size_t count = n * (size_t)g.S;
  std::vector<int32_t> hot_slots(n, -1), cold_slots(n, -1);
  for (size_t slot = 0; slot < p.hot.size(); ++slot) {
    const int expert = p.hot[slot];
    if (expert == -1 && g.dynamic_expert_cache)
      continue;
    if (expert < 0 || expert >= c.n_experts)
      return false;
    hot_slots[(size_t)expert] =
        g.vram_reference_experts &&
                (size_t)expert < p.vram_slot_by_expert.size()
            ? p.vram_slot_by_expert[(size_t)expert]
            : (int32_t)slot;
  }
  if (!g.dynamic_expert_cache) {
    for (size_t slot = 0; slot < p.cold.size(); ++slot) {
      const int expert = p.cold[slot];
      if (expert < 0 || expert >= c.n_experts ||
          hot_slots[(size_t)expert] >= 0)
        return false;
      cold_slots[(size_t)expert] =
          g.vram_reference_experts && partial_layer_ram_ready(g, layer)
              ? expert
              : (int32_t)slot;
    }
  }
  // Per-expert addressing for each branch: an expert is held by the hot branch
  // when it has a physical slot, otherwise by the cold branch.  The sentinel and
  // mask semantics come from moe_partial::HotColdSplit (include/moe_partial_route.hpp).
  std::vector<int32_t> hot_index(n, -1), cold_index(n, -1);
  for (size_t expert = 0; expert < n; ++expert) {
    const int32_t hs = hot_slots[expert];
    const int32_t cs = g.dynamic_expert_cache ? (int32_t)expert
                                             : cold_slots[expert];
    if (!g.dynamic_expert_cache && ((hs >= 0) == (cs >= 0)))
      return false;
    if (hs >= 0)
      hot_index[expert] = hs;
    else
      cold_index[expert] = cs;
  }
  moe_partial::HotColdSplit split;
  if (!moe_partial::HotColdSplit::build(hot_index, cold_index, g.S,
                                        g.S == 1, /*exact_partition=*/true,
                                        &split))
    return false;
  const std::vector<int32_t> &hot_map = split.hot().route_ids();
  const std::vector<int32_t> &cold_map = split.cold().route_ids();
  const std::vector<float> &hot_mask = split.hot().route_mask();
  const std::vector<float> &cold_mask = split.cold().route_mask();

  const auto set = [](ggml_tensor *tensor, const void *data, size_t bytes) {
    return tensor && tensor->data && ggml_nbytes(tensor) == bytes &&
           cudaMemcpy(tensor->data, data, bytes, cudaMemcpyHostToDevice) ==
               cudaSuccess;
  };
  if (!set(p.hot_map, hot_map.data(), count * sizeof(int32_t)) ||
      !set(p.hot_mask, hot_mask.data(), count * sizeof(float)) ||
      !set(p.cold_map, cold_map.data(), count * sizeof(int32_t)) ||
      !set(p.cold_mask, cold_mask.data(), count * sizeof(float))) {
    const cudaError_t err = cudaGetLastError();
    LOG_WARN("[GGML-GLM5NEXT-CACHE] route-map upload failed: %s\n",
             cudaGetErrorString(err));
    return false;
  }
  return true;
}

int dynamic_upload_budget_override() {
  const char *value =
      std::getenv("MOESTREAM_GLM5NEXT_DYNAMIC_CACHE_UPLOADS_PER_TOKEN");
  if (!value || !value[0])
    return -1;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0)
    return -1;
  return (int)std::min<long>(parsed, 128);
}

double dynamic_upload_fraction() {
  const char *value =
      std::getenv("MOESTREAM_GLM5NEXT_DYNAMIC_CACHE_SWAP_FRAC");
  if (!value || !value[0])
    return 0.25;
  char *end = nullptr;
  const double parsed = std::strtod(value, &end);
  return end != value && *end == '\0' && parsed >= 0.0 && parsed <= 1.0
             ? parsed
             : 0.25;
}

bool ensure_dynamic_upload_worker(FullGraph &g) {
  if (g.dynamic_upload)
    return true;
  size_t staging_capacity = 0;
  const auto add_layer = [&](const PartialExpertLayer &p) {
    size_t bytes = 0;
    const auto add_pair = [&](const ggml_tensor *hot,
                              const ggml_tensor *source) -> bool {
      if (!hot || !source || !hot->data || !source->data ||
          hot->nb[2] == 0 || hot->nb[2] != source->nb[2] ||
          hot->nb[2] > SIZE_MAX - bytes)
        return false;
      bytes += hot->nb[2];
      return true;
    };
    bool valid = false;
    if (p.hot_gate_up || p.cold_gate_up) {
      valid = add_pair(p.hot_gate_up, p.cold_gate_up) &&
              add_pair(p.hot_down, p.cold_down);
    } else {
      valid = add_pair(p.hot_gate, p.cold_gate) &&
              add_pair(p.hot_up, p.cold_up) &&
              add_pair(p.hot_down, p.cold_down);
    }
    if (valid)
      staging_capacity = std::max(staging_capacity, bytes);
  };
  for (int layer = 0; layer < (int)g.partial_experts.size(); ++layer)
    if (!g.partial_experts[(size_t)layer].hot.empty())
      add_layer(g.partial_experts[(size_t)layer]);
  if (staging_capacity == 0 ||
      !MemoryGuard::try_reserve_host_bytes(staging_capacity)) {
    LOG_WARN("[GGML-GLM5NEXT-CACHE] async upload staging admission refused "
             "(%.2f MiB)\n",
             (double)staging_capacity / (1024.0 * 1024.0));
    return false;
  }

  auto state = std::make_unique<DynamicUploadState>();
  state->host_reservation_bytes = staging_capacity;
  if (posix_memalign(&state->staging, 4096, staging_capacity) != 0 ||
      !state->staging) {
    MemoryGuard::release_host_bytes(state->host_reservation_bytes);
    state->host_reservation_bytes = 0;
    return false;
  }
  state->staging_capacity = staging_capacity;
  if (cudaSetDevice(g.device) != cudaSuccess ||
      cudaHostRegister(state->staging, state->staging_capacity,
                       cudaHostRegisterPortable) != cudaSuccess) {
    const cudaError_t err = cudaGetLastError();
    LOG_WARN("[GGML-GLM5NEXT-CACHE] pinned upload staging unavailable: %s\n",
             cudaGetErrorString(err));
    std::free(state->staging);
    MemoryGuard::release_host_bytes(state->host_reservation_bytes);
    return false;
  }
  if (cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    const cudaError_t err = cudaGetLastError();
    LOG_WARN("[GGML-GLM5NEXT-CACHE] upload stream creation failed: %s\n",
             cudaGetErrorString(err));
    cudaHostUnregister(state->staging);
    std::free(state->staging);
    MemoryGuard::release_host_bytes(state->host_reservation_bytes);
    return false;
  }

  DynamicUploadState *raw = state.get();
  const int device = g.device;
  try {
    state->worker = std::thread([raw, device]() {
      if (cudaSetDevice(device) != cudaSuccess) {
        std::lock_guard<std::mutex> lock(raw->mutex);
        raw->stopping = true;
        return;
      }
      for (;;) {
        DynamicUploadJob job;
        {
          std::unique_lock<std::mutex> lock(raw->mutex);
          raw->cv.wait(lock, [&]() {
            return raw->stopping || !raw->queue.empty();
          });
          if (raw->stopping)
            return;
          job = std::move(raw->queue.front());
          raw->queue.pop_front();
          raw->active = true;
        }

        const auto upload_start = std::chrono::steady_clock::now();
        bool success = true;
        size_t upload_bytes = 0;
        for (const DynamicUploadCopy &copy : job.copies) {
          if (!copy.source || !copy.destination || copy.bytes == 0 ||
              copy.staging_offset > raw->staging_capacity ||
              copy.bytes > raw->staging_capacity - copy.staging_offset ||
              copy.bytes > SIZE_MAX - upload_bytes) {
            success = false;
            break;
          }
          upload_bytes += copy.bytes;
          auto *stage = static_cast<uint8_t *>(raw->staging) +
                        copy.staging_offset;
          std::memcpy(stage, copy.source, copy.bytes);
          if (cudaMemcpyAsync(copy.destination, stage, copy.bytes,
                              cudaMemcpyHostToDevice, raw->stream) !=
              cudaSuccess) {
            success = false;
            break;
          }
        }
        // Some copies may already be queued when a later enqueue fails. The
        // staging buffer must not be reused until every submitted copy has
        // stopped reading from it, even on this error path.
        if (cudaStreamSynchronize(raw->stream) != cudaSuccess)
          success = false;
        if (!success) {
          const cudaError_t err = cudaGetLastError();
          LOG_WARN("[GGML-GLM5NEXT-CACHE] async expert upload failed "
                   "layer=%d expert=%d: %s\n",
                   job.layer, job.incoming_expert, cudaGetErrorString(err));
        }
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() -
                                      upload_start)
                                      .count();
        {
          std::lock_guard<std::mutex> lock(raw->mutex);
          raw->completed.push_back({job.layer, job.hot_slot,
                                    job.incoming_expert, success,
                                    upload_bytes, elapsed_ms});
          raw->active = false;
          if (!success)
            raw->stopping = true;
        }
      }
    });
  } catch (...) {
    cudaStreamDestroy(state->stream);
    cudaHostUnregister(state->staging);
    std::free(state->staging);
    MemoryGuard::release_host_bytes(state->host_reservation_bytes);
    return false;
  }
  LOG_INFO("[GGML-GLM5NEXT-CACHE] async H2D worker ready: staging=%.2f MiB "
           "device=%d\n",
           (double)staging_capacity / (1024.0 * 1024.0), g.device);
  g.dynamic_upload = std::move(state);
  return true;
}

bool make_dynamic_upload_job(PartialExpertLayer &p, int layer, int hot_slot,
                             int incoming_expert, int evicted_expert,
                             DynamicUploadJob *job) {
  if (!job || hot_slot < 0 || (size_t)hot_slot >= p.hot.size() ||
      incoming_expert < 0)
    return false;
  job->layer = layer;
  job->hot_slot = hot_slot;
  job->incoming_expert = incoming_expert;
  job->evicted_expert = evicted_expert;
  size_t offset = 0;
  const auto add_pair = [&](ggml_tensor *hot, ggml_tensor *source) -> bool {
    if (!hot || !source || !hot->data || !source->data ||
        hot->nb[2] == 0 || hot->nb[2] != source->nb[2] ||
        incoming_expert >= source->ne[2] || hot_slot >= hot->ne[2] ||
        (size_t)incoming_expert > SIZE_MAX / source->nb[2] ||
        (size_t)hot_slot > SIZE_MAX / hot->nb[2] ||
        hot->nb[2] > SIZE_MAX - offset)
      return false;
    const size_t bytes = hot->nb[2];
    const auto *src = static_cast<const uint8_t *>(source->data) +
                      (size_t)incoming_expert * source->nb[2];
    auto *dst = static_cast<uint8_t *>(hot->data) +
                (size_t)hot_slot * hot->nb[2];
    job->copies.push_back({src, dst, bytes, offset});
    offset += bytes;
    return true;
  };
  if (p.hot_gate_up || p.cold_gate_up) {
    return add_pair(p.hot_gate_up, p.cold_gate_up) &&
           add_pair(p.hot_down, p.cold_down);
  }
  return add_pair(p.hot_gate, p.cold_gate) &&
         add_pair(p.hot_up, p.cold_up) &&
         add_pair(p.hot_down, p.cold_down);
}

bool publish_dynamic_uploads(FullGraph &g) {
  if (!g.dynamic_upload)
    return true;
  std::vector<DynamicUploadResult> completed;
  {
    std::lock_guard<std::mutex> lock(g.dynamic_upload->mutex);
    completed.swap(g.dynamic_upload->completed);
  }
  for (const DynamicUploadResult &result : completed) {
    if (result.layer < 0 ||
        (size_t)result.layer >= g.partial_experts.size())
      continue;
    PartialExpertLayer &p = g.partial_experts[(size_t)result.layer];
    if (result.hot_slot < 0 || (size_t)result.hot_slot >= p.hot.size() ||
        (size_t)result.hot_slot >= p.slot_in_flight.size() ||
        result.incoming_expert < 0 ||
        (size_t)result.incoming_expert >= p.expert_in_flight.size()) {
      g.ready = false;
      return false;
    }
    p.slot_in_flight[(size_t)result.hot_slot] = 0;
    p.expert_in_flight[(size_t)result.incoming_expert] = 0;
    if (!result.success) {
      // The slot was invalidated before the worker started. Keep it empty and
      // let the unchanged full host source serve both the evicted and incoming
      // routes; it can be reused by a later upload.
      continue;
    }
    if (result.elapsed_ms > 0.0 && std::isfinite(result.elapsed_ms)) {
      g.dynamic_upload_ms_per_job =
          g.dynamic_upload_ms_per_job > 0.0
              ? 0.8 * g.dynamic_upload_ms_per_job + 0.2 * result.elapsed_ms
              : result.elapsed_ms;
    }
    if (p.hot[(size_t)result.hot_slot] != -1) {
      LOG_ERROR("[GGML-GLM5NEXT-CACHE] completed upload targeted a live slot "
                "layer=%d slot=%d\n",
                result.layer, result.hot_slot);
      g.ready = false;
      return false;
    }
    const auto cold = std::find(p.cold.begin(), p.cold.end(),
                                result.incoming_expert);
    if (cold == p.cold.end()) {
      LOG_ERROR("[GGML-GLM5NEXT-CACHE] completed expert missing from cold "
                "set layer=%d expert=%d\n",
                result.layer, result.incoming_expert);
      g.ready = false;
      return false;
    }
    p.hot[(size_t)result.hot_slot] = result.incoming_expert;
    p.cold.erase(cold);
    p.slot_last_use[(size_t)result.hot_slot] =
        (int)std::min<uint64_t>(g.dynamic_clock, INT_MAX);
    p.miss_count[(size_t)result.incoming_expert] = 0;
    if (!upload_partial_maps(g, p)) {
      g.ready = false;
      return false;
    }
    ++g.dynamic_promotions;
  }
  return true;
}

// MOESTREAM_GLM5NEXT_ROUTE_TAP=<file>: accumulate per-(layer,expert) decode
// routing counts and rewrite <file> ("layer expert count" lines, the same
// format as the .moestream_usage files) every 25 tokens.  Needs
// MOESTREAM_GLM5NEXT_CACHE_STATS=1 (which creates the per-layer id snapshots).
// Existing counts in <file> are loaded first so runs accumulate.
struct RouteTap {
  std::string path;
  std::vector<std::vector<uint32_t>> counts;
  int tokens = 0;
  bool loaded = false;
};

RouteTap &route_tap() {
  static RouteTap t = [] {
    RouteTap r;
    if (const char *e = std::getenv("MOESTREAM_GLM5NEXT_ROUTE_TAP"))
      r.path = e;
    return r;
  }();
  return t;
}

void route_tap_flush(RouteTap &t) {
  std::FILE *f = std::fopen(t.path.c_str(), "w");
  if (!f)
    return;
  for (size_t l = 0; l < t.counts.size(); ++l)
    for (size_t e = 0; e < t.counts[l].size(); ++e)
      if (t.counts[l][e])
        std::fprintf(f, "%zu %zu %u\n", l, e, t.counts[l][e]);
  std::fclose(f);
}

void update_partial_routes(FullGraph &g, int pos_base, double compute_ms) {
  if (!g.partial_expert_residency || g.moe_route_ids.empty())
    return;
  const auto &c = g.engine->model_file.architecture_config;
  const uint64_t clock = ++g.dynamic_clock;
  RouteTap &tap = route_tap();
  if (!tap.path.empty() && !tap.loaded) {
    tap.loaded = true;
    tap.counts.assign((size_t)c.n_layers,
                      std::vector<uint32_t>((size_t)c.n_experts, 0));
    if (std::FILE *f = std::fopen(tap.path.c_str(), "r")) {
      unsigned l, e, n;
      while (std::fscanf(f, "%u %u %u", &l, &e, &n) == 3)
        if (l < tap.counts.size() && e < tap.counts[l].size())
          tap.counts[l][e] += n;
      std::fclose(f);
    }
  }
  for (const auto &[layer, route_tensor] : g.moe_route_ids) {
    if (!route_tensor || layer < 0 ||
        (size_t)layer >= g.partial_experts.size())
      continue;
    std::vector<int32_t> route_ids((size_t)c.topk);
    ggml_backend_tensor_get(route_tensor, route_ids.data(), 0,
                            route_ids.size() * sizeof(int32_t));
    PartialExpertLayer &p = g.partial_experts[(size_t)layer];
    for (int32_t expert : route_ids) {
      if (expert < 0 || expert >= c.n_experts)
        continue;
      if (!tap.path.empty() && (size_t)layer < tap.counts.size())
        ++tap.counts[(size_t)layer][(size_t)expert];
      ++p.route_count;
      if ((size_t)expert < p.expert_in_flight.size() &&
          p.expert_in_flight[(size_t)expert]) {
        if (p.miss_count[(size_t)expert] < UINT32_MAX)
          ++p.miss_count[(size_t)expert];
        continue;
      }
      const auto hot = std::find(p.hot.begin(), p.hot.end(), expert);
      if (hot != p.hot.end()) {
        ++p.hot_route_count;
        const size_t slot = (size_t)(hot - p.hot.begin());
        if (slot < p.slot_last_use.size())
          p.slot_last_use[slot] = (int)std::min<uint64_t>(clock, INT_MAX);
        if ((size_t)expert < p.miss_count.size())
          p.miss_count[(size_t)expert] = 0;
      } else if ((size_t)expert < p.miss_count.size() &&
                 p.miss_count[(size_t)expert] < UINT32_MAX) {
        ++p.miss_count[(size_t)expert];
      }
    }
  }

  if (!tap.path.empty() && ++tap.tokens % 25 == 0)
    route_tap_flush(tap);
  if (tap.tokens > 0 && tap.tokens % 100 == 0 && !tap.path.empty()) {
    std::string line;
    for (int l = c.first_dense; l < c.n_layers; ++l) {
      const PartialExpertLayer &p = g.partial_experts[(size_t)l];
      if (!p.hot.empty() && !p.cold.empty() && p.route_count) {
        char b[64];
        std::snprintf(b, sizeof(b), " L%d=%.1f%%", l,
                      100.0 * (double)p.hot_route_count / (double)p.route_count);
        line += b;
      }
    }
    if (!line.empty())
      LOG_WARN("[GLM5NEXT-PARTIAL-HIT] tokens=%d per-layer hot-route hit:%s\n",
               tap.tokens, line.c_str());
  }

  if (g.dynamic_expert_cache) {
    if (compute_ms > 0.0 && std::isfinite(compute_ms)) {
      g.dynamic_compute_ms_ema =
          g.dynamic_compute_ms_ema > 0.0
              ? 0.8 * g.dynamic_compute_ms_ema + 0.2 * compute_ms
              : compute_ms;
    }
    // Publish only at this post-compute boundary. The just-finished token's
    // route statistics then reflect the map it actually executed with.
    if (!publish_dynamic_uploads(g))
      return;
    if (g.dynamic_upload) {
      std::lock_guard<std::mutex> lock(g.dynamic_upload->mutex);
      if (g.dynamic_upload->stopping)
        g.dynamic_upload_disabled = true;
    }
    int budget = dynamic_upload_budget_override();
    if (budget < 0 && !g.dynamic_upload_disabled) {
      if (!ensure_dynamic_upload_worker(g)) {
        g.dynamic_upload_disabled = true;
      } else {
        size_t active_layers = 0;
        for (int layer = c.first_dense; layer < c.n_layers; ++layer) {
          const PartialExpertLayer &p = g.partial_experts[(size_t)layer];
          active_layers += !p.hot.empty() && !p.cold.empty();
        }
        if (g.dynamic_upload_ms_per_job <= 0.0) {
          // Neurall-style bootstrap: enqueue one initial candidate per layer,
          // then derive the steady-state budget from observed copy time.
          budget = (int)std::min<size_t>(active_layers, 128);
        } else {
          size_t pending = 0;
          {
            std::lock_guard<std::mutex> lock(g.dynamic_upload->mutex);
            pending = g.dynamic_upload->queue.size() +
                      g.dynamic_upload->completed.size() +
                      (g.dynamic_upload->active ? 1 : 0);
          }
          const double affordable =
              dynamic_upload_fraction() * g.dynamic_compute_ms_ema /
              g.dynamic_upload_ms_per_job;
          const size_t total_budget =
              std::min<size_t>(active_layers,
                               (size_t)std::max(0.0, std::floor(affordable)));
          budget = total_budget > pending
                       ? (int)std::min<size_t>(total_budget - pending, 128)
                       : 0;
        }
      }
    }
    if (budget < 0)
      budget = 0;
    if (budget > 0 && !g.dynamic_upload_disabled) {
      if (!g.dynamic_upload && !ensure_dynamic_upload_worker(g)) {
        g.dynamic_upload_disabled = true;
      } else {
      int scheduled = 0;
      DynamicUploadState &state = *g.dynamic_upload;
      size_t queued = 0;
      {
        std::lock_guard<std::mutex> lock(state.mutex);
        queued = state.queue.size() + (state.active ? 1 : 0);
      }
      const size_t queue_limit = (size_t)std::max(2, budget * 2);
      for (int layer = c.first_dense;
           layer < c.n_layers && scheduled < budget &&
           queued < queue_limit;
           ++layer) {
        PartialExpertLayer &p = g.partial_experts[(size_t)layer];
        if (p.hot.empty() || p.cold.empty() ||
            p.slot_last_use.size() != p.hot.size() ||
            p.slot_in_flight.size() != p.hot.size() ||
            p.miss_count.size() != (size_t)c.n_experts ||
            p.expert_in_flight.size() != (size_t)c.n_experts)
          continue;
        int candidate_index = -1;
        uint32_t candidate_hits = 1; // Ignore one-off routes.
        for (size_t i = 0; i < p.cold.size(); ++i) {
          const int expert = p.cold[i];
          if (expert < 0 || expert >= c.n_experts ||
              p.expert_in_flight[(size_t)expert])
            continue;
          const uint32_t hits = p.miss_count[(size_t)expert];
          if (hits > candidate_hits) {
            candidate_hits = hits;
            candidate_index = (int)i;
          }
        }
        if (candidate_index < 0)
          continue;

        int hot_slot = -1;
        for (size_t i = 0; i < p.hot.size(); ++i) {
          if (p.hot[i] < 0 && !p.slot_in_flight[i]) {
            hot_slot = (int)i;
            break;
          }
        }
        if (hot_slot < 0) {
          int oldest = INT_MAX;
          for (size_t i = 0; i < p.hot.size(); ++i) {
            if (p.slot_in_flight[i] || p.hot[i] < 0)
              continue;
            if (p.slot_last_use[i] < oldest) {
              oldest = p.slot_last_use[i];
              hot_slot = (int)i;
            }
          }
        }
        if (hot_slot < 0)
          continue;

        const int incoming = p.cold[(size_t)candidate_index];
        const int evicted = p.hot[(size_t)hot_slot];
        DynamicUploadJob job;
        if (!make_dynamic_upload_job(p, layer, hot_slot, incoming, evicted,
                                     &job)) {
          LOG_WARN("[GGML-GLM5NEXT-CACHE] unable to form full-source upload "
                   "layer=%d incoming=%d\n",
                   layer, incoming);
          continue;
        }
        if (evicted >= 0 &&
            std::find(p.cold.begin(), p.cold.end(), evicted) == p.cold.end())
          p.cold.push_back(evicted);
        p.hot[(size_t)hot_slot] = -1;
        p.slot_in_flight[(size_t)hot_slot] = 1;
        p.expert_in_flight[(size_t)incoming] = 1;
        if (!upload_partial_maps(g, p)) {
          g.ready = false;
          return;
        }
        {
          std::lock_guard<std::mutex> lock(state.mutex);
          if (state.stopping || state.queue.size() >= queue_limit) {
            p.hot[(size_t)hot_slot] = evicted;
            p.slot_in_flight[(size_t)hot_slot] = 0;
            p.expert_in_flight[(size_t)incoming] = 0;
            if (evicted >= 0)
              p.cold.erase(std::remove(p.cold.begin(), p.cold.end(), evicted),
                           p.cold.end());
            if (!upload_partial_maps(g, p))
              g.ready = false;
            continue;
          }
          state.queue.push_back(std::move(job));
          state.cv.notify_one();
        }
        ++scheduled;
        ++queued;
      }
      if (scheduled > 0)
        LOG_INFO("[GGML-GLM5NEXT-CACHE] pos=%d queued=%d/%d async expert "
                 "uploads\n",
                 pos_base, scheduled, budget);
      }
    }
  }

  if ((pos_base + 1) % 32 == 0) {
    uint64_t routes = 0;
    uint64_t hot_routes = 0;
    for (int layer = c.first_dense; layer < c.n_layers; ++layer) {
      const PartialExpertLayer &p = g.partial_experts[(size_t)layer];
      routes += p.route_count;
      hot_routes += p.hot_route_count;
    }
    LOG_INFO("[GGML-GLM5NEXT-CACHE] pos=%d route hit=%.1f%% (%llu/%llu) "
             "dynamic_promotions=%llu h2d_ema=%.3f ms/job "
             "compute_ema=%.3f ms/token\n",
             pos_base,
             routes ? 100.0 * (double)hot_routes / routes : 0.0,
             (unsigned long long)hot_routes,
             (unsigned long long)routes,
             (unsigned long long)g.dynamic_promotions,
             g.dynamic_upload_ms_per_job, g.dynamic_compute_ms_ema);
  }
}

bool hold_graph_vram_experts(FullGraph &g) {
  if (!g.vram_reference_experts || g.vram_expert_holds.empty())
    return true;
  if (!g.topology)
    return false;
  size_t held = 0;
  for (const FullGraph::VramExpertHold &ref : g.vram_expert_holds) {
    if (!g.topology->hold_for_compute(ref.layer, ref.expert,
                                      ref.device_ptr, false)) {
      while (held > 0) {
        const auto &prior = g.vram_expert_holds[--held];
        g.topology->release_hold(prior.layer, prior.expert,
                                 prior.device_ptr, false);
      }
      LOG_ERROR("[GGML-GLM5NEXT-CACHE] unable to hold aliased VRAM expert "
                "layer=%d expert=%d for graph compute\n",
                ref.layer, ref.expert);
      return false;
    }
    ++held;
  }
  return true;
}

void release_graph_vram_experts(FullGraph &g) {
  if (!g.vram_reference_experts || !g.topology)
    return;
  for (auto it = g.vram_expert_holds.rbegin();
       it != g.vram_expert_holds.rend(); ++it)
    g.topology->release_hold(it->layer, it->expert, it->device_ptr, false);
}

ggml_tensor *static_full_weight(FullGraph &g, ggml_context *ctx,
                                const std::string &name) {
  // DSA's attn_k_b and attn_v_b are intentionally absent from the legacy
  // resident manifest because that path builds CPU WUK/WUV caches.  In this
  // graph they are ordinary static device weights.
  return load_full_weight(g, ctx, name, false);
}

ggml_tensor *graph_ram_weight(FullGraph &g, ggml_context *ctx, int layer,
                              const char *stem) {
  const auto fail = [&](const char *why) -> ggml_tensor * {
    LOG_WARN(
        "[GGML-GLM5NEXT-FULL] graph RAM weight failed layer=%d stem=%s: %s\n",
        layer, stem ? stem : "<null>", why);
    return nullptr;
  };
  if (!g.engine)
    return fail("no engine");
  if (!g.engine->graph_ram_layer_ready(layer))
    return fail("arena is not ready");
  const auto *arena = g.engine->graph_ram_arena(layer);
  if (!arena || !arena->base || arena->n_experts <= 0)
    return fail("missing arena");

  const std::string name = "blk." + std::to_string(layer) + "." + stem;
  const GGUFTensor *meta = model_tensor(*g.engine, name);
  if (!meta || !type_ok(meta->type) || meta->shape.size() != 3 ||
      meta->size_bytes == 0)
    return fail("missing or unsupported GGUF tensor metadata");

  size_t offset = 0;
  if (std::strcmp(stem, "ffn_up_exps.weight") == 0) {
    offset = arena->gate_bytes;
  } else if (std::strcmp(stem, "ffn_down_exps.weight") == 0) {
    if (arena->gate_bytes > SIZE_MAX - arena->up_bytes)
      return fail("projection offset overflow");
    offset = arena->gate_bytes + arena->up_bytes;
  } else if (std::strcmp(stem, "ffn_gate_exps.weight") != 0) {
    return fail("unknown projection stem");
  }
  // meta->size_bytes is the GGUF tensor's FULL size across every expert in
  // the layer (shape [.., .., n_experts]) -- the arena's stride budget is
  // for ONE expert's slice. Comparing the whole-tensor size against a
  // single-expert budget always fails (a 288-expert tensor is ~288x the
  // per-expert slice) -- this is exactly what made every graph_ram-ready
  // MoE layer decline at build_ffn, real-hardware-confirmed via the
  // "projection does not fit in expert stride" warning firing for all
  // three projections at the very first routed-expert layer. Divide down
  // to the per-expert size (matching how prepare_graph_ram_arenas derived
  // arena->gate_bytes/up_bytes/down_bytes in the first place) before
  // checking it against the remaining per-expert stride.
  if ((size_t)arena->n_experts <= 0 ||
      meta->size_bytes % (size_t)arena->n_experts != 0)
    return fail("tensor size not evenly divisible by expert count");
  const size_t per_expert_bytes = meta->size_bytes / (size_t)arena->n_experts;
  if (offset > arena->stride || per_expert_bytes > arena->stride - offset)
    return fail("projection does not fit in expert stride");

  std::vector<int64_t> shape;
  shape.reserve(meta->shape.size());
  for (uint64_t dim : meta->shape)
    shape.push_back((int64_t)dim);
  ggml_tensor *t = new_tensor(ctx, (ggml_type)meta->type, shape);
  if (!t)
    return fail("GGML tensor creation failed");
  if (ggml_nbytes(t) != meta->size_bytes)
    return fail("GGML tensor size differs from GGUF tensor size");

  // The same host arena is exposed through three GGML buffer handles. Each
  // tensor starts at its projection offset, while nb[2] skips one complete
  // [gate|up|down] expert slice. This is the external-weight layout already
  // used by the regular MoE GGML backend, with ownership retained by the
  // StreamingEngine rather than by the graph.
  ggml_backend_buffer_t buffer =
      ggml_backend_cpu_buffer_from_ptr(arena->base, arena->bytes);
  if (!buffer)
    return fail("CPU external buffer creation failed");
  ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  t->nb[2] = arena->stride;
  t->nb[3] = t->nb[2] * (size_t)t->ne[2];
  t->data = arena->base + offset;
  t->buffer = buffer;
  if (ggml_backend_buffer_init_tensor(buffer, t) != GGML_STATUS_SUCCESS) {
    ggml_backend_buffer_free(buffer);
    return fail("CPU external buffer tensor initialization failed");
  }
  g.host_buffers.push_back(buffer);
  g.host_reference_bytes += meta->size_bytes;
  ggml_set_name(t, (name + ".graph_ram").c_str());
  return t;
}


ggml_tensor *offload_weight(FullGraph &g, ggml_context *ctx, int layer,
                            const char *stem) {
  OffloadState *o = get_offload(*g.engine);
  if (!o)
    return nullptr;
  const int plane = std::strcmp(stem, "ffn_gate_exps.weight") == 0   ? 0
                    : std::strcmp(stem, "ffn_up_exps.weight") == 0   ? 1
                    : std::strcmp(stem, "ffn_down_exps.weight") == 0 ? 2
                                                                     : -1;
  const std::string name = "blk." + std::to_string(layer) + "." + stem;
  const GGUFTensor *meta = model_tensor(*g.engine, name);
  if (plane < 0 || !meta || !type_ok(meta->type) || meta->shape.size() != 3)
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : meta->shape)
    shape.push_back((int64_t)d);
  ggml_tensor *t = new_tensor(ctx, (ggml_type)meta->type, shape);
  if (!t || t->nb[2] != o->arena->stride(layer, plane) ||
      (uint64_t)t->ne[2] != (uint64_t)o->n_experts) {
    LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] %s: tensor stride %zu != arena stride\n",
              name.c_str(), t ? t->nb[2] : (size_t)0);
    return nullptr;
  }
  const size_t len = (size_t)t->nb[2] * (size_t)t->ne[2];
  ggml_backend_buffer_t buffer =
      ggml_backend_cpu_buffer_from_ptr(o->arena->ptr(layer, 0, plane), len);
  if (!buffer)
    return nullptr;
  ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  t->data = o->arena->ptr(layer, 0, plane);
  t->buffer = buffer;
  if (ggml_backend_buffer_init_tensor(buffer, t) != GGML_STATUS_SUCCESS) {
    ggml_backend_buffer_free(buffer);
    return nullptr;
  }
  g.host_buffers.push_back(buffer);
  ggml_set_name(t, (name + ".offload").c_str());
  return t;
}

ggml_tensor *offload_miss_node(FullGraph &g, ggml_context *ctx, int layer,
                               ggml_tensor *ids) {
  OffloadState *o = get_offload(*g.engine);
  if (!o || !ids)
    return nullptr;
  std::lock_guard<std::mutex> lock(g_offload_mutex);
  o->sites.push_back({o, layer, g.S > 1});
  ggml_tensor *out =
      ggml_map_custom1(ctx, ids, offload_miss_cb, 1, &o->sites.back());
  if (out)
    ggml_set_name(out, ("glm5next_offload_ids_" + std::to_string(layer)).c_str());
  return out;
}

ggml_tensor *expert_weight_impl(FullGraph &g, ggml_context *ctx, int layer,
                               const char *stem) {
  if (offload_layer(layer))
    return offload_weight(g, ctx, layer, stem);
  if (partial_layer(g, layer) &&
      !(g.vram_reference_experts && partial_layer_ram_ready(g, layer)))
    return load_cold_weight(g, ctx, layer, stem);
  if (partial_layer_ram_ready(g, layer)) {
    if (ggml_tensor *t = graph_ram_weight(g, ctx, layer, stem))
      return t;
  }
  const bool on_device =
      layer >= g.first_dense && layer < g.first_dense + g.device_expert_layers;
  return load_full_weight(g, ctx, "blk." + std::to_string(layer) + "." + stem,
                          on_device, true);
}

ggml_tensor *expert_gate_up_weight_impl(FullGraph &g, ggml_context *ctx,
                                       int layer) {
  if (offload_layer(layer))
    return nullptr; // separate projection planes inside the offload arena
  if (g.vram_reference_experts && partial_layer_ram_ready(g, layer))
    return nullptr;
  if (partial_layer(g, layer))
    return load_gate_up_weight(g, ctx, layer, false);
  // The shared RAM arena stores gate and up as separate planes inside each
  // expert slice. Force the separate projection views so expert e's gate and
  // up bytes are addressed with the same fixed stride.
  if (partial_layer_ram_ready(g, layer))
    return nullptr;
  const bool on_device =
      layer >= g.first_dense && layer < g.first_dense + g.device_expert_layers;
  return load_gate_up_weight(g, ctx, layer, on_device);
}


// Optional second-GPU expert residency (env MOESTREAM_GLM5NEXT_GPU1_LAYERS,
// "first-last"): decode-graph expert tensors of those layers are copied to a
// CUDA buffer on device 1 so the scheduler runs their MUL_MAT_ID there.
bool gpu1_layer(const FullGraph &g, int layer) {
  return g.decode_mode && g.gpu1 && layer >= g.gpu1_first &&
         layer <= g.gpu1_last;
}

ggml_tensor *move_expert_to_gpu1(FullGraph &g, ggml_context *ctx, int layer,
                                 ggml_tensor *host) {
  if (!host || !gpu1_layer(g, layer) || host->ne[3] != 1)
    return host;
  ggml_tensor *t = ggml_new_tensor_3d(ctx, host->type, host->ne[0], host->ne[1],
                                      host->ne[2]);
  if (!t)
    return host;
  const auto buft = ggml_backend_get_default_buffer_type(g.gpu1);
  ggml_backend_buffer_t buf =
      ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(t) + 4096);
  if (!buf) {
    LOG_WARN("[GGML-GLM5NEXT-FULL] GPU1 expert buffer declined layer=%d "
             "(%.2f GiB); keeping host tensor\n",
             layer, (double)ggml_nbytes(t) / (1024.0 * 1024.0 * 1024.0));
    return host;
  }
  ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  if (ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf)) !=
      GGML_STATUS_SUCCESS) {
    ggml_backend_buffer_free(buf);
    return host;
  }
  const size_t per_expert = ggml_row_size(host->type, host->ne[0]) *
                            (size_t)host->ne[1];
  for (int64_t e = 0; e < host->ne[2]; ++e)
    ggml_backend_tensor_set(t, (const char *)host->data + (size_t)e * host->nb[2],
                            (size_t)e * t->nb[2], per_expert);
  ggml_set_name(t, host->name);
  g.gpu1_buffers.push_back(buf);
  g.gpu1_bytes += ggml_nbytes(t);
  return t;
}

ggml_tensor *expert_weight(FullGraph &g, ggml_context *ctx, int layer,
                           const char *stem) {
  return move_expert_to_gpu1(g, ctx, layer,
                             expert_weight_impl(g, ctx, layer, stem));
}

ggml_tensor *expert_gate_up_weight(FullGraph &g, ggml_context *ctx, int layer) {
  return move_expert_to_gpu1(g, ctx, layer,
                             expert_gate_up_weight_impl(g, ctx, layer));
}

size_t expert_source_bytes(const StreamingEngine &engine,
                           int device_expert_layers) {
  const auto &c = engine.model_file.architecture_config;
  size_t total = 0;
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    if (l < c.first_dense + device_expert_layers)
      continue;
    if (offload_layer(l) || engine.graph_ram_layer_ready(l))
      continue;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(l) + "." + stem);
      if (!m ||
          total > std::numeric_limits<size_t>::max() - (size_t)m->size_bytes)
        return 0;
      total += (size_t)m->size_bytes;
    }
  }
  return total;
}

size_t expert_source_bytes(const FullGraph &g) {
  if (!g.partial_expert_residency)
    return expert_source_bytes(*g.engine, g.device_expert_layers);

  // Neurall-style dynamic caching keeps the original expert records in the
  // host source arena while hot copies come and go in VRAM. This avoids the
  // old eviction path's device-to-host readback and keeps every cold fallback
  // address stable across cache updates.
  if (g.dynamic_expert_cache)
    return expert_source_bytes(*g.engine, g.device_expert_layers);

  const auto &c = g.engine->model_file.architecture_config;
  size_t total = 0;
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    if (l < c.first_dense + g.device_expert_layers || offload_layer(l) ||
        g.engine->graph_ram_layer_ready(l))
      continue;
    if (c.n_experts <= 0 || l >= (int)g.partial_experts.size())
      return 0;
    const size_t cold = g.partial_experts[(size_t)l].cold.size();
    if (cold == 0)
      continue;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(*g.engine, "blk." + std::to_string(l) + "." + stem);
      if (!m || m->size_bytes % (size_t)c.n_experts != 0)
        return 0;
      const size_t record = (size_t)m->size_bytes / (size_t)c.n_experts;
      if (record > SIZE_MAX / cold ||
          total > SIZE_MAX - record * cold)
        return 0;
      total += record * cold;
    }
  }
  return total;
}

bool compact_partial_source(FullGraph &g) {
  if (!g.partial_expert_residency || g.partial_sources_compacted)
    return true;
  if (g.dynamic_expert_cache) {
    // Dynamic mode deliberately retains full-expert source views. The cold
    // graph maps routed IDs to their original expert indices, so no in-place
    // cold-only compaction is appropriate here.
    g.partial_sources_compacted = true;
    return true;
  }
  const auto &c = g.engine->model_file.architecture_config;
  g.expert_source_full_views = g.expert_source_views;
  g.expert_source_full_view_bytes = g.expert_source_view_bytes;

  std::unordered_map<std::string, void *> cold_views;
  std::unordered_map<std::string, size_t> cold_view_bytes;
  size_t compact_used = 0;
  size_t compact_tensors = 0;
  const size_t alignment = 4096;

  const auto compact_one = [&](const std::string &name,
                               const PartialExpertLayer &p) -> bool {
    auto src_it = g.expert_source_full_views.find(name);
    auto size_it = g.expert_source_full_view_bytes.find(name);
    if (src_it == g.expert_source_full_views.end() ||
        size_it == g.expert_source_full_view_bytes.end())
      return false;
    if (c.n_experts <= 0 || size_it->second % (size_t)c.n_experts != 0)
      return false;
    const size_t record = size_it->second / (size_t)c.n_experts;
    const size_t bytes = record * p.cold.size();
    if (bytes == 0)
      return true;
    const size_t off = (compact_used + alignment - 1) & ~(alignment - 1);
    if (off > g.expert_source_capacity ||
        bytes > g.expert_source_capacity - off)
      return false;
    if (!MemoryGuard::try_reserve_host_bytes(bytes))
      return false;
    void *scratch = nullptr;
    if (posix_memalign(&scratch, alignment, bytes) != 0 || !scratch) {
      MemoryGuard::release_host_bytes(bytes);
      return false;
    }
    const uint8_t *src = static_cast<const uint8_t *>(src_it->second);
    uint8_t *tmp = static_cast<uint8_t *>(scratch);
    for (size_t i = 0; i < p.cold.size(); ++i) {
      std::memcpy(tmp + i * record, src + (size_t)p.cold[i] * record, record);
    }
    std::memmove(static_cast<uint8_t *>(g.expert_source_base) + off, tmp,
                 bytes);
    std::free(scratch);
    MemoryGuard::release_host_bytes(bytes);
    cold_views.emplace(name,
                       static_cast<uint8_t *>(g.expert_source_base) + off);
    cold_view_bytes.emplace(name, bytes);
    compact_used = off + bytes;
    ++compact_tensors;
    return true;
  };

  for (int l = c.first_dense; l < c.n_layers; ++l) {
    const PartialExpertLayer &p = g.partial_experts[(size_t)l];
    const std::string prefix = "blk." + std::to_string(l) + ".";
    PackedGateUpMeta packed;
    if (packed_gate_up_meta(*g.engine, l, &packed)) {
      if (!compact_one(prefix + "ffn_gate_up_exps.weight", p) ||
          !compact_one(prefix + "ffn_down_exps.weight", p))
        return false;
    } else {
      for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                               "ffn_down_exps.weight"}) {
        if (!compact_one(prefix + stem, p))
          return false;
      }
    }
  }
  g.expert_source_views.swap(cold_views);
  g.expert_source_view_bytes.swap(cold_view_bytes);
  g.expert_source_used = compact_used;
  g.expert_source_tensors = compact_tensors;
  g.partial_sources_compacted = true;
  LOG_INFO("[GGML-FULL] compacted cold source: %.2f GiB, %zu tensors; "
           "full source remains one in-place DDR allocation\n",
           (double)compact_used / (1024.0 * 1024.0 * 1024.0), compact_tensors);
  return true;
}

bool preload_mmap_expert_source(FullGraph &g) {
  if (!g.dynamic_expert_cache ||
      (!g.engine->model_file.mmap_active() &&
       !g.engine->model_file.enable_mmap()))
    return false;

  const auto &c = g.engine->model_file.architecture_config;
  const int device_end = c.first_dense + g.device_expert_layers;
  size_t mapped_bytes = 0;
  const auto add_view = [&](const GGUFTensor *meta) -> bool {
    if (!meta || !type_ok(meta->type) || meta->size_bytes == 0 ||
        meta->size_bytes > SIZE_MAX ||
        (size_t)meta->size_bytes > SIZE_MAX - mapped_bytes)
      return false;
    const uint8_t *mapped = g.engine->model_file.mmap_bytes(
        *meta, meta->offset, (size_t)meta->size_bytes);
    if (!mapped)
      return false;
    const bool inserted = g.expert_source_views
                              .emplace(meta->name,
                                       const_cast<uint8_t *>(mapped))
                              .second;
    if (!inserted ||
        !g.expert_source_view_bytes
             .emplace(meta->name, (size_t)meta->size_bytes)
             .second)
      return false;
    mapped_bytes += (size_t)meta->size_bytes;
    ++g.expert_source_tensors;
    return true;
  };

  for (int layer = c.first_dense; layer < c.n_layers; ++layer) {
    if (layer < device_end || g.engine->graph_ram_layer_ready(layer))
      continue;
    PackedGateUpMeta packed;
    if (packed_gate_up_meta(*g.engine, layer, &packed)) {
      // Keep the file's native separate gate/up planes.  Interleaving them
      // into a temporary packed tensor would recreate the full-size host
      // allocation this mmap path is specifically avoiding.
      if (!add_view(packed.gate) || !add_view(packed.up))
        return false;
    } else {
      const std::string prefix = "blk." + std::to_string(layer) + ".";
      if (!add_view(model_tensor(*g.engine, prefix + "ffn_gate_exps.weight")) ||
          !add_view(model_tensor(*g.engine, prefix + "ffn_up_exps.weight")))
        return false;
    }
    const std::string down_name =
        "blk." + std::to_string(layer) + ".ffn_down_exps.weight";
    if (!add_view(model_tensor(*g.engine, down_name)))
      return false;
  }

  if (mapped_bytes == 0 || mapped_bytes != g.expert_source_capacity) {
    LOG_WARN("[GGML-GLM5NEXT-FULL] mapped source size mismatch: mapped=%.2f "
             "GiB expected=%.2f GiB\n",
             (double)mapped_bytes / (1024.0 * 1024.0 * 1024.0),
             (double)g.expert_source_capacity / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  g.expert_source_used = mapped_bytes;
  LOG_INFO("[GGML-GLM5NEXT-FULL] dynamic expert source mapped read-only: "
           "%.2f GiB across %zu tensors; pages remain OS-reclaimable\n",
           (double)mapped_bytes / (1024.0 * 1024.0 * 1024.0),
           g.expert_source_tensors);
  return true;
}

bool preload_expert_source(FullGraph &g) {
  if (g.expert_source_mmap)
    return preload_mmap_expert_source(g);

  const auto &c = g.engine->model_file.architecture_config;
  struct Job {
    const GGUFTensor *meta = nullptr;
    void *dst = nullptr;
    std::string name;
    const GGUFTensor *packed_meta = nullptr;
    const std::vector<int> *selected_experts = nullptr;
  };
  std::vector<Job> jobs;
  jobs.reserve((size_t)std::max(0, c.n_layers - c.first_dense) * 3);

  const int device_end = c.first_dense + g.device_expert_layers;
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    if (l < device_end)
      continue;
    if (offload_layer(l) || g.engine->graph_ram_layer_ready(l))
      continue;
    const std::vector<int> *selected_experts = nullptr;
    if (g.partial_expert_residency && !g.dynamic_expert_cache) {
      if (l >= (int)g.partial_experts.size())
        return false;
      selected_experts = &g.partial_experts[(size_t)l].cold;
    }
    PackedGateUpMeta packed;
    if (packed_gate_up_meta(*g.engine, l, &packed)) {
      size_t bytes = packed.bytes;
      if (selected_experts) {
        if (c.n_experts <= 0 || packed.bytes % (size_t)c.n_experts != 0)
          return false;
        bytes = packed.bytes / (size_t)c.n_experts *
                selected_experts->size();
      }
      if (bytes > 0) {
        void *dst = alloc_expert_source(g, bytes);
        if (!dst) {
          LOG_ERROR("[GGML-GLM5NEXT-FULL] DDR source arena exhausted at %s\n",
                    packed.name.c_str());
          return false;
        }
        g.expert_source_views.emplace(packed.name, dst);
        g.expert_source_view_bytes.emplace(packed.name, bytes);
        jobs.push_back(
            {packed.gate, dst, packed.name, packed.up, selected_experts});
      }
    } else {
      for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight"}) {
        const std::string name = "blk." + std::to_string(l) + "." + stem;
        const GGUFTensor *m = model_tensor(*g.engine, name);
        if (!m || !type_ok(m->type) || m->size_bytes == 0) {
          LOG_ERROR("[GGML-GLM5NEXT-FULL] missing routed tensor %s\n",
                  name.c_str());
          return false;
        }
        size_t bytes = (size_t)m->size_bytes;
        if (selected_experts) {
          if (c.n_experts <= 0 || m->size_bytes % (size_t)c.n_experts != 0)
            return false;
          bytes = (size_t)m->size_bytes / (size_t)c.n_experts *
                  selected_experts->size();
        }
        if (bytes > 0) {
          void *dst = alloc_expert_source(g, bytes);
          if (!dst) {
            LOG_ERROR(
                "[GGML-GLM5NEXT-FULL] DDR source arena exhausted at %s\n",
                name.c_str());
            return false;
          }
          g.expert_source_views.emplace(name, dst);
          g.expert_source_view_bytes.emplace(name, bytes);
          jobs.push_back({m, dst, name, nullptr, selected_experts});
        }
      }
    }

    const std::string down_name =
        "blk." + std::to_string(l) + ".ffn_down_exps.weight";
    const GGUFTensor *down = model_tensor(*g.engine, down_name);
    if (!down || !type_ok(down->type) || down->size_bytes == 0) {
      LOG_ERROR("[GGML-GLM5NEXT-FULL] missing routed tensor %s\n",
                down_name.c_str());
      return false;
    }
    size_t down_bytes = (size_t)down->size_bytes;
    if (selected_experts) {
      if (c.n_experts <= 0 || down->size_bytes % (size_t)c.n_experts != 0)
        return false;
      down_bytes = (size_t)down->size_bytes / (size_t)c.n_experts *
                   selected_experts->size();
    }
    if (down_bytes > 0) {
      void *down_dst = alloc_expert_source(g, down_bytes);
      if (!down_dst) {
        LOG_ERROR("[GGML-GLM5NEXT-FULL] DDR source arena exhausted at %s\n",
                  down_name.c_str());
        return false;
      }
      g.expert_source_views.emplace(down_name, down_dst);
      g.expert_source_view_bytes.emplace(down_name, down_bytes);
      jobs.push_back({down, down_dst, down_name, nullptr, selected_experts});
    }
  }
  if (jobs.empty())
  {
    if (g.partial_expert_residency)
      g.partial_sources_compacted = true;
    return true;
  }

  const auto start = std::chrono::steady_clock::now();
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  std::string first_error;
  std::mutex error_mutex;
  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  // Packed gate/up tensors need a temporary gate+up buffer while they are
  // interleaved into the graph arena.  Four readers made that temporary peak
  // large enough to compete with the 78 GiB source arena on a 128 GiB host.
  // This is boot-time I/O, so cap the default at two; an explicit override is
  // available for a machine with more headroom.
  unsigned read_threads = 2;
  if (const char *env = std::getenv("MOESTREAM_FULL_GRAPH_READ_THREADS")) {
    char *end = nullptr;
    const long n = std::strtol(env, &end, 10);
    if (end != env && *end == '\0' && n > 0)
      read_threads = (unsigned)std::min<long>(n, 16);
  }
  const unsigned n_threads =
      std::min<unsigned>(read_threads, std::min<unsigned>(hw, jobs.size()));
  auto worker = [&]() {
    for (;;) {
      const size_t i = next.fetch_add(1, std::memory_order_relaxed);
      if (i >= jobs.size())
        return;
      if (failed.load(std::memory_order_relaxed))
        return;
      // This is a one-time source-load phase, not a per-allocation hot path.
      // Poll between small batches so a page-cache-heavy unaligned GGUF can
      // fail closed before the host reaches swap pressure.
      if ((i & 15u) == 0 && MemoryGuard::allocation_blocked()) {
        if (!failed.exchange(true, std::memory_order_relaxed)) {
          std::lock_guard<std::mutex> lock(error_mutex);
          first_error = "host memory watermark during source preload";
        }
        return;
      }
      const Job &job = jobs[i];
      const bool ok = job.selected_experts
                          ? (job.packed_meta
                                 ? read_pack_gate_up_selected(
                                       *g.engine, *job.meta, *job.packed_meta,
                                       c.n_experts, *job.selected_experts,
                                       job.dst)
                                 : read_tensor_selected_experts(
                                       *g.engine, *job.meta, c.n_experts,
                                       *job.selected_experts, job.dst))
                          : (job.packed_meta
                                 ? read_pack_gate_up(*g.engine, *job.meta,
                                                     *job.packed_meta, job.dst)
                                 : read_tensor_direct(*g.engine, *job.meta,
                                                      job.dst));
      if (!ok) {
        if (!failed.exchange(true, std::memory_order_relaxed)) {
          std::lock_guard<std::mutex> lock(error_mutex);
          first_error = job.name;
        }
        return;
      }
    }
  };
  std::vector<std::thread> workers;
  workers.reserve(n_threads);
  for (unsigned i = 0; i < n_threads; ++i)
    workers.emplace_back(worker);
  for (auto &worker_thread : workers)
    worker_thread.join();
  if (failed.load(std::memory_order_relaxed)) {
    LOG_ERROR("[GGML-GLM5NEXT-FULL] parallel DDR source read failed: %s\n",
              first_error.c_str());
    return false;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  LOG_INFO("[GGML-GLM5NEXT-FULL] parallel DDR source read: %.2f GiB in %.3f s "
           "(%.2f GiB/s, %zu tensors, %u readers)\n",
           (double)g.expert_source_used / (1024.0 * 1024.0 * 1024.0), seconds,
           seconds > 0.0 ? (double)g.expert_source_used /
                               (seconds * 1024.0 * 1024.0 * 1024.0)
                         : 0.0,
           jobs.size(), n_threads);

  // Static partial residency is read directly into the compact cold layout.
  // Dynamic mode intentionally retains every original record as the stable
  // host backing store for asynchronous H2D cache fills.
  if (g.partial_expert_residency && !g.dynamic_expert_cache)
    g.partial_sources_compacted = true;
  if (!compact_partial_source(g)) {
    LOG_ERROR("[GGML-FULL] partial expert source compaction failed\n");
    return false;
  }

  // A pageable source makes cudaMemcpyAsync pay the driver's staging cost at
  // every graph split.  That was the dominant measured GLM graph cost:
  // 8.50 s spent submitting 65.1 GiB of copies while the scheduler itself
  // needed only 1.30 s.  Register a bounded set of the already-loaded source
  // slices once, preserving the graph's single arena and avoiding a second
  // full-size host allocation.  The default cap is deliberately below the
  // 10--30 GiB aggregate registration failure range observed on this box;
  // callers can lower or raise it with MOESTREAM_FULL_GRAPH_PIN_GB, while a
  // registration failure stops the pass and leaves the remaining slices
  // pageable rather than retrying until the machine is under pressure.
  size_t pin_budget = g.engine->ram_pinned_budget_bytes;
  const char *pin_env = std::getenv("MOESTREAM_FULL_GRAPH_PIN_GB");
  if (pin_env && pin_env[0]) {
    char *end = nullptr;
    const double gb = std::strtod(pin_env, &end);
    if (end != pin_env && *end == '\0' && gb >= 0.0) {
      const double bytes = gb * 1024.0 * 1024.0 * 1024.0;
      pin_budget = bytes >= (double)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
    }
  } else if (pin_budget > 24ull * 1024ull * 1024ull * 1024ull) {
    pin_budget = 24ull * 1024ull * 1024ull * 1024ull;
  }

  size_t pinned = 0;
  size_t attempts = 0;
  size_t failures = 0;
  if (pin_budget > 0) {
    const auto try_pin = [&](const std::string &name, void *ptr, size_t bytes) {
      if (!ptr || bytes == 0 || pinned > pin_budget ||
          bytes > pin_budget - pinned)
        return false;
      ++attempts;
      const cudaError_t err =
          cudaHostRegister(ptr, bytes, cudaHostRegisterPortable);
      if (err != cudaSuccess) {
        ++failures;
        cudaGetLastError();
        LOG_WARN("[GGML-GLM5NEXT-FULL] source pin stopped at %.2f GiB: "
                 "cudaHostRegister(%s, %.2f GiB) failed: %s\n",
                 (double)pinned / (1024.0 * 1024.0 * 1024.0), name.c_str(),
                 (double)bytes / (1024.0 * 1024.0 * 1024.0),
                 cudaGetErrorString(err));
        return false;
      }
      g.expert_source_pinned_ranges.push_back({ptr, bytes, true});
      pinned += bytes;
      return true;
    };

    bool stop = false;
    for (int l = c.first_dense; l < c.n_layers && !stop; ++l) {
      if (l < device_end)
        continue;
      PackedGateUpMeta packed;
      const std::string packed_name =
          "blk." + std::to_string(l) + ".ffn_gate_up_exps.weight";
      const std::string down_name =
          "blk." + std::to_string(l) + ".ffn_down_exps.weight";
      auto packed_it = g.expert_source_views.find(packed_name);
      auto down_it = g.expert_source_views.find(down_name);
      if (packed_gate_up_meta(*g.engine, l, &packed) &&
          packed_it != g.expert_source_views.end() &&
          down_it != g.expert_source_views.end()) {
        // Keep both projections of a layer together when the budget allows;
        // this preserves the normal gate/up and down copy behavior instead of
        // converting half the layer to pageable-source timing.
        const size_t packed_bytes = g.expert_source_view_bytes[packed_name];
        const size_t down_bytes = g.expert_source_view_bytes[down_name];
        const size_t layer_bytes = packed_bytes + down_bytes;
        if (pinned > pin_budget || layer_bytes > pin_budget - pinned)
          break;
        if (!try_pin(packed_name, packed_it->second, packed_bytes)) {
          stop = true;
          break;
        }
        if (!try_pin(down_name, down_it->second, down_bytes)) {
          stop = true;
          break;
        }
      } else {
        // Older/unusual files retain the separate three-projection source
        // layout.  Pin complete per-layer triplets under the same cap.
        size_t layer_bytes = 0;
        std::vector<std::pair<std::string, size_t>> parts;
        for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                                 "ffn_down_exps.weight"}) {
          const std::string name = "blk." + std::to_string(l) + "." + stem;
          const GGUFTensor *m = model_tensor(*g.engine, name);
          auto it = g.expert_source_views.find(name);
          if (!m || it == g.expert_source_views.end()) {
            parts.clear();
            break;
          }
          const size_t view_bytes = g.expert_source_view_bytes[name];
          if (layer_bytes > SIZE_MAX - view_bytes) {
            parts.clear();
            break;
          }
          layer_bytes += view_bytes;
          parts.emplace_back(name, view_bytes);
        }
        if (parts.empty() || pinned > pin_budget ||
            layer_bytes > pin_budget - pinned)
          break;
        for (const auto &[name, bytes] : parts) {
          if (!try_pin(name, g.expert_source_views[name], bytes)) {
            stop = true;
            break;
          }
        }
      }
    }
  }
  g.host_pinned_bytes += pinned;
  g.host_pin_failures += failures;
  LOG_INFO("[GGML-GLM5NEXT-FULL] source pin: %.2f/%.2f GiB, %zu ranges, "
           "%zu attempts, %zu failures (bounded, one-time)\n",
           (double)pinned / (1024.0 * 1024.0 * 1024.0),
           (double)pin_budget / (1024.0 * 1024.0 * 1024.0),
           g.expert_source_pinned_ranges.size(), attempts, failures);
  return true;
}

size_t device_expert_bytes(const FullGraph &g) {
  if (g.vram_reference_experts)
    return 0;
  const auto &c = g.engine->model_file.architecture_config;
  size_t total = 0;
  if (g.partial_expert_residency) {
    for (int l = c.first_dense; l < c.n_layers; ++l) {
      const PartialExpertLayer &p = g.partial_experts[(size_t)l];
      if (p.hot.empty())
        continue;
      PackedGateUpMeta packed;
      const std::string prefix = "blk." + std::to_string(l) + ".";
      if (packed_gate_up_meta(*g.engine, l, &packed)) {
        const GGUFTensor *down =
            model_tensor(*g.engine, prefix + "ffn_down_exps.weight");
        if (!down || c.n_experts <= 0 ||
            packed.bytes % (size_t)c.n_experts != 0 ||
            down->size_bytes % (size_t)c.n_experts != 0)
          return 0;
        const size_t per_gate_up = packed.bytes / (size_t)c.n_experts;
        const size_t per_down = (size_t)down->size_bytes / (size_t)c.n_experts;
        if (per_gate_up > SIZE_MAX / p.hot.size() ||
            total > SIZE_MAX - per_gate_up * p.hot.size())
          return 0;
        total += per_gate_up * p.hot.size();
        if (per_down > SIZE_MAX / p.hot.size() ||
            total > SIZE_MAX - per_down * p.hot.size())
          return 0;
        total += per_down * p.hot.size();
      } else {
        for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                                 "ffn_down_exps.weight"}) {
          const GGUFTensor *m = model_tensor(*g.engine, prefix + stem);
          if (!m || c.n_experts <= 0 ||
              m->size_bytes % (size_t)c.n_experts != 0)
            return 0;
          const size_t per = (size_t)m->size_bytes / (size_t)c.n_experts;
          if (per > SIZE_MAX / p.hot.size() ||
              total > SIZE_MAX - per * p.hot.size())
            return 0;
          total += per * p.hot.size();
        }
      }
    }
    return total;
  }
  const int end = std::min(c.n_layers, c.first_dense + g.device_expert_layers);
  for (int l = c.first_dense; l < end; ++l) {
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(*g.engine, "blk." + std::to_string(l) + "." + stem);
      if (!m ||
          total > std::numeric_limits<size_t>::max() - (size_t)m->size_bytes)
        return 0;
      total += (size_t)m->size_bytes;
    }
  }
  return total;
}

size_t dense_arena_bytes(const FullGraph &g) {
  size_t total = 0;
  for (const auto &[name, q] : g.engine->resident_layers) {
    if (!q.host_ptr || !graph_dense_weight_name(name))
      continue;
    if (name == "output.weight" && !g.decode_mode)
      continue;
    if (total > std::numeric_limits<size_t>::max() - q.meta.size_bytes - 4096)
      return 0;
    total += q.meta.size_bytes + 4096;
  }
  const auto &c = g.engine->model_file.architecture_config;
  for (int l = 0; l < c.n_layers; ++l) {
    if (glm5next_is_kda_layer(c, l))
      continue;
    for (const char *stem : {"attn_k_b.weight", "attn_v_b.weight"}) {
      const std::string name = "blk." + std::to_string(l) + "." + stem;
      // These tensors are normally in resident_layers already (the legacy
      // path keeps them host-side to build WUK/WUV).  Only add them here when
      // a manifest variant omitted them and static_full_weight() will read
      // them directly during graph construction.
      if (resident(*g.engine, name))
        continue;
      const GGUFTensor *m = model_tensor(*g.engine, name);
      if (m) {
        if (total > std::numeric_limits<size_t>::max() - m->size_bytes - 4096)
          return 0;
        total += (size_t)m->size_bytes + 4096;
      }
    }
  }
  if (g.partial_expert_residency) {
    // Two I32 ID maps and two F32 masks per MoE layer.  They are tiny, but
    // account for them explicitly so a nearly-full CUDA dense arena does not
    // rely on its generic slack by accident.
    const size_t per_layer = (size_t)c.n_experts * (size_t)g.S *
                             (2 * sizeof(int32_t) + 2 * sizeof(float));
    if ((size_t)std::max(0, c.n_layers - c.first_dense) >
        (SIZE_MAX - total) / (per_layer + 4096))
      return 0;
    total +=
        (size_t)std::max(0, c.n_layers - c.first_dense) * (per_layer + 4096);
  }
  return total + (1ull << 20);
}

bool init_dense_arena(FullGraph &g) {
  const size_t bytes = dense_arena_bytes(g);
  if (bytes == 0)
    return false;
  const auto buft = ggml_backend_get_default_buffer_type(g.cuda);
  g.dense_buffer = ggml_backend_buft_alloc_buffer(buft, bytes);
  if (!g.dense_buffer) {
    LOG_WARN("[GGML-GLM5NEXT-FULL] dense arena allocation declined: %.2f GiB\n",
             (double)bytes / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  ggml_backend_buffer_set_usage(g.dense_buffer,
                                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  g.dense_base = ggml_backend_buffer_get_base(g.dense_buffer);
  g.dense_capacity = ggml_backend_buffer_get_size(g.dense_buffer);
  return true;
}

bool init_decode_state_arena(FullGraph &g) {
  if (!g.decode_mode || !g.cuda || g.kv_capacity <= 0 || g.kv_lora <= 0)
    return false;
  const auto &c = g.engine->model_file.architecture_config;
  size_t bytes = 1ull << 20; // alignment/slack for backend tensor allocations
  auto add_tensor = [&bytes](size_t tensor_bytes) {
    constexpr size_t kAlignment = 4096;
    if (tensor_bytes > SIZE_MAX - bytes - kAlignment)
      return false;
    bytes += tensor_bytes + kAlignment;
    return true;
  };
  for (int l = 0; l < g.n_layers; ++l) {
    if (glm5next_is_kda_layer(c, l)) {
      const size_t state_n = (size_t)g.hd * g.hd * g.H;
      const size_t hist_n = (size_t)(g.d_conv - 1) * 3 * g.d_inner;
      if (state_n > SIZE_MAX / sizeof(float) ||
          hist_n > SIZE_MAX / sizeof(float) ||
          !add_tensor(state_n * sizeof(float)) ||
          !add_tensor(hist_n * sizeof(float)))
        return false;
    } else {
      if ((size_t)g.kv_capacity >
          SIZE_MAX / (size_t)g.kv_lora / sizeof(ggml_fp16_t) ||
          !add_tensor((size_t)g.kv_capacity * g.kv_lora *
                      sizeof(ggml_fp16_t)))
        return false;
    }
  }
  if (g.sparse_decode) {
    if (g.indexer_key_length <= 0 || g.indexer_kpool <= 1 ||
        g.kv_capacity % g.indexer_kpool != 0)
      return false;
    const size_t capacity = (size_t)g.kv_capacity;
    const size_t key_length = (size_t)g.indexer_key_length;
    const size_t pool_count = capacity / (size_t)g.indexer_kpool;
    if (key_length > SIZE_MAX / 2 / sizeof(float) / capacity ||
        key_length > SIZE_MAX / sizeof(float) / pool_count)
      return false;
    for (int l = 0; l < g.n_layers; ++l) {
      if (glm5next_is_kda_layer(c, l))
        continue;
      const std::string prefix = "blk." + std::to_string(l) + ".";
      if (!model_tensor(*g.engine, prefix + "indexer.proj.weight"))
        continue;
      if (!add_tensor(2 * key_length * capacity * sizeof(float)) ||
          !add_tensor(key_length * pool_count * sizeof(float)))
        return false;
    }
  }
  const auto buft = ggml_backend_get_default_buffer_type(g.cuda);
  g.device_state_buffer = ggml_backend_buft_alloc_buffer(buft, bytes);
  if (!g.device_state_buffer) {
    LOG_WARN("[GGML-GLM5NEXT-FULL-DECODE] device state arena declined: "
             "%.2f GiB (KV capacity=%d)\n",
             (double)bytes / (1024.0 * 1024.0 * 1024.0), g.kv_capacity);
    return false;
  }
  ggml_backend_buffer_set_usage(g.device_state_buffer,
                                GGML_BACKEND_BUFFER_USAGE_COMPUTE);
  g.device_state_base = ggml_backend_buffer_get_base(g.device_state_buffer);
  g.device_state_capacity = ggml_backend_buffer_get_size(g.device_state_buffer);
  g.device_state_used = 0;
  LOG_INFO("[GGML-GLM5NEXT-FULL-DECODE] device state arena=%.2f GiB, "
           "KV capacity=%d\n",
           (double)g.device_state_capacity / (1024.0 * 1024.0 * 1024.0),
           g.kv_capacity);
  return true;
}

ggml_tensor *new_decode_state_tensor(FullGraph &g, ggml_context *ctx,
                                     ggml_type type,
                                     const std::vector<int64_t> &shape,
                                     const char *name) {
  ggml_tensor *t = new_tensor(ctx, type, shape);
  if (!t || !g.device_state_buffer)
    return nullptr;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_buffer_get_type(g.device_state_buffer);
  const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
  const size_t alignment =
      std::max<size_t>(256, ggml_backend_buft_get_alignment(buft));
  const size_t aligned =
      (g.device_state_used + alignment - 1) & ~(alignment - 1);
  if (aligned > g.device_state_capacity ||
      alloc_size > g.device_state_capacity - aligned)
    return nullptr;
  void *addr = static_cast<uint8_t *>(g.device_state_base) + aligned;
  if (ggml_backend_tensor_alloc(g.device_state_buffer, t, addr) !=
      GGML_STATUS_SUCCESS)
    return nullptr;
  g.device_state_used = aligned + alloc_size;
  ggml_set_name(t, name);
  return t;
}

bool init_expert_arena(FullGraph &g) {
  if (!g.partial_expert_residency && g.device_expert_layers <= 0)
    return true;
  if (g.vram_reference_experts) {
    const bool has_hot_refs = std::any_of(
        g.partial_experts.begin(), g.partial_experts.end(),
        [](const PartialExpertLayer &p) { return !p.hot.empty(); });
    if (!has_hot_refs)
      return true;
    // GGML needs a CUDA buffer handle to classify these immutable source
    // tensors, but their data pointers belong to GpuTopology's cache pool.
    // This tiny owned allocation is metadata/handle backing only; no expert
    // weights are copied into it.
    constexpr size_t kReferenceBufferBytes = 1ull << 20;
    const auto buft = ggml_backend_get_default_buffer_type(g.cuda);
    g.expert_buffer = ggml_backend_buft_alloc_buffer(
        buft, kReferenceBufferBytes);
    if (!g.expert_buffer) {
      LOG_WARN("[GGML-GLM5NEXT-FULL] could not create CUDA buffer handle for "
               "VRAM expert references\n");
      return false;
    }
    ggml_backend_buffer_set_usage(g.expert_buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    g.expert_base = ggml_backend_buffer_get_base(g.expert_buffer);
    g.expert_capacity = ggml_backend_buffer_get_size(g.expert_buffer);
      LOG_INFO("[GGML-GLM5NEXT-FULL] VRAM expert weights are aliased from "
               "resident cache slots; graph-owned weight bytes=0, descriptor "
             "buffer=%.2f MiB\n",
             (double)g.expert_capacity / (1024.0 * 1024.0));
    return true;
  }
  const size_t bytes = device_expert_bytes(g);
  if (bytes == 0)
    return false;
  const size_t capacity =
      bytes + (size_t)std::max(1, g.n_layers) * 3 * 256 + (1ull << 20);
  const auto buft = ggml_backend_get_default_buffer_type(g.cuda);
  g.expert_buffer = ggml_backend_buft_alloc_buffer(buft, capacity);
  if (!g.expert_buffer) {
    LOG_WARN(
        "[GGML-GLM5NEXT-FULL] expert arena allocation declined: %.2f GiB\n",
        (double)capacity / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  ggml_backend_buffer_set_usage(g.expert_buffer,
                                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  g.expert_base = ggml_backend_buffer_get_base(g.expert_buffer);
  g.expert_capacity = ggml_backend_buffer_get_size(g.expert_buffer);
  LOG_INFO("[GGML-GLM5NEXT-FULL] device expert arena=%.2f GiB%s\n",
           (double)g.expert_capacity / (1024.0 * 1024.0 * 1024.0),
           g.partial_expert_residency ? " for partial hot expert set"
                                      : " for prefix MoE layers");
  return true;
}

bool init_expert_source(FullGraph &g, size_t source_bytes) {
  // Leave one page of alignment slack per projection.  The reservation is
  // still based on the exact GGUF bytes; this small fixed overhead is covered
  // by the graph's existing watermark and is reclaimed with the graph.
  constexpr size_t kSlack = 1ull << 20;
  if (source_bytes == 0)
    return true;
  if (source_bytes > SIZE_MAX - kSlack)
    return false;
  const size_t capacity = source_bytes + kSlack;
  void *p = nullptr;
  if (posix_memalign(&p, 4096, capacity) != 0 || !p) {
    LOG_WARN("[GGML-GLM5NEXT-FULL] DDR expert source allocation declined: %.2f "
             "GiB\n",
             (double)capacity / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  // The source reservation is admitted by prepare() before this allocation;
  // do not charge the same arena a second time as an ordinary owned buffer.
  g.owned.push_back({p, capacity, false, 0});
  g.expert_source_base = p;
  g.expert_source_capacity = capacity;
  g.expert_source_used = 0;
  g.expert_source_tensors = 0;
  LOG_INFO("[GGML-GLM5NEXT-FULL] DDR expert source arena=%.2f GiB (one "
           "allocation)\n",
           (double)capacity / (1024.0 * 1024.0 * 1024.0));
  return true;
}

bool source_configuration_matches(const FullGraph &source,
                                  const FullGraph &candidate) {
  if (!source.has_expert_source() || source.engine != candidate.engine ||
      source.topology != candidate.topology ||
      source.vram_reference_experts != candidate.vram_reference_experts ||
      source.device_expert_layers != candidate.device_expert_layers ||
      source.partial_expert_residency != candidate.partial_expert_residency ||
      source.dynamic_expert_cache != candidate.dynamic_expert_cache ||
      source.partial_usage_file != candidate.partial_usage_file ||
      source.partial_experts.size() != candidate.partial_experts.size())
    return false;
  for (size_t i = 0; i < source.partial_experts.size(); ++i) {
    const PartialExpertLayer &a = source.partial_experts[i];
    const PartialExpertLayer &b = candidate.partial_experts[i];
    if (a.hot != b.hot || a.cold != b.cold ||
        a.vram_slot_by_expert != b.vram_slot_by_expert ||
        a.vram_layer_base != b.vram_layer_base ||
        a.vram_layer_slots != b.vram_layer_slots ||
        a.vram_slot_stride != b.vram_slot_stride)
      return false;
  }
  if (source.vram_reference_bytes != candidate.vram_reference_bytes)
    return false;
  return true;
}

bool prepare_expert_source_only(FullGraph &g, const char *tag) {
  const size_t source = expert_source_bytes(g);
  constexpr size_t kSourceSlack = 1ull << 20;
  // 10 GiB, per explicit user instruction after the 2026-09-09 crash (was
  // 8 GiB) -- align every host-memory watermark in this file to the same
  // floor rather than several different ad hoc numbers.
  constexpr size_t kWatermark = 10ull * 1024ull * 1024ull * 1024ull;
  const size_t host_live = graph_host_live_bytes(*g.engine);
  const long pre_call_available_mb = MemoryGuard::read_available_mb();
  const size_t pre_call_reserved = MemoryGuard::reserved_host_bytes();
  LOG_INFO("[%s] admission attempt: source=%.2f GiB existing(host_live)=%.2f "
           "GiB ledger_reserved=%.2f GiB MemAvailable=%ld MiB (sampled "
           "immediately before the try_reserve_host_bytes call)\n",
           tag, (double)source / (1024.0 * 1024.0 * 1024.0),
           (double)host_live / (1024.0 * 1024.0 * 1024.0),
           (double)pre_call_reserved / (1024.0 * 1024.0 * 1024.0),
           pre_call_available_mb);
  if (g.dynamic_expert_cache) {
    // The per-token GLM cache needs an immutable source for every expert, but
    // that source need not be another eagerly populated 78 GiB anonymous
    // arena. Keep GGUF shard pages file-backed and reclaimable; the cold CPU
    // branch and the pinned H2D staging worker fault only the ranges they use.
    if (source == 0 || source > SIZE_MAX - kSourceSlack ||
        !MemoryGuard::try_reserve_host_bytes(1, kWatermark)) {
      LOG_WARN(
          "[%s] mapped-source admission refused source=%.2f GiB existing="
          "%.2f GiB reserve=%.2f GiB budget=%.2f GB MemAvailable=%ld MiB "
          "(live, not watchdog-stale) stop_mb=%ld\n",
          tag, (double)source / (1024.0 * 1024.0 * 1024.0),
          (double)host_live / (1024.0 * 1024.0 * 1024.0),
          (double)kWatermark / (1024.0 * 1024.0 * 1024.0),
          (double)MemoryGuard::host_budget_bytes() / 1.0e9,
          MemoryGuard::read_available_mb(), MemoryGuard::stop_limit_mb());
      return false;
    }
    MemoryGuard::release_host_bytes(1);
    g.expert_source_mmap = true;
    g.expert_source_capacity = source; // logical bytes, not one contiguous span
    if (!preload_expert_source(g)) {
      LOG_WARN("[%s] could not establish mapped expert views; graph will "
               "use the legacy path\n",
               tag);
      return false;
    }
    LOG_INFO("[%s] mapped-source admission: %.2f GiB logical source, "
             "existing=%.2f GiB reserve=%.2f GiB budget=%.2f GB; no full "
             "anonymous source copy\n",
             tag, (double)source / (1024.0 * 1024.0 * 1024.0),
             (double)host_live / (1024.0 * 1024.0 * 1024.0),
             (double)kWatermark / (1024.0 * 1024.0 * 1024.0),
             (double)MemoryGuard::host_budget_bytes() / 1.0e9);
    return true;
  }
  if (source > SIZE_MAX - kSourceSlack ||
      (source > 0 && !MemoryGuard::try_reserve_host_bytes(source + kSourceSlack,
                                                          kWatermark))) {
    LOG_WARN(
        "[%s] boot host admission refused source=%.2f GiB existing(host_live)="
        "%.2f GiB ledger_reserved=%.2f GiB reserve=%.2f GiB budget=%.2f GB "
        "MemAvailable=%ld MiB (live, not watchdog-stale) stop_mb=%ld "
        "blocked=%d\n",
        tag, (double)source / (1024.0 * 1024.0 * 1024.0),
        (double)host_live / (1024.0 * 1024.0 * 1024.0),
        (double)MemoryGuard::reserved_host_bytes() / (1024.0 * 1024.0 * 1024.0),
        (double)kWatermark / (1024.0 * 1024.0 * 1024.0),
        (double)MemoryGuard::host_budget_bytes() / 1.0e9,
        MemoryGuard::read_available_mb(), MemoryGuard::stop_limit_mb(),
        (int)MemoryGuard::allocation_blocked());
    return false;
  }
  g.host_reservation_bytes = source + (source ? kSourceSlack : 0);
  LOG_INFO("[%s] boot host admission: source=%.2f GiB existing=%.2f GiB "
           "reserve=%.2f GiB budget=%.2f GB\n",
           tag, (double)source / (1024.0 * 1024.0 * 1024.0),
           (double)host_live / (1024.0 * 1024.0 * 1024.0),
           (double)kWatermark / (1024.0 * 1024.0 * 1024.0),
           (double)MemoryGuard::host_budget_bytes() / 1.0e9);
  if (!init_expert_source(g, source) || !preload_expert_source(g)) {
    LOG_WARN("[%s] boot expert source preload failed; graph will use the "
             "legacy path\n",
             tag);
    return false;
  }
  LOG_INFO("[%s] boot expert source ready: %.2f GiB DDR, %zu tensors; "
           "prompt graph remains lazy\n",
           tag, (double)g.expert_source_used / (1024.0 * 1024.0 * 1024.0),
           g.expert_source_tensors);
  return true;
}

ggml_tensor *hc_affine(ggml_context *ctx, ggml_tensor *x, ggml_tensor *scale,
                       ggml_tensor *base) {
  return ggml_add(ctx, ggml_mul(ctx, x, scale), base);
}

ggml_tensor *hc_mean(FullGraph &g, ggml_context *ctx, ggml_tensor *x) {
  ggml_tensor *out = ggml_view_2d(ctx, x, g.D, g.S, x->nb[2], 0);
  for (int h = 1; h < g.HC; ++h)
    out = ggml_add(
        ctx, out,
        ggml_view_2d(ctx, x, g.D, g.S, x->nb[2], (size_t)h * x->nb[1]));
  return ggml_scale(ctx, out, 1.0f / (float)g.HC);
}

ggml_tensor *hc_contig(ggml_context *ctx, ggml_tensor *t) {
  return t && !ggml_is_contiguous(t) ? ggml_cont(ctx, t) : t;
}

// MOESTREAM_GLM5NEXT_FUSED_HC: unset/"1" = all fused ops; otherwise any of
// the letters c (hc_comb), p (hc_pre sum), o (hc_post) enables just those.
bool fused_hc_op(char which) {
  // Default ON. MOESTREAM_GLM5NEXT_UNFUSED_HC=1 restores the composed-op path.
  static const bool off = std::getenv("MOESTREAM_GLM5NEXT_UNFUSED_HC") != nullptr;
  if (off)
    return false;
  static const char *e = std::getenv("MOESTREAM_GLM5NEXT_FUSED_HC");
  if (!e || e[0] == '1')
    return true;
  return std::strchr(e, which) != nullptr;
}

ggml_tensor *hc_pre(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                    ggml_tensor *fn, ggml_tensor *scale, ggml_tensor *base,
                    ggml_tensor **post, ggml_tensor **comb) {
  const auto &c = g.engine->model_file.architecture_config;
  const int mix_dim = (2 + g.HC) * g.HC;
  ggml_tensor *flat = ggml_reshape_2d(ctx, x, g.D * g.HC, g.S);
  ggml_tensor *flat_norm = ggml_rms_norm(ctx, flat, c.eps);
  ggml_tensor *mixes = ggml_mul_mat(ctx, fn, flat_norm);

  ggml_tensor *scale_pre = ggml_view_1d(ctx, scale, 1, 0);
  ggml_tensor *scale_post = ggml_view_1d(ctx, scale, 1, sizeof(float));
  ggml_tensor *scale_comb = ggml_view_1d(ctx, scale, 1, 2 * sizeof(float));
  ggml_tensor *base_pre = ggml_view_1d(ctx, base, g.HC, 0);
  ggml_tensor *base_post = ggml_view_1d(ctx, base, g.HC, g.HC * sizeof(float));
  ggml_tensor *base_comb =
      ggml_view_1d(ctx, base, g.HC * g.HC, 2 * g.HC * sizeof(float));

  ggml_tensor *pre = ggml_view_2d(ctx, mixes, g.HC, g.S, mixes->nb[1], 0);
  pre = ggml_scale_bias(
      ctx, ggml_sigmoid(ctx, hc_affine(ctx, pre, scale_pre, base_pre)), 1.0f,
      c.hc_eps);
  *post = ggml_view_2d(ctx, mixes, g.HC, g.S, mixes->nb[1],
                       (size_t)g.HC * mixes->nb[0]);
  *post = ggml_scale(
      ctx, ggml_sigmoid(ctx, hc_affine(ctx, *post, scale_post, base_post)),
      2.0f);
  const bool f_comb = fused_hc_op('c'), f_pre = fused_hc_op('p');
  if (f_comb) {
    *comb = ggml_dsv4_hc_comb(ctx, hc_contig(ctx, mixes), hc_contig(ctx, scale),
                              hc_contig(ctx, base), c.hc_eps,
                              (int32_t)c.hc_sinkhorn_iters);
  } else {
    *comb = ggml_view_2d(ctx, mixes, g.HC * g.HC, g.S, mixes->nb[1],
                         (size_t)2 * g.HC * mixes->nb[0]);
    *comb = hc_affine(ctx, *comb, scale_comb, base_comb);
    *comb = ggml_reshape_3d(ctx, *comb, g.HC, g.HC, g.S);
    ggml_tensor *eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx, eps, c.hc_eps);
    *comb = ggml_add(ctx, ggml_soft_max(ctx, *comb), eps);
    auto norm_cols = [&]() {
      ggml_tensor *p = ggml_cont(ctx, ggml_permute(ctx, *comb, 1, 0, 2, 3));
      ggml_tensor *sum = ggml_add(ctx, ggml_sum_rows(ctx, p), eps);
      sum = ggml_permute(ctx, sum, 1, 0, 2, 3);
      *comb = ggml_div(ctx, *comb, sum);
    };
    auto norm_rows = [&]() {
      ggml_tensor *sum = ggml_add(ctx, ggml_sum_rows(ctx, *comb), eps);
      *comb = ggml_div(ctx, *comb, sum);
    };
    norm_cols();
    for (int i = 1; i < c.hc_sinkhorn_iters; ++i) {
      norm_rows();
      norm_cols();
    }
  }
  if (f_pre)
    return ggml_dsv4_hc_pre(ctx, hc_contig(ctx, x), hc_contig(ctx, pre));

  ggml_tensor *result = nullptr;
  for (int h = 0; h < g.HC; ++h) {
    ggml_tensor *xh =
        ggml_view_2d(ctx, x, g.D, g.S, x->nb[2], (size_t)h * x->nb[1]);
    ggml_tensor *ph =
        ggml_view_2d(ctx, pre, 1, g.S, pre->nb[1], (size_t)h * pre->nb[0]);
    ggml_tensor *term = ggml_mul(ctx, xh, ph);
    result = result ? ggml_add(ctx, result, term) : term;
  }
  return result;
}

ggml_tensor *hc_post(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                     ggml_tensor *residual, ggml_tensor *post,
                     ggml_tensor *comb) {
  if (fused_hc_op('o'))
    return ggml_dsv4_hc_post(ctx, hc_contig(ctx, x), hc_contig(ctx, residual),
                             hc_contig(ctx, post), hc_contig(ctx, comb));
  ggml_tensor *out = nullptr;
  for (int dst = 0; dst < g.HC; ++dst) {
    ggml_tensor *pd =
        ggml_view_2d(ctx, post, 1, g.S, post->nb[1], (size_t)dst * post->nb[0]);
    ggml_tensor *cur = ggml_mul(ctx, x, pd);
    for (int src = 0; src < g.HC; ++src) {
      ggml_tensor *rs = ggml_view_2d(ctx, residual, g.D, g.S, residual->nb[2],
                                     (size_t)src * residual->nb[1]);
      ggml_tensor *cw =
          ggml_view_2d(ctx, comb, 1, g.S, comb->nb[2],
                       (size_t)dst * comb->nb[0] + (size_t)src * comb->nb[1]);
      cur = ggml_add(ctx, cur, ggml_mul(ctx, rs, cw));
    }
    ggml_tensor *stream = ggml_reshape_3d(ctx, cur, g.D, 1, g.S);
    out = out ? ggml_concat(ctx, out, stream, 1) : stream;
  }
  return out;
}

ggml_tensor *build_kda(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *wq, ggml_tensor *wk,
                       ggml_tensor *wv, ggml_tensor *cq, ggml_tensor *ck,
                       ggml_tensor *cv, ggml_tensor *fa, ggml_tensor *fb,
                       ggml_tensor *ga, ggml_tensor *gb, ggml_tensor *beta_w,
                       ggml_tensor *a, ggml_tensor *dt, ggml_tensor *norm_w,
                       ggml_tensor *out_w, ggml_tensor *hist_in,
                       ggml_tensor *state_in, ggml_tensor **state_out,
                       ggml_tensor **hist_out) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_tensor *q = ggml_mul_mat(ctx, wq, cur);
  ggml_tensor *k = ggml_mul_mat(ctx, wk, cur);
  ggml_tensor *v = ggml_mul_mat(ctx, wv, cur);
  ggml_tensor *qkv = ggml_concat(ctx, ggml_concat(ctx, q, k, 0), v, 0);
  qkv = ggml_reshape_3d(ctx, qkv, 3 * g.d_inner, g.S, 1);

  ggml_tensor *conv_w = ggml_concat(
      ctx,
      ggml_concat(ctx, ggml_reshape_2d(ctx, cq, g.d_conv, g.d_inner),
                  ggml_reshape_2d(ctx, ck, g.d_conv, g.d_inner), 1),
      ggml_reshape_2d(ctx, cv, g.d_conv, g.d_inner), 1);
  ggml_tensor *conv_in = ggml_concat(ctx, hist_in, ggml_transpose(ctx, qkv), 0);
  ggml_tensor *conv = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_in, conv_w));
  const size_t row = ggml_row_size(conv->type, 3 * g.d_inner);
  const size_t head_row = ggml_row_size(conv->type, g.hd);
  q = ggml_view_4d(ctx, conv, g.hd, g.H, g.S, 1, head_row, row, row * g.S, 0);
  k = ggml_view_4d(ctx, conv, g.hd, g.H, g.S, 1, head_row, row, row * g.S,
                   ggml_row_size(conv->type, g.d_inner));
  v = ggml_view_4d(ctx, conv, g.hd, g.H, g.S, 1, head_row, row, row * g.S,
                   ggml_row_size(conv->type, 2 * g.d_inner));
  q = ggml_l2_norm(ctx, q, 1e-6f);
  k = ggml_l2_norm(ctx, k, 1e-6f);

  ggml_tensor *decay = ggml_mul_mat(ctx, fb, ggml_mul_mat(ctx, fa, cur));
  decay = ggml_add(ctx, decay, dt);
  decay = ggml_reshape_3d(ctx, decay, g.hd, g.H, g.S);
  decay = ggml_mul(ctx, decay, ggml_reshape_3d(ctx, a, 1, g.H, 1));
  decay = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, decay, -1.0f)),
                     c.kda_gate_lower_bound);
  decay = ggml_reshape_4d(ctx, decay, g.hd, g.H, g.S, 1);

  ggml_tensor *beta = ggml_mul_mat(ctx, beta_w, cur);
  beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, beta, 1, g.H, g.S, 1));
  ggml_tensor *scan =
      ggml_gated_delta_net(ctx, q, k, v, decay, beta, state_in, 1);

  const size_t value_row = ggml_row_size(scan->type, g.d_inner);
  const size_t state_row = ggml_row_size(scan->type, g.hd * g.hd);
  // ggml-cuda's gated_delta_net kernel stores the per-head state TRANSPOSED
  // relative to the [k][v] math convention this codebase's own legacy CPU
  // qwen4exp_gdn_delta_step (and, by the identical pattern, the KDA
  // recurrence) use -- confirmed directly in
  // ggml/src/ggml-cuda/gated_delta_net.cu's own comment ("state is stored
  // transposed: M[col][i] = S[i][col]", where col indexes v and i indexes
  // k), i.e. the raw view below has ne[0]=k (contiguous), ne[1]=v. Legacy's
  // own flat layout (glm5next_recur_state[l], engine.hpp) is st[k*hd+v] --
  // v contiguous, k slower -- the opposite axis order. Left unfixed, the
  // flat ggml_backend_tensor_get+memcpy in glm5next_full_graph_run silently
  // hands decode a transposed state matrix: same byte count, same total
  // size (hd==hd), no crash, just wrong values from the very first decode
  // token onward. Transpose+materialize here so the exported tensor's own
  // byte layout already matches what the flat copy-out assumes.
  ggml_tensor *state_raw =
      ggml_view_4d(ctx, scan, g.hd, g.hd, g.H, 1, head_row, state_row,
                   state_row * g.H, (size_t)g.d_inner * g.S * sizeof(float));
  *state_out = g.decode_mode ? ggml_cont(ctx, state_raw)
                             : ggml_cont(ctx, ggml_transpose(ctx, state_raw));
  ggml_tensor *core = ggml_view_4d(ctx, scan, g.hd, g.H, g.S, 1, head_row,
                                   value_row, value_row * g.S, 0);
  ggml_tensor *norm = ggml_mul(ctx, ggml_rms_norm(ctx, core, c.eps), norm_w);
  ggml_tensor *gate = ggml_mul_mat(ctx, gb, ggml_mul_mat(ctx, ga, cur));
  gate = ggml_reshape_3d(ctx, gate, g.hd, g.H, g.S);
  ggml_tensor *gated = ggml_mul(ctx, norm, ggml_sigmoid(ctx, gate));
  ggml_tensor *out =
      ggml_mul_mat(ctx, out_w, ggml_reshape_2d(ctx, gated, g.d_inner, g.S));
  *hist_out =
      ggml_cont(ctx, ggml_view_3d(ctx, conv_in, g.d_conv - 1, 3 * g.d_inner, 1,
                                  conv_in->nb[1], conv_in->nb[2],
                                  ggml_row_size(conv_in->type, g.S)));
  (void)layer;
  return out;
}

Glm5nextIndexerLayer *glm5next_indexer_state(FullGraph &g, int layer) {
  for (Glm5nextIndexerLayer &state : g.glm5next_indexer_layers)
    if (state.layer == layer)
      return &state;
  return nullptr;
}

bool glm5next_indexer_owner(const FullGraph &g, int layer) {
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  return model_tensor(*g.engine, prefix + "indexer.proj.weight") != nullptr;
}

ggml_tensor *glm5next_indexer_pool(ggml_context *ctx, ggml_tensor *keys,
                                   ggml_tensor *gates, ggml_tensor *ape,
                                   int key_length, int kpool, int n_pools) {
  if (!keys || !gates || !ape || key_length <= 0 || kpool <= 1 ||
      n_pools <= 0)
    return nullptr;
  const int64_t pooled_tokens = (int64_t)kpool * n_pools;
  const int64_t pooled_elements = (int64_t)key_length * pooled_tokens;
  const auto complete_pool_prefix = [&](ggml_tensor *t) -> ggml_tensor * {
    if (!t || t->ne[0] != key_length || t->ne[2] != 1 || t->ne[3] != 1 ||
        t->ne[1] < pooled_tokens)
      return nullptr;
    if (ggml_nelements(t) == pooled_elements)
      return t;
    // Prefill also caches key/gate rows for an incomplete tail so decode can
    // finish that pool later.  Pool summaries cover only complete KPool
    // blocks; trim the trailing token(s) before reshaping instead of asking
    // GGML to reshape key_length*S elements into key_length*kpool*n_pools.
    ggml_tensor *prefix =
        ggml_view_2d(ctx, t, key_length, pooled_tokens, t->nb[1], 0);
    return prefix ? ggml_cont(ctx, prefix) : nullptr;
  };
  keys = complete_pool_prefix(keys);
  gates = complete_pool_prefix(gates);
  if (!keys || !gates)
    return nullptr;
  ggml_tensor *key_blocks =
      ggml_reshape_3d(ctx, keys, key_length, kpool, n_pools);
  ggml_tensor *gate_blocks =
      ggml_reshape_3d(ctx, gates, key_length, kpool, n_pools);
  ggml_tensor *logits = ggml_add(ctx, gate_blocks, ape);
  logits = ggml_cont(ctx, ggml_permute(ctx, logits, 1, 0, 2, 3));
  ggml_tensor *prob = ggml_soft_max(ctx, logits);
  ggml_tensor *values =
      ggml_cont(ctx, ggml_permute(ctx, key_blocks, 1, 0, 2, 3));
  ggml_tensor *pooled = ggml_sum_rows(ctx, ggml_mul(ctx, prob, values));
  return ggml_reshape_2d(ctx, pooled, key_length, n_pools);
}

bool build_glm5next_indexer_prefill(FullGraph &g, ggml_context *ctx,
                                    ggml_tensor *cur, int layer) {
  const int key_length = g.indexer_key_length;
  const int kpool = g.indexer_kpool;
  const int n_pools = key_length > 0 && kpool > 1 ? g.S / kpool : 0;
  if (key_length <= 0 || kpool <= 1)
    return false;
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  ggml_tensor *w_key = resident_tensor(g, ctx, prefix + "indexer.attn_k.weight");
  ggml_tensor *w_norm =
      resident_tensor(g, ctx, prefix + "indexer.k_norm.weight");
  ggml_tensor *w_gate =
      resident_tensor(g, ctx, prefix + "indexer_compressor_gate.weight");
  ggml_tensor *w_ape =
      resident_tensor(g, ctx, prefix + "indexer_compressor_ape.weight");
  ggml_tensor *w_bias = nullptr;
  if (model_tensor(*g.engine, prefix + "indexer.k_norm.bias"))
    w_bias = resident_tensor(g, ctx, prefix + "indexer.k_norm.bias");
  if (!w_key || !w_norm || !w_gate || !w_ape ||
      (model_tensor(*g.engine, prefix + "indexer.k_norm.bias") && !w_bias))
    return false;

  const auto &c = g.engine->model_file.architecture_config;
  const float norm_eps = g.engine->model_file.get_config_float(
      "attention.layer_norm_epsilon", c.eps);
  ggml_tensor *keys = ggml_norm(ctx, ggml_mul_mat(ctx, w_key, cur), norm_eps);
  keys = ggml_mul(ctx, keys, w_norm);
  if (w_bias)
    keys = ggml_add(ctx, keys, w_bias);
  ggml_tensor *gates = ggml_mul_mat(ctx, w_gate, cur);
  ggml_tensor *packed = ggml_concat(ctx, keys, gates, 0);
  ggml_tensor *pooled = n_pools > 0
                            ? glm5next_indexer_pool(ctx, keys, gates, w_ape,
                                                    key_length, kpool, n_pools)
                            : nullptr;
  if (!packed || (n_pools > 0 && !pooled))
    return false;
  g.glm5next_indexer_layers.push_back(
      {layer, packed, pooled, nullptr, nullptr});
  return true;
}

bool build_glm5next_indexer_decode(FullGraph &g, ggml_context *ctx,
                                   ggml_tensor *cur, ggml_tensor *qr,
                                   int layer, ggml_tensor **sel_out,
                                   ggml_tensor **mask_out) {
  Glm5nextIndexerLayer *state = glm5next_indexer_state(g, layer);
  if (!state || !state->key_gate_cache || !state->pooled_cache ||
      !g.decode_indices || !g.indexer_pool_mask ||
      !g.indexer_pool_member_ids || !g.indexer_pool_update_rows ||
      !g.indexer_pool_update_index ||
      (g.indexer_select_tail &&
       (!g.indexer_tail_indices || !g.indexer_tail_mask)))
    return false;
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  ggml_tensor *w_key = resident_tensor(g, ctx, prefix + "indexer.attn_k.weight");
  ggml_tensor *w_norm =
      resident_tensor(g, ctx, prefix + "indexer.k_norm.weight");
  ggml_tensor *w_bias = nullptr;
  if (model_tensor(*g.engine, prefix + "indexer.k_norm.bias"))
    w_bias = resident_tensor(g, ctx, prefix + "indexer.k_norm.bias");
  ggml_tensor *w_gate =
      resident_tensor(g, ctx, prefix + "indexer_compressor_gate.weight");
  ggml_tensor *w_ape =
      resident_tensor(g, ctx, prefix + "indexer_compressor_ape.weight");
  ggml_tensor *w_q =
      resident_tensor(g, ctx, prefix + "indexer.attn_q_b.weight");
  ggml_tensor *w_proj =
      resident_tensor(g, ctx, prefix + "indexer.proj.weight");
  if (!w_key || !w_norm || !w_gate || !w_ape || !w_q || !w_proj ||
      (model_tensor(*g.engine, prefix + "indexer.k_norm.bias") && !w_bias))
    return false;

  const auto &c = g.engine->model_file.architecture_config;
  const float norm_eps = g.engine->model_file.get_config_float(
      "attention.layer_norm_epsilon", c.eps);
  ggml_tensor *keys = ggml_norm(ctx, ggml_mul_mat(ctx, w_key, cur), norm_eps);
  keys = ggml_mul(ctx, keys, w_norm);
  if (w_bias)
    keys = ggml_add(ctx, keys, w_bias);
  ggml_tensor *gates = ggml_mul_mat(ctx, w_gate, cur);
  ggml_tensor *packed = ggml_concat(ctx, keys, gates, 0);
  ggml_tensor *raw_updated =
      ggml_set_rows(ctx, state->key_gate_cache, packed, g.decode_indices);
  if (!raw_updated)
    return false;
  ggml_tensor *pool_rows =
      ggml_get_rows(ctx, raw_updated, g.indexer_pool_update_rows);
  const int key_length = g.indexer_key_length;
  const size_t gate_offset = ggml_row_size(pool_rows->type, key_length);
  ggml_tensor *pool_keys =
      ggml_cont(ctx, ggml_view_2d(ctx, pool_rows, key_length, g.indexer_kpool,
                                  pool_rows->nb[1], 0));
  ggml_tensor *pool_gates =
      ggml_cont(ctx, ggml_view_2d(ctx, pool_rows, key_length, g.indexer_kpool,
                                  pool_rows->nb[1], gate_offset));
  ggml_tensor *pool = glm5next_indexer_pool(ctx, pool_keys, pool_gates, w_ape,
                                           key_length, g.indexer_kpool, 1);
  if (!pool)
    return false;
  ggml_tensor *pool_updated = ggml_set_rows(
      ctx, state->pooled_cache, pool, g.indexer_pool_update_index);

  ggml_tensor *indexer_q = ggml_mul_mat(ctx, w_q, qr);
  indexer_q = ggml_reshape_3d(ctx, indexer_q, key_length,
                              g.indexer_heads, 1);
  ggml_tensor *indexer_weights = ggml_mul_mat(ctx, w_proj, cur);
  indexer_weights = ggml_scale(
      ctx, indexer_weights,
      1.0f / std::sqrt((float)key_length * g.indexer_heads));

  const int pool_count = g.kv_capacity / g.indexer_kpool;
  ggml_tensor *pool_keys_all =
      ggml_reshape_3d(ctx, pool_updated, key_length, 1, pool_count);
  ggml_tensor *k_p = ggml_permute(ctx, pool_keys_all, 0, 2, 1, 3);
  ggml_tensor *q_p = ggml_permute(ctx, indexer_q, 0, 2, 1, 3);
  ggml_tensor *kq = ggml_mul_mat(ctx, k_p, q_p);
  kq = ggml_cont(ctx, ggml_permute(ctx, kq, 2, 1, 0, 3));
  ggml_tensor *weights_3d =
      ggml_reshape_3d(ctx, indexer_weights, g.indexer_heads, 1, 1);
  ggml_tensor *score = ggml_mul(ctx, ggml_relu(ctx, kq), weights_3d);
  score = ggml_sum_rows(ctx, score);
  score = ggml_cont(ctx, ggml_permute(ctx, score, 2, 1, 0, 3));
  score = ggml_add(ctx, score, g.indexer_pool_mask);
  const int n_top_pools =
      std::min(pool_count, g.indexer_top_k / g.indexer_kpool);
  if (n_top_pools <= 0)
    return false;
  ggml_tensor *top_pools = ggml_top_k(ctx, score, n_top_pools);
  ggml_tensor *selected_scores = ggml_get_rows(
      ctx, ggml_reshape_3d(ctx, score, 1, pool_count, 1), top_pools);
  ggml_tensor *selection_order = ggml_argsort(
      ctx, ggml_reshape_2d(ctx, selected_scores, n_top_pools, 1),
      GGML_SORT_ORDER_DESC);
  top_pools = ggml_get_rows(
      ctx,
      ggml_reshape_3d(ctx, ggml_cast(ctx, top_pools, GGML_TYPE_F32), 1,
                      n_top_pools, 1),
      selection_order);
  top_pools = ggml_cast(
      ctx, ggml_cont(ctx, ggml_reshape_2d(ctx, top_pools, n_top_pools, 1)),
      GGML_TYPE_I32);
  ggml_tensor *selected_rows =
      ggml_get_rows(ctx, g.indexer_pool_member_ids, top_pools);
  selected_rows = ggml_reshape_2d(
      ctx, selected_rows, n_top_pools * g.indexer_kpool, 1);

  ggml_tensor *pool_mask_2d =
      ggml_reshape_2d(ctx, g.indexer_pool_mask, 1, pool_count);
  ggml_tensor *selected_pool_mask =
      ggml_get_rows(ctx, pool_mask_2d, top_pools);
  selected_pool_mask = ggml_reshape_4d(ctx, selected_pool_mask, 1,
                                       n_top_pools, 1, 1);
  selected_pool_mask = ggml_repeat_4d(
      ctx, selected_pool_mask, g.indexer_kpool, n_top_pools, 1, 1);
  selected_pool_mask = ggml_reshape_2d(
      ctx, selected_pool_mask, n_top_pools * g.indexer_kpool, 1);
  ggml_tensor *selected = selected_rows;
  ggml_tensor *gather_mask = selected_pool_mask;
  const int tail_rows =
      g.indexer_select_tail ? g.indexer_kpool - 1 : 0;
  if (tail_rows > 0) {
    selected = ggml_concat(ctx, selected, g.indexer_tail_indices, 0);
    gather_mask = ggml_concat(ctx, gather_mask, g.indexer_tail_mask, 0);
  }
  const int n_selected = n_top_pools * g.indexer_kpool + tail_rows;
  *sel_out = selected;
  *mask_out = ggml_reshape_4d(ctx, gather_mask, n_selected, 1, 1, 1);
  return *mask_out != nullptr;
}

ggml_tensor *build_dsa(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *wqa, ggml_tensor *wqan,
                       ggml_tensor *wqb, ggml_tensor *wkva, ggml_tensor *wkvan,
                       ggml_tensor *wkb, ggml_tensor *wvb, ggml_tensor *wo,
                       ggml_tensor **kv_out,
                       ggml_tensor *kv_cache = nullptr,
                       ggml_tensor **prev_indexer_selection = nullptr,
                       ggml_tensor **prev_indexer_mask = nullptr) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_tensor *qr = ggml_rms_norm(ctx, ggml_mul_mat(ctx, wqa, cur), c.eps);
  qr = ggml_mul(ctx, qr, wqan);
  ggml_tensor *q = ggml_mul_mat(ctx, wqb, qr);
  q = ggml_reshape_3d(ctx, q, g.qk_nope, g.H, g.S);
  ggml_tensor *kv = ggml_mul_mat(ctx, wkva, cur);
  kv = ggml_mul(ctx, ggml_rms_norm(ctx, kv, c.eps), wkvan);
  *kv_out = kv;

  if (!g.decode_mode && glm5next_indexer_owner(g, layer) &&
      !build_glm5next_indexer_prefill(g, ctx, cur, layer))
    return nullptr;

  q = ggml_permute(ctx, q, 0, 2, 1, 3);
  q = ggml_mul_mat(ctx, wkb, q);
  q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
  const int64_t kv_tokens = g.decode_mode
                                ? g.kv_capacity
                                : (((int64_t)g.S + 255) & ~((int64_t)255));
  ggml_tensor *latent = nullptr;
  ggml_tensor *kv_cache_updated = nullptr;
  if (g.decode_mode) {
    if (!kv_cache || !g.decode_indices)
      return nullptr;
    ggml_tensor *cache_row = ggml_reshape_2d(ctx, kv, g.kv_lora, 1);
    kv_cache_updated = ggml_set_rows(ctx, kv_cache, cache_row, g.decode_indices);
    if (!kv_cache_updated)
      return nullptr;
    latent = ggml_reshape_3d(ctx, kv_cache_updated, g.kv_lora, 1,
                             g.kv_capacity);
  } else {
    latent = ggml_reshape_3d(ctx, kv, g.kv_lora, 1, g.S);
  }
  if (!g.decode_mode && kv_tokens > g.S) {
    // ggml-cuda's 512-wide FATTN kernels require the KV sequence length to
    // be a multiple of FATTN_KQ_STRIDE (256).  llama.cpp's normal KV-cache path
    // already supplies a padded cache.  This standalone no-cache graph must
    // provide the same layout explicitly; zero padding is masked out below.
    latent = ggml_pad(ctx, latent, 0, 0, (int)(kv_tokens - g.S), 0);
  }

  // Keep this layout sequence identical to llama.cpp's build_attn_mha().
  // In particular, the FATTN CUDA dispatch expects the absorbed MQA key and
  // value to be permuted after q has been split into streams, and it expects
  // F32 latent K/V to be explicitly converted to F16.  The previous manual
  // path omitted those details, causing every DSA FLASH_ATTN_EXT node to
  // fall back to the CPU scheduler backend.
  const bool v_trans = latent->nb[1] > latent->nb[2];
  const int64_t n_stream = latent->ne[3];
  q = ggml_view_4d(ctx, q, q->ne[0], q->ne[1], q->ne[2] / n_stream, n_stream,
                   q->nb[1], q->nb[2], q->nb[3] / n_stream, 0);
  q = ggml_permute(ctx, q, 0, 2, 1, 3);
  ggml_tensor *k = ggml_permute(ctx, latent, 0, 2, 1, 3);
  ggml_tensor *v = ggml_permute(ctx, latent, 0, 2, 1, 3);
  if (v_trans)
    v = ggml_transpose(ctx, v);
  if (k->type == GGML_TYPE_F32)
    k = ggml_cast(ctx, k, GGML_TYPE_F16);
  if (v->type == GGML_TYPE_F32)
    v = ggml_cast(ctx, v, GGML_TYPE_F16);

  ggml_tensor *attn = nullptr;
  if (g.decode_mode && g.sparse_decode) {
    if (!prev_indexer_selection || !prev_indexer_mask)
      return nullptr;
    if (glm5next_indexer_owner(g, layer)) {
      ggml_tensor *selection = nullptr;
      ggml_tensor *selection_mask = nullptr;
      if (!build_glm5next_indexer_decode(g, ctx, cur, qr, layer, &selection,
                                         &selection_mask))
        return nullptr;
      *prev_indexer_selection = selection;
      *prev_indexer_mask = selection_mask;
    }
    ggml_tensor *selection = *prev_indexer_selection;
    ggml_tensor *selection_mask = *prev_indexer_mask;
    if (!selection || !selection_mask)
      return nullptr;
    ggml_tensor *gathered = ggml_get_rows(ctx, kv_cache_updated, selection);
    gathered = ggml_cast(ctx, gathered, GGML_TYPE_F32);
    const int64_t n_selected = selection->ne[0];
    ggml_tensor *k_g =
        ggml_reshape_4d(ctx, gathered, g.kv_lora, n_selected, 1, 1);
    ggml_tensor *kq = ggml_mul_mat(ctx, k_g, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(
        ctx, kq, selection_mask,
        1.0f / std::sqrt((float)(g.qk_nope > 0 ? g.qk_nope : 1)), 0.0f);
    ggml_tensor *v_t = ggml_cont(ctx, ggml_transpose(ctx, k_g));
    ggml_tensor *kqv = ggml_mul_mat(ctx, v_t, kq);
    kqv = ggml_mul_mat(ctx, wvb, kqv);
    attn = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
    attn = ggml_reshape_2d(ctx, attn, g.v_head * g.H, g.S);
  } else {
    attn = ggml_flash_attn_ext(
        ctx, q, k, v, g.attn_mask,
        1.0f / std::sqrt((float)(g.qk_nope > 0 ? g.qk_nope : 1)), 0.0f,
        0.0f);
  }
  if (!g.sparse_decode && g.dsa_layers.empty()) {
    LOG_INFO("[GGML-GLM5NEXT-FULL] DSA FATTN layer=%d support_cuda=%d "
             "q=(%lld,%lld,%lld,%lld)/%s k=(%lld,%lld,%lld,%lld)/%s "
             "v=(%lld,%lld,%lld,%lld)/%s mask=(%lld,%lld,%lld,%lld)/%s\n",
             layer, ggml_backend_supports_op(g.cuda, attn) ? 1 : 0,
             (long long)q->ne[0], (long long)q->ne[1], (long long)q->ne[2],
             (long long)q->ne[3], ggml_type_name(q->type), (long long)k->ne[0],
             (long long)k->ne[1], (long long)k->ne[2], (long long)k->ne[3],
             ggml_type_name(k->type), (long long)v->ne[0], (long long)v->ne[1],
             (long long)v->ne[2], (long long)v->ne[3], ggml_type_name(v->type),
             (long long)g.attn_mask->ne[0], (long long)g.attn_mask->ne[1],
             (long long)g.attn_mask->ne[2], (long long)g.attn_mask->ne[3],
             ggml_type_name(g.attn_mask->type));
  }
  // Same v_mla path as build_attn_mha(): FATTN [kv_lora,S,H] is interpreted
  // as [kv_lora,H,S] for WUV, then made contiguous before the 2D projection.
  ggml_tensor *value = attn;
  if (!(g.decode_mode && g.sparse_decode)) {
    value = ggml_permute(ctx, attn, 0, 2, 1, 3);
    value = ggml_mul_mat(ctx, wvb, value);
    value = ggml_permute(ctx, value, 0, 2, 1, 3);
    value = ggml_cont(ctx, value);
    value = ggml_reshape_2d(ctx, value, g.v_head * g.H, g.S);
  }
  (void)layer;
  return ggml_mul_mat(ctx, wo, value);
}


bool stage_layer_active(const FullGraph &g, int layer) {
  const auto &c = g.engine->model_file.architecture_config;
  return g.decode_mode && g.S == 1 && pilot_stage_k() > 0 && offload_layer(layer) && layer < c.n_layers &&
         layer - 1 >= c.first_dense;
}

// 0 for cold experts, -1e9 for experts that live in VRAM (hot, GPU1 hot): the pilot only ranks what the CPU would otherwise compute.
ggml_tensor *stage_hot_mask(FullGraph &g, ggml_context *ctx, int layer) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_tensor *t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.n_experts);
  if (!t)
    return nullptr;
  std::vector<float> m((size_t)c.n_experts, 0.0f);
  if (PartialExpertLayer *p = partial_layer(g, layer)) {
    for (int e : p->hot)
      if (e >= 0 && e < c.n_experts)
        m[(size_t)e] = -1e9f;
    if (g.gpu1)
      for (int e : p->hot1)
        if (e >= 0 && e < c.n_experts)
          m[(size_t)e] = -1e9f;
  }
  if (!arena_alloc(g.dense_buffer, g.dense_base, g.dense_capacity, g.dense_used, t, m.data(), ggml_nbytes(t)))
    return nullptr;
  ggml_set_name(t, ("glm5next_stage_hot_mask_" + std::to_string(layer)).c_str());
  return t;
}

bool build_pilot_stage(FullGraph &g, ggml_context *ctx, ggml_tensor *cur_pre, int l) {
  const auto &c = g.engine->model_file.architecture_config;
  const int next = l + 1;
  if (!stage_layer_active(g, next))
    return true;
  StageRuntime *st = get_stage(g);
  if (!st)
    return true;
  const std::string pn = "blk." + std::to_string(next) + ".";
  ggml_tensor *fn = resident_tensor(g, ctx, pn + "ffn_norm.weight");
  ggml_tensor *router = resident_tensor(g, ctx, pn + "ffn_gate_inp.weight");
  ggml_tensor *bias = resident_tensor(g, ctx, pn + "exp_probs_b.bias");
  if (!fn || !router) {
    LOG_WARN("[GGML-GLM5NEXT-STAGE] layer %d: missing next-layer norm/router tensor\n", l);
    return false;
  }
  ggml_tensor *z = ggml_mul(ctx, ggml_rms_norm(ctx, cur_pre, c.eps), fn);
  ggml_tensor *logits = ggml_mul_mat(ctx, router, z);
  ggml_tensor *probs = nullptr;
  switch (c.gating_func) {
  case GatingFunc::Sigmoid:
    probs = ggml_sigmoid(ctx, logits);
    break;
  case GatingFunc::SqrtSoftplus:
    probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));
    break;
  case GatingFunc::Softmax:
  default:
    probs = ggml_soft_max(ctx, logits);
    break;
  }
  ggml_tensor *score = bias ? ggml_add(ctx, probs, bias) : probs;
  ggml_tensor *mask = stage_hot_mask(g, ctx, next);
  if (!mask) {
    LOG_WARN("[GGML-GLM5NEXT-STAGE] layer %d: hot-mask allocation failed (dense arena %zu/%zu)\n", next, g.dense_used, g.dense_capacity);
    return false;
  }
  score = ggml_add(ctx, score, mask);
  ggml_tensor *pred = ggml_argsort_top_k(ctx, score, st->K);
  ggml_set_name(pred, ("glm5next_stage_pred_" + std::to_string(next)).c_str());
  g.stage_pred[next] = pred;
  return true;
}

ggml_tensor *stage_ring_tensor(FullGraph &g, ggml_context *ctx, StageRuntime &st, int layer, int plane) {
  static const char *stems[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
  const GGUFTensor *m = model_tensor(*g.engine, "blk." + std::to_string(layer) + "." + stems[plane]);
  if (!m || !type_ok(m->type) || m->shape.size() != 3)
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : m->shape)
    shape.push_back((int64_t)d);
  shape[2] = st.K;
  ggml_tensor *t = new_tensor(ctx, (ggml_type)m->type, shape);
  const int set = layer % StageRuntime::kSets;
  OffloadState *off = g_offload.get();
  if (!t || !off || (uint64_t)t->nb[2] != off->arena->plane_bytes(layer, plane) || ggml_nbytes(t) > (size_t)st.K * st.plane_bytes[plane]) {
    LOG_ERROR("[GGML-GLM5NEXT-STAGE] ring shape mismatch layer %d plane %d\n", layer, plane);
    return nullptr;
  }
  if (ggml_backend_tensor_alloc(st.dev_buf[set][plane], t, st.dev_base[set][plane]) != GGML_STATUS_SUCCESS) {
    LOG_WARN("[GGML-GLM5NEXT-STAGE] ring tensor alloc failed layer %d plane %d\n", layer, plane);
    return nullptr;
  }
  ggml_set_name(t, ("glm5next_stage_ring_" + std::to_string(layer) + "_" + std::to_string(plane)).c_str());
  return t;
}

ggml_tensor *stage_miss_node(FullGraph &g, ggml_context *ctx, StageRuntime &st, int layer, ggml_tensor *cold_ids, ggml_tensor *route_ids) {
  OffloadState *o = get_offload(*g.engine);
  if (!o || !cold_ids || !route_ids)
    return nullptr;
  std::lock_guard<std::mutex> lock(g_offload_mutex);
  o->sites.push_back({o, layer, false});
  StageMissSite ms;
  ms.off = &o->sites.back();
  ms.layer = layer;
  ms.set = layer % StageRuntime::kSets;
  ggml_tensor *pred_next = nullptr;
  if (stage_layer_active(g, layer + 1)) {
    auto it = g.stage_pred.find(layer + 1);
    if (it != g.stage_pred.end()) {
      pred_next = it->second;
      ms.next_layer = layer + 1;
      ms.next_set = (layer + 1) % StageRuntime::kSets;
    }
  }
  st.miss_sites.push_back(ms);
  ggml_tensor *out = ggml_map_custom3(ctx, cold_ids, route_ids, pred_next ? pred_next : route_ids, stage_miss_cb, 1, &st.miss_sites.back());
  if (out)
    ggml_set_name(out, ("glm5next_stage_miss_" + std::to_string(layer)).c_str());
  return out;
}

ggml_tensor *build_ffn(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *forced_ids = nullptr) {
  const std::string p = "blk." + std::to_string(layer) + ".";
  const auto &c = g.engine->model_file.architecture_config;
  if (layer < c.first_dense) {
    ggml_tensor *gate = resident_tensor(g, ctx, p + "ffn_gate.weight");
    ggml_tensor *up = resident_tensor(g, ctx, p + "ffn_up.weight");
    ggml_tensor *down = resident_tensor(g, ctx, p + "ffn_down.weight");
    if (!gate || !up || !down)
      return nullptr;
    ggml_tensor *a = ggml_mul_mat(ctx, gate, cur);
    ggml_tensor *b = ggml_mul_mat(ctx, up, cur);
    ggml_tensor *act = ggml_swiglu_split(ctx, a, b);
    return ggml_mul_mat(ctx, down, act);
  }

  ggml_tensor *router = resident_tensor(g, ctx, p + "ffn_gate_inp.weight");
  ggml_tensor *bias = resident_tensor(g, ctx, p + "exp_probs_b.bias");
  ggml_tensor *gate_up = expert_gate_up_weight(g, ctx, layer);
  ggml_tensor *ge = nullptr;
  ggml_tensor *ue = nullptr;
  if (!gate_up) {
    // Keep the separate projection path for older or unusual GGUFs whose
    // gate/up tensors cannot be represented as one contiguous quantized
    // tensor.  GLM-5.3 and DeepSeek-V4 use compatible layouts and take the
    // fused path above.
    ge = expert_weight(g, ctx, layer, "ffn_gate_exps.weight");
    ue = expert_weight(g, ctx, layer, "ffn_up_exps.weight");
  }
  ggml_tensor *de = expert_weight(g, ctx, layer, "ffn_down_exps.weight");
  PartialExpertLayer *partial_weights = partial_layer(g, layer);
  if (g.decode_mode && g.dynamic_expert_cache && partial_weights) {
    partial_weights->cold_gate_up = gate_up;
    partial_weights->cold_gate = ge;
    partial_weights->cold_up = ue;
    partial_weights->cold_down = de;
  }
  if (!router)
    return nullptr;
  if (!partial_weights && (!gate_up && (!ge || !ue)) ||
      (!partial_weights && !de))
    return nullptr;
  if (partial_weights && !partial_weights->cold.empty() &&
      ((!gate_up && (!ge || !ue)) || !de))
    return nullptr;
  if (partial_weights && std::getenv("MOESTREAM_GLM5NEXT_DEBUG_PARTIAL")) {
    const bool ready = partial_layer_ram_ready(g, layer);
    LOG_WARN(
        "[PARTIAL-DEBUG] layer=%d n_experts=%d hot=%zu cold=%zu ready=%d "
        "vram_slots=%zu de_ne2=%lld ge_ne2=%lld ue_ne2=%lld gate_up_ne2=%lld "
        "hot_down_ne2=%lld\n",
        layer, c.n_experts, partial_weights->hot.size(),
        partial_weights->cold.size(), ready ? 1 : 0,
        partial_weights->vram_layer_slots, de ? (long long)de->ne[2] : -1,
        ge ? (long long)ge->ne[2] : -1, ue ? (long long)ue->ne[2] : -1,
        gate_up ? (long long)gate_up->ne[2] : -1);
  }

  ggml_tensor *logits = ggml_mul_mat(ctx, router, cur);
  ggml_tensor *probs = nullptr;
  switch (c.gating_func) {
  case GatingFunc::Sigmoid:
    probs = ggml_sigmoid(ctx, logits);
    break;
  case GatingFunc::SqrtSoftplus:
    probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));
    break;
  case GatingFunc::Softmax:
  default:
    probs = ggml_soft_max(ctx, logits);
    break;
  }
  ggml_tensor *selection = bias ? ggml_add(ctx, probs, bias) : probs;
  // DeepSeek-V4's first hash-routed layers use a token-id -> expert table
  // for selection, while the learned router still supplies the unbiased
  // weights.  The table is prepared as a host input by the V4 graph builder;
  // every other layer keeps GGML's normal biased top-k node.
  ggml_tensor *ids =
      forced_ids ? forced_ids : ggml_argsort_top_k(ctx, selection, c.topk);
  if (route_stats_enabled() && g.decode_mode &&
      g.partial_expert_residency && ids) {
    // Keep a graph output copy of each layer's IDs.  The scheduler may
    // recycle the original intermediate after MUL_MAT_ID consumes it, so
    // reading `ids` after the whole graph finishes can otherwise return
    // overwritten values for earlier layers.
    ggml_tensor *snapshot = ggml_dup(ctx, ids);
    if (!snapshot)
      return nullptr;
    ggml_set_name(snapshot,
                  ("glm5next_route_snapshot_" + std::to_string(layer)).c_str());
    // Keep the route copy in the CUDA split when the backend supports it.
    // Besides making the runtime readback explicit, this prevents 42 small
    // DUP nodes from doubling the scheduler split count in dynamic-cache
    // mode. Conditional graph harvesting will still independently enforce
    // Qwen's one-split safety gate on whichever GLM segment is considered.
    if (g.sched && ggml_backend_supports_op(g.cuda, snapshot))
      ggml_backend_sched_set_tensor_backend(g.sched, snapshot, g.cuda);
    ggml_build_forward_expand(g.graph, snapshot);
    g.moe_route_ids.emplace_back(layer, snapshot);
  }
  ggml_tensor *weights =
      ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, c.n_experts, g.S), ids);
  ggml_tensor *weights2 = ggml_reshape_2d(ctx, weights, c.topk, g.S);
  if (c.expert_weights_norm) {
    ggml_tensor *sum = ggml_clamp(ctx, ggml_sum_rows(ctx, weights2),
                                  6.103515625e-5f, INFINITY);
    weights2 = ggml_div(ctx, weights2, sum);
  }
  if (c.expert_weights_scale != 0.0f && c.expert_weights_scale != 1.0f)
    weights2 = ggml_scale(ctx, weights2, c.expert_weights_scale);
  weights = ggml_reshape_3d(ctx, weights2, 1, c.topk, g.S);
  ggml_build_forward_expand(g.graph, weights);

  float clamp = 0.0f;
  if (layer >= 0 && layer < (int)c.swiglu_clamp_exp.size())
    clamp = c.swiglu_clamp_exp[(size_t)layer];
  ggml_tensor *input3 = ggml_reshape_3d(ctx, cur, g.D, 1, g.S);
  if (g.partial_expert_residency && !prepare_partial_hot(g, ctx, layer))
    return nullptr;
  if (g.partial_expert_residency && !prepare_partial_hot1(g, ctx, layer))
    return nullptr;
  if (partial_weights && std::getenv("MOESTREAM_GLM5NEXT_DEBUG_PARTIAL")) {
    LOG_WARN(
        "[PARTIAL-DEBUG-HOT] layer=%d hot=%zu vram_slots=%zu "
        "hot_gate_ne2=%lld hot_up_ne2=%lld hot_down_ne2=%lld "
        "hot_gate_up_ne2=%lld\n",
        layer, partial_weights->hot.size(), partial_weights->vram_layer_slots,
        partial_weights->hot_gate ? (long long)partial_weights->hot_gate->ne[2]
                                  : -1,
        partial_weights->hot_up ? (long long)partial_weights->hot_up->ne[2]
                                : -1,
        partial_weights->hot_down ? (long long)partial_weights->hot_down->ne[2]
                                  : -1,
        partial_weights->hot_gate_up
            ? (long long)partial_weights->hot_gate_up->ne[2]
            : -1);
  }

  const auto run_branch = [&](ggml_tensor *branch_gate_up,
                              ggml_tensor *branch_gate, ggml_tensor *branch_up,
                              ggml_tensor *branch_down, ggml_tensor *branch_ids,
                              ggml_tensor *branch_weights) -> ggml_tensor * {
    if (!branch_down || !branch_ids || !branch_weights ||
        (!branch_gate_up && (!branch_gate || !branch_up)))
      return nullptr;
    ggml_tensor *up = nullptr;
    ggml_tensor *gate = nullptr;
    if (branch_gate_up) {
      // Match Llama.cpp's merged gate_up graph: one routed projection produces
      // [2*n_ff, top_k, S], then zero-copy views feed SwiGLU.
      ggml_tensor *gate_up_out =
          ggml_mul_mat_id(ctx, branch_gate_up, input3, branch_ids);
      const int64_t n_ff = gate_up_out->ne[0] / 2;
      gate = ggml_view_3d(ctx, gate_up_out, n_ff, gate_up_out->ne[1],
                          gate_up_out->ne[2], gate_up_out->nb[1],
                          gate_up_out->nb[2], 0);
      up = ggml_view_3d(ctx, gate_up_out, n_ff, gate_up_out->ne[1],
                        gate_up_out->ne[2], gate_up_out->nb[1],
                        gate_up_out->nb[2], (size_t)n_ff * gate_up_out->nb[0]);
    } else {
      up = ggml_mul_mat_id(ctx, branch_up, input3, branch_ids);
      gate = ggml_mul_mat_id(ctx, branch_gate, input3, branch_ids);
    }
    ggml_tensor *act = nullptr;
    if (clamp > 1e-6f && swiglu_clamp_fused()) {
      ggml_tensor *up_clamped = ggml_clamp(ctx, up, -clamp, clamp);
      ggml_tensor *gate_clamped = ggml_clamp(ctx, gate, -INFINITY, clamp);
      act = ggml_mul(ctx, ggml_silu(ctx, gate_clamped), up_clamped);
    } else {
      act = ggml_swiglu_split(ctx, gate, up);
      if (clamp > 1e-6f)
        act = ggml_clamp(ctx, act, -clamp, clamp);
    }
    ggml_tensor *experts = ggml_mul_mat_id(ctx, branch_down, act, branch_ids);
    experts = ggml_mul(ctx, experts, branch_weights);
    ggml_tensor *branch =
        ggml_cont(ctx, ggml_view_2d(ctx, experts, g.D, g.S, experts->nb[2], 0));
    for (int k = 1; k < c.topk; ++k)
      branch = ggml_add(ctx, branch,
                        ggml_view_2d(ctx, experts, g.D, g.S, experts->nb[2],
                                     (size_t)k * experts->nb[1]));
    return branch;
  };

  ggml_tensor *out = nullptr;
  // Offloaded layers: the miss node wraps the ids that reach the CPU (cold) MUL_MAT_IDs, so
  // experts held in VRAM are never loaded into DDR; negative ids are skipped by ensure().
  ggml_tensor *route_ids = ids;
  if (PartialExpertLayer *partial = partial_layer(g, layer)) {
    const bool has_hot = !partial->hot.empty();
    const bool has_cold = !partial->cold.empty();
    // Route inputs and the hot+cold merge come from the shared adapter
    // (include/moe_partial_route_ggml.hpp) so every model gets the same
    // inactive-route default.
    moe_partial::BranchInputs hot_in, cold_in;
    if (has_hot) {
      ggml_tensor *hot_map = partial_map(g, ctx, layer, true, false);
      ggml_tensor *hot_mask = partial_map(g, ctx, layer, true, true);
      if (!hot_map || !hot_mask)
        return nullptr;
      hot_in = moe_partial::make_branch_inputs(ctx, hot_map, hot_mask, route_ids,
                                               weights, c.topk, g.S);
    }
    StageRuntime *stage = (has_cold && stage_layer_active(g, layer) && g.stage_pred.count(layer)) ? get_stage(g) : nullptr;
    ggml_tensor *stage_slots = nullptr, *stage_m = nullptr, *stage_keep = nullptr;
    if (has_cold) {
      ggml_tensor *cold_map = partial_map(g, ctx, layer, false, false);
      ggml_tensor *cold_mask = partial_map(g, ctx, layer, false, true);
      if (!cold_map || !cold_mask)
        return nullptr;
      cold_in = moe_partial::make_branch_inputs(ctx, cold_map, cold_mask, route_ids,
                                                weights, c.topk, g.S);
      if (offload_layer(layer)) {
        if (stage) {
          // Packed result: cold id (>= -1) for the CPU branch, or -(slot+2) for a route served from the VRAM ring.
          ggml_tensor *packed = stage_miss_node(g, ctx, *stage, layer, cold_in.ids, route_ids);
          if (!packed)
            return nullptr;
          ggml_tensor *pf = ggml_cast(ctx, packed, GGML_TYPE_F32);
          cold_in.ids = ggml_cast(ctx, ggml_clamp(ctx, pf, -1.0f, 1.0e9f), GGML_TYPE_I32);
          ggml_tensor *slots_f = ggml_clamp(ctx, ggml_scale_bias(ctx, pf, -1.0f, -2.0f), -1.0f, (float)stage->K);
          stage_slots = ggml_cast(ctx, slots_f, GGML_TYPE_I32);
          stage_m = ggml_reshape_3d(ctx, ggml_step(ctx, ggml_scale_bias(ctx, slots_f, 1.0f, 0.5f)), 1, c.topk, 1);
          stage_keep = ggml_scale_bias(ctx, stage_m, -1.0f, 1.0f);
          cold_in.weights = ggml_mul(ctx, cold_in.weights, stage_keep);
        } else {
          cold_in.ids = offload_miss_node(g, ctx, layer, cold_in.ids);
          if (!cold_in.ids)
            return nullptr;
        }
      }
    }
    out = moe_partial::combine_hot_cold(
        ctx, has_hot, has_cold, hot_in, cold_in,
        [&](const moe_partial::BranchInputs &in) {
          return run_branch(partial->hot_gate_up, partial->hot_gate,
                            partial->hot_up, partial->hot_down, in.ids,
                            in.weights);
        },
        [&](const moe_partial::BranchInputs &in) {
          return run_branch(gate_up, ge, ue, de, in.ids, in.weights);
        });
    if (!out)
      return nullptr;
    if (partial->hot1_down) {
      ggml_tensor *m1 = partial_map_hot1(g, ctx, layer, false);
      ggml_tensor *k1 = partial_map_hot1(g, ctx, layer, true);
      if (!m1 || !k1)
        return nullptr;
      const moe_partial::BranchInputs in1 =
          moe_partial::make_branch_inputs(ctx, m1, k1, route_ids, weights, c.topk, g.S);
      ggml_tensor *b1 = run_branch(nullptr, partial->hot1_gate, partial->hot1_up,
                                   partial->hot1_down, in1.ids, in1.weights);
      if (!b1)
        return nullptr;
      out = ggml_add(ctx, out, b1);
    }
    if (stage_slots) {
      ggml_tensor *rg = stage_ring_tensor(g, ctx, *stage, layer, 0);
      ggml_tensor *ru = stage_ring_tensor(g, ctx, *stage, layer, 1);
      ggml_tensor *rd = stage_ring_tensor(g, ctx, *stage, layer, 2);
      if (!rg || !ru || !rd)
        return nullptr;
      ggml_tensor *bs = run_branch(nullptr, rg, ru, rd, stage_slots, ggml_mul(ctx, weights, stage_m));
      if (!bs)
        return nullptr;
      out = ggml_add(ctx, out, bs);
    }
  } else {
    ggml_tensor *branch_ids = route_ids;
    if (offload_layer(layer)) {
      branch_ids = offload_miss_node(g, ctx, layer, route_ids);
      if (!branch_ids)
        return nullptr;
    }
    out = run_branch(gate_up, ge, ue, de, branch_ids, weights);
    if (!out)
      return nullptr;
  }

  ggml_tensor *sg = resident_tensor(g, ctx, p + "ffn_gate_shexp.weight");
  ggml_tensor *su = resident_tensor(g, ctx, p + "ffn_up_shexp.weight");
  ggml_tensor *sd = resident_tensor(g, ctx, p + "ffn_down_shexp.weight");
  if (sg && su && sd) {
    ggml_tensor *gate_shared = ggml_mul_mat(ctx, sg, cur);
    ggml_tensor *up_shared = ggml_mul_mat(ctx, su, cur);
    const float clamp_shared =
        (layer >= 0 && layer < (int)c.swiglu_clamp_shexp.size())
            ? c.swiglu_clamp_shexp[(size_t)layer]
            : 0.0f;
    ggml_tensor *a = nullptr;
    if (clamp_shared > 1e-6f && swiglu_clamp_fused()) {
      // GLM5NEXT/reference ggml_swiglu_clamp semantics.
      up_shared = ggml_clamp(ctx, up_shared, -clamp_shared, clamp_shared);
      gate_shared = ggml_clamp(ctx, gate_shared, -INFINITY, clamp_shared);
      a = ggml_mul(ctx, ggml_silu(ctx, gate_shared), up_shared);
    } else {
      a = ggml_swiglu_split(ctx, gate_shared, up_shared);
      // DeepSeek-V4's established path clamps the completed SwiGLU product
      // when the fused GLM5NEXT flag is not active.
      if (clamp_shared > 1e-6f)
        a = ggml_clamp(ctx, a, -clamp_shared, clamp_shared);
    }
    out = ggml_add(ctx, out, ggml_mul_mat(ctx, sd, a));
  }
  return out;
}

bool build_graph(FullGraph &g) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_init_params ip = {};
  ip.mem_size = kTensorContextBytes;
  ip.no_alloc = true;
  if (!MemoryGuard::try_reserve_host_bytes(kTensorContextBytes)) {
    LOG_WARN("[GGML-GLM5NEXT-FULL-MEM] host graph context refused: %.2f MiB "
             "(reserved=%.2f GiB, budget=%.2f GB)\n",
             (double)kTensorContextBytes / (1024.0 * 1024.0),
             (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
             (double)MemoryGuard::host_budget_bytes() / 1.0e9);
    return false;
  }
  g.context_reservation_bytes = kTensorContextBytes;
  g.ctx = ggml_init(ip);
  if (!g.ctx) {
    MemoryGuard::release_host_bytes(g.context_reservation_bytes);
    g.context_reservation_bytes = 0;
    return false;
  }
  g.graph = ggml_new_graph_custom(g.ctx, kGraphSize, false);
  if (!g.graph)
    return false;

  g.input = host_input(g, g.ctx, GGML_TYPE_F32, {g.D, g.HC, g.S},
                       (size_t)g.D * g.HC * g.S * sizeof(float),
                       "glm5next_full_input");
  const int64_t kv_tokens = g.decode_mode
                                ? g.kv_capacity
                                : (((int64_t)g.S + 255) & ~((int64_t)255));
  g.attn_mask = host_input(g, g.ctx, GGML_TYPE_F16, {kv_tokens, g.S, 1, 1},
                           (size_t)kv_tokens * g.S * sizeof(ggml_fp16_t),
                           "glm5next_full_causal_mask");
  if (g.decode_mode)
    g.decode_indices = host_input(g, g.ctx, GGML_TYPE_I32, {1}, sizeof(int32_t),
                                  "glm5next_full_decode_kv_index");
  if (!g.input || !g.attn_mask || (g.decode_mode && !g.decode_indices))
    return false;
  auto *mask = static_cast<ggml_fp16_t *>(g.attn_mask->data);
  for (int q = 0; q < g.S; ++q)
    for (int k = 0; k < kv_tokens; ++k)
      mask[(size_t)q * kv_tokens + k] =
          ggml_fp32_to_fp16(g.decode_mode ? -INFINITY
                                          : (k < g.S && k <= q ? 0.0f
                                                               : -INFINITY));

  if (g.decode_mode && g.sparse_decode) {
    if (g.indexer_kpool <= 1 || g.kv_capacity % g.indexer_kpool != 0 ||
        g.indexer_key_length <= 0 || g.indexer_heads <= 0 ||
        g.indexer_top_k <= 0)
      return false;
    const int pool_count = g.kv_capacity / g.indexer_kpool;
    const int tail_rows =
        g.indexer_select_tail ? g.indexer_kpool - 1 : 0;
    g.indexer_pool_member_ids = host_input(
        g, g.ctx, GGML_TYPE_I32, {g.indexer_kpool, pool_count},
        (size_t)g.indexer_kpool * pool_count * sizeof(int32_t),
        "glm5next_indexer_pool_member_ids");
    g.indexer_pool_mask = host_input(
        g, g.ctx, GGML_TYPE_F32, {pool_count},
        (size_t)pool_count * sizeof(float), "glm5next_indexer_pool_mask");
    g.indexer_pool_update_rows = host_input(
        g, g.ctx, GGML_TYPE_I32, {g.indexer_kpool},
        (size_t)g.indexer_kpool * sizeof(int32_t),
        "glm5next_indexer_pool_update_rows");
    g.indexer_pool_update_index = host_input(
        g, g.ctx, GGML_TYPE_I32, {1}, sizeof(int32_t),
        "glm5next_indexer_pool_update_index");
    if (tail_rows > 0) {
      g.indexer_tail_indices = host_input(
          g, g.ctx, GGML_TYPE_I32, {tail_rows},
          (size_t)tail_rows * sizeof(int32_t),
          "glm5next_indexer_tail_indices");
      g.indexer_tail_mask = host_input(
          g, g.ctx, GGML_TYPE_F32, {tail_rows},
          (size_t)tail_rows * sizeof(float), "glm5next_indexer_tail_mask");
    }
    if (!g.indexer_pool_member_ids || !g.indexer_pool_mask ||
        !g.indexer_pool_update_rows || !g.indexer_pool_update_index ||
        (tail_rows > 0 &&
         (!g.indexer_tail_indices || !g.indexer_tail_mask)))
      return false;
    auto *members = static_cast<int32_t *>(g.indexer_pool_member_ids->data);
    for (int pool = 0; pool < pool_count; ++pool)
      for (int i = 0; i < g.indexer_kpool; ++i)
        members[(size_t)pool * g.indexer_kpool + i] =
            pool * g.indexer_kpool + i;
  }

  ggml_tensor *res = g.input;
  ggml_tensor *previous_indexer_selection = nullptr;
  ggml_tensor *previous_indexer_mask = nullptr;
  for (int l = 0; l < g.n_layers; ++l) {
    // The graph owns a complete DDR source arena, so this is a coarse
    // per-layer admission point rather than a check on every tensor/alloc.
    // If the background guard sees the host watermark crossed, decline the
    // experimental graph and let the caller use the bounded legacy path.
    if (MemoryGuard::allocation_blocked()) {
      LOG_WARN("[GGML-GLM5NEXT-FULL] graph build stopped at layer %d/%d: "
               "host memory watermark reached (MemAvailable=%ld MiB)\n",
               l, g.n_layers, MemoryGuard::last_available_mb());
      return false;
    }
    const std::string p = "blk." + std::to_string(l) + ".";
    ggml_tensor *post = nullptr;
    ggml_tensor *comb = nullptr;
    ggml_tensor *afn = resident_tensor(g, g.ctx, p + "hc_attn_fn.weight");
    ggml_tensor *abase = resident_tensor(g, g.ctx, p + "hc_attn_base.weight");
    ggml_tensor *ascale = resident_tensor(g, g.ctx, p + "hc_attn_scale.weight");
    if (!afn || !abase || !ascale)
      return false;
    ggml_tensor *residual = res;
    ggml_tensor *cur = hc_pre(g, g.ctx, res, afn, ascale, abase, &post, &comb);
    ggml_tensor *an = resident_tensor(g, g.ctx, p + "attn_norm.weight");
    if (!an)
      return false;
    cur = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, cur, c.eps), an);

    ggml_tensor *attn = nullptr;
    if (glm5next_is_kda_layer(c, l)) {
      ggml_tensor *wq = resident_tensor(g, g.ctx, p + "attn_q.weight");
      ggml_tensor *wk = resident_tensor(g, g.ctx, p + "attn_k.weight");
      ggml_tensor *wv = resident_tensor(g, g.ctx, p + "attn_v.weight");
      ggml_tensor *cq = resident_tensor(g, g.ctx, p + "ssm_conv1d_q.weight");
      ggml_tensor *ck = resident_tensor(g, g.ctx, p + "ssm_conv1d_k.weight");
      ggml_tensor *cv = resident_tensor(g, g.ctx, p + "ssm_conv1d_v.weight");
      ggml_tensor *fa = resident_tensor(g, g.ctx, p + "ssm_f_a.weight");
      ggml_tensor *fb = resident_tensor(g, g.ctx, p + "ssm_f_b.weight");
      ggml_tensor *ga = resident_tensor(g, g.ctx, p + "ssm_g_a.weight");
      ggml_tensor *gb = resident_tensor(g, g.ctx, p + "ssm_g_b.weight");
      ggml_tensor *be = resident_tensor(g, g.ctx, p + "ssm_beta.weight");
      ggml_tensor *aa = resident_tensor(g, g.ctx, p + "ssm_a");
      ggml_tensor *dt = resident_tensor(g, g.ctx, p + "ssm_dt.bias");
      ggml_tensor *nm = resident_tensor(g, g.ctx, p + "ssm_norm.weight");
      ggml_tensor *ow = resident_tensor(g, g.ctx, p + "attn_output.weight");
      if (!wq || !wk || !wv || !cq || !ck || !cv || !fa || !fb || !ga || !gb ||
          !be || !aa || !dt || !nm || !ow)
        return false;
      const int hist = g.d_conv - 1;
      ggml_tensor *hi = nullptr;
      ggml_tensor *si = nullptr;
      if (g.decode_mode) {
        hi = new_decode_state_tensor(
            g, g.ctx, GGML_TYPE_F32, {hist, 3 * g.d_inner, 1},
            "glm5next_decode_kda_history");
        si = new_decode_state_tensor(
            g, g.ctx, GGML_TYPE_F32, {g.hd, g.hd, g.H, 1},
            "glm5next_decode_kda_state");
        if (hi && si) {
          g.kda_hist_storage.push_back(hi);
          g.kda_state_storage.push_back(si);
        }
      } else {
        hi = host_input(
            g, g.ctx, GGML_TYPE_F32, {hist, 3 * g.d_inner, 1},
            (size_t)hist * 3 * g.d_inner * sizeof(float),
            "glm5next_kda_history");
        si = host_input(
            g, g.ctx, GGML_TYPE_F32, {g.hd, g.hd, g.H, 1},
            (size_t)g.hd * g.hd * g.H * sizeof(float), "glm5next_kda_state");
      }
      if (!hi || !si)
        return false;
      if (!g.decode_mode) {
        g.kda_hist_inputs.push_back((float *)hi->data);
        g.kda_state_inputs.push_back((float *)si->data);
      }
      ggml_tensor *so = nullptr;
      ggml_tensor *ho = nullptr;
      attn = build_kda(g, g.ctx, cur, l, wq, wk, wv, cq, ck, cv, fa, fb, ga, gb,
                       be, aa, dt, nm, ow, hi, si, &so, &ho);
      if (!attn || !so || !ho)
        return false;
      if (g.decode_mode) {
        g.kda_state_updates.push_back(ggml_cpy(g.ctx, so, si));
        g.kda_hist_updates.push_back(ggml_cpy(g.ctx, ho, hi));
        if (!g.kda_state_updates.back() || !g.kda_hist_updates.back())
          return false;
      } else {
        g.kda_state_out.push_back(so);
        g.kda_hist_out.push_back(ho);
      }
      g.kda_layers.push_back(l);
    } else {
      ggml_tensor *wqa = resident_tensor(g, g.ctx, p + "attn_q_a.weight");
      ggml_tensor *wqan = resident_tensor(g, g.ctx, p + "attn_q_a_norm.weight");
      ggml_tensor *wqb = resident_tensor(g, g.ctx, p + "attn_q_b.weight");
      ggml_tensor *wkva = resident_tensor(g, g.ctx, p + "attn_kv_a_mqa.weight");
      ggml_tensor *wkvan =
          resident_tensor(g, g.ctx, p + "attn_kv_a_norm.weight");
      ggml_tensor *wkb = static_full_weight(g, g.ctx, p + "attn_k_b.weight");
      ggml_tensor *wvb = static_full_weight(g, g.ctx, p + "attn_v_b.weight");
      ggml_tensor *wo = resident_tensor(g, g.ctx, p + "attn_output.weight");
      if (!wqa || !wqan || !wqb || !wkva || !wkvan || !wkb || !wvb || !wo)
        return false;
      ggml_tensor *kv_cache = nullptr;
      if (g.decode_mode) {
        kv_cache = new_decode_state_tensor(
            g, g.ctx, GGML_TYPE_F16, {g.kv_lora, g.kv_capacity},
            "glm5next_decode_dsa_kv");
        if (!kv_cache)
          return false;
        g.dsa_kv_storage.push_back(kv_cache);
        if (g.sparse_decode && glm5next_indexer_owner(g, l)) {
          Glm5nextIndexerLayer indexer_state;
          indexer_state.layer = l;
          indexer_state.key_gate_cache = new_decode_state_tensor(
              g, g.ctx, GGML_TYPE_F32,
              {2 * g.indexer_key_length, g.kv_capacity},
              "glm5next_decode_indexer_key_gate");
          indexer_state.pooled_cache = new_decode_state_tensor(
              g, g.ctx, GGML_TYPE_F32,
              {g.indexer_key_length, g.kv_capacity / g.indexer_kpool},
              "glm5next_decode_indexer_pooled_keys");
          if (!indexer_state.key_gate_cache || !indexer_state.pooled_cache)
            return false;
          g.glm5next_indexer_layers.push_back(indexer_state);
        }
      }
      ggml_tensor *kv = nullptr;
      attn = build_dsa(g, g.ctx, cur, l, wqa, wqan, wqb, wkva, wkvan, wkb, wvb,
                       wo, &kv, kv_cache, &previous_indexer_selection,
                       &previous_indexer_mask);
      if (!attn || !kv)
        return false;
      if (!g.decode_mode)
        g.dsa_kv_out.push_back(kv);
      g.dsa_layers.push_back(l);
    }
    res = hc_post(g, g.ctx, attn, residual, post, comb);
    if (g.decode_mode && layer_timing_env())
      g.layer_marks[res] = {l, 0};

    residual = res;
    post = comb = nullptr;
    ggml_tensor *ffn_fn = resident_tensor(g, g.ctx, p + "hc_ffn_fn.weight");
    ggml_tensor *ffn_base = resident_tensor(g, g.ctx, p + "hc_ffn_base.weight");
    ggml_tensor *ffn_scale =
        resident_tensor(g, g.ctx, p + "hc_ffn_scale.weight");
    if (!ffn_fn || !ffn_base || !ffn_scale)
      return false;
    cur = hc_pre(g, g.ctx, res, ffn_fn, ffn_scale, ffn_base, &post, &comb);
    ggml_tensor *cur_pre = cur;
    ggml_tensor *fn = resident_tensor(g, g.ctx, p + "ffn_norm.weight");
    if (!fn)
      return false;
    cur = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, cur, c.eps), fn);
    ggml_build_forward_expand(g.graph, residual);
    ggml_build_forward_expand(g.graph, post);
    ggml_build_forward_expand(g.graph, comb);
    if (!build_pilot_stage(g, g.ctx, cur_pre, l))
      return false;
    ggml_tensor *ffn = build_ffn(g, g.ctx, cur, l);
    if (!ffn) {
      LOG_WARN("[GGML-GLM5NEXT-FULL] build_ffn failed at layer=%d "
               "graph_ram_ready=%d\n",
               l, g.engine->graph_ram_layer_ready(l) ? 1 : 0);
      return false;
    }
    res = hc_post(g, g.ctx, ffn, residual, post, comb);
    if (g.decode_mode && layer_timing_env())
      g.layer_marks[res] = {l, 1};
    if (l >= c.first_dense && (l == c.first_dense || l + 1 == g.n_layers ||
                               ((l - c.first_dense + 1) % 4) == 0)) {
      LOG_INFO("[GGML-GLM5NEXT-FULL] graph source progress layer=%d/%d "
               "expert_source=%.2f GiB MemAvailable=%ld MiB\n",
               l + 1, g.n_layers,
               (double)g.expert_source_used / (1024.0 * 1024.0 * 1024.0),
               MemoryGuard::last_available_mb());
    }
  }

  g.final_streams = res;
  // The surrounding generation loop applies output_norm.weight immediately
  // before the LM-head projection.  Return the same pre-norm activation as
  // the legacy layer path; applying output_norm here would normalize twice.
  g.output = hc_mean(g, g.ctx, res);

  ggml_set_output(g.final_streams);
  ggml_build_forward_expand(g.graph, g.final_streams);
  ggml_set_output(g.output);
  ggml_build_forward_expand(g.graph, g.output);
  if (g.decode_mode && !std::getenv("MOESTREAM_GLM5NEXT_CPU_LM_HEAD")) {
    ggml_tensor *out_norm = resident_tensor(g, g.ctx, "output_norm.weight");
    ggml_tensor *out_w = resident_tensor(g, g.ctx, "output.weight");
    if (out_norm && out_w) {
      ggml_tensor *lm_in = ggml_mul(
          g.ctx,
          ggml_rms_norm(g.ctx, g.output,
                        g.engine->model_file.architecture_config.eps),
          out_norm);
      g.logits = ggml_mul_mat(g.ctx, out_w, lm_in);
      ggml_set_output(g.logits);
      ggml_build_forward_expand(g.graph, g.logits);
    }
  }
  if (g.decode_mode) {
    for (ggml_tensor *t : g.kda_state_updates)
      ggml_build_forward_expand(g.graph, t);
    for (ggml_tensor *t : g.kda_hist_updates)
      ggml_build_forward_expand(g.graph, t);
  } else {
    for (ggml_tensor *t : g.kda_state_out) {
      ggml_set_output(t);
      ggml_build_forward_expand(g.graph, t);
    }
    for (ggml_tensor *t : g.kda_hist_out) {
      ggml_set_output(t);
      ggml_build_forward_expand(g.graph, t);
    }
  }
  for (ggml_tensor *t : g.dsa_kv_out) {
    ggml_set_output(t);
    ggml_build_forward_expand(g.graph, t);
  }
  if (!g.decode_mode) {
    for (const Glm5nextIndexerLayer &state : g.glm5next_indexer_layers) {
      if (!state.key_gate_out)
        return false;
      ggml_set_output(state.key_gate_out);
      ggml_build_forward_expand(g.graph, state.key_gate_out);
      if (state.pooled_out) {
        ggml_set_output(state.pooled_out);
        ggml_build_forward_expand(g.graph, state.pooled_out);
      }
    }
  }
  if (!graph_alloc_vram_headroom_ok(g, "[GGML-GLM5NEXT-FULL]"))
    return false;
  if (!ggml_backend_sched_alloc_graph(g.sched, g.graph))
    return false;
  g.graph_workspace_bytes = ggml_backend_sched_get_buffer_size(g.sched, g.cuda);

  int cuda_nodes = 0, cpu_nodes = 0, other_nodes = 0, transitions = 0;
  ggml_backend_t previous = nullptr;
  std::unordered_map<std::string, int> non_cuda;
  for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
    ggml_tensor *node = ggml_graph_node(g.graph, i);
    ggml_backend_t b = ggml_backend_sched_get_tensor_backend(g.sched, node);
    if (b == g.cuda)
      ++cuda_nodes;
    else if (b == g.cpu) {
      ++cpu_nodes;
      ++non_cuda[ggml_op_name(node->op)];
    } else {
      ++other_nodes;
      ++non_cuda[b ? ggml_backend_name(b) : "NULL"];
    }
    if (previous && previous != b)
      ++transitions;
    previous = b;
    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
      const ggml_tensor *q = node->src[0];
      const ggml_tensor *k = node->src[1];
      const ggml_tensor *v = node->src[2];
      const ggml_tensor *m = node->src[3];
      LOG_INFO("[GGML-GLM5NEXT-FULL] FATTN node=%d backend=%s support_cuda=%d "
               "q=(%lld,%lld,%lld,%lld)/%s k=(%lld,%lld,%lld,%lld)/%s "
               "v=(%lld,%lld,%lld,%lld)/%s mask=(%lld,%lld,%lld,%lld)/%s nb="
               "%zu,%zu,%zu,%zu/%zu,%zu,%zu,%zu\n",
               i, b ? ggml_backend_name(b) : "NULL",
               ggml_backend_supports_op(g.cuda, node) ? 1 : 0,
               (long long)q->ne[0], (long long)q->ne[1], (long long)q->ne[2],
               (long long)q->ne[3], ggml_type_name(q->type),
               (long long)k->ne[0], (long long)k->ne[1], (long long)k->ne[2],
               (long long)k->ne[3], ggml_type_name(k->type),
               (long long)v->ne[0], (long long)v->ne[1], (long long)v->ne[2],
               (long long)v->ne[3], ggml_type_name(v->type),
               (long long)m->ne[0], (long long)m->ne[1], (long long)m->ne[2],
               (long long)m->ne[3], ggml_type_name(m->type), q->nb[0], q->nb[1],
               q->nb[2], q->nb[3], k->nb[0], k->nb[1], k->nb[2], k->nb[3]);
    }
  }
  LOG_INFO("[GGML-GLM5NEXT-FULL] scheduler nodes cuda=%d cpu=%d other=%d "
           "transitions=%d\n",
           cuda_nodes, cpu_nodes, other_nodes, transitions);
  if (!non_cuda.empty()) {
    std::string detail;
    for (const auto &[name, count] : non_cuda) {
      if (!detail.empty())
        detail += ", ";
      detail += name + "=" + std::to_string(count);
    }
    LOG_INFO("[GGML-GLM5NEXT-FULL] non-CUDA nodes: %s\n", detail.c_str());
  }
  LOG_INFO("[GGML-GLM5NEXT-FULL] graph built layers=%d nodes=%d splits=%d "
           "workspace=%.2f GiB KDA=%zu DSA=%zu route_snapshots=%zu\n",
           g.n_layers, ggml_graph_n_nodes(g.graph),
           ggml_backend_sched_get_n_splits(g.sched),
           (double)g.graph_workspace_bytes / (1024.0 * 1024.0 * 1024.0),
           g.kda_layers.size(), g.dsa_layers.size(), g.moe_route_ids.size());
  LOG_INFO(
      "[GGML-GLM5NEXT-FULL] dense arena=%.2f/%.2f GiB tensors=%d; "
      "graph-owned expert weights=%.2f/%.2f GiB; VRAM cache aliases=%.2f "
      "GiB (included in cache, not additional); host source=%.2f GiB, "
      "DDR refs=%.2f GiB (not additional), pinned=%.2f GiB "
      "pin_failures=%zu\n",
      (double)g.dense_used / (1024.0 * 1024.0 * 1024.0),
      (double)g.dense_capacity / (1024.0 * 1024.0 * 1024.0), g.dense_tensors,
      (double)g.expert_used / (1024.0 * 1024.0 * 1024.0),
      (double)g.expert_capacity / (1024.0 * 1024.0 * 1024.0),
      (double)g.vram_reference_bytes / (1024.0 * 1024.0 * 1024.0),
      (double)g.host_weight_bytes / (1024.0 * 1024.0 * 1024.0),
      (double)g.host_reference_bytes / (1024.0 * 1024.0 * 1024.0),
      (double)g.host_pinned_bytes / (1024.0 * 1024.0 * 1024.0),
      g.host_pin_failures);
  LOG_INFO("[GGML-GLM5NEXT-FULL-MEM] host ledger=%.2f/%.2f GB; graph source="
           "%.2f GiB + context=%.2f MiB + owned-extra=%.2f MiB; device "
           "workspace=%.2f GiB (GPU, outside host_ram_gb)\n",
           (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
           (double)MemoryGuard::host_budget_bytes() / 1.0e9,
           (double)g.expert_source_capacity / (1024.0 * 1024.0 * 1024.0),
           (double)g.context_reservation_bytes / (1024.0 * 1024.0),
           (double)(g.host_weight_bytes > g.expert_source_used
                        ? g.host_weight_bytes - g.expert_source_used
                        : 0) /
               (1024.0 * 1024.0),
           (double)g.graph_workspace_bytes / (1024.0 * 1024.0 * 1024.0));
  g.ready = true;
  return true;
}

} // namespace

bool glm5next_full_graph_enabled() { return env_enabled(); }

void glm5next_full_graph_set_vram_cache(GpuTopology *topology) {
  g_glm5next_vram_cache = topology;
}

bool glm5next_full_graph_bootstrap(StreamingEngine &engine) {
  if (!env_enabled() || engine.family_ops.family != ModelFamily::GLM5NEXT)
    return false;
  // Build and seed the offload cache at boot, not inside the first decode token.
  FullGraph candidate;
  candidate.engine = &engine;
  candidate.topology = g_glm5next_vram_cache;
  candidate.device = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  candidate.first_dense = engine.model_file.architecture_config.first_dense;
  candidate.device_expert_layers =
      std::min(glm5next_full_graph_device_expert_layers(),
               std::max(0, engine.model_file.architecture_config.n_layers -
                               candidate.first_dense));
  if (configure_partial_experts(candidate))
    candidate.device_expert_layers = 0;
  if (offload_first_layer() >= 0 &&
      !get_offload(engine, candidate.partial_expert_residency
                               ? &candidate.partial_experts
                               : nullptr))
    return false;

  if (g_glm5next_source &&
      source_configuration_matches(*g_glm5next_source, candidate)) {
    LOG_INFO("[GGML-GLM5NEXT-FULL] boot expert source already resident; "
             "reusing it\n");
    return true;
  }
  g_glm5next_source.reset();
  auto source = std::make_unique<FullGraph>();
  source->engine = candidate.engine;
  source->topology = candidate.topology;
  source->device = candidate.device;
  source->first_dense = candidate.first_dense;
  source->device_expert_layers = candidate.device_expert_layers;
  source->partial_expert_residency = candidate.partial_expert_residency;
  source->vram_reference_experts = candidate.vram_reference_experts;
  source->dynamic_expert_cache = candidate.dynamic_expert_cache;
  source->partial_sources_compacted = candidate.partial_sources_compacted;
  source->partial_hot_bytes = candidate.partial_hot_bytes;
  source->partial_cold_bytes = candidate.partial_cold_bytes;
  source->partial_hot_records = candidate.partial_hot_records;
  source->vram_reference_bytes = candidate.vram_reference_bytes;
  source->partial_usage_file = std::move(candidate.partial_usage_file);
  source->partial_experts = std::move(candidate.partial_experts);
  source->vram_expert_holds = std::move(candidate.vram_expert_holds);
  if (!prepare_expert_source_only(*source, "GGML-GLM5NEXT-FULL"))
    return false;
  g_glm5next_source = std::move(source);
  return true;
}

int glm5next_full_graph_device_expert_layers() {
  return env_enabled() ? env_device_expert_layers() : 0;
}

void glm5next_full_graph_shutdown() {
  g_graph.reset();
  for (auto &kv : g_hot1_buffers)
    ggml_backend_buffer_free(kv.second);
  g_hot1_buffers.clear();
  g_glm5next_source.reset();
  g_glm5next_decode_disabled_engine = nullptr;
}

bool glm5next_full_graph_prepare(StreamingEngine &engine, int S, int pos_base,
                                 const int *) {
  if (!env_enabled() || S <= 1 || pos_base != 0 ||
      engine.family_ops.family != ModelFamily::GLM5NEXT)
    return false;
  const auto &c = engine.model_file.architecture_config;
  if (g_graph && g_graph->ready && g_graph->engine == &engine &&
      g_graph->S == S)
    return true;
  g_glm5next_decode_disabled_engine = nullptr;

  // Keep the large source arena alive while releasing only the old
  // prompt-shaped graph.  This is what makes boot preload useful for a later
  // prompt with a different token count.
  if (g_graph && g_graph->has_expert_source()) {
    if (!g_glm5next_source)
      g_glm5next_source = std::make_unique<FullGraph>();
    if (!g_graph->move_expert_source_to(*g_glm5next_source))
      return false;
  }
  g_graph.reset();
  auto g = std::make_unique<FullGraph>();
  g->engine = &engine;
  g->topology = g_glm5next_vram_cache;
  g->device = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  g->S = S;
  g->D = c.hidden;
  g->HC = std::max(1, c.hc_mult);
  g->n_layers = c.n_layers;
  g->H = c.n_heads;
  g->hd = c.kda_head_dim;
  g->d_inner = g->H * g->hd;
  g->d_conv = c.ssm_conv_kernel;
  g->qk_nope = c.qk_nope;
  g->kv_lora = c.kv_lora;
  g->indexer_heads = c.glm5next_indexer_head_count;
  g->indexer_key_length = c.glm5next_indexer_key_length;
  g->indexer_top_k = c.glm5next_indexer_top_k;
  g->indexer_kpool = c.glm5next_indexer_kpool;
  g->indexer_select_tail =
      engine.model_file.get_config_int("attention.indexer.kpool_select_tail", 0) != 0;
  g->v_head = c.v_head;
  g->first_dense = c.first_dense;
  g->device_expert_layers = std::min(glm5next_full_graph_device_expert_layers(),
                                     std::max(0, c.n_layers - c.first_dense));
  if (configure_partial_experts(*g))
    g->device_expert_layers = 0;
  g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(g->device));
  g->cpu = ggml_backend_cpu_init();
  if (g->cpu) {
    const char *e = std::getenv("MOESTREAM_GLM5NEXT_CPU_THREADS");
    ggml_backend_cpu_set_n_threads(g->cpu, e ? std::max(1, std::atoi(e)) : 6);
  }
  if (!g->cuda || !g->cpu)
    return false;

  if (g_glm5next_source &&
      source_configuration_matches(*g_glm5next_source, *g)) {
    if (!g_glm5next_source->move_expert_source_to(*g))
      return false;
    LOG_INFO("[GGML-GLM5NEXT-FULL] reusing boot expert source: %.2f GiB DDR; "
             "no SSD reload for prompt S=%d\n",
             (double)g->expert_source_used / (1024.0 * 1024.0 * 1024.0), S);
  } else if (!prepare_expert_source_only(*g, "GGML-GLM5NEXT-FULL")) {
    return false;
  }
  if (!init_expert_arena(*g) || !init_dense_arena(*g))
    return false;
  ggml_backend_t backends[] = {g->cuda, g->cpu};
  g->sched =
      ggml_backend_sched_new(backends, nullptr, 2, kGraphSize, false, true);
  if (!g->sched)
    return false;
  if (!build_graph(*g)) {
    LOG_WARN("[GGML-GLM5NEXT-FULL] graph construction declined; using legacy "
             "path\n");
    return false;
  }
  g_graph = std::move(g);
  LOG_INFO("[GGML-GLM5NEXT-FULL] graph prepared for S=%d\n", S);
  return true;
}

static void transpose_glm5next_state(float *dst, const float *src, int hd,
                                      int heads) {
  const size_t plane = (size_t)hd * hd;
  for (int h = 0; h < heads; ++h)
    for (int row = 0; row < hd; ++row)
      for (int col = 0; col < hd; ++col)
        dst[(size_t)h * plane + (size_t)row * hd + col] =
            src[(size_t)h * plane + (size_t)col * hd + row];
}

static bool sync_glm5next_decode_state_to_host(FullGraph &g) {
  if (!g.decode_mode || !g.ready || !g.engine || g.decode_last_pos < 0)
    return false;
  StreamingEngine &engine = *g.engine;
  const size_t state_n = (size_t)g.hd * g.hd * g.H;
  const size_t hist_n = (size_t)(g.d_conv - 1) * 3 * g.d_inner;
  std::vector<float> state(state_n);
  for (size_t i = 0; i < g.kda_layers.size(); ++i) {
    const int layer = g.kda_layers[i];
    if (layer < 0 || layer >= (int)engine.glm5next_recur_state.size() ||
        layer >= (int)engine.glm5next_conv_history.size() ||
        engine.glm5next_recur_state[layer].size() != state_n ||
        engine.glm5next_conv_history[layer].size() != hist_n ||
        i >= g.kda_state_storage.size() || i >= g.kda_hist_storage.size())
      return false;
    ggml_backend_tensor_get(g.kda_state_storage[i], state.data(), 0,
                            state_n * sizeof(float));
    transpose_glm5next_state(engine.glm5next_recur_state[layer].data(),
                             state.data(), g.hd, g.H);
    ggml_backend_tensor_get(g.kda_hist_storage[i],
                            engine.glm5next_conv_history[layer].data(), 0,
                            hist_n * sizeof(float));
  }

  const int valid_rows =
      std::min(g.decode_last_pos + 1, std::max(0, engine.max_t));
  const size_t kv_n = (size_t)valid_rows * g.kv_lora;
  std::vector<ggml_fp16_t> half_cache(kv_n);
  std::vector<float> cache(kv_n);
  for (size_t i = 0; i < g.dsa_layers.size(); ++i) {
    const int layer = g.dsa_layers[i];
    if (layer < 0 || layer >= (int)engine.Lc.size() || !engine.Lc[layer] ||
        i >= g.dsa_kv_storage.size())
      return false;
    if (kv_n == 0)
      continue;
    ggml_backend_tensor_get(g.dsa_kv_storage[i], half_cache.data(), 0,
                            kv_n * sizeof(ggml_fp16_t));
    for (size_t j = 0; j < kv_n; ++j)
      cache[j] = ggml_fp16_to_fp32(half_cache[j]);
    std::memcpy(engine.Lc[layer], cache.data(), kv_n * sizeof(float));
  }
  return true;
}

bool glm5next_full_graph_decode_prepare(StreamingEngine &engine, int pos_base,
                                        const int *) {
  if (!env_enabled() || engine.mtp_enabled || pos_base < 0 ||
      engine.family_ops.family != ModelFamily::GLM5NEXT || engine.max_t <= 0 ||
      pos_base >= engine.max_t)
    return false;
  if (g_glm5next_decode_disabled_engine == &engine)
    return false;
  const auto &c = engine.model_file.architecture_config;
  const int64_t rounded_capacity =
      ((int64_t)engine.max_t + 255) & ~((int64_t)255);
  if (rounded_capacity <= 0 || rounded_capacity > INT_MAX)
    return false;
  const int capacity = static_cast<int>(rounded_capacity);

  if (g_graph && g_graph->ready && g_graph->engine == &engine &&
      g_graph->decode_mode && g_graph->kv_capacity >= pos_base + 1 &&
      g_graph->decode_last_pos + 1 == pos_base)
    return true;

  if (g_graph && g_graph->ready && g_graph->decode_mode &&
      g_graph->engine == &engine &&
      !sync_glm5next_decode_state_to_host(*g_graph)) {
    g_glm5next_decode_disabled_engine = &engine;
    return false;
  }

  auto g = std::make_unique<FullGraph>();
  g->engine = &engine;
  g->topology = g_glm5next_vram_cache;
  g->device = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  g->S = 1;
  g->decode_mode = true;
  g->kv_capacity = capacity;
  g->decode_last_pos = pos_base - 1;
  g->D = c.hidden;
  g->HC = std::max(1, c.hc_mult);
  g->n_layers = c.n_layers;
  g->H = c.n_heads;
  g->hd = c.kda_head_dim;
  g->d_inner = g->H * g->hd;
  g->d_conv = c.ssm_conv_kernel;
  g->qk_nope = c.qk_nope;
  g->kv_lora = c.kv_lora;
  g->indexer_heads = c.glm5next_indexer_head_count;
  g->indexer_key_length = c.glm5next_indexer_key_length;
  g->indexer_top_k = c.glm5next_indexer_top_k;
  g->indexer_kpool = c.glm5next_indexer_kpool;
  g->indexer_select_tail =
      engine.model_file.get_config_int("attention.indexer.kpool_select_tail", 0) != 0;
  g->v_head = c.v_head;
  g->first_dense = c.first_dense;
  g->device_expert_layers = std::min(
      glm5next_full_graph_device_expert_layers(),
      std::max(0, c.n_layers - c.first_dense));
  if (configure_partial_experts(*g))
    g->device_expert_layers = 0;

  // Sparse decode requires prompt-computed indexer keys and complete KPool
  // summaries. Carry those across the prompt->decode graph replacement only
  // when the prompt graph actually produced a matching seed for every owner.
  std::vector<Glm5nextIndexerSeed> indexer_seeds;
  int expected_indexer_owners = 0;
  if (g->indexer_kpool > 1 && g->indexer_top_k > 0 &&
      g->indexer_top_k % g->indexer_kpool == 0 &&
      g_graph && g_graph->ready && !g_graph->decode_mode &&
      g_graph->engine == &engine && g_graph->S == pos_base &&
      g_graph->indexer_key_length == g->indexer_key_length &&
      g_graph->indexer_kpool == g->indexer_kpool) {
    for (int l = 0; l < g->n_layers; ++l) {
      if (!glm5next_is_kda_layer(c, l) && glm5next_indexer_owner(*g, l))
        ++expected_indexer_owners;
    }
    bool seeds_match = expected_indexer_owners > 0 &&
                       (int)g_graph->glm5next_indexer_seeds.size() ==
                           expected_indexer_owners;
    const int expected_pools = pos_base / g->indexer_kpool;
    for (const Glm5nextIndexerSeed &seed : g_graph->glm5next_indexer_seeds) {
      seeds_match = seeds_match && seed.token_count == pos_base &&
                    seed.pool_count == expected_pools &&
                    seed.key_gate.size() ==
                        (size_t)2 * g->indexer_key_length * pos_base &&
                    seed.pooled_keys.size() ==
                        (size_t)g->indexer_key_length * expected_pools;
    }
    if (seeds_match) {
      g->sparse_decode = true;
      indexer_seeds = std::move(g_graph->glm5next_indexer_seeds);
      LOG_INFO("[GGML-GLM5NEXT-FULL-DECODE] enabling sparse KPool decode: "
               "owners=%d top_k=%d kpool=%d prompt=%d\n",
               expected_indexer_owners, g->indexer_top_k,
               g->indexer_kpool, pos_base);
    } else {
      LOG_WARN("[GGML-GLM5NEXT-FULL-DECODE] prompt indexer seeds missing or "
               "incompatible; keeping dense DSA decode\n");
    }
  }

  // Preserve the large, reusable DDR expert source while replacing the
  // prompt-shaped graph with the fixed S=1 decode topology.
  bool source_transferred = false;
  if (g_graph && g_graph->has_expert_source() &&
      source_configuration_matches(*g_graph, *g)) {
    if (!g_graph->move_expert_source_to(*g))
      return false;
    source_transferred = true;
  } else if (g_glm5next_source &&
             source_configuration_matches(*g_glm5next_source, *g)) {
    if (!g_glm5next_source->move_expert_source_to(*g))
      return false;
    source_transferred = true;
  }
  g_graph.reset();
  if (g_glm5next_source && !g_glm5next_source->has_expert_source())
    g_glm5next_source.reset();

  auto retain_source_on_failure = [&]() {
    if (!g->has_expert_source())
      return;
    if (g_glm5next_source && g_glm5next_source->has_expert_source())
      g_glm5next_source.reset();
    if (!g_glm5next_source)
      g_glm5next_source = std::make_unique<FullGraph>();
    if (!g->move_expert_source_to(*g_glm5next_source))
      LOG_ERROR("[GGML-GLM5NEXT-FULL-DECODE] failed to retain expert "
                "source after graph decline\n");
  };
  auto decline_decode_graph = [&]() {
    // Design policy: a GLM5NEXT request is served by the full graph or it fails. MOESTREAM_GLM5NEXT_GRAPH_STRICT=1
    // turns the legacy-decode fallback into a hard error (default stays permissive until prefill is graph-only too).
    if (const char *strict = std::getenv("MOESTREAM_GLM5NEXT_GRAPH_STRICT");
        strict && strict[0] == '1') {
      LOG_ERROR("[GGML-GLM5NEXT-FULL-DECODE] decode graph declined and "
                "MOESTREAM_GLM5NEXT_GRAPH_STRICT=1: failing the request "
                "instead of falling back to the layered path\n");
      std::fflush(nullptr);
      std::abort();
    }
    retain_source_on_failure();
    g_glm5next_decode_disabled_engine = &engine;
    return false;
  };

  g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(g->device));
  g->cpu = ggml_backend_cpu_init();
  if (g->cpu) {
    const char *e = std::getenv("MOESTREAM_GLM5NEXT_CPU_THREADS");
    ggml_backend_cpu_set_n_threads(g->cpu, e ? std::max(1, std::atoi(e)) : 6);
  }
  if (!g->cuda || !g->cpu)
    return decline_decode_graph();
  if (!source_transferred && g_glm5next_source &&
      source_configuration_matches(*g_glm5next_source, *g)) {
    if (!g_glm5next_source->move_expert_source_to(*g))
      return decline_decode_graph();
    source_transferred = true;
  }
  if (!source_transferred &&
      !prepare_expert_source_only(*g, "GGML-GLM5NEXT-FULL-DECODE"))
    return decline_decode_graph();
  if (!init_expert_arena(*g) || !init_dense_arena(*g) ||
      !init_decode_state_arena(*g)) {
    return decline_decode_graph();
  }
  if (const char *e = std::getenv("MOESTREAM_GLM5NEXT_GPU1_LAYERS")) {
    int a = -1, b = -2;
    if (std::sscanf(e, "%d-%d", &a, &b) == 2 && a >= 0 && b >= a) {
      g->gpu1 = static_cast<ggml_backend_t>(ggml_shared_backend(1));
      if (g->gpu1) {
        g->gpu1_first = a;
        g->gpu1_last = b;
        LOG_INFO("[GGML-GLM5NEXT-FULL-DECODE] GPU1 expert layers %d-%d\n", a, b);
      }
    }
  }
  if (!g->gpu1) {
    // GPU1 hot experts (MOESTREAM_GLM5NEXT_GPU1_HOT_GIB) need the second backend even without whole GPU1 layers.
    for (const auto &pl : g->partial_experts)
      if (!pl.hot1.empty()) {
        g->gpu1 = static_cast<ggml_backend_t>(ggml_shared_backend(1));
        break;
      }
  }
  ggml_backend_t backends[] = {g->cuda, g->gpu1, g->cpu};
  if (!g->gpu1)
    backends[1] = g->cpu;
  g->sched = ggml_backend_sched_new(backends, nullptr, g->gpu1 ? 3 : 2,
                                    kGraphSize, false, true);
  if (!g->sched || !build_graph(*g)) {
    LOG_WARN("[GGML-GLM5NEXT-FULL-DECODE] graph construction declined; "
             "using legacy decode\n");
    return decline_decode_graph();
  }

  const size_t state_n = (size_t)g->hd * g->hd * g->H;
  const size_t hist_n = (size_t)(g->d_conv - 1) * 3 * g->d_inner;
  for (size_t i = 0; i < g->kda_layers.size(); ++i) {
    const int layer = g->kda_layers[i];
    if (layer < 0 || layer >= (int)engine.glm5next_recur_state.size() ||
        layer >= (int)engine.glm5next_conv_history.size() ||
        engine.glm5next_recur_state[layer].size() != state_n ||
        engine.glm5next_conv_history[layer].size() != hist_n ||
        i >= g->kda_state_storage.size() || i >= g->kda_hist_storage.size()) {
      return decline_decode_graph();
    }
    std::vector<float> native_state(state_n);
    transpose_glm5next_state(native_state.data(),
                             engine.glm5next_recur_state[layer].data(), g->hd,
                             g->H);
    ggml_backend_tensor_set(g->kda_state_storage[i], native_state.data(), 0,
                            state_n * sizeof(float));
    ggml_backend_tensor_set(g->kda_hist_storage[i],
                            engine.glm5next_conv_history[layer].data(), 0,
                            hist_n * sizeof(float));
  }

  for (size_t i = 0; i < g->dsa_layers.size(); ++i) {
    const int layer = g->dsa_layers[i];
    if (layer < 0 || layer >= (int)engine.Lc.size() || !engine.Lc[layer] ||
        i >= g->dsa_kv_storage.size()) {
      return decline_decode_graph();
    }
    ggml_tensor *cache = g->dsa_kv_storage[i];
    if (cudaMemset(cache->data, 0, ggml_nbytes(cache)) != cudaSuccess) {
      return decline_decode_graph();
    }
    const size_t seed_rows = (size_t)pos_base * g->kv_lora;
    if (seed_rows > 0) {
      std::vector<ggml_fp16_t> half_cache(seed_rows);
      const float *src = engine.Lc[layer];
      for (size_t j = 0; j < seed_rows; ++j)
        half_cache[j] = ggml_fp32_to_fp16(src[j]);
      ggml_backend_tensor_set(cache, half_cache.data(), 0,
                              seed_rows * sizeof(ggml_fp16_t));
    }
  }

  for (Glm5nextIndexerLayer &state : g->glm5next_indexer_layers) {
    if (!state.key_gate_cache || !state.pooled_cache ||
        cudaMemset(state.key_gate_cache->data, 0,
                   ggml_nbytes(state.key_gate_cache)) != cudaSuccess ||
        cudaMemset(state.pooled_cache->data, 0,
                   ggml_nbytes(state.pooled_cache)) != cudaSuccess)
      return decline_decode_graph();
    const auto seed = std::find_if(
        indexer_seeds.begin(), indexer_seeds.end(),
        [&state](const Glm5nextIndexerSeed &candidate) {
          return candidate.layer == state.layer;
        });
    if (seed == indexer_seeds.end() ||
        seed->key_gate.size() !=
            (size_t)2 * g->indexer_key_length * pos_base ||
        seed->pooled_keys.size() !=
            (size_t)g->indexer_key_length * (pos_base / g->indexer_kpool))
      return decline_decode_graph();
    if (!seed->key_gate.empty())
      ggml_backend_tensor_set(state.key_gate_cache, seed->key_gate.data(), 0,
                              seed->key_gate.size() * sizeof(float));
    if (!seed->pooled_keys.empty())
      ggml_backend_tensor_set(state.pooled_cache, seed->pooled_keys.data(), 0,
                              seed->pooled_keys.size() * sizeof(float));
  }

  g->ready = true;
  g_graph = std::move(g);
  LOG_INFO("[GGML-GLM5NEXT-FULL-DECODE] reusable graph prepared: "
           "position=%d capacity=%d KDA=%zu DSA=%zu\n",
           pos_base, capacity, g_graph->kda_layers.size(),
           g_graph->dsa_layers.size());
  return true;
}

bool glm5next_full_graph_decode_run(StreamingEngine &engine,
                                    float *hidden_states, int pos_base,
                                    const int *) {
  if (!env_enabled() || !hidden_states || pos_base < 0 || !g_graph ||
      !g_graph->ready || !g_graph->decode_mode || g_graph->engine != &engine ||
      g_graph->S != 1 || pos_base >= g_graph->kv_capacity ||
      g_graph->decode_last_pos + 1 != pos_base)
    return false;
  FullGraph &g = *g_graph;
  struct NanProbe {
    int reported = 0;
    int seen = 0;
  };
  static NanProbe probe;
  const bool nan_probe_on =
      std::getenv("MOESTREAM_GLM5NEXT_DEBUG_NAN_NODES") != nullptr &&
      probe.reported == 0;
  if (nan_probe_on) {
    probe.seen = 0;
    ggml_backend_sched_set_eval_callback(
        g.sched,
        [](ggml_tensor *t, bool ask, void *ud) -> bool {
          auto *p = static_cast<NanProbe *>(ud);
          if (ask)
            return p->reported < 6 && (t->type == GGML_TYPE_F32 ||
                                       t->type == GGML_TYPE_F16);
          const int idx = p->seen++;
          if (p->reported >= 6)
            return true;
          const size_t n = ggml_nelements(t);
          std::vector<uint8_t> raw(ggml_nbytes(t));
          ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
          size_t bad = 0;
          if (t->type == GGML_TYPE_F32) {
            for (size_t i = 0; i < n; ++i)
              if (!std::isfinite(reinterpret_cast<const float *>(raw.data())[i]))
                ++bad;
          } else {
            for (size_t i = 0; i < n; ++i)
              if (!std::isfinite(ggml_fp16_to_fp32(
                      reinterpret_cast<const ggml_fp16_t *>(raw.data())[i])))
                ++bad;
          }
          if (bad) {
            ++p->reported;
            if (t->op == GGML_OP_MUL_MAT_ID && t->src[2] &&
                t->src[2]->type == GGML_TYPE_I32 && t->type == GGML_TYPE_F32) {
              const size_t nid = ggml_nelements(t->src[2]);
              std::vector<int32_t> ids(nid);
              ggml_backend_tensor_get(t->src[2], ids.data(), 0,
                                      nid * sizeof(int32_t));
              std::string line;
              const size_t row_len = (size_t)t->ne[0];
              for (size_t r = 0; r < (size_t)t->ne[1] && r < nid; ++r) {
                size_t rb = 0;
                for (size_t i = 0; i < row_len; ++i)
                  if (!std::isfinite(reinterpret_cast<const float *>(
                          raw.data())[r * row_len + i]))
                    ++rb;
                line += " [row" + std::to_string(r) + " id=" +
                        std::to_string(ids[r]) + " bad=" + std::to_string(rb) +
                        "]";
              }
              LOG_WARN("[NAN-NODE-IDS] w_ne2=%lld w_type=%s%s\n",
                       (long long)t->src[0]->ne[2],
                       ggml_type_name(t->src[0]->type), line.c_str());
              const ggml_tensor *w = t->src[0];
              const size_t blk_bytes = ggml_type_size(w->type);
              const size_t blk_elems = (size_t)ggml_blck_size(w->type);
              const size_t slot_blocks =
                  (size_t)w->ne[0] * (size_t)w->ne[1] / blk_elems;
              for (size_t r = 0; r < 8 && r < nid; ++r) {
                if (ids[r] < 0 || ids[r] >= w->ne[2])
                  continue;
                std::vector<uint8_t> wb(slot_blocks * blk_bytes);
                ggml_backend_tensor_get(w, wb.data(),
                                        (size_t)ids[r] * w->nb[2], wb.size());
                size_t bad_d = 0;
                for (size_t b = 0; b < slot_blocks; ++b) {
                  const ggml_fp16_t d = *reinterpret_cast<const ggml_fp16_t *>(
                      wb.data() + b * blk_bytes);
                  if (!std::isfinite(ggml_fp16_to_fp32(d)))
                    ++bad_d;
                }
                LOG_WARN("[NAN-NODE-W] row=%zu slot=%d blocks=%zu bad_d=%zu "
                         "nb2=%zu buft=%s\n",
                         r, ids[r], slot_blocks, bad_d, (size_t)w->nb[2],
                         w->buffer ? ggml_backend_buffer_name(w->buffer) : "-");
                if (r == 0 && t->src[1] && t->src[1]->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(t->src[1]) &&
                    t->src[1]->ne[0] == w->ne[0]) {
                  const auto *tr = ggml_get_type_traits(w->type);
                  std::vector<float> x((size_t)w->ne[0]);
                  ggml_backend_tensor_get(t->src[1], x.data(), 0,
                                          x.size() * sizeof(float));
                  double xmax = 0;
                  size_t xbad = 0;
                  for (float v : x) {
                    if (!std::isfinite(v)) ++xbad;
                    else xmax = std::max(xmax, (double)std::fabs(v));
                  }
                  std::vector<float> wrow((size_t)w->ne[0]);
                  const size_t row_bytes = ggml_row_size(w->type, w->ne[0]);
                  int shown = 0, ref_finite_of_bad = 0, bad_rows = 0;
                  for (size_t j = 0; j < (size_t)t->ne[0]; ++j) {
                    const float got =
                        reinterpret_cast<const float *>(raw.data())[j];
                    if (std::isfinite(got)) continue;
                    ++bad_rows;
                    tr->to_float(wb.data() + j * row_bytes, wrow.data(),
                                 w->ne[0]);
                    double ref = 0;
                    for (size_t i = 0; i < x.size(); ++i)
                      ref += (double)wrow[i] * x[i];
                    if (std::isfinite(ref)) ++ref_finite_of_bad;
                    if (shown < 3) {
                      ++shown;
                      LOG_WARN("[NAN-NODE-REF] out_idx=%zu got=%g cpu_ref=%g\n",
                               j, got, ref);
                    }
                  }
                  LOG_WARN("[NAN-NODE-REF] src1 nonfinite=%zu absmax=%g "
                           "bad_rows=%d cpu_ref_finite=%d\n",
                           xbad, xmax, bad_rows, ref_finite_of_bad);
                }
              }
            }
            LOG_WARN("[NAN-NODE] #%d op=%s name=%s bad=%zu/%zu ne=%lld,%lld,%lld,%lld "
                     "src0=%s src1=%s\n",
                     idx, ggml_op_name(t->op), t->name, bad, n,
                     (long long)t->ne[0], (long long)t->ne[1],
                     (long long)t->ne[2], (long long)t->ne[3],
                     t->src[0] ? t->src[0]->name : "-",
                     t->src[1] ? t->src[1]->name : "-");
          }
          return true;
        },
        &probe);
  }
  if (g.dynamic_expert_cache && !publish_dynamic_uploads(g)) {
    if (!sync_glm5next_decode_state_to_host(g))
      LOG_ERROR("[GGML-GLM5NEXT-FULL-DECODE] cache-map handoff failed\n");
    g_glm5next_decode_disabled_engine = &engine;
    return false;
  }
  if (engine.hc_stream_buf.size() < (size_t)g.D * g.HC ||
      !g.input || !g.attn_mask || !g.decode_indices)
    return false;

  std::memcpy(g.input->data, engine.hc_stream_buf.data(),
              (size_t)g.D * g.HC * sizeof(float));
  *static_cast<int32_t *>(g.decode_indices->data) = pos_base;
  auto *mask = static_cast<ggml_fp16_t *>(g.attn_mask->data);
  for (int k = 0; k < g.kv_capacity; ++k)
    mask[k] = ggml_fp32_to_fp16(k <= pos_base ? 0.0f : -INFINITY);

  if (g.sparse_decode) {
    const int kpool = g.indexer_kpool;
    const int pool_count = g.kv_capacity / kpool;
    const int complete_pools = (pos_base + 1) / kpool;
    const int tail_count = (pos_base + 1) % kpool;
    const int pool_start = (pos_base / kpool) * kpool;
    auto *pool_mask = static_cast<float *>(g.indexer_pool_mask->data);
    for (int pool = 0; pool < pool_count; ++pool)
      pool_mask[pool] = pool < complete_pools ? 0.0f : -INFINITY;
    auto *update_rows =
        static_cast<int32_t *>(g.indexer_pool_update_rows->data);
    for (int i = 0; i < kpool; ++i)
      update_rows[i] = std::min(pool_start + i, pos_base);
    *static_cast<int32_t *>(g.indexer_pool_update_index->data) =
        pos_base / kpool;
    if (g.indexer_select_tail) {
      auto *tail_indices =
          static_cast<int32_t *>(g.indexer_tail_indices->data);
      auto *tail_mask = static_cast<float *>(g.indexer_tail_mask->data);
      const int tail_start = complete_pools * kpool;
      for (int i = 0; i < kpool - 1; ++i) {
        const bool valid = i < tail_count;
        tail_indices[i] = valid ? tail_start + i : 0;
        tail_mask[i] = valid ? 0.0f : -INFINITY;
      }
    }
  }

  const long avail_mb = MemoryGuard::read_available_mb();
  constexpr long kMinComputeHeadroomMb = 10240;
  if (avail_mb >= 0 && avail_mb < kMinComputeHeadroomMb) {
    LOG_ERROR("[GGML-GLM5NEXT-FULL-DECODE] refusing graph compute: "
              "MemAvailable=%ld MiB < %ld MiB; falling back to legacy decode\n",
              avail_mb, kMinComputeHeadroomMb);
    if (!sync_glm5next_decode_state_to_host(g))
      LOG_ERROR("[GGML-GLM5NEXT-FULL-DECODE] host state handoff failed\n");
    g_glm5next_decode_disabled_engine = &engine;
    return false;
  }

  if (!hold_graph_vram_experts(g)) {
    g_glm5next_decode_disabled_engine = &engine;
    return false;
  }
  struct LayerTimer {
    const FullGraph *g = nullptr;
    std::chrono::steady_clock::time_point last;
    std::vector<double> attn_ms, ffn_ms;
    int tokens = 0;
  };
  static LayerTimer lt;
  const bool layer_timing_on = layer_timing_env() && !nan_probe_on;
  if (layer_timing_on) {
    lt.g = &g;
    if (lt.attn_ms.size() != (size_t)g.n_layers) {
      lt.attn_ms.assign((size_t)g.n_layers, 0.0);
      lt.ffn_ms.assign((size_t)g.n_layers, 0.0);
      lt.tokens = 0;
    }
    ggml_backend_sched_set_eval_callback(
        g.sched,
        [](ggml_tensor *t, bool ask, void *ud) -> bool {
          auto *l = static_cast<LayerTimer *>(ud);
          auto it = l->g->layer_marks.find(t);
          if (it == l->g->layer_marks.end())
            return false;
          if (ask)
            return true;
          const auto now = std::chrono::steady_clock::now();
          const double ms =
              std::chrono::duration<double, std::milli>(now - l->last).count();
          l->last = now;
          (it->second.second == 0 ? l->attn_ms : l->ffn_ms)[(size_t)it->second.first] += ms;
          return true;
        },
        &lt);
    lt.last = std::chrono::steady_clock::now();
  }
  struct OpProf {
    std::chrono::steady_clock::time_point last;
    std::map<std::string, std::pair<double, int>> agg;
    int tokens = 0;
    ggml_backend_sched_t sched = nullptr;
  };
  static OpProf op;
  static const bool op_prof_env =
      std::getenv("MOESTREAM_GLM5NEXT_OP_PROFILE") != nullptr;
  const bool op_prof_on = op_prof_env && !layer_timing_on && !nan_probe_on &&
                          op.tokens < 30;
  if (op_prof_on) {
    ggml_backend_sched_set_eval_callback(
        g.sched,
        [](ggml_tensor *t, bool ask, void *ud) -> bool {
          if (ask)
            return true;
          auto *o = static_cast<OpProf *>(ud);
          const auto now = std::chrono::steady_clock::now();
          const double ms =
              std::chrono::duration<double, std::milli>(now - o->last).count();
          o->last = now;
          std::string key = ggml_op_name(t->op);
          if (t->op == GGML_OP_MUL_MAT_ID || t->op == GGML_OP_MUL_MAT)
            key += std::string("(") + ggml_type_name(t->src[0]->type) + ")" +
                   (t->src[0]->buffer
                        ? ggml_backend_buffer_name(t->src[0]->buffer)
                        : "nobuf") +
                   "/" +
                   ggml_backend_name(
                       ggml_backend_sched_get_tensor_backend(o->sched, t));
          auto &e = o->agg[key];
          e.first += ms;
          ++e.second;
          return true;
        },
        &op);
    op.sched = g.sched;
    op.last = std::chrono::steady_clock::now();
  }
  const auto start = std::chrono::steady_clock::now();
  if (ggml_backend_sched_graph_compute_async(g.sched, g.graph) !=
      GGML_STATUS_SUCCESS) {
    ggml_backend_sched_synchronize(g.sched);
    release_graph_vram_experts(g);
    g.ready = false;
    g_glm5next_decode_disabled_engine = &engine;
    return false;
  }
  ggml_backend_sched_synchronize(g.sched);
  release_graph_vram_experts(g);
  if (op_prof_on) {
    ggml_backend_sched_set_eval_callback(g.sched, nullptr, nullptr);
    if (++op.tokens == 30) {
      std::vector<std::pair<std::string, std::pair<double, int>>> v(
          op.agg.begin(), op.agg.end());
      std::sort(v.begin(), v.end(), [](auto &a, auto &b) {
        return a.second.first > b.second.first;
      });
      for (size_t i = 0; i < v.size() && i < 25; ++i)
        LOG_WARN("[GLM5NEXT-OP-PROFILE] %-28s total/token=%.2fms nodes/token=%.1f avg=%.1fus\n",
                 v[i].first.c_str(), v[i].second.first / 30,
                 v[i].second.second / 30.0,
                 1000.0 * v[i].second.first / v[i].second.second);
    }
  }
  if (nan_probe_on) {
    ggml_backend_sched_set_eval_callback(g.sched, nullptr, nullptr);
    probe.reported = std::max(probe.reported, 1);
  }
  if (layer_timing_on) {
    ggml_backend_sched_set_eval_callback(g.sched, nullptr, nullptr);
    if (++lt.tokens % 100 == 0) {
      double ta = 0, tf = 0;
      std::string line;
      for (int l = 0; l < g.n_layers; ++l) {
        ta += lt.attn_ms[(size_t)l];
        tf += lt.ffn_ms[(size_t)l];
        char b[64];
        std::snprintf(b, sizeof(b), " L%d=%.2f/%.2f", l,
                      lt.attn_ms[(size_t)l] / lt.tokens,
                      lt.ffn_ms[(size_t)l] / lt.tokens);
        line += b;
      }
      LOG_WARN("[GLM5NEXT-LAYER-TIMING] tokens=%d mean per token: attn=%.1fms "
               "ffn=%.1fms (marked-node sync inflates absolute time)\n"
               "[GLM5NEXT-LAYER-TIMING] attn/ffn ms per layer:%s\n",
               lt.tokens, ta / lt.tokens, tf / lt.tokens, line.c_str());
    }
  }
  const auto synced = std::chrono::steady_clock::now();
  const double compute_ms = std::chrono::duration<double, std::milli>(
                                synced - start)
                                .count();
  const auto cache_update_start = std::chrono::steady_clock::now();
  if (route_stats_enabled())
    update_partial_routes(g, pos_base, compute_ms);
  const auto cache_update_done = std::chrono::steady_clock::now();
  if (!g.ready) {
    if (!sync_glm5next_decode_state_to_host(g))
      LOG_ERROR("[GGML-GLM5NEXT-FULL-DECODE] cache update state handoff "
                "failed\n");
    g_glm5next_decode_disabled_engine = &engine;
    return false;
  }
  ggml_backend_tensor_get(g.output, hidden_states, 0,
                          (size_t)g.D * sizeof(float));
  if (std::getenv("MOESTREAM_GLM5NEXT_DEBUG_DECODE_OUT")) {
    size_t bad = 0;
    double sum = 0.0, mx = 0.0;
    for (int i = 0; i < g.D; ++i) {
      const float v = hidden_states[i];
      if (!std::isfinite(v)) { ++bad; continue; }
      sum += v; mx = std::max(mx, (double)std::fabs(v));
    }
    LOG_WARN("[GLM5NEXT-DECODE-OUT] pos=%d nonfinite=%zu/%d mean=%g absmax=%g "
             "h[0..3]=%g %g %g %g\n",
             pos_base, bad, g.D, sum / g.D, mx, hidden_states[0],
             hidden_states[1], hidden_states[2], hidden_states[3]);
  }
  g.decode_last_pos = pos_base;
  engine.glm5next_full_graph_output_ready = true;
  if (g.logits && engine.qwen4exp_full_graph_logits_target &&
      engine.qwen4exp_full_graph_logits_target_size == (int)g.logits->ne[0]) {
    ggml_backend_tensor_get(g.logits, engine.qwen4exp_full_graph_logits_target,
                            0, ggml_nbytes(g.logits));
    engine.qwen4exp_full_graph_logits_ready = true;
  }
  const auto done = std::chrono::steady_clock::now();
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  LOG_INFO("[GGML-GLM5NEXT-FULL-DECODE-TIME] pos=%d sync=%.3f ms "
           "cache_update=%.3f ms output_get=%.3f ms total=%.3f ms\n",
           pos_base, ms(start, synced), ms(cache_update_start, cache_update_done),
           ms(cache_update_done, done), ms(start, done));
  return true;
}

bool glm5next_full_graph_run(StreamingEngine &engine, float *hidden_states,
                             int S, int pos_base, const int *) {
  if (!env_enabled() || !hidden_states || S <= 1 || pos_base != 0 || !g_graph ||
      !g_graph->ready || g_graph->engine != &engine || g_graph->S != S)
    return false;
  FullGraph &g = *g_graph;
  const auto &c = engine.model_file.architecture_config;
  const size_t state_n = (size_t)g.hd * g.hd * g.H;
  const size_t hist_n = (size_t)(g.d_conv - 1) * 3 * g.d_inner;

  for (size_t i = 0; i < g.kda_layers.size(); ++i) {
    const int l = g.kda_layers[i];
    if (l < 0 || l >= (int)engine.glm5next_recur_state.size() ||
        l >= (int)engine.glm5next_conv_history.size() ||
        engine.glm5next_recur_state[l].size() != state_n ||
        engine.glm5next_conv_history[l].size() != hist_n)
      return false;
    std::fill(g.kda_state_inputs[i], g.kda_state_inputs[i] + state_n, 0.0f);
    std::fill(g.kda_hist_inputs[i], g.kda_hist_inputs[i] + hist_n, 0.0f);
  }
  if (engine.hc_stream_buf.size() < (size_t)g.D * g.HC * S ||
      g.host_inputs.empty())
    return false;
  std::memcpy(g.host_inputs[0], engine.hc_stream_buf.data(),
              (size_t)g.D * g.HC * S * sizeof(float));

  // Real-hardware incident (2026-09-09): a 480-token prompt crashed the
  // whole machine (hard reboot required, no surviving kernel/OOM log --
  // this box has no persistent journald storage) with the last surviving
  // application-log line being "graph prepared for S=...", i.e. somewhere
  // in the gap between prepare() returning and this compute call. Every
  // OTHER host allocation on this path (arena prep, expert source, graph
  // context) goes through MemoryGuard::try_reserve_host_bytes first; this
  // compute call did not, and it is the last real memory-relevant, cheaply
  // avoidable step before the model correctness this session verified
  // could otherwise be undone by a repeat crash. Re-check REAL, fresh
  // MemAvailable (not the background watchdog's own possibly-500ms-stale
  // allocation_blocked() flag) immediately before committing to it -- a
  // large gap can open between the admission checks earlier in prepare()
  // (preload alone took minutes) and this moment. Not a fix for whatever
  // the real mechanism was (unconfirmed -- no kernel evidence survived),
  // just a cheap, additional, defense-in-depth backstop.
  {
    const long avail_mb = MemoryGuard::read_available_mb();
    constexpr long kMinComputeHeadroomMb =
        10240; // 10 GiB, per explicit user instruction after the 2026-09-09
               // crash
    if (avail_mb >= 0 && avail_mb < kMinComputeHeadroomMb) {
      LOG_ERROR(
          "[GGML-GLM5NEXT-FULL] refusing graph compute: MemAvailable=%ld MiB "
          "< %ld MiB safety floor -- falling back to the legacy path rather "
          "than risk the host-memory incident this check was added for "
          "(2026-09-09, see this call site's own comment)\n",
          avail_mb, kMinComputeHeadroomMb);
      return false;
    }
  }

  if (!hold_graph_vram_experts(g))
    return false;
  const auto start = std::chrono::steady_clock::now();
  if (ggml_backend_sched_graph_compute_async(g.sched, g.graph) !=
      GGML_STATUS_SUCCESS) {
    ggml_backend_sched_synchronize(g.sched);
    release_graph_vram_experts(g);
    return false;
  }
  ggml_backend_sched_synchronize(g.sched);
  release_graph_vram_experts(g);
  const auto synced = std::chrono::steady_clock::now();
  ggml_backend_tensor_get(g.output, hidden_states, 0,
                          (size_t)g.D * S * sizeof(float));

  for (size_t i = 0; i < g.kda_layers.size(); ++i) {
    const int l = g.kda_layers[i];
    ggml_backend_tensor_get(g.kda_state_out[i],
                            engine.glm5next_recur_state[l].data(), 0,
                            state_n * sizeof(float));
    ggml_backend_tensor_get(g.kda_hist_out[i],
                            engine.glm5next_conv_history[l].data(), 0,
                            hist_n * sizeof(float));
  }
  const size_t kv_n = (size_t)g.S * g.kv_lora;
  std::vector<float> kv(kv_n);
  for (size_t i = 0; i < g.dsa_layers.size(); ++i) {
    const int l = g.dsa_layers[i];
    if (l < 0 || l >= (int)engine.Lc.size() || !engine.Lc[l])
      return false;
    ggml_backend_tensor_get(g.dsa_kv_out[i], kv.data(), 0,
                            kv_n * sizeof(float));
    std::memcpy(engine.Lc[l], kv.data(), kv_n * sizeof(float));
  }
  g.glm5next_indexer_seeds.clear();
  g.glm5next_indexer_seeds.reserve(g.glm5next_indexer_layers.size());
  bool indexer_seeds_ok = true;
  for (const Glm5nextIndexerLayer &state : g.glm5next_indexer_layers) {
    if (!state.key_gate_out) {
      indexer_seeds_ok = false;
      break;
    }
    Glm5nextIndexerSeed seed;
    seed.layer = state.layer;
    seed.token_count = g.S;
    seed.pool_count = g.indexer_kpool > 1 ? g.S / g.indexer_kpool : 0;
    seed.key_gate.resize((size_t)2 * g.indexer_key_length * g.S);
    seed.pooled_keys.resize(
        (size_t)g.indexer_key_length * seed.pool_count);
    ggml_backend_tensor_get(state.key_gate_out, seed.key_gate.data(), 0,
                            seed.key_gate.size() * sizeof(float));
    if (seed.pool_count > 0) {
      if (!state.pooled_out) {
        indexer_seeds_ok = false;
        break;
      }
      ggml_backend_tensor_get(state.pooled_out, seed.pooled_keys.data(), 0,
                              seed.pooled_keys.size() * sizeof(float));
    }
    g.glm5next_indexer_seeds.push_back(std::move(seed));
  }
  if (!indexer_seeds_ok) {
    g.glm5next_indexer_seeds.clear();
    LOG_WARN("[GGML-GLM5NEXT-FULL] indexer seed capture unavailable; "
             "decode will retain dense attention\n");
  }
  const auto done = std::chrono::steady_clock::now();
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  LOG_INFO("[GGML-GLM5NEXT-FULL-TIME] sync=%.3f ms state_get=%.3f ms "
           "total=%.3f ms\n",
           ms(start, synced), ms(synced, done), ms(start, done));
  if (!g.glm5next_indexer_seeds.empty())
    LOG_INFO("[GGML-GLM5NEXT-FULL] captured KPool prompt seeds: layers=%zu "
             "tokens=%d pools=%d\n",
             g.glm5next_indexer_seeds.size(), g.S,
             g.S / std::max(1, g.indexer_kpool));
  engine.glm5next_full_graph_output_ready = true;
  (void)c;
  return true;
}

bool deepseek_v4_full_graph_enabled() { return dsv4_env_enabled(); }

int deepseek_v4_full_graph_device_expert_layers() {
  return dsv4_env_enabled() ? dsv4_env_device_expert_layers() : 0;
}

void deepseek_v4_full_graph_shutdown() { g_dsv4_graph.reset(); }

bool deepseek_v4_full_graph_prepare(StreamingEngine &engine, int S,
                                    int pos_base, const int *token_ids) {
  if (!dsv4_env_enabled() || S <= 1 || pos_base != 0 ||
      engine.family_ops.family != ModelFamily::DeepSeek_V4)
    return false;
  const auto &c = engine.model_file.architecture_config;
  // The graph deliberately attends every completed CSA block.  That is an
  // exact match while the visible block count stays below the model's
  // Lightning Indexer top-k; beyond that point the legacy path remains the
  // correctness fallback until dynamic in-graph top-k selection is added.
  if (c.dsv4_indexer_top_k > 0 && S > c.dsv4_indexer_top_k * 4) {
    LOG_WARN(
        "[GGML-DSV4-FULL] prompt S=%d exceeds CSA dense-selection limit %d; "
        "using legacy indexed attention\n",
        S, c.dsv4_indexer_top_k * 4);
    return false;
  }
  if (g_dsv4_graph && g_dsv4_graph->ready && g_dsv4_graph->engine == &engine &&
      g_dsv4_graph->S == S)
    return true;

  g_dsv4_graph.reset();
  auto g = std::make_unique<FullGraph>();
  g->engine = &engine;
  g->device = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  g->S = S;
  g->D = c.hidden;
  g->HC = std::max(1, c.hc_mult);
  g->n_layers = c.n_layers;
  g->H = c.n_heads;
  g->qk_nope = c.qk_nope;
  g->qk_rope = c.qk_rope;
  g->o_group_count = c.o_group_count;
  g->o_lora_rank = c.o_lora_rank;
  g->sliding_window = c.sliding_window;
  g->first_dense = c.first_dense;
  g->device_expert_layers =
      std::min(deepseek_v4_full_graph_device_expert_layers(),
               std::max(0, c.n_layers - c.first_dense));
  if (configure_partial_experts(*g))
    g->device_expert_layers = 0;
  g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(g->device));
  g->cpu = ggml_backend_cpu_init();
  if (g->cpu) {
    const char *e = std::getenv("MOESTREAM_GLM5NEXT_CPU_THREADS");
    ggml_backend_cpu_set_n_threads(g->cpu, e ? std::max(1, std::atoi(e)) : 6);
  }
  if (!g->cuda || !g->cpu)
    return false;

  const size_t source = expert_source_bytes(engine, g->device_expert_layers);
  const size_t host_live = graph_host_live_bytes(engine);
  constexpr size_t kSourceSlack = 1ull << 20;
  // 10 GiB, per explicit user instruction after the 2026-09-09 crash.
  constexpr size_t kWatermark = 10ull * 1024ull * 1024ull * 1024ull;
  if (source == 0 || source > SIZE_MAX - kSourceSlack ||
      !MemoryGuard::try_reserve_host_bytes(source + kSourceSlack, kWatermark)) {
    LOG_WARN(
        "[GGML-DSV4-FULL] host watermark refused source=%.2f GiB + existing "
        "host=%.2f GiB + reserve=%.2f GiB; budget=%.2f GB MemAvailable=%ld "
        "MiB\n",
        (double)source / (1024.0 * 1024.0 * 1024.0),
        (double)host_live / (1024.0 * 1024.0 * 1024.0),
        (double)kWatermark / (1024.0 * 1024.0 * 1024.0),
        (double)MemoryGuard::host_budget_bytes() / 1.0e9,
        MemoryGuard::last_available_mb());
    return false;
  }
  g->host_reservation_bytes = source + kSourceSlack;
  LOG_INFO("[GGML-DSV4-FULL] host admission: source=%.2f GiB existing=%.2f GiB "
           "reserve=%.2f GiB budget=%.2f GB\n",
           (double)source / (1024.0 * 1024.0 * 1024.0),
           (double)host_live / (1024.0 * 1024.0 * 1024.0),
           (double)kWatermark / (1024.0 * 1024.0 * 1024.0),
           (double)MemoryGuard::host_budget_bytes() / 1.0e9);
  if (!init_expert_source(*g, source) || !preload_expert_source(*g) ||
      !init_expert_arena(*g) || !init_dense_arena(*g))
    return false;
  ggml_backend_t backends[] = {g->cuda, g->cpu};
  g->sched =
      ggml_backend_sched_new(backends, nullptr, 2, kGraphSize, false, true);
  if (!g->sched || !build_dsv4_graph(*g, token_ids)) {
    LOG_WARN(
        "[GGML-DSV4-FULL] graph construction declined; using legacy path\n");
    return false;
  }
  g_dsv4_graph = std::move(g);
  LOG_INFO("[GGML-DSV4-FULL] graph prepared for S=%d\n", S);
  return true;
}

bool deepseek_v4_full_graph_run(StreamingEngine &engine, float *hidden_states,
                                int S, int pos_base, const int *) {
  if (!dsv4_env_enabled() || !hidden_states || S <= 1 || pos_base != 0 ||
      !g_dsv4_graph || !g_dsv4_graph->ready ||
      g_dsv4_graph->engine != &engine || g_dsv4_graph->S != S)
    return false;
  FullGraph &g = *g_dsv4_graph;
  const auto &c = engine.model_file.architecture_config;
  const int window = g.sliding_window > 0 ? g.sliding_window : engine.max_t;

  // The graph starts a fresh MoEStream sequence.  Clear the state first so a
  // failed/short graph run cannot leave the following legacy decode mixed
  // with stale compressed rows from a prior request.
  for (const auto &lt : g.dsv4_layers) {
    if (lt.layer < 0 || lt.layer >= (int)engine.KVc.size() ||
        !engine.KVc[lt.layer] || lt.layer >= (int)engine.dsv4_compressed.size())
      return false;
    const int ratio = lt.ratio;
    if (ratio != 0 && engine.dsv4_compressed[lt.layer].ratio != ratio)
      return false;
    if (ratio != 0)
      engine.dsv4_compressed[lt.layer].reset();
  }
  if (g.host_inputs.empty() ||
      engine.hc_stream_buf.size() < (size_t)g.D * g.HC * S)
    return false;
  std::memcpy(g.host_inputs[0], engine.hc_stream_buf.data(),
              (size_t)g.D * g.HC * S * sizeof(float));

  // Same defense-in-depth backstop as glm5next_full_graph_run's own compute
  // call (see that call site's own comment for the real-hardware incident
  // that motivated it) -- DSV4's full graph hasn't been run on real hardware
  // yet at all, so there is no evidence either way here, but the gap this
  // closes (a fresh, synchronous MemAvailable check right before the one
  // uncounted compute step in this whole admission chain) applies equally.
  {
    const long avail_mb = MemoryGuard::read_available_mb();
    constexpr long kMinComputeHeadroomMb =
        10240; // 10 GiB, per explicit user instruction after the 2026-09-09
               // crash
    if (avail_mb >= 0 && avail_mb < kMinComputeHeadroomMb) {
      LOG_ERROR("[GGML-DSV4-FULL] refusing graph compute: MemAvailable=%ld MiB "
                "< %ld MiB safety floor -- falling back to the legacy path\n",
                avail_mb, kMinComputeHeadroomMb);
      return false;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  if (ggml_backend_sched_graph_compute_async(g.sched, g.graph) !=
      GGML_STATUS_SUCCESS)
    return false;
  ggml_backend_sched_synchronize(g.sched);
  const auto synced = std::chrono::steady_clock::now();
  ggml_backend_tensor_get(g.output, hidden_states, 0,
                          (size_t)g.D * S * sizeof(float));

  if (dsv4_graph_diagnostic_enabled()) {
    for (const auto &stage : g.dsv4_stage_outputs) {
      // Attention/FFN stages are [D,S] tensors, while residual stages are
      // [D,HC,S].  Reading every stage into the residual-sized buffer used
      // to over-read the former and triggered GGML's bounds assertion.
      const size_t bytes = ggml_nbytes(stage.second);
      if (bytes == 0 || bytes % sizeof(float) != 0) {
        LOG_ERROR("[GGML-DSV4-DIAG] %s has unsupported byte size=%zu\n",
                  stage.first.c_str(), bytes);
        return false;
      }
      std::vector<float> layer_values(bytes / sizeof(float));
      ggml_backend_tensor_get(stage.second, layer_values.data(), 0, bytes);
      size_t finite = 0;
      float min_value = INFINITY;
      float max_value = -INFINITY;
      for (float value : layer_values) {
        if (std::isfinite(value)) {
          ++finite;
          min_value = std::min(min_value, value);
          max_value = std::max(max_value, value);
        }
      }
      LOG_INFO("[GGML-DSV4-DIAG] %s shape=(%lld,%lld,%lld,%lld) finite=%zu/%zu "
               "min=%g max=%g\n",
               stage.first.c_str(), (long long)stage.second->ne[0],
               (long long)stage.second->ne[1], (long long)stage.second->ne[2],
               (long long)stage.second->ne[3], finite, layer_values.size(),
               min_value, max_value);
      if (finite != layer_values.size())
        break;
    }
  }

  for (const auto &lt : g.dsv4_layers) {
    const int l = lt.layer;
    const size_t raw_n = (size_t)(g.qk_nope + g.qk_rope) * S;
    std::vector<float> raw(raw_n);
    ggml_backend_tensor_get(lt.raw_kv, raw.data(), 0, raw_n * sizeof(float));
    for (int s = 0; s < S; ++s) {
      const int slot = s % std::max(1, window);
      std::memcpy(engine.KVc[l] + (size_t)slot * (g.qk_nope + g.qk_rope),
                  raw.data() + (size_t)s * (g.qk_nope + g.qk_rope),
                  (size_t)(g.qk_nope + g.qk_rope) * sizeof(float));
    }
    if (lt.ratio == 0)
      continue;
    auto &state = engine.dsv4_compressed[l];
    const int width =
        lt.ratio == 4 ? 2 * (g.qk_nope + g.qk_rope) : (g.qk_nope + g.qk_rope);
    std::vector<float> comp((size_t)width * S);
    std::vector<float> score((size_t)width * S);
    ggml_backend_tensor_get(lt.comp_kv, comp.data(), 0,
                            comp.size() * sizeof(float));
    ggml_backend_tensor_get(lt.comp_score, score.data(), 0,
                            score.size() * sizeof(float));
    for (int s = 0; s < S; ++s) {
      const int slot = s % state.state_size;
      std::memcpy(state.state_kv.data() + (size_t)slot * width,
                  comp.data() + (size_t)s * width,
                  (size_t)width * sizeof(float));
      std::memcpy(state.state_score.data() + (size_t)slot * width,
                  score.data() + (size_t)s * width,
                  (size_t)width * sizeof(float));
    }
    const int blocks = S / lt.ratio;
    if (blocks > 0 && lt.keys) {
      std::vector<float> keys((size_t)(g.qk_nope + g.qk_rope) * blocks);
      ggml_backend_tensor_get(lt.keys, keys.data(), 0,
                              keys.size() * sizeof(float));
      std::memcpy(state.keys.data(), keys.data(),
                  std::min(state.keys.size(), keys.size()) * sizeof(float));
    }
    if (lt.indexer_comp_kv) {
      const int idim = c.dsv4_indexer_key_length;
      const int iwidth = 2 * idim;
      std::vector<float> ikv((size_t)iwidth * S);
      std::vector<float> iscore((size_t)iwidth * S);
      ggml_backend_tensor_get(lt.indexer_comp_kv, ikv.data(), 0,
                              ikv.size() * sizeof(float));
      ggml_backend_tensor_get(lt.indexer_comp_score, iscore.data(), 0,
                              iscore.size() * sizeof(float));
      for (int s = 0; s < S; ++s) {
        const int slot = s % state.state_size;
        std::memcpy(state.indexer_state_kv.data() + (size_t)slot * iwidth,
                    ikv.data() + (size_t)s * iwidth,
                    (size_t)iwidth * sizeof(float));
        std::memcpy(state.indexer_state_score.data() + (size_t)slot * iwidth,
                    iscore.data() + (size_t)s * iwidth,
                    (size_t)iwidth * sizeof(float));
      }
      if (blocks > 0 && lt.indexer_keys) {
        std::vector<float> keys((size_t)idim * blocks);
        ggml_backend_tensor_get(lt.indexer_keys, keys.data(), 0,
                                keys.size() * sizeof(float));
        std::memcpy(state.indexer_keys.data(), keys.data(),
                    std::min(state.indexer_keys.size(), keys.size()) *
                        sizeof(float));
      }
    }
  }
  const auto done = std::chrono::steady_clock::now();
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  LOG_INFO(
      "[GGML-DSV4-FULL-TIME] sync=%.3f ms state_get=%.3f ms total=%.3f ms\n",
      ms(start, synced), ms(synced, done), ms(start, done));
  engine.deepseek_v4_full_graph_output_ready = true;
  return true;
}

#else

bool glm5next_full_graph_enabled() { return false; }
bool glm5next_full_graph_bootstrap(StreamingEngine &) { return false; }
void glm5next_full_graph_set_vram_cache(GpuTopology *) {}
int glm5next_full_graph_device_expert_layers() { return 0; }
bool glm5next_full_graph_prepare(StreamingEngine &, int, int, const int *) {
  return false;
}
bool glm5next_full_graph_run(StreamingEngine &, float *, int, int,
                             const int *) {
  return false;
}
bool glm5next_full_graph_decode_prepare(StreamingEngine &, int, const int *) {
  return false;
}
bool glm5next_full_graph_decode_run(StreamingEngine &, float *, int,
                                    const int *) {
  return false;
}
void glm5next_full_graph_shutdown() {}
bool deepseek_v4_full_graph_enabled() { return false; }
int deepseek_v4_full_graph_device_expert_layers() { return 0; }
bool deepseek_v4_full_graph_prepare(StreamingEngine &, int, int, const int *) {
  return false;
}
bool deepseek_v4_full_graph_run(StreamingEngine &, float *, int, int,
                                const int *) {
  return false;
}
void deepseek_v4_full_graph_shutdown() {}

#endif
