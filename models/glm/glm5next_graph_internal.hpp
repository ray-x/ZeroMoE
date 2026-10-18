#pragma once

#include "graph_core.hpp"

// Shared internals of the GLM5NEXT / DeepSeek-V4 full-graph translation units.

#include "graph_config.hpp"
#include "pilot_ridge.hpp"
#include "dense_dispatch.hpp"
#include "direct_io.hpp"
#include "engine.hpp"
#include "expert_cache.hpp"
#include "early_ready_lane.hpp"
#include "expert_store.hpp"
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
#include "moe_routed_ggml.hpp"
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
#include <cuda_runtime.h>

#if defined(MOESTREAM_USE_GGML_CUDA)
#include <cuda_runtime.h>

namespace glm5next_graph {

constexpr int kGraphSize = 60000;
constexpr size_t kTensorContextBytes = 192ull * 1024ull * 1024ull;


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
  std::vector<float> heat;        // decayed re-reference score of each cold expert (MOESTREAM_GLM5NEXT_DYNAMIC_HEAT)
  std::vector<uint64_t> heat_at;  // token clock of the last heat update
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

using moestream::graph::DeviceArena;
using moestream::graph::HostAllocation;
using moestream::graph::alloc_host;
using moestream::graph::bind_host;
using moestream::graph::host_input;
using moestream::graph::model_tensor;
using moestream::graph::new_tensor;
using moestream::graph::resident;

struct FullGraph : moestream::graph::GraphCore {
  FullGraph() { mem_tag = "GGML-GLM5NEXT-FULL-MEM"; }

  GpuTopology *topology = nullptr;
  int d_inner = 0;
  int qk_nope = 0;
  int qk_rope = 0;
  int kv_lora = 0;
  int v_head = 0;
  int o_group_count = 0;
  int o_lora_rank = 0;
  int sliding_window = 0;
  int first_dense = 0;
  ggml_backend_t gpu1 = nullptr; // optional second CUDA device for expert layers
  int gpu1_first = -1, gpu1_last = -2;
  std::vector<ggml_backend_buffer_t> gpu1_buffers;
  std::unordered_map<int, ggml_tensor *> stage_pred; // layer -> pilot ids predicted for it (built one layer earlier)
  std::vector<ggml_backend_buffer_t> ridge_buffers;   // device copies of the pilot ridge corrections (one per corrected layer)
  std::vector<std::pair<int, ggml_tensor *>> probe_x;  // MOESTREAM_GLM5NEXT_PROBE_DUMP: (layer, pre-FFN hidden) read back after every decode token
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


  ggml_tensor *indexer_pool_member_ids = nullptr; // [kpool, n_pools], I32
  ggml_tensor *indexer_pool_mask = nullptr; // [n_pools], F32, per-token
  ggml_tensor *indexer_pool_update_rows = nullptr; // [kpool], I32
  ggml_tensor *indexer_pool_update_index = nullptr; // [1], I32
  ggml_tensor *indexer_tail_indices = nullptr; // [kpool-1], I32
  ggml_tensor *indexer_tail_mask = nullptr; // [kpool-1], F16
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
  struct Dsv41LayerTensors {
    int layer = -1;
    int ratio = 0;
    int kv_source = -1;
    int index_source = -1;
    ggml_tensor *raw_kv = nullptr;       // [qh, S], rotated raw SWA K==V
    ggml_tensor *comp_kv = nullptr;      // [qh, S], source compressor output
    ggml_tensor *comp_score = nullptr;   // [qh, S], pooling logits
    ggml_tensor *keys = nullptr;         // [qh, floor(S/ratio)]
    ggml_tensor *indexer_keys = nullptr; // [indexer_key_dim, floor(S/ratio)]
  };
  std::vector<Dsv41LayerTensors> dsv41_layers;
  ggml_tensor *dsv41_final_pre_mix = nullptr;
  std::vector<std::pair<int, ggml_tensor *>> dsv41_engram_embeddings;
  std::vector<std::pair<int, std::vector<float>>> dsv41_engram_host_rows;
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
  // Chunked prefill (pos_base > 0): the tokens before this chunk, the prompt length covered once it ran, the latent rows of
  // the earlier chunks as a host input per DSA layer, and the indexer seeds of the earlier chunks to extend.
  int pos_base = 0;
  int total_tokens = 0;
  bool ran = false; // the prefill graph completed a run (its KDA/latent/indexer state is exported)
  std::vector<std::pair<int, float *>> dsa_prev_inputs;
  std::vector<Glm5nextIndexerSeed> prev_indexer_seeds;
  // Prefill chunk whose context exceeds the indexer's top-k: attention keeps, per query, the tokens of the best-scoring
  // pools (plus the tail of the incomplete pool), as the decode graph does. The mask is built by the first indexer-owner
  // layer and reused by the DSA layers after it.
  bool sparse_prefill = false;
  ggml_tensor *prefill_pool_valid = nullptr; // [pools, S] 1 where the pool is complete for that query
  ggml_tensor *prefill_tail01 = nullptr;     // [kv_tokens, S] 1 for the tokens of the query's incomplete pool
  ggml_tensor *prefill_sel_mask = nullptr;   // [kv_tokens, S] F16 additive attention mask
  struct PrevPoolInput {
    int layer = -1;
    float *data = nullptr;
    int pools = 0;
  };
  std::vector<PrevPoolInput> indexer_prev_pool_inputs;
  std::vector<ggml_tensor *> dsa_kv_storage;
  std::vector<ggml_tensor *> kda_state_storage;
  std::vector<ggml_tensor *> kda_hist_storage;
  std::vector<ggml_tensor *> kda_state_updates;
  std::vector<ggml_tensor *> kda_hist_updates;
  std::vector<int> dsa_layers;
  std::vector<std::pair<int, ggml_tensor *>> moe_route_ids;
  // CPU lanes (MOESTREAM_GLM5NEXT_EARLY_READY): one per layer whose cold experts
  // run concurrently with that layer's GPU work.
  struct VramExpertHold {
    int layer = -1;
    int expert = -1;
    const uint8_t *device_ptr = nullptr;
  };
  std::vector<VramExpertHold> vram_expert_holds;
  std::vector<Glm5nextIndexerLayer> glm5next_indexer_layers;
  std::vector<Glm5nextIndexerSeed> glm5next_indexer_seeds;


  size_t host_reference_bytes = 0;
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
  // ggml_init(no_alloc=true) still owns a host-side context arena.  Keep it
  // in the same admission ledger as the graph's explicit host buffers; the
  // old watchdog only saw the latter and could therefore under-report the
  // graph's host peak during construction.
  size_t graph_workspace_bytes = 0;

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
    early_ready.clear();
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
    for (ggml_backend_buffer_t b : ridge_buffers)
      if (b)
        ggml_backend_buffer_free(b);
    ridge_buffers.clear();
    probe_x.clear();
    if (expert.buffer) {
      ggml_backend_buffer_free(expert.buffer);
      expert.buffer = nullptr;
    }
    if (state.buffer) {
      ggml_backend_buffer_free(state.buffer);
      state.buffer = nullptr;
    }
    if (dense.buffer) {
      ggml_backend_buffer_free(dense.buffer);
      dense.buffer = nullptr;
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
    release_core(); // host buffers, context, owned host memory, counters
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
    dsa_prev_inputs.clear();
    indexer_prev_pool_inputs.clear();
    prefill_pool_valid = prefill_tail01 = prefill_sel_mask = nullptr;
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
    dsv41_layers.clear();
    dsv41_engram_embeddings.clear();
    dsv41_engram_host_rows.clear();
    dsv4_stage_outputs.clear();
    topology = nullptr;
  }
};


// Helpers defined in ggml_glm5next_graph.cpp and used by the DeepSeek-V4 builder.
bool type_ok(GGUFType type);
bool configure_partial_experts(FullGraph &g);
size_t expert_source_bytes(const StreamingEngine &engine, int device_expert_layers);
size_t graph_host_live_bytes(const StreamingEngine &engine);
bool init_expert_source(FullGraph &g, size_t source_bytes);
bool preload_expert_source(FullGraph &g);
bool prepare_expert_source_only(FullGraph &g, const char *tag);
bool init_expert_arena(FullGraph &g);
bool init_dense_arena(FullGraph &g);
// Defined in ggml_deepseek_v4_graph.cpp, also used by the GLM builder.
bool graph_alloc_vram_headroom_ok(const FullGraph &g, const char *log_tag);
ggml_tensor *resident_tensor(FullGraph &g, ggml_context *ctx, const std::string &name);
ggml_tensor *build_ffn(FullGraph &g, ggml_context *ctx, ggml_tensor *cur, int layer, ggml_tensor *forced_ids);
ggml_tensor *hc_pre(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                    ggml_tensor *fn, ggml_tensor *scale, ggml_tensor *base,
                    ggml_tensor **post, ggml_tensor **comb,
                    ggml_tensor **pre_gate = nullptr,
                    ggml_tensor *collapse_carry = nullptr);
ggml_tensor *hc_post(FullGraph &g, ggml_context *ctx, ggml_tensor *x, ggml_tensor *residual, ggml_tensor *post, ggml_tensor *comb);

}  // namespace glm5next_graph

#endif
