#pragma once

// Internal declarations shared by the ggml_qwen4exp_graph*.cpp translation
// units (namespace qwen4exp_graph).  The public entry points stay in
// ggml_qwen4exp_graph.hpp; everything here is private to the Qwen4Exp
// full-graph implementation.

#include "graph_config.hpp"
#include "pilot_ridge.hpp"
#include "ggml_glm5next_graph.hpp"
#include "ggml_deepseek_v4_graph.hpp"

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
#include "../models/full_graph_model.hpp"
#include "moe_routed_config.hpp"
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
#if defined(MOESTREAM_USE_GGML_CUDA)

#include <cuda_runtime.h>

#include "glm5next_graph_internal.hpp"

namespace glm5next_graph {

// ---- forward declarations of the types below ----
struct OffloadState;
struct OffloadSite;
struct StageSite;
struct StageMissSite;
struct StageResult;
struct StageRuntime;
struct PackedGateUpMeta;
struct RouteTap;

// ---- function declarations (defined in ggml_qwen4exp_graph_*.cpp) ----
bool env_enabled();
bool offload_async();
int offload_first_layer();
bool offload_layer(int layer);
std::string offload_env(const char *name, const char *dflt = "");
int pilot_stage_k();
bool pilot_prefetch_on();
bool pilot_copy_on();
int pilot_prefetch_k();
int pilot_lookahead();
float pilot_min_score();
bool dynamic_cache_enabled();
bool route_stats_enabled();
int env_device_expert_layers();
bool type_ok(GGUFType type);
bool layer_timing_env();
int dynamic_upload_budget_override();
double dynamic_upload_fraction();
bool early_ready_wanted();
// Compute the cold experts that are already in DDR while the missing ones are read from the SSD.
bool split_load_wanted();
// Number of resident cold experts per layer the lane may run on a second CUDA stream (0 = off).
int gpu_assist_slots();
int gpu_assist2_device();
int gpu_assist2_slots();
// True when every routed-expert plane of `layer` has a type worth assisting on the GPU (see settings).
bool gpu_assist_layer_ok(const FullGraph &g, int layer);
// Keep the CPU lane on layers that also have a GPU1 hot branch (MOESTREAM_GLM5NEXT_LANE_WITH_GPU1, default on).
bool lane_with_gpu1_wanted();
OffloadState *get_offload(const StreamingEngine &engine, const std::vector<PartialExpertLayer> *partial = nullptr);
void offload_miss_cb(ggml_tensor *dst, const ggml_tensor *a, int ith, int, void *ud);
ggml_tensor *offload_weight(FullGraph &g, ggml_context *ctx, int layer, const char *stem);
ggml_tensor *offload_miss_node(FullGraph &g, ggml_context *ctx, int layer, ggml_tensor *ids);
// Host node over [topk, S] I32 route ids. mode 0/1: ids of experts below / at-or-above `split_at` (the upper half
// re-based to 0, everything else -1). mode 2/3: 1 where mode 0/1 holds a real id, else 0.
ggml_tensor *cold_split_node(ggml_context *ctx, ggml_tensor *ids, int split_at, int mode);
ggml_tensor *expert_weight_impl(FullGraph &g, ggml_context *ctx, int layer, const char *stem);
ggml_tensor *expert_gate_up_weight_impl(FullGraph &g, ggml_context *ctx, int layer);
bool gpu1_layer(const FullGraph &g, int layer);
ggml_tensor *move_expert_to_gpu1(FullGraph &g, ggml_context *ctx, int layer, ggml_tensor *host);
void stage_worker_main(StageRuntime *st);
StageRuntime *get_stage(FullGraph &g);
StageRuntime *get_stage_impl(FullGraph &g);
void stage_post_job(StageRuntime *st, int layer, int set, const ggml_tensor *pred);
void stage_miss_cb(ggml_tensor *dst, const ggml_tensor *a, const ggml_tensor *b, const ggml_tensor *c, int ith, int, void *ud);
bool stage_layer_active(const FullGraph &g, int layer);
ggml_tensor *stage_hot_mask(FullGraph &g, ggml_context *ctx, int layer);
bool build_pilot_stage(FullGraph &g, ggml_context *ctx, ggml_tensor *cur_pre, int l);
ggml_tensor *stage_ring_tensor(FullGraph &g, ggml_context *ctx, StageRuntime &st, int layer, int plane);
ggml_tensor *stage_miss_node(FullGraph &g, ggml_context *ctx, StageRuntime &st, int layer, ggml_tensor *cold_ids, ggml_tensor *route_ids);
void *alloc_expert_source(FullGraph &g, size_t bytes);
bool graph_dense_weight_name(const std::string &name);
bool arena_alloc(ggml_backend_buffer_t buffer, void *base, size_t capacity, size_t &used, ggml_tensor *t, const void *src, size_t src_bytes);
ggml_tensor *resident_tensor(FullGraph &g, ggml_context *ctx, const std::string &name);
size_t device_expert_bytes(const FullGraph &g);
size_t dense_arena_bytes(const FullGraph &g);
bool init_dense_arena(FullGraph &g);
bool init_decode_state_arena(FullGraph &g);
ggml_tensor *new_decode_state_tensor(FullGraph &g, ggml_context *ctx, ggml_type type, const std::vector<int64_t> &shape, const char *name);
bool init_expert_arena(FullGraph &g);
bool init_expert_source(FullGraph &g, size_t source_bytes);
bool source_configuration_matches(const FullGraph &source, const FullGraph &candidate);
bool prepare_expert_source_only(FullGraph &g, const char *tag);
ggml_tensor *load_full_weight(FullGraph &g, ggml_context *ctx, const std::string &name, bool device_expert, bool routed_expert = false);
bool packed_gate_up_meta(const StreamingEngine &engine, int layer, PackedGateUpMeta *out);
double full_graph_resident_gb(const char *name);
size_t graph_host_live_bytes(const StreamingEngine &engine);
std::string partial_usage_path(const FullGraph &g);
PartialExpertLayer *partial_layer(FullGraph &g, int layer);
const PartialExpertLayer *partial_layer(const FullGraph &g, int layer);
bool partial_layer_ram_ready(FullGraph &g, int layer);
bool read_tensor_direct(const StreamingEngine &engine, const GGUFTensor &meta, void *dst);
bool read_pack_gate_up(const StreamingEngine &engine, const GGUFTensor &gate, const GGUFTensor &up, void *dst);
bool read_tensor_range_direct(const StreamingEngine &engine, const GGUFTensor &meta, size_t relative_offset, void *dst, size_t bytes);
bool read_tensor_selected_experts(const StreamingEngine &engine, const GGUFTensor &meta, int n_experts, const std::vector<int> &selected, void *dst);
bool read_pack_gate_up_selected(const StreamingEngine &engine, const GGUFTensor &gate, const GGUFTensor &up, int n_experts, const std::vector<int> &selected, void *dst);
ggml_tensor *load_gate_up_weight(FullGraph &g, ggml_context *ctx, int layer, bool device_expert);
ggml_tensor *load_cold_weight(FullGraph &g, ggml_context *ctx, int layer, const char *stem);
template <typename Loader> bool arena_load_selected(FullGraph &g, ggml_tensor *t, Loader &&load);
ggml_tensor *make_hot_weight(FullGraph &g, ggml_context *ctx, int layer, const char *stem);
ggml_tensor *make_hot_gate_up(FullGraph &g, ggml_context *ctx, int layer);
ggml_tensor *expert_weight(FullGraph &g, ggml_context *ctx, int layer, const char *stem);
ggml_tensor *expert_gate_up_weight(FullGraph &g, ggml_context *ctx, int layer);
size_t expert_source_bytes(const StreamingEngine &engine, int device_expert_layers);
size_t expert_source_bytes(const FullGraph &g);
bool compact_partial_source(FullGraph &g);
bool preload_mmap_expert_source(FullGraph &g);
bool preload_expert_source(FullGraph &g);
bool configure_vram_reference_experts(FullGraph &g);
bool configure_partial_experts_base(FullGraph &g);
bool configure_partial_experts(FullGraph &g);
ggml_tensor *make_vram_ref_weight(FullGraph &g, ggml_context *ctx, int layer, const char *stem);
bool prepare_partial_hot(FullGraph &g, ggml_context *ctx, int layer);
ggml_tensor *partial_map_hot1(FullGraph &g, ggml_context *ctx, int layer, bool mask);
ggml_tensor *hot1_weight(FullGraph &g, ggml_context *ctx, int layer, const char *stem, int plane);
bool prepare_partial_hot1(FullGraph &g, ggml_context *ctx, int layer);
void assign_gpu1_hot(FullGraph &g, double gib);
ggml_tensor *partial_map(FullGraph &g, ggml_context *ctx, int layer, bool hot, bool mask);
bool upload_partial_maps(FullGraph &g, PartialExpertLayer &p);
bool ensure_dynamic_upload_worker(FullGraph &g);
bool make_dynamic_upload_job(PartialExpertLayer &p, int layer, int hot_slot, int incoming_expert, int evicted_expert, DynamicUploadJob *job);
bool publish_dynamic_uploads(FullGraph &g);
RouteTap &route_tap();
void route_tap_flush(RouteTap &t);
void update_partial_routes(FullGraph &g, int pos_base, double compute_ms);
bool hold_graph_vram_experts(FullGraph &g);
void release_graph_vram_experts(FullGraph &g);
ggml_tensor *static_full_weight(FullGraph &g, ggml_context *ctx, const std::string &name);
ggml_tensor *graph_ram_weight(FullGraph &g, ggml_context *ctx, int layer, const char *stem);
ggml_tensor *hc_affine(ggml_context *ctx, ggml_tensor *x, ggml_tensor *scale, ggml_tensor *base);
ggml_tensor *hc_mean(FullGraph &g, ggml_context *ctx, ggml_tensor *x);
ggml_tensor *hc_contig(ggml_context *ctx, ggml_tensor *t);
bool fused_hc_op(char which);
ggml_tensor *hc_pre(FullGraph &g, ggml_context *ctx, ggml_tensor *x, ggml_tensor *fn, ggml_tensor *scale, ggml_tensor *base, ggml_tensor **post, ggml_tensor **comb);
ggml_tensor *hc_post(FullGraph &g, ggml_context *ctx, ggml_tensor *x, ggml_tensor *residual, ggml_tensor *post, ggml_tensor *comb);
ggml_tensor *build_kda(FullGraph &g, ggml_context *ctx, ggml_tensor *cur, int layer, ggml_tensor *wq, ggml_tensor *wk, ggml_tensor *wv, ggml_tensor *cq, ggml_tensor *ck, ggml_tensor *cv, ggml_tensor *fa, ggml_tensor *fb, ggml_tensor *ga, ggml_tensor *gb, ggml_tensor *beta_w, ggml_tensor *a, ggml_tensor *dt, ggml_tensor *norm_w, ggml_tensor *out_w, ggml_tensor *hist_in, ggml_tensor *state_in, ggml_tensor **state_out, ggml_tensor **hist_out);
Glm5nextIndexerLayer *glm5next_indexer_state(FullGraph &g, int layer);
bool glm5next_indexer_owner(const FullGraph &g, int layer);
ggml_tensor *glm5next_indexer_pool(ggml_context *ctx, ggml_tensor *keys, ggml_tensor *gates, ggml_tensor *ape, int key_length, int kpool, int n_pools);
bool build_glm5next_indexer_prefill(FullGraph &g, ggml_context *ctx, ggml_tensor *cur, int layer);
bool build_glm5next_indexer_decode(FullGraph &g, ggml_context *ctx, ggml_tensor *cur, ggml_tensor *qr, int layer, ggml_tensor **sel_out, ggml_tensor **mask_out);
ggml_tensor *build_dsa(FullGraph &g, ggml_context *ctx, ggml_tensor *cur, int layer, ggml_tensor *wqa, ggml_tensor *wqan, ggml_tensor *wqb, ggml_tensor *wkva, ggml_tensor *wkvan, ggml_tensor *wkb, ggml_tensor *wvb, ggml_tensor *wo, ggml_tensor **kv_out, ggml_tensor *kv_cache = nullptr, ggml_tensor **prev_indexer_selection = nullptr, ggml_tensor **prev_indexer_mask = nullptr);
EarlyReadyLane *make_cold_lane(FullGraph &g, ggml_context *ctx, int layer, ggml_tensor *cur, ggml_tensor *ids, ggml_tensor *weights, ggml_tensor *cold_map, ggml_tensor *cold_mask, const moe_routed::ExpertWeights &cold_w, float clamp = 0.0f);
ggml_tensor *build_ffn(FullGraph &g, ggml_context *ctx, ggml_tensor *cur, int layer, ggml_tensor *forced_ids = nullptr);
bool build_graph(FullGraph &g);
void transpose_glm5next_state(float *dst, const float *src, int hd, int heads);
bool sync_glm5next_decode_state_to_host(FullGraph &g);

// ---- process-wide state (defined in ggml_qwen4exp_graph.cpp) ----
extern std::unique_ptr<FullGraph> g_graph;
extern std::unique_ptr<FullGraph> g_glm5next_source;
extern GpuTopology *g_glm5next_vram_cache;
extern StreamingEngine *g_glm5next_decode_disabled_engine;
extern std::unique_ptr<OffloadState> g_offload;
extern std::mutex g_offload_mutex;
extern std::unique_ptr<StageRuntime> g_stage;
extern bool g_stage_failed;
extern std::map<int, ggml_backend_buffer_t> g_hot1_buffers;

// ---- types ----
struct OffloadState;

struct OffloadSite {
  OffloadState *state = nullptr;
  int layer = 0;
  bool prefill = false;
  int n_tokens = 1;
  // experts of this layer served by a VRAM branch (hot / hot1): never routed to the cold branch, so never read ahead
  std::vector<uint8_t> not_cold;
};

// The GLM graph's view of the DDR expert tier: the tier itself is
// moestream::offload::ExpertStore; this adds the per-site records the miss
// nodes point at and short aliases for the store's parts.
struct OffloadState {
  int first = 0, n_layers = 0, n_experts = 0;
  std::unique_ptr<moestream::offload::ExpertStore> store;
  moestream::offload::ExpertCache *cache = nullptr;
  moestream::offload::VirtualArena *arena = nullptr;
  moestream::offload::ExpertResidency *residency = nullptr;
  std::deque<OffloadSite> sites;
  // Most recently built batched (prefill) miss site of a layer, or nullptr.
  const OffloadSite *site_for_layer(int layer) const {
    for (auto it = sites.rbegin(); it != sites.rend(); ++it)
      if (it->layer == layer && it->prefill)
        return &*it;
    return nullptr;
  }
  bool is_pinned(int layer, int expert) const { return store->is_pinned(layer, expert); }
};

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
  // Pilot candidates posted for each target layer, kept to score them against the layer's real routes (calibration telemetry).
  struct Cand {
    int n = 0;
    int32_t ids[8] = {};
    float score[8] = {};
  } cand[128];
  static constexpr int kScoreBuckets = 9;
  std::atomic<uint64_t> rank_n[8] = {}, rank_hit[8] = {}, sb_n[kScoreBuckets] = {}, sb_hit[kScoreBuckets] = {};
  static int score_bucket(float s) {
    static const float edges[kScoreBuckets - 1] = {0.01f, 0.02f, 0.05f, 0.1f, 0.2f, 0.4f, 0.8f, 1.6f};
    int b = 0;
    while (b < kScoreBuckets - 1 && s >= edges[b]) ++b;
    return b;
  }
  std::deque<StageSite> sites;
  std::deque<StageMissSite> miss_sites;
  std::atomic<uint64_t> direct{0}, jobs{0}, copied{0}, resolves{0}, staged_routes{0}, routes_seen{0}, wait_us{0}, skipped_not_resident{0},
      prefetch_loads{0}, prefetch_us{0};
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

struct StageMissSite {
  OffloadSite *off = nullptr; // this layer's offload state/site
  int layer = 0, set = 0;
  int next_layer = -1, next_set = 0; // layer whose experts the pilot input predicts (-1: last layer)
};

struct PackedGateUpMeta {
  const GGUFTensor *gate = nullptr;
  const GGUFTensor *up = nullptr;
  std::string name;
  size_t bytes = 0;
};

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


} // namespace glm5next_graph

#endif // MOESTREAM_USE_GGML_CUDA

