#pragma once

#include "graph_core.hpp"

// Internal declarations shared by the ggml_qwen4exp_graph*.cpp translation
// units (namespace qwen4exp_graph).  The public entry points stay in
// ggml_qwen4exp_graph.hpp; everything here is private to the Qwen4Exp
// full-graph implementation.

#include "graph_config.hpp"
#include "../models/full_graph_model.hpp"
#include "moe_routed_config.hpp"
#include "moe_routed_ggml.hpp"
#include "ggml_qwen4exp_graph.hpp"
#include "option_b_global_hot.hpp"

#include "dense_dispatch.hpp" // is_cpu_affinity_layer (cpu_affinity_layers config)
#include "cuda_manager.hpp"
#include "direct_io.hpp"
#include "engine.hpp"
#include "exl3_gpu.hpp"
#include "exl3_ggml_model.hpp"
#include "ggml_moe_backend.hpp"
#include "kquant_dequant.hpp"
#include "logger.hpp"
#include "memory_guard.hpp"
#include "mmid_route_skip.hpp"
#include "moe_partial_route.hpp"
#include "ple_hash.hpp"
#include "thread_profile.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <sched.h>
#include <unistd.h>
#include <vector>

#if defined(MOESTREAM_USE_GGML_CUDA)

#include <cuda_runtime.h>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "early_ready_lane.hpp"
#if defined(_OPENMP)
#include <omp.h>
#endif

extern "C" cudaStream_t moestream_ggml_backend_cuda_get_stream(
    ggml_backend_t backend);
extern "C" cudaGraph_t moestream_ggml_cuda_get_captured_graph(
    ggml_backend_t backend, const ggml_tensor *first_node);
extern "C" void *moestream_qwen4exp_unpack_staged_payload_kernel_ptr();

namespace qwen4exp_graph {

// ---- forward declarations of the types below ----
struct Qwen4ExpCudaHostNodeContext;
struct Qwen4ExpGpuStageContext;
struct Qwen4ExpUsageTrackContext;
struct Qwen4ExpHostNodeCpuExecutionGuard;
struct Qwen4ExpHostNodeCpuActivityGuard;
struct FullGraph;
enum class ExistingDeviceExpertBind;
struct ScopedGraphBuildTimer;
struct SplitExpertWeight;
struct ExpertSplitTables;

constexpr int kGraphSize = 50000;
constexpr size_t kTensorContextBytes = 128ull * 1024ull * 1024ull;

// ---- function declarations (defined in ggml_qwen4exp_graph_*.cpp) ----
bool qwen4exp_memory_trace_enabled();
long qwen4exp_proc_kb(const char *path, const char *key);
long long qwen4exp_proc_bytes(const char *path);
void qwen4exp_memory_snapshot(const char *tag, int device, const char *detail = nullptr);
void qwen4exp_cuda_failure_snapshot(const char *tag, int device, cudaError_t status);
void CUDART_CB qwen4exp_cuda_gpu_stage_callback(void *user_data);
bool qwen4exp_cuda_gpu_stage_node_enqueue(const ggml_tensor *node, cudaStream_t stream);
void CUDART_CB qwen4exp_cuda_usage_track_callback(void *user_data);
bool qwen4exp_cuda_usage_track_node_enqueue(const ggml_tensor *node, cudaStream_t stream);
bool env_host_node_gpu_reduce();
bool env_host_node_skip_cpu();
int env_partial_graph_layer();
bool env_partial_graph_enabled(int layer);
bool env_host_node_keep_cpu_prologue();
bool qwen4exp_host_node_timing_enabled();
double qwen4exp_host_node_ordinary_cpu_wait_ms_since_last();
double qwen4exp_host_node_gate_wait_ms_since_last();
double qwen4exp_cpu_split_enter_wait_ms_since_last();
uint64_t qwen4exp_fnv64(const void *data, size_t bytes);
bool qwen4exp_dump_f32(const char *path, const float *data, size_t elements);
void CUDART_CB qwen4exp_cuda_host_node_callback(void *user_data);
void CUDART_CB qwen4exp_cuda_host_node_arm_direct_workers( void *user_data);
bool env_prefill_state_handoff_enabled();
bool prefill_device_state_inputs_enabled(const StreamingEngine &engine);
bool env_enabled();
bool env_decode_borrow_prefill_arenas();
int env_device_expert_layers();
bool env_device_expert_trailing();
bool env_layer_timing_enabled();
bool env_phase_timing_enabled();
bool env_qwen4exp_host_node_enabled();
bool env_layer_timing_verbose();
bool env_graph_replay_timing_enabled();
bool env_merged_gate_up_enabled();
int env_host_node_diag_layer();
bool env_host_node_transport_diag_enabled(int layer);
int env_partial_layer();
int env_partial_layer_top_n();
float env_partial_layer_min_coverage();
bool env_partial_layer_phase2_enabled();
bool partial_prefill_capture_enabled();
std::chrono::milliseconds env_storage_wait_timeout();
bool env_adaptive_completion_policy_enabled();
bool env_decode_enabled();
int env_decode_kv_capacity(int max_t, int required);
void pin_host_node_threads(ggml_threadpool_params &p, int n_threads);
int graph_cpu_threads();
bool graph_scheduler_parallel();
bool graph_cuda_host_buffers();
bool graph_match_llama_cpp_topology(const FullGraph &graph);
bool graph_layout_debug();
bool has_ple_layer(const ArchitectureConfig &c);
bool type_ok(GGUFType type);
void verify_cuda_graphs_compiled_in(ggml_backend_t cuda_backend);
int layer_timing_parse_layer(const char *name);
bool layer_timing_eval_callback(struct ggml_tensor *t, bool ask, void *user_data);
void qwen4exp_scan_captured_graph_writes(FullGraph &g, const void *range_base, size_t range_bytes);
void qwen_probe_dump(FullGraph &g);
void log_qwen_layout_tensor(const FullGraph &g, int layer, bool cpu_only, const char *role, const ggml_tensor *tensor);
void dump_qwen_moe_layout(const FullGraph &g);
void bind_host_expert_scheduler_backend(FullGraph &g, ggml_tensor *t);
ggml_tensor *resident_tensor( FullGraph &g, ggml_context *ctx, const std::string &name, ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
bool graph_dense_weight_name(const std::string &name);
bool init_device_dense_arena(FullGraph &g);
bool init_device_state_arena(FullGraph &g);
ggml_tensor *new_device_state_tensor(FullGraph &g, ggml_type type, const std::vector<int64_t> &shape, const char *name);
bool device_expert_block_covers(const FullGraph &g, int layer);
bool graph_expert_projection_resident(const FullGraph &g, int layer, const char *stem);
bool init_device_expert_arena(FullGraph &g);
std::string expert_ptr_text(const void *ptr);
FullGraph *expert_arena_owner(FullGraph &g);
std::string device_expert_single_key(const void *host, size_t bytes, const std::string &source_tag);
void *alloc_registered_host(FullGraph &g, size_t bytes);
void plan_expert_vram_split(FullGraph &g);
void plan_partial_layer(StreamingEngine &engine);
size_t full_expert_source_bytes(const StreamingEngine &engine, int layer_limit = -1);
bool preloaded_ram_expert_source(FullGraph &g, const std::string &name, const GGUFTensor &meta, void *&ptr, size_t &bytes, size_t &expert_stride, size_t &arena_bytes);
int device_expert_resident_layer_count(const FullGraph &g);
size_t device_expert_bytes(const FullGraph &g);
ExistingDeviceExpertBind bind_existing_device_expert( FullGraph &g, ggml_tensor *t, size_t bytes, const std::string &source_tag);
bool bind_or_allocate_expert(FullGraph &g, ggml_tensor *t, const std::string &key, void **addr_out, bool *new_allocation);
bool bind_device_expert(FullGraph &g, ggml_tensor *t, const void *host, size_t bytes, const std::string &source_tag = std::string());
bool bind_device_expert_strided(FullGraph &g, ggml_tensor *t, const void *base, size_t src_stride, size_t per_expert_bytes, int n_experts);
bool bind_device_expert_gather(FullGraph &g, ggml_tensor *t, const std::vector<const void *> &src_ptrs, size_t per_expert_bytes);
bool bind_device_expert_strided_gate_up(FullGraph &g, ggml_tensor *t, const void *gate_base, size_t gate_stride, size_t gate_bytes, const void *up_base, size_t up_stride, size_t up_bytes, int n_experts);
ggml_tensor *load_full_weight(FullGraph &g, ggml_context *ctx, const std::string &name, bool device_resident = false);
ggml_tensor *expert_weight(FullGraph &g, ggml_context *ctx, int layer, const char *stem);
ggml_tensor *expert_gate_up_weight(FullGraph &g, ggml_context *ctx, int layer, bool device_resident);
ggml_tensor *build_split_gpu_tensor(FullGraph &g, ggml_context *ctx, const std::string &name, const GGUFTensor &m, const std::vector<int> &hot_ids);
SplitExpertWeight expert_weight_split(FullGraph &g, ggml_context *ctx, int layer, const char *stem);
ggml_tensor *dense_weight(FullGraph &g, ggml_context *ctx, int layer, const char *stem);
ggml_tensor *hc_mix(FullGraph &g, ggml_context *ctx, ggml_tensor *x, ggml_tensor *norm, ggml_tensor *down, ggml_tensor *up, ggml_tensor *inject, ggml_tensor **inject_out);
ggml_tensor *hc_combine(FullGraph &g, ggml_context *ctx, ggml_tensor *residual, ggml_tensor *block, ggml_tensor *inject);
ggml_tensor *build_shared(FullGraph &g, ggml_context *ctx, ggml_tensor *cur, int layer, ggml_tensor *gate, ggml_tensor *up, ggml_tensor *down, ggml_tensor *scalar_gate);
void gdn_state_transpose_host(float *dst, const float *src, int hd, int Hv);
ggml_tensor *build_gdn(FullGraph &g, ggml_context *ctx, ggml_tensor *cur, int layer, ggml_tensor *w_qkv, ggml_tensor *w_gate, ggml_tensor *w_alpha, ggml_tensor *w_beta, ggml_tensor *w_dt, ggml_tensor *w_a, ggml_tensor *w_conv, ggml_tensor *w_norm, ggml_tensor *w_out, ggml_tensor *hist_in, ggml_tensor *state_in, ggml_tensor **state_out, ggml_tensor **hist_out);
ggml_tensor *build_full_attention(FullGraph &g, ggml_context *ctx, ggml_tensor *cur, int layer, ggml_tensor *w_q, ggml_tensor *w_k, ggml_tensor *w_v, ggml_tensor *w_qn, ggml_tensor *w_kn, ggml_tensor *w_o, ggml_tensor **k_cache, ggml_tensor **v_cache);
ggml_tensor *build_full_attention_decode( FullGraph &g, ggml_context *ctx, ggml_tensor *cur, int layer, ggml_tensor *w_q, ggml_tensor *w_k, ggml_tensor *w_v, ggml_tensor *w_qn, ggml_tensor *w_kn, ggml_tensor *w_o, ggml_tensor *k_storage, ggml_tensor *v_storage);
bool prepare_ple_embedding(FullGraph &g, std::vector<float> &embedding, int S, int pos_base, const int *token_ids);
ggml_tensor *build_qwen4exp_host_node( FullGraph &g, ggml_context *ctx, ggml_tensor *cur, ggml_tensor *ids, ggml_tensor *weights, int layer, bool partial_cold_only = false, ggml_tensor *gate_exps_for_cpu = nullptr, ggml_tensor *up_exps_for_cpu = nullptr, ggml_tensor *down_exps_for_cpu = nullptr, ggml_tensor *remap_cpu_for_cpu = nullptr, ggml_tensor *mask_cpu_for_cpu = nullptr);
int build_gpu_stage_node(FullGraph &g, ggml_context *ctx, ggml_tensor *ids, int layer);
ExpertSplitTables build_expert_split_tables(FullGraph &g, ggml_context *ctx, int n_experts, const std::vector<int> &gpu_ids, bool safe_dummy_fill = false);
bool exl3_ggml_bridge_requested(const FullGraph &g, int layer, std::string &model_path);
ggml_tensor *build_moe(FullGraph &g, ggml_context *ctx, ggml_tensor *cur, int layer, ggml_tensor *router, ggml_tensor *selection_bias, ggml_tensor *gate_exps, ggml_tensor *up_exps, ggml_tensor *down_exps, ggml_tensor *gate_exps_gpu = nullptr, ggml_tensor *up_exps_gpu = nullptr, ggml_tensor *down_exps_gpu = nullptr, const std::vector<int> *split_gpu_ids = nullptr, ggml_tensor *gate_up_exps = nullptr, ggml_tensor **ids_out = nullptr, ggml_tensor **probs_out = nullptr, bool routing_only = false, ggml_tensor *fixed_route_ids = nullptr, ggml_tensor *fixed_route_weights = nullptr, ggml_tensor **route_weights_out = nullptr);
bool qwen_moe_layer_cpu_only(const FullGraph &g, int layer);
bool build_graph(FullGraph &g, const std::vector<float> *ple_embedding);
bool sync_prefill_recurrent_state_to_host(FullGraph &g);
bool sync_decode_state_to_host(FullGraph &g);
bool prepare_prefill_state_handoff(FullGraph &source);
ggml_backend_t full_graph_cuda_scheduler_backend(FullGraph &g);
bool engine_plans_decode_expert_split(const StreamingEngine &engine);
std::unique_ptr<FullGraph> extract_decode_arena_owner( FullGraph &g, StreamingEngine &engine);
size_t estimate_prefill_workspace_bytes(int S);
void manage_decode_arena_for_prefill(StreamingEngine &engine, int S);
bool feed_decode_common_inputs(FullGraph &g, int pos_base);
bool qwen4exp_stage_layer13_before_graph(FullGraph &g, int pos_base);

// ---- process-wide state (defined in ggml_qwen4exp_graph.cpp) ----
extern std::atomic<unsigned> g_qwen4exp_host_node_cpu_active;
extern std::atomic<uint64_t> g_qwen4exp_host_node_cpu_gate_wait_ns;
extern std::atomic<unsigned> g_qwen4exp_ordinary_cpu_active;
extern std::atomic<uint64_t> g_qwen4exp_host_node_ordinary_cpu_wait_ns;
extern std::atomic<uint64_t> g_qwen4exp_cpu_split_enter_wait_ns;
extern std::mutex g_qwen4exp_cpu_execution_mutex;
extern thread_local bool g_qwen4exp_build_graph_retry_smaller_expert_layers;
// Batch size the last memory pre-check worked out would fit (0 = none); read by qwen4exp_full_graph_suggested_batch().
extern int g_qwen4exp_suggested_batch;
extern bool g_qwen4exp_prefill_op_offload; // the prefill scheduler runs host experts on the GPU (set at prepare)
// What a prefill batch leaves for the next one of the same prompt: the engine and the tokens covered so far.
extern const StreamingEngine *g_qwen4exp_cont_engine;
extern int g_qwen4exp_cont_total;
extern std::unique_ptr<FullGraph> g_graph;
extern std::atomic<uint64_t> g_full_graph_generation;
extern std::unique_ptr<FullGraph> g_graph_prefill_arena_owner;
extern std::unique_ptr<FullGraph> g_graph_prefill_state_owner;
extern std::unique_ptr<FullGraph> g_graph_decode_arena_owner;
extern bool g_config_full_graph;
extern bool g_config_full_graph_decode;
extern bool g_config_prefill_state_handoff;

// ---- types ----
// A CPY node carrying this context is the graph-visible marker for a fused
// CPU-offloaded MoE layer.  The CUDA backend sees only the opaque `extra`
// pointer and calls the bridge below; the context itself stays owned by the
// FullGraph instance for the lifetime of the captured graph.
struct Qwen4ExpCudaHostNodeContext {
  static constexpr uint64_t kMagic = 0x514f46464c4f4144ull; // QOFFLOAD
  uint64_t magic = kMagic;
  GgmlQwenCpuExpertRef layer{};
  const void *gate_plane = nullptr;
  const void *up_plane = nullptr;
  const void *down_plane = nullptr;
  int n_experts = 0;
  int d_model = 0;
  int d_ff = 0;
  int topk = 0;
  size_t expert_stride = 0;
  size_t gate_stride = 0;
  size_t up_stride = 0;
  size_t down_stride = 0;
  bool projection_compact = false;
  // CPU returns unweighted [D, topk] expert outputs; the GPU weights and
  // reduces them after the marker (saves the CPU reduce tail).
  bool gpu_reduce = false;
  // The milestone-2 callback projects an immutable per-token classification
  // into private CPU-graph inputs. The original D2H route snapshot remains
  // untouched for diagnostics and exact-once accounting.
  bool partial_cold_only = false;
  std::unique_ptr<EarlyReadyLane> partial_cpu_graph;
  std::vector<int32_t> resident_expert_index;
  std::vector<int32_t> staged_expert_index;
  std::vector<int32_t> route_snapshot_ids;
  std::vector<float> route_snapshot_weights;
  std::vector<float> route_output;
  moe_partial::RouteClassification route_classification;
  uint64_t route_slot_generation = 0;
  // The full graph owns this pool. The host-node runner borrows it for the
  // callback and never frees or pauses it.
  ggml_threadpool_t shared_cpu_threadpool = nullptr;
  int shared_cpu_threads = 0;
  GgmlQwenCpuMoeRunner *cpu_runner = nullptr;
  float *host_input = nullptr;
  float *host_output = nullptr;
  int32_t *host_ids = nullptr;
  float *host_weights = nullptr;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  size_t ids_bytes = 0;
  size_t weights_bytes = 0;
  void *device_output = nullptr;
  int layer_index = -1;
  // Immutable static hot set for a planned per-expert split.  This is
  // diagnostic-only for the current full-CPU host marker, but keeping it in
  // the captured context makes it possible to prove when the marker would
  // otherwise bypass a planned GPU branch.
  std::vector<int> gpu_resident_experts;
  bool transport_diag = false;
  float *host_output_gpu_check = nullptr;
  cudaEvent_t d2h_start = nullptr;
  cudaEvent_t d2h_end = nullptr;
  cudaEvent_t h2d_start = nullptr;
  cudaEvent_t h2d_end = nullptr;
  cudaEvent_t post_h2d_check_end = nullptr;
  std::atomic<bool> last_ok{false};
  std::atomic<double> last_cpu_ms{0.0};
  std::atomic<double> last_callback_total_ms{0.0};
  std::atomic<double> last_callback_guard_wait_ms{0.0};
  std::atomic<double> last_callback_body_ms{0.0};
  std::atomic<uint64_t> last_cpu_hash{0};
  std::atomic<uint64_t> last_gpu_hash{0};
  std::atomic<bool> last_cpu_hash_valid{false};
};

// A CPY node carrying THIS context is the graph-visible marker for a
// generalized GPU-staged decode layer (see engine.qwen4exp_gpu_stage_*).
// Unlike Qwen4ExpCudaHostNodeContext above, the callback body here does a
// pure host-memory GATHER (no CPU FFN compute, no CUDA API calls -- a hard
// requirement inside a cudaLaunchHostFunc callback) -- it copies each of
// this token's real routed experts' gate/up/down bytes, in router-output
// order, out of the preloaded DDR arena into one compact pinned bounce
// buffer, which the SAME enqueue call then H2D-copies into three fixed
// compact device destinations (gate_dst/up_dst/down_dst). Slot k of every
// destination always holds real routed expert ids[k]'s bytes, which is why
// the downstream MUL_MAT_ID nodes can use a CONSTANT [0..topk-1] ids
// tensor instead of any real per-token id remap -- see build_gpu_stage_node
// and build_moe's staged-layer branch for the full invariant.
struct Qwen4ExpGpuStageContext {
  static constexpr uint64_t kMagic = 0x51474d53544147ull; // "QGMSTAG" (trunc)
  uint64_t magic = kMagic;
  int layer_index = -1;
  int n_experts = 0;
  int topk = 0;
  // Source planes/strides into the persistent, engine-owned, independently
  // pinnable DDR arena (StreamingEngine::graph_ram_arenas[layer]) -- the
  // SAME arena load_full_weight()'s general host-resident path and Option
  // B's layer-13 staging already read from. Borrowed, not owned.
  const uint8_t *gate_plane = nullptr;
  const uint8_t *up_plane = nullptr;
  const uint8_t *down_plane = nullptr;
  size_t gate_stride = 0;
  size_t up_stride = 0;
  size_t down_stride = 0;
  size_t gate_bytes = 0; // bytes for ONE expert's gate plane
  size_t up_bytes = 0;
  size_t down_bytes = 0;
  // Pinned host buffers, sized for exactly `topk` slots each.
  int32_t *host_ids = nullptr;        // topk * sizeof(int32_t)
  // Bounded fix (see qwen4exp_cuda_gpu_stage_node_enqueue's own comment):
  // gate_dst/up_dst/down_dst are bump-allocated back-to-back from the same
  // device_expert_buffer arena (build_gpu_stage_node allocates them in that
  // exact order, nothing else allocates in between), so they form one
  // contiguous-modulo-alignment span. ONE pinned host buffer mirrors that
  // same span/gaps byte-for-byte (gate at offset 0, up at
  // combined_up_offset, down at combined_down_offset), so the H2D refresh
  // is ONE cudaMemcpyAsync instead of three -- the callback still writes
  // each projection's bytes at its own offset, unchanged in substance.
  uint8_t *host_combined_bounce = nullptr; // combined_bytes total
  size_t combined_bytes = 0;
  size_t combined_up_offset = 0;
  size_t combined_down_offset = 0;
  size_t ids_bytes = 0;
  // gate_dst's ->data -- the start of the combined contiguous span; up/down
  // are reached via combined_up_offset/combined_down_offset from here.
  void *device_combined_dst = nullptr;
  // The `ids` device tensor this layer's real routing produced this build
  // (ggml_cont'd, so a flat D2H read is safe -- decode is always S==1, so
  // the argsort-view-stride pitfall documented at build_moe's ids_out site
  // does not apply here either way, but ggml_cont keeps this robust).
  cudaEvent_t d2h_start = nullptr;
  cudaEvent_t d2h_end = nullptr;
  cudaEvent_t h2d_start = nullptr;
  cudaEvent_t h2d_end = nullptr;
  std::atomic<bool> last_ok{false};
  std::atomic<double> last_callback_ms{0.0};
};

// Decode-only router-usage tap (engine.qwen4exp_track_router_usage): every
// layer's real routed `ids` is appended (via an ordinary ggml_concat node --
// a ususal graph op, no host bridge, no manual backend pin needed since it
// inherits CUDA placement from its argsort_top_k ancestor) into ONE
// persistent [topk * n_layers] I32 device accumulator
// (FullGraph::usage_track_ids), in layer order. A SINGLE marker after the
// whole layer loop D2H-copies that one buffer and bumps
// StreamingEngine::route_freq for every (layer, expert) entry in one host
// callback -- one host-bridge dispatch per token instead of n_layers, unlike
// a naive per-layer tap. See engine.qwen4exp_track_router_usage's own
// comment for why this is decode-only (S==1): the argsort-view-stride
// pitfall documented at build_moe's ids_out site does not apply here since
// there is exactly one row per layer.
struct Qwen4ExpUsageTrackContext {
  static constexpr uint64_t kMagic = 0x515552544b475431ull; // "QURTKGT1" trunc
  uint64_t magic = kMagic;
  StreamingEngine *engine = nullptr;
  int n_layers = 0;
  int n_experts = 0;
  int topk = 0;
  int32_t *host_ids = nullptr; // pinned, n_layers * topk entries
  size_t ids_bytes = 0;
  std::atomic<double> last_callback_ms{0.0};
  // Per-layer hot-set membership (empty = layer has no static split).
  std::vector<std::vector<uint8_t>> hot_mask;
  uint64_t token_count = 0;
};

struct Qwen4ExpHostNodeCpuExecutionGuard {
  const std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lock;
  explicit Qwen4ExpHostNodeCpuExecutionGuard()
      : lock(g_qwen4exp_cpu_execution_mutex) {
    const uint64_t wait_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
    g_qwen4exp_host_node_ordinary_cpu_wait_ns.fetch_add(
        wait_ns, std::memory_order_relaxed);
  }
};

struct Qwen4ExpHostNodeCpuActivityGuard {
  Qwen4ExpHostNodeCpuActivityGuard() {
    g_qwen4exp_host_node_cpu_active.fetch_add(1, std::memory_order_acq_rel);
  }
  ~Qwen4ExpHostNodeCpuActivityGuard() {
    g_qwen4exp_host_node_cpu_active.fetch_sub(1, std::memory_order_acq_rel);
  }
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
  FullGraph() { mem_tag = "GGML-QWEN-FULL-MEM"; }

  // Chunked prefill: the tokens before this batch, and per full-attention layer the host inputs holding the K/V rows of
  // the earlier batches (filled from the engine's dense KV cache before the run).
  int pos_base = 0;
  struct PrevKv {
    int layer = -1;
    float *k = nullptr, *v = nullptr;
  };
  std::vector<PrevKv> prev_kv_inputs;

  uint64_t graph_generation = 0;
  int Hkv = 0;
  int Hk_gdn = 0;
  int Hv_gdn = 0;
  int gdn_hd = 0;
  int key_dim = 0;
  int value_dim = 0;
  int conv_dim = 0;

  ggml_backend_t cuda_raw = nullptr;
  ggml_backend_t exl3_cuda_adapter = nullptr;
  bool exl3_bridge_failed = false;
  ggml_threadpool_t cpu_threadpool = nullptr;
  int cpu_threads = 0;
  std::unique_ptr<std::ofstream> early_route_trace;
  std::unique_ptr<Qwen4ExpStageShapeClassPool> early_stage_shapes;
  std::vector<std::unique_ptr<Qwen4ExpStagePool>> early_stage_pools;
  // Per-layer CPU timing instrumentation (env: MOESTREAM_QWEN4EXP_GRAPH_
  // LAYER_TIMING=1). Uses ggml_backend_sched_set_eval_callback, which fires
  // synchronously around each CPU-backend node's real compute (ggml-cpu has
  // no async concept) -- gives real per-layer/per-op CPU time that neither
  // nsys nor the legacy per-token DEBUG counters can see for full-graph
  // decode. See memory: full-graph-per-layer-timing-instrumentation-plan.md.
  // All empty/false and zero-cost when the env var is unset.
  bool layer_timing_enabled = false;
  std::vector<double> layer_timing_cpu_ms;      // indexed by layer
  std::unordered_map<std::string, double> layer_timing_op_ms; // keyed by stem
  std::unordered_map<const ggml_tensor *,
                     std::chrono::steady_clock::time_point>
      layer_timing_t0;
  long layer_timing_tokens_since_emit = 0;
  // Gap between consecutive CPU-backend nodes' ask=false and the next
  // ask=true -- the dispatch/barrier/scheduling overhead the op-bracket
  // timing above cannot see (it only times INSIDE one node's bracket).
  std::chrono::steady_clock::time_point layer_timing_last_ask_false{};
  bool layer_timing_have_last_ask_false = false;
  double layer_timing_gap_ms_total = 0.0;
  long layer_timing_gap_count = 0;
  // One-shot verbose node/backend dump (env: MOESTREAM_QWEN4EXP_GRAPH_
  // LAYER_TIMING_VERBOSE=1) -- see env_layer_timing_verbose()'s comment.
  bool layer_timing_verbose_enabled = false;
  bool layer_timing_verbose_dumped = false;
  int layer_timing_verbose_tokens_seen = 0;
  std::chrono::steady_clock::time_point layer_timing_verbose_last{};
  bool layer_timing_verbose_have_last = false;
  // Op result tensors (mul_mat_id/swiglu outputs inside build_moe()) get
  // ggml's auto-generated "node_NNNN" name, not a "blk.N.stem" name --
  // only the persistent WEIGHT leaf tensors are explicitly named by
  // expert_weight()/etc. Recorded once at graph-build time (the graph is
  // reused across tokens, so this never needs updating per-token) so the
  // eval callback can attribute a result tensor to its layer without
  // relying on tensor names at all.
  std::unordered_map<const ggml_tensor *, std::pair<int, const char *>>
      layer_timing_layer_of; // tensor -> (layer, op stem label)
  // Opt-in Layer-0 quantization-path diagnostic. These graph-owned
  // intermediates are retained only so the first compute can dump the
  // actual gate/up values; the normal path never populates or reads them.
  // Actual graph-owned outputs for the two-layer S=1 diagnostic. These are
  // deliberately not independent cpy leaves: each tensor remains on the
  // Layer-0 -> Layer-1 dependency chain and is marked as a graph output.
  // Optional exact hidden-state snapshots at the same semantic boundary as
  // llama.cpp's l_last-0/l_last-1 callbacks: after each layer's attention and
  // FFN residual combine.  These are intentionally limited to layers 0/1.
  ggml_tensor *hidden01_snapshot[2] = {nullptr, nullptr};
  bool hidden01_dumped = false;
  // Generalizes hidden01_snapshot above to ALL layers, dumped every decode
  // step (not one-shot) -- for a full layer-by-layer, token-by-token parity
  // trace against llama.cpp when hidden01's 2-layer window isn't enough
  // to localize where two engines' outputs first diverge. Gated by
  // MOESTREAM_QWEN4EXP_ALLLAYERS_DUMP; same semantic boundary (after each
  // layer's attention+FFN residual combine) as hidden01_snapshot.
  std::vector<ggml_tensor *> all_layers_snapshot;
  // Optional one-shot raw routed-MoE output dump for Layer 13. This is
  // deliberately before the shared expert and residual combine so the
  // ordinary GGML CPU path and the host-node callback are compared at the
  // same semantic boundary.
  // MOESTREAM_QWEN4EXP_PROBE_DUMP: per-layer (FFN input, router probs) taps for offline next-layer predictor studies.
  std::vector<std::tuple<int, ggml_tensor *, ggml_tensor *>> probe_taps;
  // Diagnostic (2026-09-17): cumulative time inside resident_tensor()'s own
  // cudaMemcpy H2D calls, across this ONE build_graph() call -- see memory:
  // full-graph-decode-graph-construction-5s-latency.md. Reset to 0 at the
  // top of build_graph(), read+logged at the bottom, alongside the
  // node_construction/sched_alloc_graph split already there.
  double dense_upload_ms_total = 0.0;
  // Same idea, for bind_device_expert()'s own cudaMemcpy (the 17.68 GiB,
  // 22-GPU-resident-layer upload -- the other real candidate for where
  // node_construction's time goes, besides dense_upload_ms_total above).
  double expert_upload_ms_total = 0.0;
  long expert_upload_calls = 0;
  // Diagnostic (2026-09-17): further split of node_construction beyond the
  // cudaMemcpy timers above, which only accounted for ~28-44% of the total.
  // Buckets the REST of the per-layer node-construction work (host-side
  // ggml_tensor/op building, not any device transfer) into attention-block
  // vs MoE-block, to find where the remaining ~55-72% actually goes.
  double attn_construction_ms_total = 0.0;
  double moe_construction_ms_total = 0.0;
  // Finer split of moe_construction_ms_total: time inside the weight-tensor
  // lookup calls (expert_weight_split/expert_weight/expert_gate_up_weight --
  // model_tensor lookup + preloaded_ram_expert_source + bind_host/
  // bind_device_expert*) vs build_moe()'s own ggml graph-node creation.
  double moe_weight_lookup_ms_total = 0.0;
  // Split the weight-lookup bucket into metadata/source binding and backend
  // arena binding. The H2D copy itself remains in expert_upload_ms_total.
  double expert_weight_meta_ms_total = 0.0;
  double expert_weight_source_ms_total = 0.0;
  double expert_weight_host_bind_ms_total = 0.0;
  double expert_weight_device_bind_ms_total = 0.0;
  double expert_weight_load_ms_total = 0.0;
  double expert_weight_split_ms_total = 0.0;
  // Split the VRAM-only cold graph source path: aligned host staging alloc
  // and GGUF pread before the separately measured H2D upload. This path is
  // used for layers intentionally excluded from DDR expert arenas.
  double expert_vram_stage_alloc_ms_total = 0.0;
  double expert_vram_file_read_ms_total = 0.0;
  uint64_t expert_vram_file_read_bytes = 0;
  long expert_vram_file_read_calls = 0;
  long expert_weight_host_bind_calls = 0;
  long expert_weight_device_bind_calls = 0;
  bool device_expert_partial_gate_up = false;
  // When true, the device_expert_layers-sized GPU-resident block sits at
  // the END of the model (layers [n_layers-N, n_layers)) instead of the
  // START (layers [0, N)). Added specifically to match llama.cpp's own
  // `-ncmoe N` convention: llama-bench forces layers 0..N-1 to CPU via
  // llm_ffn_block_regex(i, ...) for i in [0,N), which leaves the LAST
  // n_layers-N layers GPU-resident -- the opposite end from MoEStream's
  // historical leading-N convention. Every same-N comparison run before
  // this flag existed compared GPU residency on genuinely different
  // layers between the two engines, not just a different count. See
  // [[full-graph-mixed-decode-root-cause-cpu-backend-no-events]]'s FIFTH
  // ROUND for the discovery. device_expert_partial_gate_up's boundary-
  // layer optimization is forced off whenever this is set (see its own
  // setter) rather than generalizing its own boundary math -- v1 scope.
  bool device_expert_trailing = false;
  // Per-layer plan produced once by plan_expert_vram_split(), BEFORE
  // init_device_expert_arena(): layer -> absolute expert ids chosen for
  // VRAM residency (the hottest-by-usage subset a fixed VRAM budget could
  // afford), for layers cpu_affinity_layers/device_expert_layers did not
  // already claim. Empty/absent entry = no split for that layer (today's
  // behavior, unchanged). v1 scope: decode only (S==1) -- see build_moe.
  std::unordered_map<int, std::vector<int>> split_gpu_expert_ids;
  // Phase 2 suffix graphs borrow the immutable VRAM expert arena from the
  // prefix owner.  Expert tensors are keyed by their source layout because
  // the same logical layer can be represented as a full, strided, gathered,
  // or compact gate/up tensor depending on the graph path.
  FullGraph *expert_source_graph = nullptr;
  std::unordered_map<std::string, void *> expert_addrs;
  // Layer-13 staging proof. The graph keeps fixed compact GPU tensors; each
  // decode token refreshes one or more cold expert records from the
  // active-layer pool before graph replay.
  bool stage_enabled = false;
  bool stage_route_logged = false;
  int stage_layer = -1;
  int stage_expert = -1;
  int stage_local_index = -1;
  struct StageBinding {
    int expert = -1;
    int local_index = -1;
    const uint8_t *src_gate = nullptr;
    const uint8_t *src_up = nullptr;
    const uint8_t *src_down = nullptr;
    size_t gate_bytes = 0;
    size_t up_bytes = 0;
    size_t down_bytes = 0;
    uint8_t *dst_gate = nullptr;
    uint8_t *dst_up = nullptr;
    uint8_t *dst_down = nullptr;
    ExpertLocation *slot = nullptr;
  };
  std::vector<int> stage_experts;
  std::vector<StageBinding> stage_bindings;
  std::vector<ExpertLocation *> stage_slots;
  // Stable per-layer contexts for the opt-in CUDA host-node bridge. These
  // must outlive graph capture and every replay; they are cleared only after
  // the scheduler/graph objects have been torn down.
  std::vector<std::unique_ptr<Qwen4ExpCudaHostNodeContext>> host_node_contexts;
  // Reusable exact-cold CPU branch for the Phase-2 parent graph. Its host node
  // is inserted directly into the parent graph, alongside (never inside) the
  // resident/staged conditional bodies.
  std::vector<std::unique_ptr<Exl3GgmlBridgeLayer>> exl3_bridge_layers;
  // Generalized GPU staging for host-resident decode layers (see
  // engine.qwen4exp_gpu_stage_enabled/_layers). One binding per staged
  // layer: a fixed, topk-sized compact gate/up/down destination in the
  // shared device_expert_buffer bump arena (see device_expert_bytes()'s own
  // staging-bytes block), refreshed every decode token via the same
  // CUDA-graph host-node-callback technique the CPU-offload bridge above
  // uses -- but the callback here does a pure host-memory GATHER (real
  // routed experts' bytes copied into a compact pinned bounce buffer) in
  // place of CPU FFN compute, feeding ordinary GPU MUL_MAT_ID nodes instead
  // of producing the final answer directly. See build_gpu_stage_node().
  struct GpuStageBinding {
    int layer = -1;
    // [topk, S] I32, value = 0..topk-1, uploaded once at graph-build time,
    // never touched again -- see build_gpu_stage_node's own comment for why
    // a real per-token id remap table is unnecessary.
    ggml_tensor *ids_dispatch_const = nullptr;
    // Fixed compact device tensors, shape [cols, rows, topk] (topk, not
    // n_experts), bound once via bind_or_allocate_expert into
    // expert.buffer -- never re-bound, only their CONTENTS change
    // per token.
    ggml_tensor *gate_dst = nullptr;
    ggml_tensor *up_dst = nullptr;
    ggml_tensor *down_dst = nullptr;
    // The GGML_OP_CPY marker node that refreshes gate/up/down_dst's
    // contents each replay. build_moe attaches this as an extra src[] on
    // every MUL_MAT_ID call that reads them, purely to force correct
    // dependency ordering -- see build_gpu_stage_node's own comment.
    ggml_tensor *gpu_stage_marker_dep = nullptr;
  };
  std::vector<GpuStageBinding> gpu_stage_bindings;
  std::vector<std::unique_ptr<struct Qwen4ExpGpuStageContext>>
      gpu_stage_contexts;
  // Decode-only router-usage tap (engine.qwen4exp_track_router_usage) -- see
  // Qwen4ExpUsageTrackContext's own comment. usage_track_ids accumulates
  // every layer's real routed ids via ordinary ggml_concat nodes across the
  // layer loop; usage_track_context owns the single flush marker's pinned
  // buffer/state, built once after the loop.
  ggml_tensor *usage_track_ids = nullptr;
  std::unique_ptr<struct Qwen4ExpUsageTrackContext> usage_track_context;
  const uint8_t *stage_src_gate = nullptr;
  const uint8_t *stage_src_up = nullptr;
  const uint8_t *stage_src_down = nullptr;
  size_t stage_gate_bytes = 0;
  size_t stage_up_bytes = 0;
  size_t stage_down_bytes = 0;
  uint8_t *stage_dst_gate = nullptr;
  uint8_t *stage_dst_up = nullptr;
  uint8_t *stage_dst_down = nullptr;
  ExpertLocation *stage_slot = nullptr;
  cudaEvent_t stage_wait_done_event = nullptr;
  cudaEvent_t stage_wait_start_event = nullptr;
  cudaEvent_t stage_d2d_start_event = nullptr;
  cudaEvent_t stage_d2d_end_event = nullptr;
  cudaEvent_t stage_compute_start_event = nullptr;
  cudaEvent_t stage_compute_end_event = nullptr;
  ggml_tensor *stage_route_ids = nullptr;
  // Phase 1 partial-layer dispatch: prefill-only, non-null exactly when
  // this graph is building engine->qwen4exp_partial_layer's MoE block
  // during a non-decode (S>1) build. Marked as a graph output so
  // qwen4exp_full_graph_run can read back this prompt's real routing for
  // that layer and decide, once, whether decode may commit to the
  // GPU-only path. Shape [topk, S] I32, same layout ggml_argsort_top_k
  // itself produces.
  ggml_tensor *partial_layer_ids_out = nullptr;
  // Diagnostic sibling of partial_layer_ids_out: the partial layer's raw
  // post-softmax router distribution ([n_experts, S]), for checking
  // whether routing is genuinely peaked or effectively near-uniform noise.
  ggml_tensor *partial_layer_probs_out = nullptr;
  // Phase-2 prefix: selected, post-normalization router weights in original
  // route-slot order. Resident, staged and CPU continuations reuse these
  // values so each selected route contributes with the exact prefix weight.
  ggml_tensor *partial_layer_weights_out = nullptr;
  // Root-cause diagnostic (2026-09-18): the raw exp_probs_b.bias tensor
  // for the partial layer, exposed as a graph output so
  // qwen4exp_full_graph_run can dump its actual values -- see
  // [[qwen4exp-legacy-vs-fullgraph-routing-divergence]].
  ggml_tensor *partial_layer_bias_out = nullptr;

  // Phase 2: true per-token GPU/CPU dispatch for the partial layer,
  // validated via a standalone spike test (splitting a computation across
  // two separately-submitted ggml graphs with a persistent device-tensor
  // handoff -- see qwen4exp-full-graph-decode-benchmark-baseline memory's
  // follow-on note). Off (Full) is byte-identical to today's single-graph
  // decode for every other config; Prefix/SuffixGpu/SuffixCpu are three
  // DISTINCT FullGraph instances (own ctx/graph/sched) built once at
  // decode-prepare time, chosen per token by a cheap host-side readback of
  // the prefix's own layer routing.
  // Prefix-only: the persistent handoff buffer this instance writes `res`
  // (the residual right after the split layer's attention) into. Suffix
  // instances read the prefix instance's own phase2_handoff_buf/addr
  // directly (the prefix instance IS g_graph, phase2_mode==Prefix -- see
  // the g_graph global's own comment) and bind a fresh leaf tensor to the
  // SAME address in their own ctx -- never touch this pair themselves.
  // Host mirrors used only by the CPU continuation.  Route/HC publication
  // buffers are CUDA-owned for GPU children; CPU MUL_MAT_ID and HC nodes must
  // receive explicit D2H materializations instead of reading those device
  // addresses through a mixed scheduler edge.
  // Frozen split-layer HC/router inputs. The prefix publishes these once;
  // every Phase-2 branch aliases the same FFN input, injection vector, route
  // IDs and normalized route weights for this layer invocation.
  // SplitGpu/SplitStaged write the post-FFN residual here; Tail starts from
  // this address after the conditional child has completed.
  // SuffixGpu and SuffixCpu build the SAME layer range [split_layer+1,
  // n_layers) and therefore issue new_device_state_tensor() calls in an
  // IDENTICAL sequence (GDN/dense-attn/PLE classification per layer is a
  // static model property, not decode-time data) -- but only ONE of them
  // runs per token, so their GDN/KV/PLE state for those shared layers
  // must be the SAME physical buffer, or it would silently diverge the
  // moment dispatch alternates between them. The instance built FIRST
  // (SuffixGpu) allocates for real and records each address in
  // state_addrs, in call order; the instance built SECOND
  // (SuffixCpu, via state_source_graph pointing at the first) replays
  // that list instead of allocating its own arena slots, binding a fresh
  // leaf in ITS OWN ctx to each recorded address in turn.
  FullGraph *state_source_graph = nullptr;
  std::vector<void *> state_addrs;
  size_t state_replay_cursor = 0;
  // A decode graph may borrow the recurrent/PLE state outputs of the
  // completed S>1 prefill graph directly.  The prefill scheduler remains
  // alive while those output buffers are borrowed; KV rows are left null in
  // this table because a decode graph's larger KV capacity cannot alias the
  // small S-token prefill allocation and is seeded from the host cache.
  bool state_handoff_ready = false;
  bool prefill_host_state_deferred = false;
  bool decode_host_state_dirty = false;
  bool prefill_device_state_inputs = false;
  std::vector<void *> state_handoff_addrs;
  std::vector<ggml_backend_buffer_t> state_handoff_buffers;
  size_t state_handoff_cursor = 0;
  std::unordered_set<ggml_tensor *> state_handoff_tensors;
  // Phase 2 sibling graphs use the same read-only dense-weight arena.  The
  // source graph owns the CUDA buffer; borrowers bind their own ggml tensor
  // handles to the source addresses.  Unlike the state arena above, dense
  // weights are keyed by model tensor name because prefix and suffix graphs
  // do not materialize the same layer range.
  FullGraph *dense_source_graph = nullptr;
  std::unordered_map<std::string, void *> dense_addrs;
  // Phase-2 continuations read immutable route/input outputs from the prefix
  // graph. Aliases are backed by its persistent output buffers and are
  // consumed only after the prefix has completed for this token.
  struct CompletionPayloadBinding {
    void *gate = nullptr;
    void *up = nullptr;
    void *down = nullptr;
    size_t gate_bytes = 0;
    size_t up_bytes = 0;
    size_t down_bytes = 0;
    size_t gate_stride = 0;
    size_t up_stride = 0;
    size_t down_stride = 0;
    void *staged_route_mask = nullptr;
    void *staged_dispatch_ids = nullptr;
    ggml_tensor *staged_route_mask_tensor = nullptr;
    ggml_tensor *staged_dispatch_ids_tensor = nullptr;
    int topk = 0;
    int layer = -1;
  } completion_payload_binding;

  ggml_tensor *topology_cpu_prologue = nullptr;
  ggml_tensor *ple_embedding = nullptr;
  ggml_tensor *ple_hist_out = nullptr;
  ggml_tensor *ple_hist_storage = nullptr;
  ggml_tensor *ple_hist_update = nullptr;

  std::vector<ggml_tensor *> kv_k;
  std::vector<ggml_tensor *> kv_v;
  std::vector<int> kv_layers;
  std::vector<ggml_tensor *> gdn_state_out;
  std::vector<ggml_tensor *> gdn_hist_out;
  std::vector<ggml_tensor *> kv_state_k;
  std::vector<ggml_tensor *> kv_state_v;
  std::vector<ggml_tensor *> gdn_state_storage;
  std::vector<ggml_tensor *> gdn_hist_storage;
  std::vector<ggml_tensor *> gdn_state_updates;
  std::vector<ggml_tensor *> gdn_hist_updates;
  std::vector<int> gdn_layers;
  std::vector<float *> gdn_state_inputs;
  std::vector<float *> gdn_hist_inputs;
  int ple_layer = -1;
  float *ple_hist_input = nullptr;


  size_t graph_bytes = 0;
  // Bytes/views borrowed from StreamingEngine's already-populated graph-RAM
  // arenas.  These are not owned by this graph and must not be counted as a
  // second host-side expert copy.
  size_t host_ram_alias_bytes = 0;
  size_t host_ram_alias_tensors = 0;
  // ggml_init(no_alloc=true) still allocates a host-side context arena. Keep
  // that arena in the process-wide admission ledger as well, otherwise graph
  // construction can exceed host_ram_gb between explicit host-buffer
  // allocations.
  // The source admission is made once before graph construction, while this
  // graph materializes that source as many individual host allocations.  This
  // remainder prevents alloc_host() from charging those same bytes twice.

  ~FullGraph() { destroy(); }

  void destroy() {
    if (sched) {
      // qwen4exp_full_graph_run synchronizes before returning.  Avoid calling
      // into the borrowed CUDA backend during process teardown: the shared
      // backend can already be in CUDA runtime shutdown at this point.
      ggml_backend_sched_free(sched);
      sched = nullptr;
    }
    if (exl3_cuda_adapter) {
      ggml_backend_free(exl3_cuda_adapter);
      exl3_cuda_adapter = nullptr;
    }
    if (early_stage_shapes) early_stage_shapes->shutdown();
    early_stage_pools.clear();
    early_stage_shapes.reset();
    early_ready.clear();
    early_route_trace.reset();
    // Host-node CPU runners borrow this graph-owned pool.  Detach them after
    // the scheduler is gone, but before the pool itself is destroyed.  The
    // runner's graph plan is synchronous, so no callback can still be using
    // it once graph teardown begins.
    for (auto &host_node : host_node_contexts) {
      if (host_node && host_node->cpu_runner &&
          host_node->shared_cpu_threadpool) {
        host_node->cpu_runner->unbind_external_threadpool(
            host_node->shared_cpu_threadpool);
        host_node->cpu_runner = nullptr;
      }
    }
    if (stage_wait_done_event || stage_wait_start_event ||
        stage_d2d_start_event ||
        stage_d2d_end_event || stage_compute_start_event ||
        stage_compute_end_event) {
      cudaSetDevice(device);
      if (stage_wait_done_event)
        cudaEventDestroy(stage_wait_done_event);
      if (stage_wait_start_event)
        cudaEventDestroy(stage_wait_start_event);
      if (stage_d2d_start_event)
        cudaEventDestroy(stage_d2d_start_event);
      if (stage_d2d_end_event)
        cudaEventDestroy(stage_d2d_end_event);
      if (stage_compute_start_event)
        cudaEventDestroy(stage_compute_start_event);
      if (stage_compute_end_event)
        cudaEventDestroy(stage_compute_end_event);
    }
    stage_wait_done_event = nullptr;
    stage_wait_start_event = nullptr;
    stage_d2d_start_event = nullptr;
    stage_d2d_end_event = nullptr;
    stage_compute_start_event = nullptr;
    stage_compute_end_event = nullptr;
    stage_slot = nullptr;
    stage_slots.clear();
    stage_bindings.clear();
    stage_experts.clear();
    stage_route_ids = nullptr;
    stage_enabled = false;
    if (cpu && cpu_threadpool)
      ggml_backend_cpu_set_threadpool(cpu, nullptr);
    if (cpu_threadpool) {
      ggml_threadpool_free(cpu_threadpool);
      cpu_threadpool = nullptr;
    }
    if (cpu) {
      ggml_backend_free(cpu);
      cpu = nullptr;
    }
    if (expert.buffer && !expert_source_graph) {
      ggml_backend_buffer_free(expert.buffer);
    }
    expert.buffer = nullptr;
    expert.base = nullptr;
    expert.capacity = 0;
    expert.used = 0;
    if (dense.buffer && !dense_source_graph) {
      ggml_backend_buffer_free(dense.buffer);
    }
    dense.buffer = nullptr;
    dense.base = nullptr;
    dense.capacity = 0;
    dense.used = 0;
    dense.tensors = 0;
    if (state.buffer) {
      ggml_backend_buffer_free(state.buffer);
      state.buffer = nullptr;
    }
    state.base = nullptr;
    state.capacity = 0;
    state.used = 0;
    device_expert_layers = 0;
    device_expert_partial_gate_up = false;
    split_gpu_expert_ids.clear();
    expert_source_graph = nullptr;
    expert_addrs.clear();
    partial_layer_ids_out = nullptr;
    partial_layer_probs_out = nullptr;
    partial_layer_weights_out = nullptr;
    state_source_graph = nullptr;
    state_addrs.clear();
    state_replay_cursor = 0;
    state_handoff_ready = false;
    prefill_host_state_deferred = false;
    decode_host_state_dirty = false;
    prefill_device_state_inputs = false;
    state_handoff_addrs.clear();
    state_handoff_buffers.clear();
    state_handoff_cursor = 0;
    state_handoff_tensors.clear();
    dense_source_graph = nullptr;
    dense_addrs.clear();
    for (auto &host_node : host_node_contexts) {
      if (!host_node)
        continue;
      if (host_node->d2h_start)
        cudaEventDestroy(host_node->d2h_start);
      if (host_node->d2h_end)
        cudaEventDestroy(host_node->d2h_end);
      if (host_node->h2d_start)
        cudaEventDestroy(host_node->h2d_start);
      if (host_node->h2d_end)
        cudaEventDestroy(host_node->h2d_end);
      if (host_node->post_h2d_check_end)
        cudaEventDestroy(host_node->post_h2d_check_end);
      host_node->d2h_start = nullptr;
      host_node->d2h_end = nullptr;
      host_node->h2d_start = nullptr;
      host_node->h2d_end = nullptr;
      host_node->post_h2d_check_end = nullptr;
    }
    host_node_contexts.clear();
    exl3_bridge_layers.clear();
    release_core(); // host buffers, context, owned host memory, counters
    cpu_threads = 0;
    ple_embedding = nullptr;
    ple_hist_out = nullptr;
    ple_hist_storage = nullptr;
    ple_hist_update = nullptr;
    kv_k.clear();
    kv_v.clear();
    kv_layers.clear();
    gdn_state_out.clear();
    gdn_hist_out.clear();
    kv_state_k.clear();
    kv_state_v.clear();
    gdn_state_storage.clear();
    gdn_hist_storage.clear();
    gdn_state_updates.clear();
    gdn_hist_updates.clear();
    gdn_layers.clear();
    gdn_state_inputs.clear();
    gdn_hist_inputs.clear();
    ple_layer = -1;
    ple_hist_input = nullptr;
    graph_bytes = 0;
    host_ram_alias_bytes = 0;
    host_ram_alias_tensors = 0;
  }
};

enum class ExistingDeviceExpertBind { Missing, Bound, Failed };

struct ScopedGraphBuildTimer {
  double &total_ms;
  const std::chrono::steady_clock::time_point started =
      std::chrono::steady_clock::now();

  explicit ScopedGraphBuildTimer(double &total) : total_ms(total) {}
  ~ScopedGraphBuildTimer() {
    total_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started)
                    .count();
  }
};

struct SplitExpertWeight {
  ggml_tensor *cpu = nullptr; // full [cols,rows,n_expert] -- today's tensor,
                              // unchanged, used either as-is (no split) or
                              // as build_moe's "CPU branch" input (split).
  ggml_tensor *gpu = nullptr; // [cols,rows,M], only for a genuine split.
};

// Builds the two small I32/F32 constant lookup tables a genuinely split
// layer's mul_mat_id pair needs, once, at graph-build time (never
// refreshed per decode step -- these encode a STATIC expert->device
// assignment, exactly like device_expert_layers already is). Shapes are
// [1, n_experts, 1] to satisfy ggml_get_rows' a->ne[2]==b->ne[1] contract
// against `ids`' own [topk, S] shape with S==1 (decode-only, enforced by
// this function's only caller). remap_gpu[e] = local index in [0,M) for a
// hot expert and -1 for a cold one; remap_cpu[e] = e for a cold expert and
// -1 for a hot one. GGML MUL_MAT_ID treats negative ids as inactive route
// slots, so each backend skips the other backend's routes before the matmul.
// mask_gpu/mask_cpu remain as a defensive numerical guard at the merge.
struct ExpertSplitTables {
  ggml_tensor *remap_gpu = nullptr;
  ggml_tensor *remap_cpu = nullptr;
  ggml_tensor *mask_gpu = nullptr;
  ggml_tensor *mask_cpu = nullptr;
};


} // namespace qwen4exp_graph

#endif // MOESTREAM_USE_GGML_CUDA

