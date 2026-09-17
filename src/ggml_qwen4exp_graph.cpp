// Real multi-layer Qwen4Exp GGML graph.
//
// This is intentionally a separate bridge rather than a rewrite of the
// established execution path.  Llama.cpp's important property is that the
// C++ caller constructs one dependency graph for the entire trunk and then
// gives it to ggml_backend_sched_graph_compute_async().  The scheduler may
// split that graph internally around host-backed expert weights, but the
// caller does not return to C++ after every layer.  This file reproduces that
// shape for MoEStream's model representation.

#include "ggml_qwen4exp_graph.hpp"

#include "dense_dispatch.hpp" // is_cpu_affinity_layer (cpu_affinity_layers config)
#include "direct_io.hpp"
#include "engine.hpp"
#include "ggml_moe_backend.hpp"
#include "kquant_dequant.hpp"
#include "logger.hpp"
#include "memory_guard.hpp"
#include "ple_hash.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(MOESTREAM_USE_GGML_CUDA)

#include <cuda_runtime.h>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

constexpr int kGraphSize = 50000;
constexpr size_t kTensorContextBytes = 128ull * 1024ull * 1024ull;

struct HostAllocation {
  void *ptr = nullptr;
  size_t bytes = 0;
  bool pinned = false;
  size_t reservation_bytes = 0;
};

struct FullGraph {
  StreamingEngine *engine = nullptr;
  int device = -1;
  int S = 0;
  int D = 0;
  int HC = 0;
  int n_layers = 0;
  int H = 0;
  int Hkv = 0;
  int hd = 0;
  int Hk_gdn = 0;
  int Hv_gdn = 0;
  int gdn_hd = 0;
  int key_dim = 0;
  int value_dim = 0;
  int conv_dim = 0;
  int d_conv = 0;
  bool decode_mode = false;
  int kv_capacity = 0;
  int decode_last_pos = -1;

  ggml_backend_t cuda = nullptr;
  ggml_backend_t cpu = nullptr;
  ggml_threadpool_t cpu_threadpool = nullptr;
  int cpu_threads = 0;
  ggml_backend_sched_t sched = nullptr;
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
  // Op result tensors (mul_mat_id/swiglu outputs inside build_moe()) get
  // ggml's auto-generated "node_NNNN" name, not a "blk.N.stem" name --
  // only the persistent WEIGHT leaf tensors are explicitly named by
  // expert_weight()/etc. Recorded once at graph-build time (the graph is
  // reused across tokens, so this never needs updating per-token) so the
  // eval callback can attribute a result tensor to its layer without
  // relying on tensor names at all.
  std::unordered_map<const ggml_tensor *, std::pair<int, const char *>>
      layer_timing_layer_of; // tensor -> (layer, op stem label)
  ggml_backend_buffer_t device_expert_buffer = nullptr;
  void *device_expert_base = nullptr;
  size_t device_expert_capacity = 0;
  size_t device_expert_used = 0;
  ggml_backend_buffer_t device_dense_buffer = nullptr;
  void *device_dense_base = nullptr;
  size_t device_dense_capacity = 0;
  size_t device_dense_used = 0;
  int device_dense_tensors = 0;
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
  ggml_backend_buffer_t device_state_buffer = nullptr;
  void *device_state_base = nullptr;
  size_t device_state_capacity = 0;
  size_t device_state_used = 0;
  int device_expert_layers = 0;
  bool device_expert_partial_gate_up = false;
  // When true, the device_expert_layers-sized GPU-resident block sits at
  // the END of the model (layers [n_layers-N, n_layers)) instead of the
  // START (layers [0, N)). Added specifically to match beellama's own
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
  ggml_context *ctx = nullptr;
  ggml_cgraph *graph = nullptr;

  ggml_tensor *input = nullptr;         // [D, HC, S], CPU input
  ggml_tensor *topology_cpu_prologue = nullptr;
  ggml_tensor *output = nullptr;        // [D, S], final HC output
  ggml_tensor *logits = nullptr;        // [vocab, S], decode-only output
  ggml_tensor *final_streams = nullptr; // [D, HC, S], for state/debug writeback
  ggml_tensor *positions = nullptr;     // [S], I32
  ggml_tensor *attn_mask = nullptr;     // [S, S], F16 (flash-attn contract)
  ggml_tensor *decode_indices = nullptr; // [Hkv], I32 row indices for KV append
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

  std::vector<HostAllocation> owned;
  std::vector<ggml_backend_buffer_t> host_buffers;
  std::vector<void *> host_inputs;

  size_t graph_bytes = 0;
  size_t host_weight_bytes = 0;
  // Bytes/views borrowed from StreamingEngine's already-populated graph-RAM
  // arenas.  These are not owned by this graph and must not be counted as a
  // second host-side expert copy.
  size_t host_ram_alias_bytes = 0;
  size_t host_ram_alias_tensors = 0;
  size_t host_pinned_bytes = 0;
  size_t host_pin_failures = 0;
  size_t host_reservation_bytes = 0;
  // ggml_init(no_alloc=true) still allocates a host-side context arena. Keep
  // that arena in the process-wide admission ledger as well, otherwise graph
  // construction can exceed host_ram_gb between explicit host-buffer
  // allocations.
  size_t context_reservation_bytes = 0;
  // The source admission is made once before graph construction, while this
  // graph materializes that source as many individual host allocations.  This
  // remainder prevents alloc_host() from charging those same bytes twice.
  size_t host_precharged_bytes = 0;
  bool ready = false;

  ~FullGraph() { destroy(); }

  void destroy() {
    if (sched) {
      // qwen4exp_full_graph_run synchronizes before returning.  Avoid calling
      // into the borrowed CUDA backend during process teardown: the shared
      // backend can already be in CUDA runtime shutdown at this point.
      ggml_backend_sched_free(sched);
      sched = nullptr;
    }
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
    if (device_expert_buffer) {
      ggml_backend_buffer_free(device_expert_buffer);
      device_expert_buffer = nullptr;
    }
    device_expert_base = nullptr;
    device_expert_capacity = 0;
    device_expert_used = 0;
    if (device_dense_buffer) {
      ggml_backend_buffer_free(device_dense_buffer);
      device_dense_buffer = nullptr;
    }
    device_dense_base = nullptr;
    device_dense_capacity = 0;
    device_dense_used = 0;
    device_dense_tensors = 0;
    if (device_state_buffer) {
      ggml_backend_buffer_free(device_state_buffer);
      device_state_buffer = nullptr;
    }
    device_state_base = nullptr;
    device_state_capacity = 0;
    device_state_used = 0;
    device_expert_layers = 0;
    device_expert_partial_gate_up = false;
    split_gpu_expert_ids.clear();
    for (ggml_backend_buffer_t b : host_buffers) {
      if (b)
        ggml_backend_buffer_free(b);
    }
    host_buffers.clear();
    if (ctx) {
      ggml_free(ctx);
      ctx = nullptr;
    }
    for (HostAllocation &a : owned) {
      if (a.pinned)
        cudaHostUnregister(a.ptr);
      if (a.ptr)
        std::free(a.ptr);
      MemoryGuard::release_host_bytes(a.reservation_bytes);
      a.ptr = nullptr;
      a.pinned = false;
    }
    owned.clear();
    host_inputs.clear();
    cuda = nullptr;
    cpu = nullptr;
    cpu_threads = 0;
    graph = nullptr;
    input = output = logits = final_streams = positions = attn_mask =
        decode_indices = ple_embedding = nullptr;
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
    ready = false;
    graph_bytes = 0;
    host_weight_bytes = 0;
    host_ram_alias_bytes = 0;
    host_ram_alias_tensors = 0;
    host_pinned_bytes = 0;
    host_pin_failures = 0;
    MemoryGuard::release_host_bytes(host_reservation_bytes);
    host_reservation_bytes = 0;
    MemoryGuard::release_host_bytes(context_reservation_bytes);
    context_reservation_bytes = 0;
    host_precharged_bytes = 0;
    engine = nullptr;
    S = 0;
    decode_mode = false;
    kv_capacity = 0;
    decode_last_pos = -1;
  }
};

std::unique_ptr<FullGraph> g_graph;

bool env_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOESTREAM_QWEN4EXP_FULL_GRAPH");
    return v && v[0] == '1';
  }();
  return enabled;
}

int env_device_expert_layers() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_LAYERS");
  if (!v || !v[0])
    return 0;
  char *end = nullptr;
  long n = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || n <= 0)
    return 0;
  return static_cast<int>(std::min<long>(n, 48));
}

// Matches beellama's `-ncmoe N` convention (GPU-resident block at the END
// of the model) instead of MoEStream's historical leading-N convention.
// See FullGraph::device_expert_trailing's own comment.
bool env_device_expert_trailing() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_TRAILING");
  return v && v[0] == '1';
}

// See FullGraph::layer_timing_enabled's own comment.
bool env_layer_timing_enabled() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_GRAPH_LAYER_TIMING");
  return v && v[0] == '1';
}

// The merged gate/up representation is intentionally runtime-switchable while
// its CUDA and CPU kernel cost is being compared with the established
// three-projection graph.  It is opt-in until the doubled-row MUL_MAT_ID path
// matches the separate projection path; setting this to 1 enables it without
// changing placement or routing.
bool env_merged_gate_up_enabled() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_GRAPH_MERGED_GATE_UP");
  return v && v[0] == '1';
}

int layer_timing_parse_layer(const char *name);

// Parses the "blk.<N>." prefix ggml_qwen4exp_graph.cpp's own tensor names
// always use (see expert_weight/dense_weight/etc.) -- returns -1 if the
// name doesn't match (dense/global tensors with no per-layer index, e.g.
// the embedding or output head). Mirrors load_full_weight's own parsing.
int layer_timing_parse_layer(const char *name) {
  if (!name || std::strncmp(name, "blk.", 4) != 0)
    return -1;
  char *end = nullptr;
  const long n = std::strtol(name + 4, &end, 10);
  if (end == name + 4 || !end || *end != '.')
    return -1;
  return static_cast<int>(n);
}

// ggml_backend_sched_eval_callback: fires with ask=true immediately before
// a node runs, ask=false immediately after. For CPU-backend nodes this
// synchronously brackets real compute time (ggml-cpu has no async
// concept); for CUDA-backend nodes it would only bracket the async launch,
// so those are skipped entirely -- GPU-side timing already exists from the
// nsys work elsewhere in this investigation. See FullGraph::layer_timing_*
// fields and the memory note this feature's plan came from.
bool layer_timing_eval_callback(struct ggml_tensor *t, bool ask,
                                void *user_data) {
  FullGraph *g = static_cast<FullGraph *>(user_data);
  if (!g || !t)
    return true;
  if (ggml_backend_sched_get_tensor_backend(g->sched, t) != g->cpu)
    return true; // GPU node or unassigned -- not timeable this way, skip
  const auto now = std::chrono::steady_clock::now();
  if (ask) {
    // Gap since the LAST CPU node's ask=false -- dispatch/barrier/
    // scheduling overhead the op-bracket timing below cannot see, since
    // it only covers what happens INSIDE one node's own bracket.
    if (g->layer_timing_have_last_ask_false) {
      g->layer_timing_gap_ms_total +=
          std::chrono::duration<double, std::milli>(
              now - g->layer_timing_last_ask_false)
              .count();
      ++g->layer_timing_gap_count;
    }
    g->layer_timing_t0[t] = now;
    return true;
  }
  g->layer_timing_last_ask_false = now;
  g->layer_timing_have_last_ask_false = true;
  auto it = g->layer_timing_t0.find(t);
  if (it == g->layer_timing_t0.end())
    return true; // no matching ask=true (shouldn't happen, be defensive)
  const double ms =
      std::chrono::duration<double, std::milli>(now - it->second).count();
  g->layer_timing_t0.erase(it);
  // Op result tensors (mul_mat_id/swiglu outputs) get ggml's auto "node_
  // NNNN" name, not a "blk.N.stem" one -- only build_moe()'s own explicit
  // layer_timing_layer_of recording (populated at graph-build time, see
  // its own comment) can attribute these correctly. Fall back to name
  // parsing for anything build_moe() didn't tag (e.g. weight leaves, if
  // this callback ever fires on one) so the op-level map still groups
  // something meaningful rather than silently dropping it.
  auto tag_it = g->layer_timing_layer_of.find(t);
  const bool tagged = tag_it != g->layer_timing_layer_of.end();
  const int layer = tagged ? tag_it->second.first
                           : layer_timing_parse_layer(ggml_get_name(t));
  if (layer >= 0 && layer < static_cast<int>(g->layer_timing_cpu_ms.size()))
    g->layer_timing_cpu_ms[static_cast<size_t>(layer)] += ms;
  std::string stem;
  if (tagged) {
    stem = tag_it->second.second;
  } else {
    const char *name = ggml_get_name(t);
    stem = name;
    if (layer >= 0) {
      // Strip the "blk.<N>." prefix so the op-level map groups by stem
      // across all layers, e.g. "ffn_down_exps.weight" regardless of
      // which layer -- same convention expert_weight() itself uses.
      const char *dot = std::strchr(name + 4, '.');
      if (dot && dot[1] != '\0')
        stem = dot + 1;
    }
  }
  g->layer_timing_op_ms[stem] += ms;
  return true;
}

bool env_decode_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOESTREAM_QWEN4EXP_FULL_GRAPH_DECODE");
    return v && v[0] == '1';
  }();
  return enabled;
}

int env_decode_kv_capacity(int max_t, int required) {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_FULL_GRAPH_DECODE_KV_CAPACITY");
  int requested = 2048;
  if (v && v[0]) {
    char *end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end != v && *end == '\0' && parsed > 0)
      requested = static_cast<int>(std::min<long>(parsed, max_t));
  }
  return std::min(max_t, std::max(required, requested));
}

int graph_cpu_threads() {
#if defined(_OPENMP)
  const int available = std::max(1, omp_get_max_threads());
#else
  const int available = 1;
#endif
  const char *v = std::getenv("MOESTREAM_QWEN_CPU_THREADS");
  if (v && v[0]) {
    char *end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end != v && *end == '\0' && parsed > 0)
      return std::min(available, static_cast<int>(parsed));
  }
  return available;
}

// REVERTED (2026-09-17): a prior commit defaulted this to false, reasoning
// "the reference Bee scheduler uses pipeline_parallel=false for this
// graph... so the event/four-copy mode adds bookkeeping without useful
// overlap." That reasoning does not hold up against Bee's own source
// (llama-context.cpp): `pipeline_parallel` there is an UNRELATED feature --
// it requires `model.n_devices() > 1` (multi-GPU tensor split) AND
// `!model.has_tensor_overrides()`, and Bee's own `-ncmoe` flag sets tensor
// overrides for exactly the CPU-offloaded MoE tensors this graph mirrors.
// So Bee's `pipeline_parallel=false` in every comparison this project has
// ever run is not a deliberate choice about CPU/GPU MoE handoff at all --
// it is bee's own multi-GPU pipelining feature being auto-disabled by
// `-ncmoe`, for reasons having nothing to do with this graph's performance.
// Confirmed via direct measurement this graph's OWN parallel=false default
// was a real regression (11.76 vs 12.20-12.86 tok/s, N=22 mixed decode) --
// this is the fix from qwen4exp-mixed-gpu-decode-gap.md
// (ggml_backend_sched_new's parallel=false leaves CUDA sync events NULL,
// forcing a blocking stream-sync every CPU/GPU transition instead of a
// cheap event wait), previously verified at ~15.7->18.5-18.9 tok/s and
// accidentally undone by that commit. Default back to true; set
// MOESTREAM_QWEN4EXP_GRAPH_PARALLEL=0 for the blocking-sync A/B.
bool graph_scheduler_parallel() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_GRAPH_PARALLEL");
  return !(v && v[0] == '0');
}

// Bee gives the CPU scheduler slot the CUDA device's host buffer type. That
// makes graph-owned CPU activations pinned host allocations, so the many
// CPU<->CUDA boundary copies do not fall through the pageable-host staging
// path. This is the normal full-graph mode; set the environment variable to
// 0 for an explicit pageable-buffer A/B.
bool graph_cuda_host_buffers() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_GRAPH_CUDA_HOST_BUFFERS");
  return !(v && v[0] == '0');
}

// Bee's full graph begins with the CPU token-embedding GET_ROWS node, then
// crosses to CUDA for the first transformer block.  MoEStream receives the
// already-expanded hidden state from the engine, so it cannot include that
// embedding operation without duplicating model work.  Keep the same
// CPU->GPU graph boundary with one CPU contiguous staging node instead.  This
// is a topology-compatibility node, not a numerical transformation; set the
// variable to 0 for the original MoEStream graph shape when doing an A/B.
bool graph_match_bee_topology() {
  const char *v =
      std::getenv("MOESTREAM_QWEN4EXP_GRAPH_MATCH_BEE_TOPOLOGY");
  return !(v && v[0] == '0');
}

// Emit one-time tensor metadata for CPU MoE comparisons.  The same
// BEE_GGML_PROFILE_LAYOUT switch is consumed by Bee's GGML scheduler so the
// two implementations can be compared without changing their execution.
bool graph_layout_debug() {
  const char *v = std::getenv("BEE_GGML_PROFILE_LAYOUT");
  return v && v[0] == '1';
}

bool has_ple_layer(const ArchitectureConfig &c) {
  return !c.ple_layers.empty() && c.ple_n_heads > 0 &&
         c.ple_n_embd_per_layer > 0;
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
  if (bytes == 0)
    return nullptr;
  size_t reservation = 0;
  if (g.host_precharged_bytes >= bytes) {
    g.host_precharged_bytes -= bytes;
  } else {
    reservation = bytes - g.host_precharged_bytes;
    g.host_precharged_bytes = 0;
    if (!MemoryGuard::try_reserve_host_bytes(reservation)) {
      LOG_WARN("[GGML-QWEN-FULL-MEM] host graph allocation refused: %.2f MB "
               "(reserved=%.2f GB, budget=%.2f GB)\n",
               bytes / 1e6, MemoryGuard::reserved_host_bytes() / 1e9,
               MemoryGuard::host_budget_bytes() / 1e9);
      return nullptr;
    }
  }
  const int err = posix_memalign(&p, 4096, bytes);
  if (err != 0 || !p) {
    MemoryGuard::release_host_bytes(reservation);
    return nullptr;
  }
  g.owned.push_back({p, bytes, false, reservation});
  return p;
}

ggml_tensor *host_input(FullGraph &g, ggml_context *ctx, ggml_type type,
                        const std::vector<int64_t> &shape, size_t bytes,
                        const char *name) {
  ggml_tensor *t = new_tensor(ctx, type, shape);
  if (!t)
    return nullptr;
  void *p = alloc_host(g, std::max(bytes, ggml_nbytes(t)));
  if (!p || !bind_host(g, t, p, std::max(bytes, ggml_nbytes(t)),
                       GGML_BACKEND_BUFFER_USAGE_COMPUTE))
    return nullptr;
  ggml_set_input(t);
  ggml_set_name(t, name);
  g.host_inputs.push_back(p);
  return t;
}

const QuantizedTensor *resident(const StreamingEngine &engine,
                                const std::string &name) {
  auto it = engine.resident_layers.find(name);
  if (it == engine.resident_layers.end() || !it->second.host_ptr)
    return nullptr;
  return &it->second;
}

bool graph_dense_weight_name(const std::string &name);

ggml_tensor *resident_tensor(
    FullGraph &g, ggml_context *ctx, const std::string &name,
    ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
  const QuantizedTensor *q = resident(*g.engine, name);
  if (!q || !type_ok(q->meta.type))
    return nullptr;
  std::vector<int64_t> shape;
  shape.reserve(q->meta.shape.size());
  for (uint64_t d : q->meta.shape)
    shape.push_back(static_cast<int64_t>(d));
  ggml_tensor *t = new_tensor(ctx, static_cast<ggml_type>(q->meta.type), shape);
  if (!t || q->meta.size_bytes < ggml_nbytes(t))
    return nullptr;
  if (g.device_dense_buffer && usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
      graph_dense_weight_name(name)) {
    const ggml_backend_buffer_type_t buft =
        ggml_backend_buffer_get_type(g.device_dense_buffer);
    const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
    const size_t alignment =
        std::max<size_t>(256, ggml_backend_buft_get_alignment(buft));
    const size_t aligned =
        (g.device_dense_used + alignment - 1) & ~(alignment - 1);
    if (aligned > g.device_dense_capacity ||
        alloc_size > g.device_dense_capacity - aligned)
      return nullptr;
    void *addr = static_cast<uint8_t *>(g.device_dense_base) + aligned;
    if (ggml_backend_tensor_alloc(g.device_dense_buffer, t, addr) !=
        GGML_STATUS_SUCCESS)
      return nullptr;
    const auto t_memcpy_start = std::chrono::steady_clock::now();
    const cudaError_t err = cudaMemcpy(addr, q->host_ptr, q->meta.size_bytes,
                                       cudaMemcpyHostToDevice);
    g.dense_upload_ms_total += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() -
                                  t_memcpy_start)
                                  .count();
    if (err != cudaSuccess) {
      LOG_ERROR("[GGML-QWEN-FULL] dense device upload failed for %s: %s\n",
                name.c_str(), cudaGetErrorString(err));
      return nullptr;
    }
    g.device_dense_used = aligned + alloc_size;
    ++g.device_dense_tensors;
  } else if (!bind_host(g, t, q->host_ptr, q->meta.size_bytes, usage)) {
    return nullptr;
  }
  ggml_set_name(t, name.c_str());
  return t;
}

bool graph_dense_weight_name(const std::string &name) {
  // Keep this list aligned with the names consumed by build_graph(). Routed
  // *_exps tensors have their own residency policy; indexer/NextN tensors are
  // not part of this v1 graph, and absorbed-MLA tensors are unused here.
  if (name.find("_exps.weight") != std::string::npos ||
      name.find("indexer") != std::string::npos ||
      name.find("nextn") != std::string::npos ||
      name.find("attn_k_b.weight") != std::string::npos ||
      name.find("attn_v_b.weight") != std::string::npos ||
      name.find("attn_output_a.weight") != std::string::npos)
    return false;
  return name.find("attn") != std::string::npos ||
         name.find("hc_") != std::string::npos ||
         name.find("ssm_") != std::string::npos ||
         name.find("ple_") != std::string::npos ||
         name.find("norm") != std::string::npos ||
         name.find("gate_inp") != std::string::npos ||
         name.find("shexp") != std::string::npos ||
         name.find("ffn_gate.weight") != std::string::npos ||
         name.find("ffn_up.weight") != std::string::npos ||
         name.find("ffn_down.weight") != std::string::npos ||
         name.find("exp_probs_b.bias") != std::string::npos ||
         name == "output_hc_norm.weight" || name == "output_hc_down.weight" ||
         name == "output_hc_up.weight" || name == "output.weight" ||
         name == "output_norm.weight";
}

bool init_device_dense_arena(FullGraph &g) {
  size_t bytes = 0;
  int tensors = 0;
  for (const auto &[name, q] : g.engine->resident_layers) {
    if (!q.host_ptr || !graph_dense_weight_name(name))
      continue;
    constexpr size_t kTensorSlack = 4096;
    if (bytes > SIZE_MAX - q.meta.size_bytes - kTensorSlack)
      return false;
    bytes += q.meta.size_bytes + kTensorSlack;
    ++tensors;
  }
  if (bytes == 0)
    return false;
  constexpr size_t kArenaSlack = 1ull << 20;
  if (bytes > SIZE_MAX - kArenaSlack)
    return false;
  const size_t capacity = bytes + kArenaSlack;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_get_default_buffer_type(g.cuda);
  g.device_dense_buffer = ggml_backend_buft_alloc_buffer(buft, capacity);
  if (!g.device_dense_buffer) {
    LOG_WARN(
        "[GGML-QWEN-FULL] dense device arena allocation declined: %.2f GiB\n",
        static_cast<double>(capacity) / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  ggml_backend_buffer_set_usage(g.device_dense_buffer,
                                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  g.device_dense_base = ggml_backend_buffer_get_base(g.device_dense_buffer);
  g.device_dense_capacity = ggml_backend_buffer_get_size(g.device_dense_buffer);
  g.device_dense_used = 0;
  g.device_dense_tensors = 0;
  LOG_INFO("[GGML-QWEN-FULL] dense device arena=%.2f GiB for %d graph weight "
           "tensors\n",
           static_cast<double>(g.device_dense_capacity) /
               (1024.0 * 1024.0 * 1024.0),
           tensors);
  return true;
}

bool init_device_state_arena(FullGraph &g) {
  if (!g.decode_mode || g.kv_capacity <= 0)
    return true;
  const auto &c = g.engine->model_file.architecture_config;
  size_t bytes = 0;
  auto add = [&](size_t n) {
    constexpr size_t kSlack = 4096;
    if (n > SIZE_MAX - kSlack || bytes > SIZE_MAX - n - kSlack)
      return false;
    bytes += n + kSlack;
    return true;
  };
  for (int l = 0; l < g.n_layers; ++l) {
    if (!qwen4exp_is_gdn_layer(c, l)) {
      const size_t kv = static_cast<size_t>(g.kv_capacity) * g.Hkv * g.hd *
                        sizeof(float);
      if (!add(kv) || !add(kv))
        return false;
    } else {
      const size_t state = static_cast<size_t>(g.gdn_hd) * g.gdn_hd *
                           g.Hv_gdn * sizeof(float);
      const size_t hist = static_cast<size_t>(g.d_conv - 1) * g.conv_dim *
                          sizeof(float);
      if (!add(state) || !add(hist))
        return false;
    }
    if (qwen4exp_is_ple_layer(c, l)) {
      const size_t ple_hist = static_cast<size_t>(c.ple_conv_kernel - 1) *
                              c.ple_ngram_size * g.D * g.HC * sizeof(float);
      if (!add(ple_hist))
        return false;
    }
  }
  constexpr size_t kArenaSlack = 1ull << 20;
  if (bytes > SIZE_MAX - kArenaSlack)
    return false;
  const size_t capacity = bytes + kArenaSlack;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_get_default_buffer_type(g.cuda);
  g.device_state_buffer = ggml_backend_buft_alloc_buffer(buft, capacity);
  if (!g.device_state_buffer) {
    LOG_WARN("[GGML-QWEN-FULL-DECODE] device state arena declined: %.2f GiB\n",
             static_cast<double>(capacity) / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  ggml_backend_buffer_set_usage(g.device_state_buffer,
                                GGML_BACKEND_BUFFER_USAGE_COMPUTE);
  g.device_state_base = ggml_backend_buffer_get_base(g.device_state_buffer);
  g.device_state_capacity = ggml_backend_buffer_get_size(g.device_state_buffer);
  g.device_state_used = 0;
  LOG_INFO("[GGML-QWEN-FULL-DECODE] device state arena=%.2f GiB, KV capacity=%d\n",
           static_cast<double>(g.device_state_capacity) /
               (1024.0 * 1024.0 * 1024.0),
           g.kv_capacity);
  return true;
}

ggml_tensor *new_device_state_tensor(FullGraph &g, ggml_type type,
                                     const std::vector<int64_t> &shape,
                                     const char *name) {
  if (!g.device_state_buffer)
    return nullptr;
  ggml_tensor *t = new_tensor(g.ctx, type, shape);
  if (!t)
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

// True for a layer inside the device_expert_layers-sized GPU-resident
// block, wherever that block sits (leading, the historical default, or
// trailing, matching beellama's `-ncmoe` convention -- see
// FullGraph::device_expert_trailing's own comment). Single source of
// truth for the block's bounds so device_expert_bytes() and
// plan_expert_vram_split()'s eligibility filter can't drift from
// graph_expert_projection_resident()'s own notion of which layers are
// covered.
bool device_expert_block_covers(const FullGraph &g, int layer) {
  if (g.device_expert_layers <= 0)
    return false;
  const int n_layers = g.engine->model_file.architecture_config.n_layers;
  if (g.device_expert_trailing)
    return layer >= n_layers - g.device_expert_layers;
  return layer < g.device_expert_layers;
}

bool graph_expert_projection_resident(const FullGraph &g, int layer,
                                      const char *stem) {
  // cpu_affinity_layers (config): force this layer's routed-expert weights
  // to stay CPU-resident regardless of device_expert_layers, mirroring the
  // legacy dispatch path's own is_cpu_affinity_layer() contract ("never
  // pin or pre-warm VRAM for these layers"). Checked first so it overrides
  // the leading-N device_expert_layers count below even if this layer would
  // otherwise have qualified for VRAM residency.
  if (is_cpu_affinity_layer(layer))
    return false;
  if (layer >= 0 && device_expert_block_covers(g, layer))
    return true;
  // device_expert_partial_gate_up's boundary-layer optimization is only
  // meaningful for the leading convention (the boundary is the first CPU
  // layer just after the leading GPU block, layer==device_expert_layers);
  // it is forced off in trailing mode at the config-set call site instead
  // of generalizing this boundary math, so this line never fires there.
  return g.device_expert_partial_gate_up && layer == g.device_expert_layers &&
         std::strcmp(stem, "ffn_down_exps.weight") != 0;
}

// Per-expert VRAM split planner (qwen4exp_graph_expert_vram_budget_gb).
// Generalizes cpu_affinity_layers/device_expert_layers' whole-layer choice:
// instead of an all-or-nothing residency decision per layer, spend a fixed
// VRAM budget on the globally hottest routed experts (by real usage, same
// ranking data pin_topk_per_layer/global_topn already consult), across any
// layer NOT already claimed by cpu_affinity_layers or device_expert_layers/
// device_expert_partial_gate_up -- those two mechanisms are checked FIRST
// and always win, so this planner never double-places a layer they already
// resolved. Must run before init_device_expert_arena() so the arena can
// size itself for the resulting split_gpu_expert_ids. v1 scope: decode only
// (g.S==1) -- see build_moe()'s own S==1 guard for why; a config with the
// budget set during prefill (S>1) simply plans nothing here, no error.
void plan_expert_vram_split(FullGraph &g) {
  g.split_gpu_expert_ids.clear();
  if (g.S != 1)
    return;
  const double budget_gb = g.engine->qwen4exp_graph_expert_vram_budget_gb;
  if (budget_gb <= 0.0)
    return; // feature off -- identical to today, zero planning cost

  const std::string path = g.engine->qwen4exp_graph_pin_file.empty()
                               ? std::string(".moestream_usage")
                               : g.engine->qwen4exp_graph_pin_file;
  auto [entries, maxc] = StreamingEngine::load_usage(path);
  (void)maxc;
  if (entries.empty()) {
    LOG_INFO("[GGML-QWEN-FULL-SPLIT] no usage data at %s -- per-expert VRAM "
             "split stays off this run.\n",
             path.c_str());
    return;
  }

  const auto &c = g.engine->model_file.architecture_config;
  const int n_layers = c.n_layers;
  const int n_experts = c.n_experts;

  // Bounds-check + drop any (layer,expert) already resolved by
  // cpu_affinity_layers/device_expert_layers/device_expert_partial_gate_up --
  // same stale-usage-file tolerance as main.cpp's sanitize_usage_entries.
  std::vector<StreamingEngine::UsageEntry> eligible;
  eligible.reserve(entries.size());
  for (const auto &e : entries) {
    if (e.layer < 0 || e.layer >= n_layers || e.expert < 0 ||
        e.expert >= n_experts)
      continue;
    if (is_cpu_affinity_layer(e.layer))
      continue;
    if (device_expert_block_covers(g, e.layer))
      continue;
    if (g.device_expert_partial_gate_up && e.layer == g.device_expert_layers)
      continue; // boundary layer's own asymmetric gate/up-only residency
    eligible.push_back(e);
  }
  if (eligible.empty()) {
    LOG_INFO("[GGML-QWEN-FULL-SPLIT] no eligible layers (all claimed by "
             "cpu_affinity_layers/device_expert_layers) -- per-expert VRAM "
             "split stays off this run.\n");
    return;
  }

  // Single global priority queue over all (layer, expert, count) triples --
  // spends the fixed budget on the model's genuinely hottest experts first,
  // rather than splitting it evenly per layer. Stable tie-break by
  // (layer, expert) for determinism, same reasoning as main.cpp's [PIN]
  // block's own stable_sort.
  std::stable_sort(eligible.begin(), eligible.end(),
                   [](const StreamingEngine::UsageEntry &a,
                      const StreamingEngine::UsageEntry &b) {
                     if (a.count != b.count)
                       return a.count > b.count;
                     if (a.layer != b.layer)
                       return a.layer < b.layer;
                     return a.expert < b.expert;
                   });

  size_t remaining_budget =
      static_cast<size_t>(budget_gb * 1024.0 * 1024.0 * 1024.0);
  std::unordered_map<int, std::unordered_set<int>> chosen;
  size_t experts_placed = 0;
  size_t bytes_spent = 0;
  for (const auto &e : eligible) {
    if (remaining_budget == 0)
      break;
    auto &layer_set = chosen[e.layer];
    if (layer_set.count(e.expert))
      continue; // usage file is expected unique per (layer,expert); don't trust that blindly
    const size_t cost = g.engine->compute_expert_slab_bytes(e.layer);
    if (cost == 0 || cost > remaining_budget)
      continue; // doesn't fit -- skip, keep scanning cheaper/later entries
    layer_set.insert(e.expert);
    remaining_budget -= cost;
    bytes_spent += cost;
    ++experts_placed;
  }

  for (auto &[layer, ids_set] : chosen) {
    if (ids_set.empty())
      continue;
    std::vector<int> &dst = g.split_gpu_expert_ids[layer];
    dst.assign(ids_set.begin(), ids_set.end());
    std::sort(dst.begin(), dst.end()); // deterministic local-index order
  }

  LOG_INFO("[GGML-QWEN-FULL-SPLIT] budget=%.3f GiB spent=%.3f GiB across %zu "
           "layer(s), %zu expert(s) total\n",
           budget_gb, bytes_spent / (1024.0 * 1024.0 * 1024.0),
           g.split_gpu_expert_ids.size(), experts_placed);
}

const GGUFTensor *model_tensor(const StreamingEngine &engine,
                               const std::string &name) {
  auto it = engine.model_file.tensors.find(name);
  return it == engine.model_file.tensors.end() ? nullptr : &it->second;
}

size_t full_expert_source_bytes(const StreamingEngine &engine) {
  const auto &c = engine.model_file.architecture_config;
  size_t total = 0;
  for (int layer = 0; layer < c.n_layers; ++layer) {
    // When main.cpp prepared a complete graph-RAM arena, the three packed
    // expert planes for this layer are already resident in DDR.  The full
    // graph aliases those bytes below; do not pre-charge or allocate a second
    // graph-owned source set for the same layer.
    if (engine.graph_ram_layer_ready(layer))
      continue;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(layer) + "." + stem);
      if (!m || m->size_bytes == 0)
        return 0;
      if (static_cast<size_t>(m->size_bytes) >
          std::numeric_limits<size_t>::max() - total)
        return 0;
      total += static_cast<size_t>(m->size_bytes);
    }
  }
  return total;
}

// Resolve one packed Qwen expert projection to the complete, preloaded
// graph-RAM arena. This is deliberately Qwen-specific. The arena may use the
// legacy [gate|up|down] per-expert layout or the projection-compact layout
// (three separate expert-axis planes); in both cases this returns a strided
// view without making a graph-owned copy.
bool preloaded_ram_expert_source(FullGraph &g, const std::string &name,
                                 const GGUFTensor &meta, void *&ptr,
                                 size_t &bytes, size_t &expert_stride,
                                 size_t &arena_bytes) {
  const std::string prefix = "blk.";
  if (name.compare(0, prefix.size(), prefix) != 0)
    return false;

  char *end = nullptr;
  const long parsed_layer = std::strtol(name.c_str() + prefix.size(), &end, 10);
  if (end == name.c_str() + prefix.size() || parsed_layer < 0 || *end != '.')
    return false;
  const int layer = static_cast<int>(parsed_layer);
  const std::string suffix(end + 1);
  const auto *arena = g.engine->graph_ram_arena(layer);
  if (!arena)
    return false;
  size_t plane_offset = 0;
  size_t plane_bytes = 0;
  size_t plane_stride = 0;
  if (suffix == "ffn_gate_exps.weight") {
    plane_offset = arena->gate_offset;
    plane_bytes = arena->gate_bytes;
    plane_stride = arena->gate_stride;
  } else if (suffix == "ffn_up_exps.weight") {
    plane_offset = arena->projection_compact ? arena->up_offset
                                             : arena->gate_bytes;
    plane_bytes = arena->up_bytes;
    plane_stride = arena->up_stride;
  } else if (suffix == "ffn_down_exps.weight") {
    plane_offset = arena->projection_compact
                       ? arena->down_offset
                       : arena->gate_bytes + arena->up_bytes;
    plane_bytes = arena->down_bytes;
    plane_stride = arena->down_stride;
  } else {
    return false;
  }

  if (!g.engine->graph_ram_layer_ready(layer))
    return false;
  const int n_experts = g.engine->model_file.architecture_config.n_experts;
  const size_t n_exp = static_cast<size_t>(std::max(0, n_experts));
  if (!arena || !arena->base || n_experts <= 0 ||
      arena->n_experts != n_experts || plane_bytes == 0 ||
      plane_stride == 0 ||
      plane_offset > arena->bytes ||
      plane_bytes > arena->bytes - plane_offset ||
      n_exp > SIZE_MAX / plane_bytes ||
      meta.size_bytes != plane_bytes * n_exp || plane_stride < plane_bytes ||
      static_cast<size_t>(n_experts - 1) >
          (SIZE_MAX - plane_bytes) / plane_stride)
    return false;

  const size_t view_bytes =
      plane_stride * static_cast<size_t>(n_experts - 1) + plane_bytes;
  if (plane_offset > arena->bytes || view_bytes > arena->bytes - plane_offset)
    return false;

  ptr = arena->base + plane_offset;
  bytes = view_bytes;
  expert_stride = plane_stride;
  arena_bytes = arena->bytes;
  return true;
}

size_t device_expert_bytes(const FullGraph &g) {
  const auto &engine = *g.engine;
  const auto &c = engine.model_file.architecture_config;
  const int limit = std::min(std::max(0, g.device_expert_layers), c.n_layers);
  // Block bounds mirror device_expert_block_covers() exactly: leading mode
  // walks [0,limit), trailing mode walks [n_layers-limit,n_layers). Kept as
  // an explicit loop (not a device_expert_block_covers() call per layer)
  // since this function also needs the leading-only boundary-layer index
  // (`limit`) below, unchanged from before trailing mode existed.
  const int block_start = g.device_expert_trailing ? c.n_layers - limit : 0;
  const int block_end = g.device_expert_trailing ? c.n_layers : limit;
  size_t total = 0;
  for (int layer = block_start; layer < block_end; ++layer) {
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(layer) + "." + stem);
      if (!m || m->size_bytes == 0 ||
          total > SIZE_MAX - static_cast<size_t>(m->size_bytes))
        return 0;
      total += static_cast<size_t>(m->size_bytes);
    }
  }
  // Boundary-layer (partial_gate_up) accounting is leading-only by
  // construction -- device_expert_partial_gate_up is forced off whenever
  // device_expert_trailing is set (see that setter), so limit < c.n_layers
  // here always refers to the leading block's own boundary.
  if (g.device_expert_partial_gate_up && limit < c.n_layers) {
    const int layer = limit;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(layer) + "." + stem);
      if (!m || m->size_bytes == 0 ||
          total > SIZE_MAX - static_cast<size_t>(m->size_bytes))
        return 0;
      total += static_cast<size_t>(m->size_bytes);
    }
  }
  // plan_expert_vram_split()'s per-layer split tensors: M hot experts'
  // combined gate+up+down bytes per split layer. These layers are disjoint
  // from the leading-N/boundary accounting above by construction (the
  // planner never places a layer device_expert_layers/
  // device_expert_partial_gate_up already claimed).
  for (const auto &[layer, ids] : g.split_gpu_expert_ids) {
    if (ids.empty())
      continue;
    const size_t per_expert = engine.compute_expert_slab_bytes(layer);
    if (per_expert == 0)
      continue;
    if (ids.size() > SIZE_MAX / per_expert)
      return 0;
    const size_t layer_bytes = per_expert * ids.size();
    if (total > SIZE_MAX - layer_bytes)
      return 0;
    total += layer_bytes;
  }
  return total;
}

bool init_device_expert_arena(FullGraph &g) {
  const bool split_active = !g.split_gpu_expert_ids.empty();
  if (g.device_expert_layers <= 0 && !split_active)
    return true;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_get_default_buffer_type(g.cuda);
  const int requested_layers = g.device_expert_layers;
  // A request for all 48 routed layers is valid as a placement preference but
  // cannot fit on the 24-GiB card.  Clamp only whole-layer residency requests
  // to the largest arena that actually allocates; the graph then keeps the
  // remaining layers on the DDR/CPU backend instead of abandoning the entire
  // full graph. Per-expert split allocations remain strict until their own
  // budget is made independently retryable.
  const bool allow_layer_clamp = requested_layers > 0 && !split_active;
  for (;;) {
    const size_t bytes = device_expert_bytes(g);
    if (bytes == 0) {
      if (g.device_expert_layers <= 0 && !split_active) {
        g.split_gpu_expert_ids.clear();
        return true;
      }
      if (allow_layer_clamp && g.device_expert_layers > 1) {
        --g.device_expert_layers;
        continue;
      }
      if (allow_layer_clamp && g.device_expert_layers == 1) {
        g.device_expert_layers = 0;
        g.device_expert_partial_gate_up = false;
        LOG_WARN("[GGML-QWEN-FULL] no whole expert layer fits in the "
                 "device arena; keeping all routed experts in DDR/CPU\n");
        return true;
      }
      return false;
    }
    constexpr size_t kAlignment = 256;
    size_t tensor_slack =
        static_cast<size_t>(g.device_expert_layers) * 3 * kAlignment;
    // Each split layer allocates 3 more small tensors (gate/up/down) into the
    // same bump arena -- same per-tensor alignment slack accounting as above.
    tensor_slack += g.split_gpu_expert_ids.size() * 3 * kAlignment;
    constexpr size_t kArenaSlack = 1ull << 20;
    if (bytes > SIZE_MAX - tensor_slack - kArenaSlack) {
      if (allow_layer_clamp && g.device_expert_layers > 1) {
        --g.device_expert_layers;
        continue;
      }
      return false;
    }
    const size_t capacity = bytes + tensor_slack + kArenaSlack;
    g.device_expert_buffer = ggml_backend_buft_alloc_buffer(buft, capacity);
    if (g.device_expert_buffer)
      {
        // build_graph() applies the same one-GiB free-VRAM floor before the
        // scheduler reserves its workspace.  Apply it here too, while the
        // whole-layer request is still retryable; otherwise a 24-layer arena
        // can allocate successfully and only fail later during graph
        // construction, returning the entire request to the legacy path.
        size_t free_bytes = 0, total_bytes = 0;
        const cudaError_t meminfo =
            cudaMemGetInfo(&free_bytes, &total_bytes);
        constexpr size_t kGraphFreeFloor = 1ull * 1024ull * 1024ull * 1024ull;
        if (meminfo == cudaSuccess && free_bytes < kGraphFreeFloor &&
            allow_layer_clamp && g.device_expert_layers > 1) {
          ggml_backend_buffer_free(g.device_expert_buffer);
          g.device_expert_buffer = nullptr;
          g.device_expert_base = nullptr;
          g.device_expert_capacity = 0;
          g.device_expert_used = 0;
          --g.device_expert_layers;
          continue;
        }
        break;
      }
    if (allow_layer_clamp && g.device_expert_layers > 1) {
      --g.device_expert_layers;
      continue;
    }
    if (allow_layer_clamp && g.device_expert_layers == 1) {
      g.device_expert_layers = 0;
      g.device_expert_partial_gate_up = false;
      LOG_WARN("[GGML-QWEN-FULL] device expert arena could not fit even "
               "one whole layer; keeping all routed experts in DDR/CPU\n");
      return true;
    }
    LOG_WARN(
        "[GGML-QWEN-FULL] device expert arena allocation declined: %.2f GiB\n",
        static_cast<double>(capacity) / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  if (g.device_expert_layers != requested_layers)
    LOG_WARN("[GGML-QWEN-FULL] requested %d device expert layers, but only "
             "%d whole layers fit; remaining layers stay in DDR/CPU\n",
             requested_layers, g.device_expert_layers);
  ggml_backend_buffer_set_usage(g.device_expert_buffer,
                                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  g.device_expert_base = ggml_backend_buffer_get_base(g.device_expert_buffer);
  g.device_expert_capacity =
      ggml_backend_buffer_get_size(g.device_expert_buffer);
  g.device_expert_used = 0;
  {
    const int n_layers = g.engine->model_file.architecture_config.n_layers;
    const int block_start =
        g.device_expert_trailing ? n_layers - g.device_expert_layers : 0;
    const int block_end =
        g.device_expert_trailing ? n_layers : g.device_expert_layers;
    LOG_INFO(
        "[GGML-QWEN-FULL] device expert arena=%.2f GiB for layers [%d,%d)%s%s "
        "split_layers=%zu\n",
        static_cast<double>(g.device_expert_capacity) /
            (1024.0 * 1024.0 * 1024.0),
        block_start, block_end,
        g.device_expert_partial_gate_up ? " + next-layer gate/up" : "",
        split_active ? " + per-expert split" : "",
        g.split_gpu_expert_ids.size());
  }
  return true;
}

bool bind_device_expert(FullGraph &g, ggml_tensor *t, const void *host,
                        size_t bytes) {
  if (!g.device_expert_buffer || !t || !host || bytes < ggml_nbytes(t))
    return false;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_buffer_get_type(g.device_expert_buffer);
  const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
  const size_t alignment =
      std::max<size_t>(256, ggml_backend_buft_get_alignment(buft));
  const size_t aligned =
      (g.device_expert_used + alignment - 1) & ~(alignment - 1);
  if (aligned > g.device_expert_capacity ||
      alloc_size > g.device_expert_capacity - aligned)
    return false;
  void *addr = static_cast<uint8_t *>(g.device_expert_base) + aligned;
  if (ggml_backend_tensor_alloc(g.device_expert_buffer, t, addr) !=
      GGML_STATUS_SUCCESS)
    return false;
  const auto t_memcpy_start = std::chrono::steady_clock::now();
  const cudaError_t err = cudaMemcpy(addr, host, bytes, cudaMemcpyHostToDevice);
  g.expert_upload_ms_total += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() -
                                  t_memcpy_start)
                                  .count();
  ++g.expert_upload_calls;
  if (err != cudaSuccess) {
    LOG_ERROR("[GGML-QWEN-FULL] device expert upload failed: %s\n",
              cudaGetErrorString(err));
    return false;
  }
  g.device_expert_used = aligned + alloc_size;
  return true;
}

// Same bump-allocation contract as bind_device_expert() above, but the
// source is one uniform-stride run of n_experts planes inside a RAM arena
// (base + e*src_stride for e in [0,n_experts)) rather than one contiguous
// host buffer -- the arena interleaves gate|up|down per expert, so a single
// plane's bytes across experts are NOT contiguous (see load_full_weight()'s
// device_resident arena path). A per-expert cudaMemcpy loop (like
// bind_device_expert_gather() below) works but pays real per-call CUDA API
// overhead n_experts times (measured: made node_construction WORSE, not
// better, at this model's expert count). cudaMemcpy2D expresses the whole
// strided transfer as ONE driver call (width=per_expert_bytes,
// height=n_experts, spitch=src_stride, dpitch=per_expert_bytes since the
// destination is freshly bump-allocated and therefore compact).
bool bind_device_expert_strided(FullGraph &g, ggml_tensor *t, const void *base,
                                size_t src_stride, size_t per_expert_bytes,
                                int n_experts) {
  if (!g.device_expert_buffer || !t || !base || src_stride == 0 ||
      per_expert_bytes == 0 || n_experts <= 0 || src_stride < per_expert_bytes)
    return false;
  if (static_cast<size_t>(n_experts) > SIZE_MAX / per_expert_bytes)
    return false;
  const size_t total_bytes =
      per_expert_bytes * static_cast<size_t>(n_experts);
  if (total_bytes < ggml_nbytes(t))
    return false;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_buffer_get_type(g.device_expert_buffer);
  const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
  const size_t alignment =
      std::max<size_t>(256, ggml_backend_buft_get_alignment(buft));
  const size_t aligned =
      (g.device_expert_used + alignment - 1) & ~(alignment - 1);
  if (aligned > g.device_expert_capacity ||
      alloc_size > g.device_expert_capacity - aligned)
    return false;
  void *addr = static_cast<uint8_t *>(g.device_expert_base) + aligned;
  if (ggml_backend_tensor_alloc(g.device_expert_buffer, t, addr) !=
      GGML_STATUS_SUCCESS)
    return false;
  const auto t_memcpy_start = std::chrono::steady_clock::now();
  const cudaError_t err =
      cudaMemcpy2D(addr, per_expert_bytes, base, src_stride, per_expert_bytes,
                  static_cast<size_t>(n_experts), cudaMemcpyHostToDevice);
  g.expert_upload_ms_total += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() -
                                  t_memcpy_start)
                                  .count();
  ++g.expert_upload_calls;
  if (err != cudaSuccess) {
    LOG_ERROR("[GGML-QWEN-FULL] device expert strided upload failed: %s\n",
              cudaGetErrorString(err));
    return false;
  }
  g.device_expert_used = aligned + alloc_size;
  return true;
}

// Same bump-allocation contract as bind_device_expert() above, but the
// destination tensor `t` (shape [cols,rows,M], M == src_ptrs.size()) is
// filled from M separately-sourced host planes instead of one contiguous
// range -- the M hot experts plan_expert_vram_split() chose are not
// contiguous in the original expert axis. The destination IS contiguous
// (freshly allocated here), so only the source side needs per-expert
// copies; local index k in `t`'s expert axis holds src_ptrs[k]'s bytes.
bool bind_device_expert_gather(FullGraph &g, ggml_tensor *t,
                               const std::vector<const void *> &src_ptrs,
                               size_t per_expert_bytes) {
  if (!g.device_expert_buffer || !t || src_ptrs.empty() ||
      per_expert_bytes == 0)
    return false;
  if (src_ptrs.size() > SIZE_MAX / per_expert_bytes)
    return false;
  const size_t total_bytes = per_expert_bytes * src_ptrs.size();
  if (total_bytes < ggml_nbytes(t))
    return false;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_buffer_get_type(g.device_expert_buffer);
  const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
  const size_t alignment =
      std::max<size_t>(256, ggml_backend_buft_get_alignment(buft));
  const size_t aligned =
      (g.device_expert_used + alignment - 1) & ~(alignment - 1);
  if (aligned > g.device_expert_capacity ||
      alloc_size > g.device_expert_capacity - aligned)
    return false;
  void *addr = static_cast<uint8_t *>(g.device_expert_base) + aligned;
  if (ggml_backend_tensor_alloc(g.device_expert_buffer, t, addr) !=
      GGML_STATUS_SUCCESS)
    return false;
  for (size_t k = 0; k < src_ptrs.size(); ++k) {
    void *dst = static_cast<uint8_t *>(addr) + k * per_expert_bytes;
    const cudaError_t err =
        cudaMemcpy(dst, src_ptrs[k], per_expert_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      LOG_ERROR(
          "[GGML-QWEN-FULL-SPLIT] device expert gather-copy failed at "
          "local index %zu/%zu: %s\n",
          k, src_ptrs.size(), cudaGetErrorString(err));
      // Bytes already copied for k'<k stay allocated (arena is a bump
      // allocator with no partial free) but the caller treats a false
      // return as "this tensor is unusable" and falls back to CPU-only for
      // this layer -- the wasted arena bytes are the same bounded, one-time
      // cost bind_device_expert()'s own failure path already accepts.
      return false;
    }
  }
  g.device_expert_used = aligned + alloc_size;
  return true;
}

// Projection-compact counterpart of bind_device_expert_gather(). Gate and up
// are separate host planes, so copy the two source slices into each compact
// VRAM expert record. This is a startup-only graph construction operation; it
// avoids rebuilding a large interleaved DDR copy merely to preserve the GPU
// gate/up fusion.
// Uniform-stride counterpart to bind_device_expert_gather_gate_up() below --
// same interleaved-destination contract (dst holds [gate_e|up_e] per expert,
// compact), but the two sources are each one uniform-stride run
// (gate_base+e*gate_stride, up_base+e*up_stride) rather than arbitrary
// per-expert pointers, so each plane's whole transfer is one cudaMemcpy2D
// call instead of n_experts separate ones (same rationale as
// bind_device_expert_strided() above -- measured real win on this model's
// expert count).
bool bind_device_expert_strided_gate_up(FullGraph &g, ggml_tensor *t,
                                        const void *gate_base,
                                        size_t gate_stride, size_t gate_bytes,
                                        const void *up_base, size_t up_stride,
                                        size_t up_bytes, int n_experts) {
  if (!g.device_expert_buffer || !t || !gate_base || !up_base ||
      gate_bytes == 0 || up_bytes == 0 || n_experts <= 0 ||
      gate_stride < gate_bytes || up_stride < up_bytes ||
      gate_bytes > SIZE_MAX - up_bytes)
    return false;
  const size_t per_expert_bytes = gate_bytes + up_bytes;
  if (static_cast<size_t>(n_experts) > SIZE_MAX / per_expert_bytes)
    return false;
  const size_t total_bytes =
      per_expert_bytes * static_cast<size_t>(n_experts);
  if (total_bytes < ggml_nbytes(t))
    return false;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_buffer_get_type(g.device_expert_buffer);
  const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
  const size_t alignment =
      std::max<size_t>(256, ggml_backend_buft_get_alignment(buft));
  const size_t aligned =
      (g.device_expert_used + alignment - 1) & ~(alignment - 1);
  if (aligned > g.device_expert_capacity ||
      alloc_size > g.device_expert_capacity - aligned)
    return false;
  void *addr = static_cast<uint8_t *>(g.device_expert_base) + aligned;
  if (ggml_backend_tensor_alloc(g.device_expert_buffer, t, addr) !=
      GGML_STATUS_SUCCESS)
    return false;
  const auto t_memcpy_start = std::chrono::steady_clock::now();
  cudaError_t err =
      cudaMemcpy2D(addr, per_expert_bytes, gate_base, gate_stride, gate_bytes,
                  static_cast<size_t>(n_experts), cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemcpy2D(static_cast<uint8_t *>(addr) + gate_bytes,
                       per_expert_bytes, up_base, up_stride, up_bytes,
                       static_cast<size_t>(n_experts), cudaMemcpyHostToDevice);
  g.expert_upload_ms_total += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() -
                                  t_memcpy_start)
                                  .count();
  ++g.expert_upload_calls;
  if (err != cudaSuccess) {
    LOG_ERROR("[GGML-QWEN-FULL] compact gate/up strided upload failed: %s\n",
              cudaGetErrorString(err));
    return false;
  }
  g.device_expert_used = aligned + alloc_size;
  return true;
}


ggml_tensor *load_full_weight(FullGraph &g, ggml_context *ctx,
                              const std::string &name,
                              bool device_resident = false) {
  const GGUFTensor *m = model_tensor(*g.engine, name);
  if (!m || !type_ok(m->type) || m->shape.size() < 2)
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : m->shape)
    shape.push_back(static_cast<int64_t>(d));
  ggml_tensor *t = new_tensor(ctx, static_cast<ggml_type>(m->type), shape);
  if (!t)
    return nullptr;
  const size_t bytes = static_cast<size_t>(m->size_bytes);
  const size_t required = std::max(bytes, ggml_nbytes(t));

  // A Qwen full graph must consume the same complete DDR expert planes that
  // the RAM tier already loaded.  In that configuration the source is a
  // fixed-stride arena, so adjust only the expert-axis stride; no selected
  // weights are copied and no GGUF mapping is consulted.
  //
  // 2026-09-17: this arena source used to be consulted ONLY for
  // !device_resident (the host-alias path just below). A device_resident
  // caller (device_expert_layers' whole-layer VRAM block) fell through
  // unconditionally to the pread_direct_aware() fallback further down --
  // a REAL re-read from disk/page-cache of the entire device-resident expert
  // block, on every single build_graph() call, even though the RAM tier had
  // already loaded and validated (graph_ram_layer_ready()) these exact
  // bytes moments earlier. Root-caused via a chrono split of node_construction
  // (see memory: full-graph-decode-graph-construction-5s-latency.md) that
  // isolated ~3.1s of the ~5.3s decode-build cost to this function, entirely
  // OUTSIDE the already-measured dense/expert cudaMemcpy timers -- consistent
  // with device_expert_bytes() (~17.68 GiB at device_expert_layers=22) moving
  // through a page-cache-speed host memcpy that no counter was watching.
  // Fixed by transferring straight from the arena into VRAM via
  // bind_device_expert_strided() (one cudaMemcpy2D per tensor, since the
  // arena's per-plane expert stride is uniform), skipping the host
  // round-trip entirely instead of just skipping the disk read. A first cut
  // used a per-expert cudaMemcpy loop (matching expert_gate_up_weight()'s
  // pre-existing bind_device_expert_gather() mechanism) but measured WORSE
  // than the disk-read baseline at this model's expert count -- n_experts
  // separate CUDA API calls cost more than the read they replaced;
  // expert_gate_up_weight()'s own device_resident branch had the same
  // per-call-overhead problem and was converted to
  // bind_device_expert_strided()/bind_device_expert_strided_gate_up() too.
  {
    void *ram_ptr = nullptr;
    size_t ram_bytes = 0;
    size_t ram_stride = 0;
    size_t ram_arena_bytes = 0;
    if (preloaded_ram_expert_source(g, name, *m, ram_ptr, ram_bytes, ram_stride,
                                    ram_arena_bytes)) {
      if (shape.size() < 3 ||
          t->ne[2] != g.engine->model_file.architecture_config.n_experts ||
          ram_stride < t->nb[2])
        return nullptr;
      if (device_resident) {
        const int n_experts = static_cast<int>(t->ne[2]);
        if (n_experts <= 0 || bytes % static_cast<size_t>(n_experts) != 0)
          return nullptr;
        const size_t per_expert_bytes =
            bytes / static_cast<size_t>(n_experts);
        if (!bind_device_expert_strided(g, t, ram_ptr, ram_stride,
                                        per_expert_bytes, n_experts))
          return nullptr;
        ggml_set_name(t, name.c_str());
        return t;
      }

      t->nb[2] = ram_stride;
      t->nb[3] = ram_stride * static_cast<size_t>(t->ne[2]);
      if (!bind_host(g, t, ram_ptr, ram_bytes,
                     GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
        return nullptr;
      // The three projection views overlap the same one-layer arena. Count
      // the backing arena once, on the gate view, rather than summing the
      // three strided view spans and reporting a fictitious 3x allocation.
      if (name.find("ffn_gate_exps.weight") != std::string::npos)
        g.host_ram_alias_bytes += ram_arena_bytes;
      ++g.host_ram_alias_tensors;
      ggml_set_name(t, name.c_str());
      return t;
    }

    // A graph-RAM arena is an explicit ownership decision: once it exists,
    // this Qwen layer is supposed to use the already-preloaded DDR bytes.
    // Do not quietly turn an incomplete arena into a second graph-owned
    // startup read from the GGUF file.  Rejecting the optional graph keeps
    // the normal RAM/cache dispatcher as the correctness fallback and makes
    // the source policy visible instead of hiding an SSD read behind this
    // path.  This guard is deliberately Qwen-full-graph-only; other model
    // families have their own graph source ownership rules.
    int graph_ram_layer = -1;
    if (name.compare(0, 4, "blk.") == 0) {
      char *end = nullptr;
      const long parsed = std::strtol(name.c_str() + 4, &end, 10);
      if (end != name.c_str() + 4 && end && *end == '.')
        graph_ram_layer = static_cast<int>(parsed);
    }
    if (graph_ram_layer >= 0 && g.engine->qwen_ram_preload_all_experts) {
      LOG_WARN("[GGML-QWEN-FULL] refusing file-backed expert source for "
               "layer=%d: ram_preload_all_experts requires the preloaded "
               "DDR arena\n",
               graph_ram_layer);
      return nullptr;
    }
    if (graph_ram_layer >= 0 &&
        g.engine->graph_ram_layer_allocated(graph_ram_layer) &&
        !g.engine->graph_ram_layer_ready(graph_ram_layer)) {
      LOG_WARN("[GGML-QWEN-FULL] refusing file-backed expert source for "
               "layer=%d: preloaded DDR arena is allocated but not ready\n",
               graph_ram_layer);
      return nullptr;
    }
  }

  void *p = alloc_host(g, required);
  if (!p)
    return nullptr;
  // Fallback for graph configurations without a complete preloaded RAM
  // arena. This is a startup copy into graph-owned DDR via pread; it is not
  // the decode-time CPU source used by the DDR-affinity path above.
  const int fd = g.engine->model_file.fd_for_tensor(*m);
  const ssize_t got = direct_io::pread_direct_aware(
      fd, g.engine->model_file.direct_io_active_for(*m),
      g.engine->model_file.direct_io_alignment, p, bytes, m->offset);
  if (got < 0 || static_cast<size_t>(got) != bytes) {
    LOG_ERROR("[GGML-QWEN-FULL] short read for %s: %zd/%zu\n", name.c_str(),
              got, bytes);
    return nullptr;
  }
  if (device_resident) {
    if (!bind_device_expert(g, t, p, bytes))
      return nullptr;
  } else if (!bind_host(g, t, p, required, GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
    return nullptr;
  } else {
    // GGML's CUDA scheduler uses set_tensor_async for host-backed expert
    // weights. Register each graph-owned source once, after it has been
    // faulted in by the read above, so those copies can use DMA directly.
    // Registration is outside the measured prefill window and is best effort:
    // this machine has a finite aggregate CUDA host-registration ceiling.
    const size_t owned_index = g.owned.size() - 1;
    const cudaError_t pin_err =
        cudaHostRegister(p, required, cudaHostRegisterPortable);
    if (pin_err == cudaSuccess) {
      g.owned[owned_index].pinned = true;
      g.host_pinned_bytes += required;
    } else {
      ++g.host_pin_failures;
      cudaGetLastError();
      if (g.host_pin_failures == 1) {
        LOG_WARN(
            "[GGML-QWEN-FULL] host expert pinning declined at %.2f GiB: %s; "
            "remaining sources stay pageable\n",
            static_cast<double>(g.host_pinned_bytes) /
                (1024.0 * 1024.0 * 1024.0),
            cudaGetErrorString(pin_err));
      }
    }
  }
  g.host_weight_bytes += bytes;
  ggml_set_name(t, name.c_str());
  return t;
}

ggml_tensor *expert_weight(FullGraph &g, ggml_context *ctx, int layer,
                           const char *stem) {
  return load_full_weight(g, ctx, "blk." + std::to_string(layer) + "." + stem,
                          graph_expert_projection_resident(g, layer, stem));
}

// When gate and up have the same quantized row layout, expose them as one
// [cols, 2*rows, n_expert] MUL_MAT_ID weight. GPU graph layers retain this
// merge by gathering the two compact host planes into their final VRAM tensor.
// CPU graph layers use separate views: gate and up are separate planes in the
// projection-compact arena, and making a merged host tensor would recreate the
// large duplicate this layout is intended to remove.
ggml_tensor *expert_gate_up_weight(FullGraph &g, ggml_context *ctx, int layer,
                                   bool device_resident) {
  const std::string base = "blk." + std::to_string(layer) + ".";
  const GGUFTensor *gate = model_tensor(
      *g.engine, base + "ffn_gate_exps.weight");
  const GGUFTensor *up =
      model_tensor(*g.engine, base + "ffn_up_exps.weight");
  const int n_experts = g.engine->model_file.architecture_config.n_experts;
  if (!gate || !up || n_experts <= 0 || gate->type != up->type ||
      gate->shape.size() != 3 || up->shape != gate->shape ||
      gate->shape[2] != static_cast<uint64_t>(n_experts) ||
      gate->size_bytes == 0 || gate->size_bytes != up->size_bytes ||
      gate->size_bytes % static_cast<uint64_t>(n_experts) != 0)
    return nullptr;

  const auto *arena = g.engine->graph_ram_arena(layer);
  if (!arena || !g.engine->graph_ram_layer_ready(layer) || !arena->base ||
      arena->n_experts != n_experts || arena->gate_bytes == 0 ||
      arena->gate_bytes != arena->up_bytes ||
      arena->gate_bytes !=
          static_cast<size_t>(gate->size_bytes) /
              static_cast<size_t>(n_experts) ||
      arena->up_bytes !=
              static_cast<size_t>(up->size_bytes) /
              static_cast<size_t>(n_experts) ||
      arena->gate_bytes > SIZE_MAX - arena->up_bytes)
    return nullptr;

  if (!device_resident && arena->projection_compact)
    return nullptr;

  const size_t plane_bytes = arena->gate_bytes + arena->up_bytes;
  if (plane_bytes == 0)
    return nullptr;
  const size_t legacy_view_bytes =
      arena->stride != 0 &&
              static_cast<size_t>(n_experts - 1) <=
                  (SIZE_MAX - plane_bytes) / arena->stride
          ? arena->stride * static_cast<size_t>(n_experts - 1) + plane_bytes
          : 0;
  if (!arena->projection_compact &&
      (arena->stride < plane_bytes || legacy_view_bytes == 0 ||
       legacy_view_bytes > arena->bytes))
    return nullptr;

  std::vector<int64_t> shape;
  shape.reserve(gate->shape.size());
  for (uint64_t d : gate->shape)
    shape.push_back(static_cast<int64_t>(d));
  if (shape[1] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                       2)
    return nullptr;
  shape[1] *= 2;
  ggml_tensor *t = new_tensor(ctx, static_cast<ggml_type>(gate->type), shape);
  if (!t || t->ne[0] != static_cast<int64_t>(gate->shape[0]) ||
      t->ne[1] != static_cast<int64_t>(gate->shape[1]) * 2 ||
      t->ne[2] != n_experts)
    return nullptr;

  if (device_resident) {
    // Keep ggml's default compact expert-axis strides here; using a DDR
    // arena stride on this descriptor would make ggml_nbytes(t) include the
    // skipped down plane/padding and reject the valid VRAM allocation.
    if (arena->projection_compact) {
      if (!bind_device_expert_strided_gate_up(
              g, t, arena->base + arena->gate_offset, arena->gate_stride,
              arena->gate_bytes, arena->base + arena->up_offset,
              arena->up_stride, arena->up_bytes, n_experts))
        return nullptr;
    } else {
      // Uniform expert-axis stride -- same shape bind_device_expert_strided()
      // (load_full_weight()'s own device_resident arena path) already
      // handles with one cudaMemcpy2D instead of an n_experts-call loop.
      if (!bind_device_expert_strided(g, t, arena->base, arena->stride,
                                      plane_bytes, n_experts))
        return nullptr;
    }
  } else {
    // The host view aliases the interleaved DDR arena, so its expert-axis
    // stride must skip the following down projection and any alignment
    // padding in each source slab.
    t->nb[2] = arena->stride;
    t->nb[3] = arena->stride * static_cast<size_t>(n_experts);
    if (!bind_host(g, t, arena->base, legacy_view_bytes,
                   GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
      return nullptr;
    g.host_ram_alias_bytes += arena->bytes;
    ++g.host_ram_alias_tensors;
  }
  ggml_set_name(t, (base + "ffn_gate_up_exps.weight").c_str());
  return t;
}

// Builds the [cols,rows,M] VRAM-resident tensor for a genuinely split layer
// (0<M<n_expert), gathering the M hot experts named by `hot_ids` straight
// out of the SAME preloaded RAM arena bytes the (unchanged) full-width CPU
// tensor for this same (layer,stem) already aliases -- reuses
// preloaded_ram_expert_source()'s existing base+stride resolution, no new
// stride math. Returns nullptr (never a hard error -- caller falls back to
// CPU-only for this layer) whenever the arena isn't ready, e.g. during
// prefill/non-decode graph builds that don't unconditionally require it.
ggml_tensor *build_split_gpu_tensor(FullGraph &g, ggml_context *ctx,
                                    const std::string &name,
                                    const GGUFTensor &m,
                                    const std::vector<int> &hot_ids) {
  void *ptr = nullptr;
  size_t bytes = 0, stride = 0, arena_bytes = 0;
  if (!preloaded_ram_expert_source(g, name, m, ptr, bytes, stride,
                                   arena_bytes))
    return nullptr;
  const int n_experts = g.engine->model_file.architecture_config.n_experts;
  if (n_experts <= 0 || m.size_bytes % static_cast<uint64_t>(n_experts) != 0)
    return nullptr;
  const size_t plane_bytes =
      static_cast<size_t>(m.size_bytes) / static_cast<size_t>(n_experts);
  if (plane_bytes == 0)
    return nullptr;

  std::vector<int64_t> shape;
  shape.reserve(m.shape.size());
  for (uint64_t d : m.shape)
    shape.push_back(static_cast<int64_t>(d));
  if (shape.size() < 3 || shape[2] != n_experts)
    return nullptr;
  shape[2] = static_cast<int64_t>(hot_ids.size());
  ggml_tensor *t = new_tensor(ctx, static_cast<ggml_type>(m.type), shape);
  if (!t)
    return nullptr;

  std::vector<const void *> src_ptrs;
  src_ptrs.reserve(hot_ids.size());
  for (int e : hot_ids) {
    if (e < 0 || e >= n_experts)
      return nullptr;
    src_ptrs.push_back(static_cast<const uint8_t *>(ptr) +
                       static_cast<size_t>(e) * stride);
  }
  if (!bind_device_expert_gather(g, t, src_ptrs, plane_bytes))
    return nullptr;
  ggml_set_name(t, (name + ".split_gpu").c_str());
  return t;
}

struct SplitExpertWeight {
  ggml_tensor *cpu = nullptr; // full [cols,rows,n_expert] -- today's tensor,
                              // unchanged, used either as-is (no split) or
                              // as build_moe's "CPU branch" input (split).
  ggml_tensor *gpu = nullptr; // [cols,rows,M], only for a genuine split.
};

// Split-aware replacement for expert_weight() at build_graph()'s MoE call
// site. Behaves EXACTLY like expert_weight() (returns only `.cpu`, `.gpu`
// stays null) for every layer graph_expert_projection_resident() already
// resolves (whole-layer CPU, or whole-layer/boundary VRAM-resident) --
// those paths are untouched. Only consults g.split_gpu_expert_ids, and
// only when g.S==1 (decode; see plan_expert_vram_split's own S==1 guard),
// so a split config never affects prefill's tensor construction at all.
SplitExpertWeight expert_weight_split(FullGraph &g, ggml_context *ctx,
                                      int layer, const char *stem) {
  SplitExpertWeight r;
  const std::string name = "blk." + std::to_string(layer) + "." + stem;
  if (graph_expert_projection_resident(g, layer, stem)) {
    r.cpu = load_full_weight(g, ctx, name, /*device_resident=*/true);
    return r; // today's whole-layer/boundary VRAM path, unchanged
  }
  r.cpu = load_full_weight(g, ctx, name, /*device_resident=*/false);
  if (!r.cpu || g.S != 1)
    return r; // today's whole-layer CPU path, unchanged; or non-decode build
  auto it = g.split_gpu_expert_ids.find(layer);
  if (it == g.split_gpu_expert_ids.end() || it->second.empty())
    return r; // no split planned for this layer -- today's path, unchanged
  const GGUFTensor *m = model_tensor(*g.engine, name);
  if (!m)
    return r;
  r.gpu = build_split_gpu_tensor(g, ctx, name, *m, it->second);
  if (!r.gpu)
    LOG_WARN("[GGML-QWEN-FULL-SPLIT] layer=%d %s: planned split unavailable "
             "this run (RAM arena not ready) -- falling back to CPU-only "
             "for this layer\n",
             layer, stem);
  return r;
}

ggml_tensor *dense_weight(FullGraph &g, ggml_context *ctx, int layer,
                          const char *stem) {
  return resident_tensor(g, ctx, "blk." + std::to_string(layer) + "." + stem);
}

ggml_tensor *hc_mix(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                    ggml_tensor *norm, ggml_tensor *down, ggml_tensor *up,
                    ggml_tensor *inject, ggml_tensor **inject_out) {
  ggml_tensor *xn =
      ggml_rms_norm(ctx, x, g.engine->model_file.architecture_config.eps);
  xn = ggml_reshape_2d(ctx, xn, g.D * g.HC, g.S);
  xn = ggml_mul(ctx, xn, norm);
  ggml_tensor *lo = ggml_mul_mat(ctx, down, xn);
  lo = ggml_silu(ctx, ggml_scale(ctx, lo, 1.0f / static_cast<float>(g.HC)));
  ggml_tensor *gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, up, lo));
  ggml_tensor *gated = ggml_mul(ctx, xn, gate);
  ggml_tensor *g3 = ggml_reshape_3d(ctx, gated, g.D, g.HC, g.S);

  ggml_tensor *mixed =
      ggml_cont(ctx, ggml_view_2d(ctx, g3, g.D, g.S,
                                  ggml_row_size(g3->type, g.D) * g.HC, 0));
  for (int h = 1; h < g.HC; ++h) {
    ggml_tensor *s =
        ggml_view_2d(ctx, g3, g.D, g.S, ggml_row_size(g3->type, g.D) * g.HC,
                     ggml_row_size(g3->type, g.D) * h);
    mixed = ggml_add(ctx, mixed, s);
  }
  mixed = ggml_scale(ctx, mixed, 1.0f / static_cast<float>(g.HC));
  if (inject_out)
    *inject_out = inject ? ggml_mul_mat(ctx, inject, xn) : nullptr;
  return mixed;
}

ggml_tensor *hc_combine(FullGraph &g, ggml_context *ctx, ggml_tensor *residual,
                        ggml_tensor *block, ggml_tensor *inject) {
  ggml_tensor *w = ggml_scale(
      ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / g.HC)), 2.0f);
  w = ggml_reshape_3d(ctx, w, 1, g.HC, g.S);
  ggml_tensor *b = ggml_reshape_3d(ctx, block, g.D, 1, g.S);
  b = ggml_repeat_4d(ctx, b, g.D, g.HC, g.S, 1);
  return ggml_add(ctx, residual, ggml_mul(ctx, b, w));
}

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

ExpertSplitTables build_expert_split_tables(FullGraph &g, ggml_context *ctx,
                                            int n_experts,
                                            const std::vector<int> &gpu_ids) {
  ExpertSplitTables r;
  if (n_experts <= 0)
    return r;
  std::vector<int32_t> remap_gpu(static_cast<size_t>(n_experts), -1);
  std::vector<int32_t> remap_cpu(static_cast<size_t>(n_experts), 0);
  std::vector<float> mgpu(static_cast<size_t>(n_experts), 0.0f);
  std::vector<float> mcpu(static_cast<size_t>(n_experts), 1.0f);
  for (int e = 0; e < n_experts; ++e)
    remap_cpu[static_cast<size_t>(e)] = e;
  for (size_t k = 0; k < gpu_ids.size(); ++k) {
    const int e = gpu_ids[k];
    if (e < 0 || e >= n_experts)
      continue;
    remap_gpu[static_cast<size_t>(e)] = static_cast<int32_t>(k);
    remap_cpu[static_cast<size_t>(e)] = -1;
    mgpu[static_cast<size_t>(e)] = 1.0f;
    mcpu[static_cast<size_t>(e)] = 0.0f;
  }
  const std::vector<int64_t> shape = {1, n_experts, 1};
  r.remap_gpu = host_input(g, ctx, GGML_TYPE_I32, shape,
                           remap_gpu.size() * sizeof(int32_t),
                           "split_remap_gpu");
  r.remap_cpu = host_input(g, ctx, GGML_TYPE_I32, shape,
                           remap_cpu.size() * sizeof(int32_t),
                           "split_remap_cpu");
  r.mask_gpu = host_input(g, ctx, GGML_TYPE_F32, shape,
                          mgpu.size() * sizeof(float), "split_mask_gpu");
  r.mask_cpu = host_input(g, ctx, GGML_TYPE_F32, shape,
                          mcpu.size() * sizeof(float), "split_mask_cpu");
  if (!r.remap_gpu || !r.remap_cpu || !r.mask_gpu || !r.mask_cpu)
    return {};
  // host_input() binds these to already-allocated host memory (->data is
  // live immediately, same idiom this file already uses for g.positions);
  // written once here since a split layer's expert->device assignment
  // never changes for this graph's lifetime.
  std::memcpy(r.remap_gpu->data, remap_gpu.data(),
              remap_gpu.size() * sizeof(int32_t));
  std::memcpy(r.remap_cpu->data, remap_cpu.data(),
              remap_cpu.size() * sizeof(int32_t));
  std::memcpy(r.mask_gpu->data, mgpu.data(), mgpu.size() * sizeof(float));
  std::memcpy(r.mask_cpu->data, mcpu.data(), mcpu.size() * sizeof(float));
  return r;
}

ggml_tensor *build_moe(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *router,
                       ggml_tensor *selection_bias, ggml_tensor *gate_exps,
                       ggml_tensor *up_exps, ggml_tensor *down_exps,
                       ggml_tensor *gate_exps_gpu = nullptr,
                       ggml_tensor *up_exps_gpu = nullptr,
                       ggml_tensor *down_exps_gpu = nullptr,
                       const std::vector<int> *split_gpu_ids = nullptr,
                       ggml_tensor *gate_up_exps = nullptr) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_tensor *logits = ggml_mul_mat(ctx, router, cur);
  ggml_tensor *probs = ggml_soft_max(ctx, logits);
  ggml_tensor *selection =
      selection_bias ? ggml_add(ctx, probs, selection_bias) : probs;
  ggml_tensor *ids = ggml_argsort_top_k(ctx, selection, c.topk);
  ggml_tensor *probs3 = ggml_reshape_3d(ctx, probs, 1, c.n_experts, g.S);
  ggml_tensor *weights = ggml_get_rows(ctx, probs3, ids);
  ggml_tensor *weights2 = ggml_reshape_2d(ctx, weights, c.topk, g.S);
  if (c.expert_weights_norm) {
    ggml_tensor *sum = ggml_sum_rows(ctx, weights2);
    sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
    weights2 = ggml_div(ctx, weights2, sum);
  }
  if (c.expert_weights_scale != 0.0f && c.expert_weights_scale != 1.0f)
    weights2 = ggml_scale(ctx, weights2, c.expert_weights_scale);
  weights = ggml_reshape_3d(ctx, weights2, 1, c.topk, g.S);
  ggml_build_forward_expand(g.graph, weights);

  ggml_tensor *input3 = ggml_reshape_3d(ctx, cur, g.D, 1, g.S);
  ggml_tensor *experts = nullptr;

  // Genuine per-expert split path: route IDs are remapped independently for
  // each backend. Negative IDs are inactive slots understood by the GGML
  // MUL_MAT_ID kernels, so each branch computes only its assigned routes.
  // The masks remain as a defensive numerical guard at the merge. Every
  // other layer/config takes the unchanged single-mul_mat_id path below.
  if (split_gpu_ids && !split_gpu_ids->empty() && gate_exps_gpu &&
      up_exps_gpu && down_exps_gpu) {
    ExpertSplitTables tab =
        build_expert_split_tables(g, ctx, c.n_experts, *split_gpu_ids);
    if (tab.remap_gpu && tab.remap_cpu && tab.mask_gpu && tab.mask_cpu) {
      ggml_tensor *ids_gpu_raw = ggml_get_rows(ctx, tab.remap_gpu, ids);
      ggml_tensor *ids_gpu = ggml_reshape_2d(ctx, ids_gpu_raw, c.topk, g.S);
      ggml_tensor *ids_cpu_raw = ggml_get_rows(ctx, tab.remap_cpu, ids);
      ggml_tensor *ids_cpu = ggml_reshape_2d(ctx, ids_cpu_raw, c.topk, g.S);
      ggml_tensor *weights_gpu =
          ggml_mul(ctx, weights, ggml_get_rows(ctx, tab.mask_gpu, ids));
      ggml_tensor *weights_cpu =
          ggml_mul(ctx, weights, ggml_get_rows(ctx, tab.mask_cpu, ids));

      ggml_tensor *up_g = ggml_mul_mat_id(ctx, up_exps_gpu, input3, ids_gpu);
      ggml_tensor *gate_g = ggml_mul_mat_id(ctx, gate_exps_gpu, input3, ids_gpu);
      ggml_tensor *act_g = ggml_swiglu_split(ctx, gate_g, up_g);
      ggml_tensor *experts_gpu =
          ggml_mul_mat_id(ctx, down_exps_gpu, act_g, ids_gpu);

      ggml_tensor *up_c = ggml_mul_mat_id(ctx, up_exps, input3, ids_cpu);
      ggml_tensor *gate_c = ggml_mul_mat_id(ctx, gate_exps, input3, ids_cpu);
      ggml_tensor *act_c = ggml_swiglu_split(ctx, gate_c, up_c);
      ggml_tensor *experts_cpu =
          ggml_mul_mat_id(ctx, down_exps, act_c, ids_cpu);

      experts = ggml_add(ctx, ggml_mul(ctx, experts_gpu, weights_gpu),
                         ggml_mul(ctx, experts_cpu, weights_cpu));
    }
  }

  if (!experts) {
    ggml_tensor *up = nullptr;
    ggml_tensor *gate = nullptr;
    ggml_tensor *gate_up = nullptr; // only set on the fused path
    if (gate_up_exps) {
      // Bee's Qwen graph and the CPU runner use one routed matmul for the
      // adjacent gate+up planes, then split the result into two views.  The
      // tensor is a zero-copy view over the shared DDR arena for CPU layers;
      // GPU layers may use the same representation when available.
      gate_up = ggml_mul_mat_id(ctx, gate_up_exps, input3, ids);
      if (gate_up) {
        gate = ggml_view_3d(ctx, gate_up, gate_up->ne[0] / 2,
                            gate_up->ne[1], gate_up->ne[2], gate_up->nb[1],
                            gate_up->nb[2], 0);
        up = ggml_view_3d(ctx, gate_up, gate_up->ne[0] / 2,
                          gate_up->ne[1], gate_up->ne[2], gate_up->nb[1],
                          gate_up->nb[2],
                          static_cast<size_t>(gate_up->ne[0] / 2) *
                              gate_up->nb[0]);
      }
    } else {
      // Separate-tensor fallback for layers whose source arena cannot be
      // presented as adjacent, type-matched gate+up planes.
      up = ggml_mul_mat_id(ctx, up_exps, input3, ids);
      gate = ggml_mul_mat_id(ctx, gate_exps, input3, ids);
    }
    ggml_tensor *act = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor *down = ggml_mul_mat_id(ctx, down_exps, act, ids);
    experts = ggml_mul(ctx, down, weights);
    if (g.layer_timing_enabled) {
      // See FullGraph::layer_timing_layer_of's own comment -- these are
      // the actual CPU-dispatched op result tensors when this layer is
      // CPU-resident. On the fused path, `gate`/`up` are VIEWS of
      // `gate_up` (near-zero real compute, but the eval callback still
      // fires an ask=true/ask=false pair around them since they inherit
      // gate_up's CPU backend assignment) -- tag the real compute node
      // (`gate_up`) instead of double-counting the views, so the gap
      // count reflects genuine dispatch handoffs, not view bookkeeping.
      if (gate_up)
        g.layer_timing_layer_of[gate_up] = {layer, "moe_gate_up_fused"};
      else {
        if (up)
          g.layer_timing_layer_of[up] = {layer, "moe_up"};
        if (gate)
          g.layer_timing_layer_of[gate] = {layer, "moe_gate"};
      }
      g.layer_timing_layer_of[act] = {layer, "moe_swiglu"};
      g.layer_timing_layer_of[down] = {layer, "moe_down"};
    }
  }

  ggml_tensor *out = ggml_view_2d(ctx, experts, g.D, g.S, experts->nb[2], 0);
  out = ggml_cont(ctx, out);
  for (int k = 1; k < c.topk; ++k) {
    ggml_tensor *e = ggml_view_2d(ctx, experts, g.D, g.S, experts->nb[2],
                                  static_cast<size_t>(k) * experts->nb[1]);
    out = ggml_add(ctx, out, e);
  }
  (void)layer;
  return out;
}

bool qwen_moe_layer_cpu_only(const FullGraph &g, int layer) {
  const auto split_it = g.split_gpu_expert_ids.find(layer);
  if (split_it != g.split_gpu_expert_ids.end() && !split_it->second.empty())
    return false;
  return !graph_expert_projection_resident(g, layer, "ffn_gate_exps.weight") &&
         !graph_expert_projection_resident(g, layer, "ffn_up_exps.weight") &&
         !graph_expert_projection_resident(g, layer, "ffn_down_exps.weight");
}

void log_qwen_layout_tensor(const FullGraph &g, int layer, bool cpu_only,
                            const char *role, const ggml_tensor *tensor) {
  if (!tensor)
    return;
  ggml_backend_buffer_t buffer = tensor->buffer;
  if (!buffer && tensor->view_src)
    buffer = tensor->view_src->buffer;
  const char *buft = buffer
                         ? ggml_backend_buft_name(
                               ggml_backend_buffer_get_type(buffer))
                         : "unallocated";
  const char *backend_name = "unassigned";
  if (g.sched) {
    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(
        g.sched, const_cast<ggml_tensor *>(tensor));
    if (backend)
      backend_name = ggml_backend_name(backend);
  }
  const char *pinned = "unknown";
  if (buffer && ggml_backend_buffer_is_host(buffer)) {
    void *base = ggml_backend_buffer_get_base(buffer);
    unsigned int flags = 0;
    if (base && cudaHostGetFlags(&flags, base) == cudaSuccess) {
      pinned = "yes";
    } else {
      cudaGetLastError();
      pinned = "no";
    }
  }
  LOG_INFO(
      "[GGML-QWEN-LAYOUT] layer=%d cpu_only=%d role=%s op=%s name=%s "
      "type=%s nbytes=%zu ne=[%lld,%lld,%lld,%lld] "
      "nb=[%zu,%zu,%zu,%zu] buft=%s backend=%s pinned=%s\n",
      layer, cpu_only ? 1 : 0, role, ggml_op_name(tensor->op),
      ggml_get_name(tensor), ggml_type_name(tensor->type),
      ggml_nbytes(tensor), (long long) tensor->ne[0],
      (long long) tensor->ne[1], (long long) tensor->ne[2],
      (long long) tensor->ne[3], tensor->nb[0], tensor->nb[1], tensor->nb[2],
      tensor->nb[3], buft, backend_name, pinned);
}

void dump_qwen_moe_layout(const FullGraph &g) {
  if (!graph_layout_debug() || !g.graph || !g.sched)
    return;
  std::unordered_set<const ggml_tensor *> logged;
  for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
    ggml_tensor *node = ggml_graph_node(g.graph, i);
    if (!node || node->op != GGML_OP_MUL_MAT_ID || !node->src[0])
      continue;
    ggml_backend_t backend =
        ggml_backend_sched_get_tensor_backend(g.sched, node);
    if (backend != g.cpu)
      continue;
    const int layer = layer_timing_parse_layer(ggml_get_name(node->src[0]));
    if (layer < 0)
      continue;
    const bool cpu_only = qwen_moe_layer_cpu_only(g, layer);
    const char *roles[] = {"weight", "input", "ids"};
    for (int src = 0; src < 3; ++src) {
      ggml_tensor *tensor = node->src[src];
      if (!tensor || !logged.insert(tensor).second)
        continue;
      log_qwen_layout_tensor(g, layer, cpu_only, roles[src], tensor);
    }
  }
}

ggml_tensor *build_shared(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                          int layer, ggml_tensor *gate, ggml_tensor *up,
                          ggml_tensor *down, ggml_tensor *scalar_gate) {
  if (!gate || !up || !down)
    return nullptr;
  ggml_tensor *gate_out = ggml_mul_mat(ctx, gate, cur);
  ggml_tensor *up_out = ggml_mul_mat(ctx, up, cur);
  ggml_tensor *act = ggml_swiglu_split(ctx, gate_out, up_out);
  ggml_tensor *out = ggml_mul_mat(ctx, down, act);
  if (scalar_gate) {
    ggml_tensor *sg = ggml_sigmoid(ctx, ggml_mul_mat(ctx, scalar_gate, cur));
    out = ggml_mul(ctx, out, sg);
  }
  (void)layer;
  return out;
}

ggml_tensor *build_gdn(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *w_qkv, ggml_tensor *w_gate,
                       ggml_tensor *w_alpha, ggml_tensor *w_beta,
                       ggml_tensor *w_dt, ggml_tensor *w_a, ggml_tensor *w_conv,
                       ggml_tensor *w_norm, ggml_tensor *w_out,
                       ggml_tensor *hist_in, ggml_tensor *state_in,
                       ggml_tensor **state_out, ggml_tensor **hist_out) {
  ggml_tensor *qkv = ggml_mul_mat(ctx, w_qkv, cur);
  ggml_tensor *z = ggml_mul_mat(ctx, w_gate, cur);
  ggml_tensor *alpha = ggml_mul_mat(ctx, w_alpha, cur);
  ggml_tensor *beta_raw = ggml_mul_mat(ctx, w_beta, cur);
  ggml_tensor *glog =
      ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, w_dt)), w_a);
  ggml_tensor *beta = ggml_sigmoid(ctx, beta_raw);

  ggml_tensor *conv_in = ggml_concat(ctx, hist_in, ggml_transpose(ctx, qkv), 0);
  ggml_tensor *conv = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_in, w_conv));
  const size_t row = ggml_row_size(GGML_TYPE_F32, g.conv_dim);
  ggml_tensor *q =
      ggml_view_4d(ctx, conv, g.gdn_hd, g.Hk_gdn, g.S, 1,
                   ggml_row_size(GGML_TYPE_F32, g.gdn_hd), row, row * g.S, 0);
  ggml_tensor *k =
      ggml_view_4d(ctx, conv, g.gdn_hd, g.Hk_gdn, g.S, 1,
                   ggml_row_size(GGML_TYPE_F32, g.gdn_hd), row, row * g.S,
                   ggml_row_size(GGML_TYPE_F32, g.key_dim));
  ggml_tensor *v =
      ggml_view_4d(ctx, conv, g.gdn_hd, g.Hv_gdn, g.S, 1,
                   ggml_row_size(GGML_TYPE_F32, g.gdn_hd), row, row * g.S,
                   ggml_row_size(GGML_TYPE_F32, 2 * g.key_dim));
  q = ggml_l2_norm(ctx, q, g.engine->model_file.architecture_config.eps);
  k = ggml_l2_norm(ctx, k, g.engine->model_file.architecture_config.eps);
  glog = ggml_reshape_4d(ctx, glog, 1, g.Hv_gdn, g.S, 1);
  beta = ggml_reshape_4d(ctx, beta, 1, g.Hv_gdn, g.S, 1);
  ggml_tensor *gdn =
      ggml_gated_delta_net(ctx, q, k, v, glog, beta, state_in, 1);

  const size_t hd_row = ggml_row_size(GGML_TYPE_F32, g.gdn_hd);
  const size_t value_row = ggml_row_size(GGML_TYPE_F32, g.value_dim);
  const size_t state_row = ggml_row_size(GGML_TYPE_F32, g.gdn_hd * g.gdn_hd);
  ggml_tensor *core = ggml_view_4d(ctx, gdn, g.gdn_hd, g.Hv_gdn, g.S, 1, hd_row,
                                   value_row, value_row * g.S, 0);
  // ggml-cuda's gated_delta_net kernel stores the per-head state TRANSPOSED
  // relative to the [k][v] math convention this codebase's own legacy CPU
  // qwen4exp_gdn_delta_step uses -- confirmed directly in
  // ggml/src/ggml-cuda/gated_delta_net.cu's own comment ("state is stored
  // transposed: M[col][i] = S[i][col]", where col indexes v and i indexes
  // k), i.e. the raw view below has ne[0]=k (contiguous), ne[1]=v. Legacy's
  // own flat layout (qwen4exp_recur_state[l], engine.hpp) is st[k*hd+v] --
  // v contiguous, k slower -- the opposite axis order. Left unfixed, the
  // flat ggml_backend_tensor_get+memcpy in qwen4exp_full_graph_run silently
  // hands decode a transposed state matrix: same byte count, same total
  // size (gdn_hd==gdn_hd), no crash, just wrong values from the very first
  // decode token onward. Transpose+materialize here so the exported
  // tensor's own byte layout already matches what the flat copy-out assumes.
  ggml_tensor *state_raw =
      ggml_view_4d(ctx, gdn, g.gdn_hd, g.gdn_hd, g.Hv_gdn, 1, hd_row, state_row,
                   state_row * g.Hv_gdn,
                   static_cast<size_t>(g.value_dim) * g.S * sizeof(float));
  *state_out = ggml_cont(ctx, ggml_transpose(ctx, state_raw));
  ggml_tensor *norm = ggml_mul(
      ctx,
      ggml_rms_norm(ctx, core, g.engine->model_file.architecture_config.eps),
      w_norm);
  ggml_tensor *z4 = ggml_reshape_4d(ctx, z, g.gdn_hd, g.Hv_gdn, g.S, 1);
  ggml_tensor *gated = ggml_mul(ctx, norm, ggml_sigmoid(ctx, z4));
  ggml_tensor *out =
      ggml_mul_mat(ctx, w_out, ggml_reshape_2d(ctx, gated, g.value_dim, g.S));
  *hist_out =
      ggml_cont(ctx, ggml_view_3d(ctx, conv_in, g.d_conv - 1, g.conv_dim, 1,
                                  conv_in->nb[1], conv_in->nb[2],
                                  ggml_row_size(conv_in->type, g.S)));
  (void)layer;
  return out;
}

ggml_tensor *build_full_attention(FullGraph &g, ggml_context *ctx,
                                  ggml_tensor *cur, int layer, ggml_tensor *w_q,
                                  ggml_tensor *w_k, ggml_tensor *w_v,
                                  ggml_tensor *w_qn, ggml_tensor *w_kn,
                                  ggml_tensor *w_o, ggml_tensor **k_cache,
                                  ggml_tensor **v_cache) {
  ggml_tensor *qfull = ggml_mul_mat(ctx, w_q, cur);
  ggml_tensor *q =
      ggml_view_3d(ctx, qfull, g.hd, g.H, g.S, sizeof(float) * g.hd * 2,
                   sizeof(float) * g.hd * 2 * g.H, 0);
  q = ggml_mul(
      ctx, ggml_rms_norm(ctx, q, g.engine->model_file.architecture_config.eps),
      w_qn);
  ggml_tensor *k =
      ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w_k, cur), g.hd, g.Hkv, g.S);
  k = ggml_mul(
      ctx, ggml_rms_norm(ctx, k, g.engine->model_file.architecture_config.eps),
      w_kn);
  ggml_tensor *v =
      ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w_v, cur), g.hd, g.Hkv, g.S);
  const int n_rot = g.engine->model_file.architecture_config.rope_n_rot > 0
                        ? g.engine->model_file.architecture_config.rope_n_rot
                        : g.hd;
  q = ggml_rope(ctx, q, g.positions, n_rot, GGML_ROPE_TYPE_NORMAL);
  k = ggml_rope(ctx, k, g.positions, n_rot, GGML_ROPE_TYPE_NORMAL);
  // The host dense KV cache stores the same normalized and rotated K/V values
  // that Bee's graph writes to its cache.  Keep these F32 tensors before the
  // layout-only permute/cast used by flash attention.
  if (k_cache)
    *k_cache = k;
  if (v_cache)
    *v_cache = v;
  q = ggml_permute(ctx, q, 0, 2, 1, 3);
  k = ggml_permute(ctx, k, 0, 2, 1, 3);
  v = ggml_permute(ctx, v, 0, 2, 1, 3);
  // The CUDA MMA flash-attention implementation intentionally consumes Q as
  // F32 and converts/loads K and V as F16.  Casting Q to F16 selects the MMA
  // path and then trips its Q-type assertion during graph execution.
  if (k->type == GGML_TYPE_F32)
    k = ggml_cast(ctx, k, GGML_TYPE_F16);
  if (v->type == GGML_TYPE_F32)
    v = ggml_cast(ctx, v, GGML_TYPE_F16);
  ggml_tensor *attn = ggml_flash_attn_ext(
      ctx, q, k, v, g.attn_mask, 1.0f / std::sqrt(static_cast<float>(g.hd)),
      0.0f, 0.0f);
  ggml_tensor *gate =
      ggml_view_3d(ctx, qfull, g.hd, g.H, g.S, sizeof(float) * g.hd * 2,
                   sizeof(float) * g.hd * 2 * g.H, sizeof(float) * g.hd);
  gate = ggml_reshape_3d(ctx, ggml_cont_2d(ctx, gate, g.H * g.hd, g.S), g.hd,
                         g.H, g.S);
  attn = ggml_mul(ctx, attn, ggml_sigmoid(ctx, gate));
  attn = ggml_reshape_2d(ctx, attn, g.H * g.hd, g.S);
  (void)layer;
  return ggml_mul_mat(ctx, w_o, attn);
}

// Decode variant of build_full_attention().  K/V are appended directly to a
// device-resident cache with one set_rows node.  The graph therefore carries
// the cache update, attention, and output projection in the same scheduler
// submission; no host K/V readback is needed between tokens.
ggml_tensor *build_full_attention_decode(
    FullGraph &g, ggml_context *ctx, ggml_tensor *cur, int layer,
    ggml_tensor *w_q, ggml_tensor *w_k, ggml_tensor *w_v, ggml_tensor *w_qn,
    ggml_tensor *w_kn, ggml_tensor *w_o, ggml_tensor *k_storage,
    ggml_tensor *v_storage) {
  ggml_tensor *qfull = ggml_mul_mat(ctx, w_q, cur);
  ggml_tensor *q = ggml_view_3d(
      ctx, qfull, g.hd, g.H, 1, sizeof(float) * g.hd * 2,
      sizeof(float) * g.hd * 2 * g.H, 0);
  q = ggml_mul(
      ctx, ggml_rms_norm(ctx, q, g.engine->model_file.architecture_config.eps),
      w_qn);
  ggml_tensor *k = ggml_reshape_3d(
      ctx, ggml_mul_mat(ctx, w_k, cur), g.hd, g.Hkv, 1);
  k = ggml_mul(
      ctx, ggml_rms_norm(ctx, k, g.engine->model_file.architecture_config.eps),
      w_kn);
  ggml_tensor *v = ggml_reshape_3d(
      ctx, ggml_mul_mat(ctx, w_v, cur), g.hd, g.Hkv, 1);
  const int n_rot = g.engine->model_file.architecture_config.rope_n_rot > 0
                        ? g.engine->model_file.architecture_config.rope_n_rot
                        : g.hd;
  q = ggml_rope(ctx, q, g.positions, n_rot, GGML_ROPE_TYPE_NORMAL);
  k = ggml_rope(ctx, k, g.positions, n_rot, GGML_ROPE_TYPE_NORMAL);

  ggml_tensor *k2 = ggml_reshape_2d(ctx, k, g.hd, g.Hkv);
  ggml_tensor *v2 = ggml_reshape_2d(ctx, v, g.hd, g.Hkv);
  ggml_tensor *kset = ggml_set_rows(ctx, k_storage, k2, g.decode_indices);
  ggml_tensor *vset = ggml_set_rows(ctx, v_storage, v2, g.decode_indices);
  ggml_tensor *kcache = ggml_reshape_4d(
      ctx, kset, g.hd, g.kv_capacity, g.Hkv, 1);
  ggml_tensor *vcache = ggml_reshape_4d(
      ctx, vset, g.hd, g.kv_capacity, g.Hkv, 1);

  q = ggml_permute(ctx, q, 0, 2, 1, 3);
  if (kcache->type == GGML_TYPE_F32)
    kcache = ggml_cast(ctx, kcache, GGML_TYPE_F16);
  if (vcache->type == GGML_TYPE_F32)
    vcache = ggml_cast(ctx, vcache, GGML_TYPE_F16);
  ggml_tensor *attn = ggml_flash_attn_ext(
      ctx, q, kcache, vcache, g.attn_mask,
      1.0f / std::sqrt(static_cast<float>(g.hd)), 0.0f, 0.0f);
  ggml_tensor *gate = ggml_view_3d(
      ctx, qfull, g.hd, g.H, 1, sizeof(float) * g.hd * 2,
      sizeof(float) * g.hd * 2 * g.H, sizeof(float) * g.hd);
  gate = ggml_reshape_3d(ctx, ggml_cont_2d(ctx, gate, g.H * g.hd, 1), g.hd,
                         g.H, 1);
  attn = ggml_mul(ctx, attn, ggml_sigmoid(ctx, gate));
  attn = ggml_reshape_2d(ctx, attn, g.H * g.hd, 1);
  (void)layer;
  return ggml_mul_mat(ctx, w_o, attn);
}

bool prepare_ple_embedding(FullGraph &g, std::vector<float> &embedding, int S,
                           int pos_base, const int *token_ids) {
  const auto &c = g.engine->model_file.architecture_config;
  if (!has_ple_layer(c) || !token_ids ||
      (pos_base != 0 && !(g.decode_mode && S == 1)))
    return !has_ple_layer(c);
  const GGUFTensor *table =
      model_tensor(*g.engine, "per_layer_token_embd.weight");
  if (!table || table->shape.size() != 2)
    return false;
  const int row_dim = static_cast<int>(table->shape[0]);
  const int rb = row_bytes(table->type, row_dim);
  if (rb <= 0 || row_dim != c.ple_n_embd_per_layer)
    return false;

  std::vector<int32_t> rows(static_cast<size_t>(S) * c.ple_n_heads);
  const int64_t eos = c.ple_eos_token_id;
  for (int i = 0; i < S; ++i) {
    int64_t ctx[8] = {};
    if (c.ple_ngram_size > 8)
      return false;
    ctx[0] = token_ids[i];
    bool cut = false;
    for (int s = 1; s < c.ple_ngram_size; ++s) {
      if (g.decode_mode) {
        const int64_t pos = pos_base + i;
        const int64_t back = s - i;
        const auto &history = g.engine->qwen4exp_ple_tok_hist;
        if (g.engine->qwen4exp_ple_next_pos == pos && back > 0 &&
            back <= static_cast<int64_t>(history.size())) {
          const size_t index = history.size() - static_cast<size_t>(back);
          ctx[s] = history[index];
        } else {
          ctx[s] = eos;
        }
      } else {
        const int j = i - s;
        ctx[s] = j >= 0 ? token_ids[j] : eos;
      }
      if (cut || ctx[s] == eos) {
        ctx[s] = eos;
        cut = true;
      }
    }
    for (int n = 2; n <= c.ple_ngram_size; ++n) {
      int base = (n - 2) * c.ple_heads_per_ngram;
      for (int h = 0; h < c.ple_heads_per_ngram; ++h) {
        int hi = base + h;
        rows[static_cast<size_t>(i) * c.ple_n_heads + hi] =
            static_cast<int32_t>(
                ple_ngram_row(ctx, n, c.ple_layer_multipliers.data(),
                              c.ple_head_vocab_sizes[static_cast<size_t>(hi)],
                              c.ple_head_offsets[static_cast<size_t>(hi)]));
      }
    }
  }

  std::unordered_map<int32_t, std::vector<float>> cache;
  void *raw_mem = nullptr;
  if (posix_memalign(&raw_mem, 4096, static_cast<size_t>(rb)) != 0 ||
      !raw_mem)
    return false;
  std::unique_ptr<uint8_t, decltype(&std::free)> raw_guard(
      static_cast<uint8_t *>(raw_mem), &std::free);
  auto *raw = raw_guard.get();
  const int fd = g.engine->model_file.fd_for_tensor(*table);
  for (int32_t r : rows) {
    if (cache.count(r))
      continue;
    const ssize_t got = direct_io::pread_direct_aware(
        fd, g.engine->model_file.direct_io_active_for(*table),
        g.engine->model_file.direct_io_alignment, raw, static_cast<size_t>(rb),
        table->offset + static_cast<uint64_t>(r) * static_cast<size_t>(rb));
    if (got != static_cast<ssize_t>(rb))
      return false;
    std::vector<float> row(static_cast<size_t>(row_dim));
    if (!dequant_row(table->type, raw, row.data(), row_dim))
      return false;
    cache.emplace(r, std::move(row));
  }
  embedding.resize(static_cast<size_t>(S) * c.ple_n_heads * row_dim);
  for (int i = 0; i < S; ++i) {
    for (int h = 0; h < c.ple_n_heads; ++h) {
      const auto &row =
          cache.at(rows[static_cast<size_t>(i) * c.ple_n_heads + h]);
      std::memcpy(embedding.data() +
                      (static_cast<size_t>(i) * c.ple_n_heads + h) * row_dim,
                  row.data(), static_cast<size_t>(row_dim) * sizeof(float));
    }
  }
  return true;
}

bool build_graph(FullGraph &g, const std::vector<float> *ple_embedding) {
  // Timing split (2026-09-17): full-graph decode's FIRST call pays a real
  // ~5.3s one-time cost, invisible in any --ngen 400 average (6000/400=
  // 15ms/token) but a genuine multi-second latency cliff for interactive
  // use (every new conversation/context reset pays it). See memory:
  // full-graph-decode-graph-construction-5s-latency.md. This split isolates
  // OUR host-side node-construction C++ (this function's own body) from
  // ggml's own ggml_backend_sched_alloc_graph() call below, since the fix
  // differs completely depending on which one dominates.
  const auto t_build_graph_start = std::chrono::steady_clock::now();
  g.dense_upload_ms_total = 0.0;
  g.expert_upload_ms_total = 0.0;
  g.expert_upload_calls = 0;
  g.attn_construction_ms_total = 0.0;
  g.moe_construction_ms_total = 0.0;
  g.moe_weight_lookup_ms_total = 0.0;
  const auto &c = g.engine->model_file.architecture_config;
  ggml_init_params ip = {};
  ip.mem_size = kTensorContextBytes;
  ip.no_alloc = true;
  if (!MemoryGuard::try_reserve_host_bytes(kTensorContextBytes)) {
    LOG_WARN("[GGML-QWEN-FULL-MEM] host graph context refused: %.2f MiB "
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
                       static_cast<size_t>(g.D) * g.HC * g.S * sizeof(float),
                       "qwen_full_input");
  g.positions = host_input(g, g.ctx, GGML_TYPE_I32, {g.S},
                           static_cast<size_t>(g.S) * sizeof(int32_t),
                           "qwen_full_positions");
  if (g.decode_mode) {
    g.attn_mask = host_input(
        g, g.ctx, GGML_TYPE_F16, {g.kv_capacity, 1, 1, 1},
        static_cast<size_t>(g.kv_capacity) * sizeof(ggml_fp16_t),
        "qwen_full_decode_mask");
    g.decode_indices = host_input(
        g, g.ctx, GGML_TYPE_I32, {g.Hkv},
        static_cast<size_t>(g.Hkv) * sizeof(int32_t),
        "qwen_full_decode_kv_indices");
  } else {
    g.attn_mask = host_input(
        g, g.ctx, GGML_TYPE_F16, {g.S, g.S},
        static_cast<size_t>(g.S) * g.S * sizeof(ggml_fp16_t),
        "qwen_full_causal_mask");
  }
  if (!g.input || !g.positions || !g.attn_mask ||
      (g.decode_mode && !g.decode_indices))
    return false;
  auto *pos = static_cast<int32_t *>(g.positions->data);
  auto *mask = static_cast<ggml_fp16_t *>(g.attn_mask->data);
  for (int i = 0; i < g.S; ++i)
    pos[i] = i;
  if (g.decode_mode) {
    for (int k = 0; k < g.kv_capacity; ++k)
      mask[k] = ggml_fp32_to_fp16(0.0f);
    auto *indices = static_cast<int32_t *>(g.decode_indices->data);
    for (int h = 0; h < g.Hkv; ++h)
      indices[h] = h;
  } else {
    for (int q = 0; q < g.S; ++q)
      for (int k = 0; k < g.S; ++k)
        mask[static_cast<size_t>(q) * g.S + k] =
            ggml_fp32_to_fp16(k <= q ? 0.0f : -INFINITY);
  }

  ggml_tensor *res = g.input;
  if (graph_match_bee_topology()) {
    // The scheduler normally starts directly on the first GPU-weighted
    // operation because g.input is already a CPU buffer.  An explicit CPU
    // CONT makes the initial CPU->GPU handoff visible as the same leading
    // CPU/GPU split that Bee has around model.input_embed/hc_init.
    res = ggml_cont(g.ctx, res);
    if (!res)
      return false;
    ggml_set_name(res, "qwen_full_bee_cpu_prologue");
    g.topology_cpu_prologue = res;
    if (g.sched)
      ggml_backend_sched_set_tensor_backend(g.sched, res, g.cpu);
  }
  for (int l = 0; l < g.n_layers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    if (qwen4exp_is_ple_layer(c, l)) {
      if (!ple_embedding)
        return false;
      g.ple_embedding =
          host_input(g, g.ctx, GGML_TYPE_F32,
                     {c.ple_n_embd_per_layer * c.ple_n_heads, g.S},
                     static_cast<size_t>(c.ple_n_embd_per_layer) *
                         c.ple_n_heads * g.S * sizeof(float),
                     "qwen_full_ple_embedding");
      if (!g.ple_embedding)
        return false;
      std::memcpy(g.ple_embedding->data, ple_embedding->data(),
                  ple_embedding->size() * sizeof(float));
      g.ple_layer = l;
      // PLE graph is constructed below with the same equations as Bee's
      // build_ple.  The input embedding is host gathered, while its learned
      // projections and the rest of the operation stay in this graph.
      ggml_tensor *ple_key = resident_tensor(g, g.ctx, p + "ple_key.weight");
      ggml_tensor *ple_value =
          resident_tensor(g, g.ctx, p + "ple_value.weight");
      ggml_tensor *ple_nk =
          resident_tensor(g, g.ctx, p + "ple_norm_key.weight");
      ggml_tensor *ple_nq =
          resident_tensor(g, g.ctx, p + "ple_norm_query.weight");
      ggml_tensor *ple_nc =
          resident_tensor(g, g.ctx, p + "ple_norm_conv.weight");
      ggml_tensor *ple_conv =
          resident_tensor(g, g.ctx, p + "ple_conv1d.weight");
      if (!ple_key || !ple_value || !ple_nk || !ple_nq || !ple_nc || !ple_conv)
        return false;
      ggml_tensor *pk = ggml_mul_mat(g.ctx, ple_key, g.ple_embedding);
      ggml_tensor *pv = ggml_mul_mat(g.ctx, ple_value, g.ple_embedding);
      auto group_norm = [&](ggml_tensor *x, ggml_tensor *w) {
        x = ggml_reshape_3d(g.ctx, x, g.D, g.HC, g.S);
        x = ggml_rms_norm(g.ctx, x, c.eps);
        x = ggml_reshape_2d(g.ctx, x, g.D * g.HC, g.S);
        x = ggml_mul(g.ctx, x, w);
        return ggml_reshape_3d(g.ctx, x, g.D, g.HC, g.S);
      };
      pk = group_norm(pk, ple_nk);
      ggml_tensor *pq = group_norm(res, ple_nq);
      ggml_tensor *score =
          ggml_scale(g.ctx, ggml_sum_rows(g.ctx, ggml_mul(g.ctx, pk, pq)),
                     1.0f / std::sqrt(static_cast<float>(g.D)));
      ggml_tensor *gate = ggml_sigmoid(
          g.ctx,
          ggml_mul(g.ctx, ggml_sgn(g.ctx, score),
                   ggml_sqrt(g.ctx, ggml_clamp(g.ctx, ggml_abs(g.ctx, score),
                                               1e-6f, INFINITY))));
      ggml_tensor *v3 = ggml_repeat_4d(
          g.ctx, ggml_reshape_3d(g.ctx, pv, g.D, 1, g.S), g.D, g.HC, g.S, 1);
      ggml_tensor *gated = ggml_mul(g.ctx, v3, gate);
      ggml_tensor *normed =
          group_norm(ggml_reshape_2d(g.ctx, gated, g.D * g.HC, g.S), ple_nc);
      normed = ggml_reshape_2d(g.ctx, normed, g.D * g.HC, g.S);

      const int hist = (c.ple_conv_kernel - 1) * c.ple_ngram_size;
      ggml_tensor *ph = nullptr;
      if (g.decode_mode) {
        ph = new_device_state_tensor(
            g, GGML_TYPE_F32, {hist, g.D * g.HC, 1},
            "qwen_full_decode_ple_history");
        g.ple_hist_storage = ph;
      } else {
        ph = host_input(
            g, g.ctx, GGML_TYPE_F32, {hist, g.D * g.HC, 1},
            static_cast<size_t>(hist) * g.D * g.HC * sizeof(float),
            "qwen_full_ple_history");
      }
      if (!ph)
        return false;
      if (!g.decode_mode)
        g.ple_hist_input = static_cast<float *>(ph->data);
      if (!g.decode_mode &&
          l < static_cast<int>(g.engine->qwen4exp_ple_conv_history.size())) {
        auto &h = g.engine->qwen4exp_ple_conv_history[l];
        if (h.size() == static_cast<size_t>(hist) * g.D * g.HC)
          std::memcpy(ph->data, h.data(), h.size() * sizeof(float));
      }
      ggml_tensor *padded =
          ggml_concat(g.ctx, ph, ggml_transpose(g.ctx, normed), 0);
      ggml_tensor *conv_out = nullptr;
      for (int k = 0; k < c.ple_conv_kernel; ++k) {
        const int start = hist - (c.ple_conv_kernel - 1 - k) * c.ple_ngram_size;
        ggml_tensor *shifted = ggml_cont(
            g.ctx,
            ggml_transpose(g.ctx,
                           ggml_view_3d(g.ctx, padded, g.S, g.D * g.HC, 1,
                                        padded->nb[1], padded->nb[2],
                                        ggml_row_size(padded->type, start))));
        ggml_tensor *wk = ggml_cont(
            g.ctx, ggml_view_2d(g.ctx, ple_conv, 1, g.D * g.HC, ple_conv->nb[1],
                                static_cast<size_t>(k) * ple_conv->nb[0]));
        wk = ggml_reshape_1d(g.ctx, wk, g.D * g.HC);
        if (wk->type != GGML_TYPE_F32)
          wk = ggml_cast(g.ctx, wk, GGML_TYPE_F32);
        ggml_tensor *term = ggml_mul(g.ctx, shifted, wk);
        conv_out = conv_out ? ggml_add(g.ctx, conv_out, term) : term;
      }
      conv_out =
          ggml_reshape_3d(g.ctx, ggml_silu(g.ctx, conv_out), g.D, g.HC, g.S);
      res = ggml_add(g.ctx, res, ggml_add(g.ctx, gated, conv_out));
      g.ple_hist_out = ggml_cont(
          g.ctx, ggml_view_3d(g.ctx, padded, hist, g.D * g.HC, 1, padded->nb[1],
                              padded->nb[2], ggml_row_size(padded->type, g.S)));
      if (g.decode_mode) {
        g.ple_hist_update =
            ggml_cpy(g.ctx, g.ple_hist_out, g.ple_hist_storage);
      }
    }

    const auto t_attn_start = std::chrono::steady_clock::now();
    ggml_tensor *inject = nullptr;
    ggml_tensor *hn = resident_tensor(g, g.ctx, p + "hc_attn_norm.weight");
    ggml_tensor *hdn = resident_tensor(g, g.ctx, p + "hc_attn_down.weight");
    ggml_tensor *hup = resident_tensor(g, g.ctx, p + "hc_attn_up.weight");
    ggml_tensor *hinj = resident_tensor(g, g.ctx, p + "hc_attn_inject.weight");
    if (!hn || !hdn || !hup || !hinj)
      return false;
    ggml_tensor *attn_in = hc_mix(g, g.ctx, res, hn, hdn, hup, hinj, &inject);
    ggml_tensor *attn_out = nullptr;
    if (qwen4exp_is_gdn_layer(c, l)) {
      ggml_tensor *qkv = resident_tensor(g, g.ctx, p + "attn_qkv.weight");
      ggml_tensor *zg = resident_tensor(g, g.ctx, p + "attn_gate.weight");
      ggml_tensor *al = resident_tensor(g, g.ctx, p + "ssm_alpha.weight");
      ggml_tensor *be = resident_tensor(g, g.ctx, p + "ssm_beta.weight");
      ggml_tensor *dt = resident_tensor(g, g.ctx, p + "ssm_dt.bias");
      ggml_tensor *aa = resident_tensor(g, g.ctx, p + "ssm_a");
      ggml_tensor *cv = resident_tensor(g, g.ctx, p + "ssm_conv1d.weight");
      ggml_tensor *nm = resident_tensor(g, g.ctx, p + "ssm_norm.weight");
      ggml_tensor *oo = resident_tensor(g, g.ctx, p + "ssm_out.weight");
      if (!qkv || !zg || !al || !be || !dt || !aa || !cv || !nm || !oo)
        return false;
      const int hist = g.d_conv - 1;
      ggml_tensor *hi = nullptr;
      ggml_tensor *si = nullptr;
      if (g.decode_mode) {
        hi = new_device_state_tensor(g, GGML_TYPE_F32, {hist, g.conv_dim, 1},
                                      "qwen_full_decode_gdn_history");
        si = new_device_state_tensor(
            g, GGML_TYPE_F32, {g.gdn_hd, g.gdn_hd, g.Hv_gdn, 1},
            "qwen_full_decode_gdn_state");
        g.gdn_hist_storage.push_back(hi);
        g.gdn_state_storage.push_back(si);
      } else {
        hi = host_input(
            g, g.ctx, GGML_TYPE_F32, {hist, g.conv_dim, 1},
            static_cast<size_t>(hist) * g.conv_dim * sizeof(float),
            "qwen_full_gdn_history");
        si = host_input(
            g, g.ctx, GGML_TYPE_F32, {g.gdn_hd, g.gdn_hd, g.Hv_gdn, 1},
            static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn *
                sizeof(float),
            "qwen_full_gdn_state");
      }
      if (!hi || !si)
        return false;
      if (g.decode_mode) {
        g.gdn_hist_updates.push_back(nullptr);
        g.gdn_state_updates.push_back(nullptr);
      } else {
        g.gdn_hist_inputs.push_back(static_cast<float *>(hi->data));
        g.gdn_state_inputs.push_back(static_cast<float *>(si->data));
        if (l < static_cast<int>(g.engine->qwen4exp_conv_history.size()) &&
            g.engine->qwen4exp_conv_history[l].size() ==
                static_cast<size_t>(hist) * g.conv_dim)
          std::memcpy(hi->data, g.engine->qwen4exp_conv_history[l].data(),
                      static_cast<size_t>(hist) * g.conv_dim * sizeof(float));
        if (l < static_cast<int>(g.engine->qwen4exp_recur_state.size()) &&
            g.engine->qwen4exp_recur_state[l].size() ==
                static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn)
          std::memcpy(si->data, g.engine->qwen4exp_recur_state[l].data(),
                      static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn *
                          sizeof(float));
      }
      ggml_tensor *so = nullptr, *ho = nullptr;
      attn_out = build_gdn(g, g.ctx, attn_in, l, qkv, zg, al, be, dt, aa, cv,
                           nm, oo, hi, si, &so, &ho);
      if (g.decode_mode) {
        g.gdn_state_updates.back() =
            ggml_cpy(g.ctx, so, g.gdn_state_storage.back());
        g.gdn_hist_updates.back() =
            ggml_cpy(g.ctx, ho, g.gdn_hist_storage.back());
      } else {
        g.gdn_state_out.push_back(so);
        g.gdn_hist_out.push_back(ho);
      }
      g.gdn_layers.push_back(l);
    } else {
      ggml_tensor *wq = dense_weight(g, g.ctx, l, "attn_q.weight");
      ggml_tensor *wk = dense_weight(g, g.ctx, l, "attn_k.weight");
      ggml_tensor *wv = dense_weight(g, g.ctx, l, "attn_v.weight");
      ggml_tensor *wqn = dense_weight(g, g.ctx, l, "attn_q_norm.weight");
      ggml_tensor *wkn = dense_weight(g, g.ctx, l, "attn_k_norm.weight");
      ggml_tensor *wo = dense_weight(g, g.ctx, l, "attn_output.weight");
      if (!wq || !wk || !wv || !wqn || !wkn || !wo)
        return false;
      if (g.decode_mode) {
        ggml_tensor *k_cache = new_device_state_tensor(
            g, GGML_TYPE_F32, {g.hd, g.kv_capacity * g.Hkv},
            "qwen_full_decode_kv_k");
        ggml_tensor *v_cache = new_device_state_tensor(
            g, GGML_TYPE_F32, {g.hd, g.kv_capacity * g.Hkv},
            "qwen_full_decode_kv_v");
        if (!k_cache || !v_cache)
          return false;
        g.kv_state_k.push_back(k_cache);
        g.kv_state_v.push_back(v_cache);
        attn_out = build_full_attention_decode(
            g, g.ctx, attn_in, l, wq, wk, wv, wqn, wkn, wo, k_cache, v_cache);
      } else {
        ggml_tensor *k_cache = nullptr;
        ggml_tensor *v_cache = nullptr;
        attn_out = build_full_attention(g, g.ctx, attn_in, l, wq, wk, wv, wqn,
                                        wkn, wo, &k_cache, &v_cache);
        if (!k_cache || !v_cache)
          return false;
        g.kv_k.push_back(k_cache);
        g.kv_v.push_back(v_cache);
      }
      g.kv_layers.push_back(l);
    }
    res = hc_combine(g, g.ctx, res, attn_out, inject);
    g.attn_construction_ms_total += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_attn_start).count();

    const auto t_moe_start = std::chrono::steady_clock::now();
    inject = nullptr;
    hn = resident_tensor(g, g.ctx, p + "hc_ffn_norm.weight");
    hdn = resident_tensor(g, g.ctx, p + "hc_ffn_down.weight");
    hup = resident_tensor(g, g.ctx, p + "hc_ffn_up.weight");
    hinj = resident_tensor(g, g.ctx, p + "hc_ffn_inject.weight");
    if (!hn || !hdn || !hup || !hinj)
      return false;
    ggml_tensor *ffn_in = hc_mix(g, g.ctx, res, hn, hdn, hup, hinj, &inject);
    ggml_tensor *router = resident_tensor(g, g.ctx, p + "ffn_gate_inp.weight");
    ggml_tensor *bias = resident_tensor(g, g.ctx, p + "exp_probs_b.bias");
    if (!router)
      return false;

    // The per-expert split path still needs separate gate/up tensors because
    // its two branches have different expert-axis remaps.  For ordinary
    // whole-layer CPU/GPU placement, prefer the adjacent gate+up view: on
    // graph-RAM layers this is zero-copy and reduces three routed matmuls to
    // two, matching Bee's reference graph.
    const auto split_it = g.split_gpu_expert_ids.find(l);
    const bool split_planned =
        g.S == 1 && split_it != g.split_gpu_expert_ids.end() &&
        !split_it->second.empty();
    ggml_tensor *moe = nullptr;
    if (!split_planned && env_merged_gate_up_enabled()) {
      const auto t_wlookup_start = std::chrono::steady_clock::now();
      const bool gate_up_device =
          graph_expert_projection_resident(g, l, "ffn_gate_exps.weight");
      ggml_tensor *gate_up =
          expert_gate_up_weight(g, g.ctx, l, gate_up_device);
      ggml_tensor *down = expert_weight(g, g.ctx, l,
                                        "ffn_down_exps.weight");
      g.moe_weight_lookup_ms_total +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - t_wlookup_start)
              .count();
      if (gate_up && down) {
        moe = build_moe(g, g.ctx, ffn_in, l, router, bias, nullptr, nullptr,
                        down, nullptr, nullptr, nullptr, nullptr, gate_up);
      }
    }

    if (!moe) {
      const auto t_wlookup_start = std::chrono::steady_clock::now();
      SplitExpertWeight ge =
          expert_weight_split(g, g.ctx, l, "ffn_gate_exps.weight");
      SplitExpertWeight ue =
          expert_weight_split(g, g.ctx, l, "ffn_up_exps.weight");
      SplitExpertWeight de =
          expert_weight_split(g, g.ctx, l, "ffn_down_exps.weight");
      g.moe_weight_lookup_ms_total +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - t_wlookup_start)
              .count();
      if (!ge.cpu || !ue.cpu || !de.cpu)
        return false;
      // v1 scope: uniform M across gate/up/down for a layer -- only treat
      // this as a real split if ALL THREE projections' gpu tensor succeeded
      // (expert_weight_split() already falls back to null on any failure).
      const bool split_ok = ge.gpu && ue.gpu && de.gpu;
      const std::vector<int> *split_ids =
          split_ok ? &g.split_gpu_expert_ids.at(l) : nullptr;
      moe = build_moe(g, g.ctx, ffn_in, l, router, bias, ge.cpu, ue.cpu,
                      de.cpu, split_ok ? ge.gpu : nullptr,
                      split_ok ? ue.gpu : nullptr,
                      split_ok ? de.gpu : nullptr, split_ids);
    }
    ggml_tensor *sg =
        resident_tensor(g, g.ctx, p + "ffn_gate_inp_shexp.weight");
    ggml_tensor *sge = resident_tensor(g, g.ctx, p + "ffn_gate_shexp.weight");
    ggml_tensor *sup = resident_tensor(g, g.ctx, p + "ffn_up_shexp.weight");
    ggml_tensor *sdo = resident_tensor(g, g.ctx, p + "ffn_down_shexp.weight");
    if (sge && sup && sdo) {
      ggml_tensor *shared =
          build_shared(g, g.ctx, ffn_in, l, sge, sup, sdo, sg);
      moe = ggml_add(g.ctx, moe, shared);
    }
    res = hc_combine(g, g.ctx, res, moe, inject);
    g.moe_construction_ms_total += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_moe_start).count();
  }

  ggml_tensor *on = resident_tensor(g, g.ctx, "output_hc_norm.weight");
  ggml_tensor *od = resident_tensor(g, g.ctx, "output_hc_down.weight");
  ggml_tensor *ou = resident_tensor(g, g.ctx, "output_hc_up.weight");
  if (!on || !od || !ou)
    return false;
  g.final_streams = res;
  g.output = hc_mix(g, g.ctx, res, on, od, ou, nullptr, nullptr);
  if (!g.decode_mode) {
    ggml_set_output(g.final_streams);
    ggml_build_forward_expand(g.graph, g.final_streams);
  }
  ggml_set_output(g.output);
  ggml_build_forward_expand(g.graph, g.output);
  if (g.decode_mode) {
    ggml_tensor *lm_in = g.output;
    ggml_tensor *out_norm =
        resident_tensor(g, g.ctx, "output_norm.weight");
    if (out_norm) {
      lm_in = ggml_mul(
          g.ctx,
          ggml_rms_norm(g.ctx, lm_in,
                        g.engine->model_file.architecture_config.eps),
          out_norm);
    }
    ggml_tensor *out_weight =
        resident_tensor(g, g.ctx, "output.weight");
    if (!out_weight)
      return false;
    g.logits = ggml_mul_mat(g.ctx, out_weight, lm_in);
    ggml_set_output(g.logits);
    ggml_build_forward_expand(g.graph, g.logits);
  }
  if (!g.decode_mode) {
    for (ggml_tensor *t : g.kv_k) {
      ggml_set_output(t);
      ggml_build_forward_expand(g.graph, t);
    }
    for (ggml_tensor *t : g.kv_v) {
      ggml_set_output(t);
      ggml_build_forward_expand(g.graph, t);
    }
    for (ggml_tensor *t : g.gdn_state_out) {
      ggml_set_output(t);
      ggml_build_forward_expand(g.graph, t);
    }
    for (ggml_tensor *t : g.gdn_hist_out) {
      ggml_set_output(t);
      ggml_build_forward_expand(g.graph, t);
    }
    if (g.ple_hist_out) {
      ggml_set_output(g.ple_hist_out);
      ggml_build_forward_expand(g.graph, g.ple_hist_out);
    }
  } else {
    // State copies are graph nodes, but are intentionally not graph outputs:
    // decode keeps GDN/KV/PLE state on the CUDA backend between submissions.
    for (ggml_tensor *t : g.gdn_state_updates)
      if (t)
        ggml_build_forward_expand(g.graph, t);
    for (ggml_tensor *t : g.gdn_hist_updates)
      if (t)
        ggml_build_forward_expand(g.graph, t);
    if (g.ple_hist_update)
      ggml_build_forward_expand(g.graph, g.ple_hist_update);
  }
  // Real, measured incident (2026-09-09): ggml_backend_sched_alloc_graph's
  // internal ggml_gallocr_reserve_n_impl did not fail gracefully on a real
  // cudaMalloc OOM on beellama's older ggml fork -- it logged "cudaMalloc
  // failed: out of memory" and then segfaulted the whole process (exit 139)
  // instead of returning a checkable failure, at the exact point where
  // main.cpp's own fallback VRAM cache reservation (a fixed 4GB, unaware of
  // device_expert_layers' own VRAM use) left the scheduler's own reserve()
  // short by ~1.8GB. main.cpp now sizes that fallback cache against real
  // free VRAM and this family's own device_expert_layers estimate -- THAT
  // was the actual root cause and is already fixed there. This check stays
  // as a second, independent line of defense: refuse to even attempt the
  // allocator call when real free VRAM is below a floor, and decline to the
  // legacy path instead of risking the same crash again.
  //
  // 2026-09-15: tried twice to replace this flat constant with something
  // smarter -- both reverted, see below -- then settled on this smaller,
  // still-flat, but now EVIDENCE-BACKED value instead of an arbitrary one.
  // (1) Lowering 2 GiB -> 512 MiB: crashed at device_expert_layers=24
  // (needed >=650.24 MiB there -- the exact size of the single buffer
  // ggml_gallocr_reserve_n_impl tried and failed to cudaMalloc). (2)
  // Querying the REAL requirement via ggml_backend_sched_reserve_size() (a
  // safe, no_alloc=true measure pass that cannot itself OOM) reported
  // 0.640 GiB at N=22 -- consistent with N=24's 0.635 GiB failure point
  // above, i.e. the real requirement is roughly STABLE around ~0.64 GiB,
  // not scaling dramatically with layer count as first assumed from just
  // those two data points. But reusing that measure call's split_graph
  // pass on the SAME g.graph object that a real
  // ggml_backend_sched_alloc_graph() call re-splits moments later leaves
  // something behind that a plain ggml_backend_sched_reset() in between
  // does NOT fully clear: real-hardware repro, twice, with and without that
  // reset -- a hard `GGML_ASSERT(src0->type == src1->type)` abort inside
  // ggml_cuda_op_concat on the FIRST real graph compute afterward (prefill,
  // S=7). Do not retry live-measuring this way without either a genuinely
  // separate measure-graph object (llama.cpp's own convention -- never
  // reuse/re-split the exact cgraph that will actually execute) or a
  // from-scratch ggml_backend_sched for the real run after measuring on a
  // throwaway one.
  //
  // Given the two real measurements above both land at ~0.63-0.64 GiB,
  // 1 GiB keeps a real ~1.5-1.6x margin over observed need (vs 2 GiB's
  // ~3.1x, which cost every device_expert_layers config ~1 GiB of pure
  // waste) while recovering enough headroom for device_expert_layers=22
  // to fit (was declining with only 1.83 GiB free). Re-widen if a real
  // OOM/crash resurfaces at some untested N; don't lower below ~0.8 GiB
  // without a fresh measurement -- these two points aren't enough to prove
  // it never goes higher for a different N or model.
  {
    size_t free_b = 0, total_b = 0;
    int prev_device = -1;
    cudaGetDevice(&prev_device);
    cudaSetDevice(g.device);
    const cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (prev_device >= 0)
      cudaSetDevice(prev_device);
    constexpr size_t kMinGraphAllocHeadroomBytes =
        1ull * 1024ull * 1024ull * 1024ull; // 1 GiB
    if (err != cudaSuccess || free_b < kMinGraphAllocHeadroomBytes) {
      LOG_ERROR(
          "[GGML-QWEN-FULL] refusing graph allocator reserve: device %d "
          "free=%.2f GiB "
          "< %.2f GiB safety floor (device expert arena + dense arena already "
          "committed %.2f GiB) -- falling back to the legacy path\n",
          g.device,
          err == cudaSuccess
              ? static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0)
              : -1.0,
          static_cast<double>(kMinGraphAllocHeadroomBytes) /
              (1024.0 * 1024.0 * 1024.0),
          static_cast<double>(g.device_expert_capacity +
                              g.device_dense_capacity) /
              (1024.0 * 1024.0 * 1024.0));
      return false;
    }
  }
  const double t_construction_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() -
                                       t_build_graph_start)
                                       .count();
  const auto t_alloc_start = std::chrono::steady_clock::now();
  if (!ggml_backend_sched_alloc_graph(g.sched, g.graph))
    return false;
  const double t_alloc_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() -
                                t_alloc_start)
                                .count();
  LOG_INFO("[GGML-QWEN-FULL-BUILD-TIME] node_construction=%.1fms "
           "(of which dense_cudaMemcpy=%.1fms/%d tensors, "
           "expert_cudaMemcpy=%.1fms/%ld calls, attn_block=%.1fms, "
           "moe_block=%.1fms [weight_lookup=%.1fms]) sched_alloc_graph=%.1fms\n",
           t_construction_ms, g.dense_upload_ms_total,
           g.device_dense_tensors, g.expert_upload_ms_total,
           g.expert_upload_calls, g.attn_construction_ms_total,
           g.moe_construction_ms_total, g.moe_weight_lookup_ms_total,
           t_alloc_ms);
  g.graph_bytes = ggml_backend_sched_get_buffer_size(g.sched, g.cuda);
  dump_qwen_moe_layout(g);

  // The graph is only useful for performance if the scheduler keeps most of
  // the dependency chain on CUDA.  A high split count can otherwise hide a
  // CPU fallback for an op or a view/copy boundary, which recreates the
  // per-layer host orchestration that this graph is meant to remove.
  int cuda_nodes = 0;
  int cpu_nodes = 0;
  int other_nodes = 0;
  int backend_transitions = 0;
  ggml_backend_t previous_backend = nullptr;
  std::unordered_map<std::string, int> non_cuda_ops;
  for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
    ggml_tensor *node = ggml_graph_node(g.graph, i);
    ggml_backend_t backend =
        ggml_backend_sched_get_tensor_backend(g.sched, node);
    if (backend == g.cuda) {
      ++cuda_nodes;
    } else if (backend == g.cpu) {
      ++cpu_nodes;
      ++non_cuda_ops[ggml_op_name(node->op)];
    } else {
      ++other_nodes;
      ++non_cuda_ops[backend ? ggml_backend_name(backend) : "NULL"];
    }
    if (previous_backend && backend != previous_backend)
      ++backend_transitions;
    previous_backend = backend;
  }
  LOG_INFO("[GGML-QWEN-FULL] scheduler nodes cuda=%d cpu=%d other=%d "
           "transitions=%d\n",
           cuda_nodes, cpu_nodes, other_nodes, backend_transitions);
  if (!non_cuda_ops.empty()) {
    std::string detail;
    for (const auto &entry : non_cuda_ops) {
      if (!detail.empty())
        detail += ", ";
      detail += entry.first + "=" + std::to_string(entry.second);
    }
    LOG_INFO("[GGML-QWEN-FULL] non-CUDA node ops: %s\n", detail.c_str());
  }
  LOG_INFO("[GGML-QWEN-FULL] graph built layers=%d nodes=%d splits=%d "
           "workspace=%.2f GiB\n",
           g.n_layers, ggml_graph_n_nodes(g.graph),
           ggml_backend_sched_get_n_splits(g.sched),
           static_cast<double>(g.graph_bytes) / (1024.0 * 1024.0 * 1024.0));
  LOG_INFO(
      "[GGML-QWEN-FULL] dense device arena used=%.2f/%.2f GiB; %d tensors\n",
      static_cast<double>(g.device_dense_used) / (1024.0 * 1024.0 * 1024.0),
      static_cast<double>(g.device_dense_capacity) / (1024.0 * 1024.0 * 1024.0),
      g.device_dense_tensors);
  if (g.device_expert_layers > 0) {
    const int complete_records = g.device_expert_layers * c.n_experts;
    LOG_INFO(
        "[GGML-QWEN-FULL] device expert arena used=%.2f/%.2f GiB; "
        "%d complete layer(s), %d expert records, all 3 projections resident\n",
        static_cast<double>(g.device_expert_used) / (1024.0 * 1024.0 * 1024.0),
        static_cast<double>(g.device_expert_capacity) /
            (1024.0 * 1024.0 * 1024.0),
        g.device_expert_layers, complete_records);
  }
  LOG_INFO(
      "[GGML-QWEN-FULL] host expert source=%.2f GiB graph-owned DDR copy; "
      "%.2f GiB aliased from preloaded DDR arena (%zu projection views)\n",
      static_cast<double>(g.host_weight_bytes) / (1024.0 * 1024.0 * 1024.0),
      static_cast<double>(g.host_ram_alias_bytes) / (1024.0 * 1024.0 * 1024.0),
      g.host_ram_alias_tensors);
  if (g.host_weight_bytes > 0) {
    LOG_INFO("[GGML-QWEN-FULL] host expert source pinned=%.2f GiB; "
             "pin failures=%zu\n",
             static_cast<double>(g.host_pinned_bytes) /
                 (1024.0 * 1024.0 * 1024.0),
             g.host_pin_failures);
  }
  LOG_INFO("[GGML-QWEN-FULL-MEM] host ledger=%.2f/%.2f GB; graph source="
           "%.2f GiB + context=%.2f MiB + owned-extra=%.2f MiB; device "
           "workspace=%.2f GiB (GPU, outside host_ram_gb)\n",
           (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
           (double)MemoryGuard::host_budget_bytes() / 1.0e9,
           (double)g.host_weight_bytes / (1024.0 * 1024.0 * 1024.0),
           (double)g.context_reservation_bytes / (1024.0 * 1024.0),
           (double)(g.host_weight_bytes > g.host_reservation_bytes
                        ? g.host_weight_bytes - g.host_reservation_bytes
                        : 0) /
               (1024.0 * 1024.0),
           (double)g.graph_bytes / (1024.0 * 1024.0 * 1024.0));
  g.ready = true;
  return true;
}

bool sync_decode_state_to_host(FullGraph &g) {
  if (!g.decode_mode || !g.ready || !g.engine)
    return true;
  StreamingEngine &engine = *g.engine;
  const size_t state_n = static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn;
  const size_t hist_n = static_cast<size_t>(g.d_conv - 1) * g.conv_dim;
  for (size_t i = 0; i < g.gdn_layers.size(); ++i) {
    const int l = g.gdn_layers[i];
    if (i >= g.gdn_state_storage.size() || i >= g.gdn_hist_storage.size() ||
        l < 0 || l >= static_cast<int>(engine.qwen4exp_recur_state.size()) ||
        l >= static_cast<int>(engine.qwen4exp_conv_history.size()) ||
        engine.qwen4exp_recur_state[l].size() != state_n ||
        engine.qwen4exp_conv_history[l].size() != hist_n)
      return false;
    ggml_backend_tensor_get(g.gdn_state_storage[i],
                            engine.qwen4exp_recur_state[l].data(), 0,
                            state_n * sizeof(float));
    ggml_backend_tensor_get(g.gdn_hist_storage[i],
                            engine.qwen4exp_conv_history[l].data(), 0,
                            hist_n * sizeof(float));
  }
  const size_t row = static_cast<size_t>(g.Hkv) * g.hd;
  const int rows = std::min(std::max(0, g.decode_last_pos + 1), g.kv_capacity);
  for (size_t i = 0; i < g.kv_layers.size(); ++i) {
    const int l = g.kv_layers[i];
    if (i >= g.kv_state_k.size() || i >= g.kv_state_v.size() || l < 0 ||
        l >= static_cast<int>(engine.Kc.size()) ||
        l >= static_cast<int>(engine.Vc.size()) || !engine.Kc[l] ||
        !engine.Vc[l])
      return false;
    const size_t bytes = static_cast<size_t>(rows) * row * sizeof(float);
    if (bytes > 0) {
      ggml_backend_tensor_get(g.kv_state_k[i], engine.Kc[l], 0, bytes);
      ggml_backend_tensor_get(g.kv_state_v[i], engine.Vc[l], 0, bytes);
    }
  }
  if (g.ple_layer >= 0 && g.ple_hist_storage) {
    const auto &c = engine.model_file.architecture_config;
    const size_t ple_n = static_cast<size_t>(c.ple_conv_kernel - 1) *
                         c.ple_ngram_size * g.HC * g.D;
    if (g.ple_layer >= static_cast<int>(engine.qwen4exp_ple_conv_history.size()) ||
        engine.qwen4exp_ple_conv_history[g.ple_layer].size() != ple_n)
      return false;
    ggml_backend_tensor_get(g.ple_hist_storage,
                            engine.qwen4exp_ple_conv_history[g.ple_layer].data(),
                            0, ple_n * sizeof(float));
  }
  return true;
}

} // namespace

bool qwen4exp_full_graph_enabled() { return env_enabled(); }

int qwen4exp_full_graph_device_expert_layers() {
  return env_enabled() ? env_device_expert_layers() : 0;
}

bool qwen4exp_full_graph_decode_enabled() {
  return env_enabled() && env_decode_enabled();
}

void qwen4exp_full_graph_shutdown() { g_graph.reset(); }

bool qwen4exp_full_graph_prepare(StreamingEngine &engine, int S, int pos_base,
                                 const int *token_ids) {
  if (!env_enabled() || S <= 1 || pos_base != 0 ||
      engine.family_ops.family != ModelFamily::Qwen4Exp)
    return false;
  const auto &c = engine.model_file.architecture_config;
  if (engine.qwen_ram_preload_all_experts) {
    for (int layer = 0; layer < c.n_layers; ++layer) {
      if (!engine.graph_ram_layer_ready(layer)) {
        LOG_WARN("[GGML-QWEN-FULL] disabled: ram_preload_all_experts "
                 "requires a ready DDR arena for every layer (missing "
                 "layer=%d); no GGUF mmap fallback\n",
                 layer);
        return false;
      }
    }
  }
  const int dev = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  if (g_graph && g_graph->ready && g_graph->engine == &engine &&
      g_graph->S == S) {
    // PLE rows are request data, so refresh that input when the graph shape
    // is reused for a different prompt.  The model graph itself remains
    // persistent, exactly like Bee's shape cache.
    if (has_ple_layer(c)) {
      std::vector<float> ple;
      if (!prepare_ple_embedding(*g_graph, ple, S, pos_base, token_ids) ||
          !g_graph->ple_embedding ||
          ple.size() * sizeof(float) > ggml_nbytes(g_graph->ple_embedding))
        return false;
      std::memcpy(g_graph->ple_embedding->data, ple.data(),
                  ple.size() * sizeof(float));
    }
    return true;
  }
  g_graph.reset();
  auto g = std::make_unique<FullGraph>();
  g->engine = &engine;
  g->device = dev;
  g->S = S;
  g->D = c.hidden;
  g->HC = c.hc_mult;
  g->n_layers = c.n_layers;
  g->H = c.n_heads;
  g->Hkv = std::max(1, c.n_head_kv);
  g->hd = c.head_dim;
  g->Hk_gdn = c.ssm_group_count;
  g->Hv_gdn = c.ssm_time_step_rank;
  g->gdn_hd = c.ssm_state_size;
  g->key_dim = g->Hk_gdn * g->gdn_hd;
  g->value_dim = c.ssm_inner_size;
  g->conv_dim = 2 * g->key_dim + g->value_dim;
  g->d_conv = c.ssm_conv_kernel;
  g->device_expert_layers =
      std::min(qwen4exp_full_graph_device_expert_layers(), c.n_layers);
  g->device_expert_trailing = env_device_expert_trailing();
  const char *partial_gate_up =
      std::getenv("MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_PARTIAL_GATE_UP");
  // partial_gate_up's boundary math (layer==device_expert_layers, the
  // first CPU layer just past a LEADING block) has no trailing-mode
  // equivalent yet -- force it off rather than apply leading-only math
  // to a trailing block. See device_expert_trailing's own comment.
  g->device_expert_partial_gate_up = g->device_expert_layers > 0 &&
                                     !g->device_expert_trailing &&
                                     partial_gate_up &&
                                     partial_gate_up[0] == '1';
  plan_expert_vram_split(*g); // must run before init_device_expert_arena
  g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(dev));
  g->cpu = ggml_backend_cpu_init();
  if (!g->cuda || !g->cpu)
    return false;
  g->cpu_threads = graph_cpu_threads();
  ggml_backend_cpu_set_n_threads(g->cpu, g->cpu_threads);
  ggml_threadpool_params cpu_tp_params =
      ggml_threadpool_params_default(g->cpu_threads);
  g->cpu_threadpool = ggml_threadpool_new(&cpu_tp_params);
  if (g->cpu_threadpool)
    ggml_backend_cpu_set_threadpool(g->cpu, g->cpu_threadpool);
  LOG_INFO("[GGML-QWEN-FULL] CPU backend configured with %d threads%s\n",
           g->cpu_threads,
           g->cpu_threadpool ? " (persistent threadpool)" : "");

  // Admit the graph-owned expert source set once, before any of its
  // individual host buffers are allocated.  Layers with a complete
  // preloaded graph-RAM arena are excluded: their expert planes are aliased
  // directly from the existing DDR cache below, so this graph does not make
  // a second copy.  Any remaining layers use one graph-owned DDR copy rather
  // than a decode-time GGUF mmap source.  The extra watermark covers the
  // GGML host context, prompt/state inputs, and allocator/I/O fluctuation;
  // no /proc/meminfo read is performed in the per-buffer allocation loop.
  const size_t expert_bytes = full_expert_source_bytes(engine);
  // 10 GiB, per explicit user instruction after the 2026-09-09 crash (was
  // 4 GiB -- the smallest, least-conservative watermark of any admission
  // check in either graph file; aligned to the same 10 GiB floor as the
  // rest).
  constexpr size_t kGraphHostWatermark = 10ull * 1024ull * 1024ull * 1024ull;
  if (expert_bytes > 0 &&
      !MemoryGuard::try_reserve_host_bytes(expert_bytes, kGraphHostWatermark)) {
    LOG_WARN("[GGML-QWEN-FULL] host watermark refused graph source=%.2f GiB + "
             "%.2f GiB reserve; "
             "MemAvailable=%ld MiB\n",
             static_cast<double>(expert_bytes) / (1024.0 * 1024.0 * 1024.0),
             static_cast<double>(kGraphHostWatermark) /
                 (1024.0 * 1024.0 * 1024.0),
             MemoryGuard::last_available_mb());
    return false;
  }
  g->host_reservation_bytes = expert_bytes;
  g->host_precharged_bytes = expert_bytes;
  LOG_INFO("[GGML-QWEN-FULL] host source admission: graph-owned=%.2f GiB + "
           "%.2f GiB reserve; "
           "complete preloaded DDR arenas are aliased directly\n",
           static_cast<double>(expert_bytes) / (1024.0 * 1024.0 * 1024.0),
           static_cast<double>(kGraphHostWatermark) /
               (1024.0 * 1024.0 * 1024.0));
  if (!init_device_dense_arena(*g))
    return false;
  // Reserve fixed dense weights before choosing the variable-size expert
  // arena. Otherwise an oversized all-GPU request can consume the remaining
  // VRAM with experts, appear to fit after clamping, and then fail here when
  // the dense arena is allocated.
  if (!init_device_expert_arena(*g))
    return false;
  ggml_backend_t backends[] = {g->cuda, g->cpu};
  ggml_backend_buffer_type_t sched_bufts[2] = {};
  ggml_backend_buffer_type_t *sched_bufts_ptr = nullptr;
  if (graph_cuda_host_buffers()) {
    sched_bufts[0] = ggml_backend_get_default_buffer_type(g->cuda);
    sched_bufts[1] = ggml_backend_dev_host_buffer_type(
        ggml_backend_get_device(g->cuda));
    if (sched_bufts[1]) {
      sched_bufts_ptr = sched_bufts;
      LOG_INFO("[GGML-QWEN-FULL] scheduler CPU buffer=%s (CUDA host buffer)\n",
               ggml_backend_buft_name(sched_bufts[1]));
    }
  }
  // Match Bee's pipeline_parallel=false scheduler mode by default.  The
  // graph's CPU/GPU transitions are dependency-serial; the event/four-copy
  // mode is retained as an explicit opt-in for comparison.
  g->sched =
      ggml_backend_sched_new(backends, sched_bufts_ptr, 2, kGraphSize,
                             graph_scheduler_parallel(), true);
  if (!g->sched)
    return false;
  LOG_INFO("[GGML-QWEN-FULL] scheduler pipeline_parallel=%s copies=%d\n",
           graph_scheduler_parallel() ? "true" : "false",
           ggml_backend_sched_get_n_copies(g->sched));
  g->layer_timing_enabled = env_layer_timing_enabled();
  if (g->layer_timing_enabled) {
    g->layer_timing_cpu_ms.assign(static_cast<size_t>(g->n_layers), 0.0);
    ggml_backend_sched_set_eval_callback(g->sched, layer_timing_eval_callback,
                                         g.get());
  }

  // The model graph starts from the already expanded HC embedding.  PLE's
  // hash gather is prepared once here, outside the measured prefill window;
  // learned PLE projections and its causal convolution remain graph nodes.
  std::vector<float> ple;
  if (!prepare_ple_embedding(*g, ple, S, pos_base, token_ids) &&
      has_ple_layer(c)) {
    LOG_WARN("[GGML-QWEN-FULL] PLE embedding preparation declined\n");
    return false;
  }
  if (!build_graph(*g, has_ple_layer(c) ? &ple : nullptr)) {
    LOG_WARN("[GGML-QWEN-FULL] graph construction declined; using fallback\n");
    return false;
  }
  g_graph = std::move(g);
  return true;
}

bool qwen4exp_full_graph_decode_prepare(StreamingEngine &engine, int pos_base,
                                        const int *token_ids) {
  if (!qwen4exp_full_graph_decode_enabled() || pos_base < 0 ||
      engine.family_ops.family != ModelFamily::Qwen4Exp)
    return false;
  const auto &c = engine.model_file.architecture_config;
  if (engine.max_t <= 0 || pos_base >= engine.max_t)
    return false;

  // The decode graph seeds recurrent/KV state from the graph's validated
  // prefill writeback.  Do not silently combine it with the existing layered
  // GPU attention/GDN state owners; that would create two divergent state
  // stores.  A later capacity growth is allowed because it first downloads
  // the current decode graph state below.
  const bool continuing_decode =
      g_graph && g_graph->ready && g_graph->engine == &engine &&
      g_graph->decode_mode;
  const bool has_graph_prefill =
      g_graph && g_graph->ready && g_graph->engine == &engine &&
      !g_graph->decode_mode;
  if (!continuing_decode && !has_graph_prefill) {
    LOG_WARN("[GGML-QWEN-FULL-DECODE] disabled: decode state has not been "
             "seeded by the opt-in full-graph prefill\n");
    return false;
  }

  // Decode full-graph mode is intentionally stricter than the ordinary graph:
  // all expert planes must already be in the complete DDR graph arena.  This
  // keeps a graph submission from silently introducing an SSD/GGUF source or
  // a per-token lazy-load boundary.  Models with lazy expert ownership stay on
  // the established layered path.
  for (int layer = 0; layer < c.n_layers; ++layer) {
    if (!engine.graph_ram_layer_ready(layer)) {
      LOG_WARN("[GGML-QWEN-FULL-DECODE] disabled: layer=%d has no complete "
               "preloaded DDR expert arena; layered decode remains active\n",
               layer);
      return false;
    }
  }

  const int required = pos_base + 1;
  const int dev = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  if (g_graph && g_graph->ready && g_graph->engine == &engine &&
      g_graph->decode_mode && g_graph->kv_capacity >= required &&
      g_graph->decode_last_pos + 1 == pos_base) {
    if (has_ple_layer(c)) {
      std::vector<float> ple;
      if (!prepare_ple_embedding(*g_graph, ple, 1, pos_base, token_ids) ||
          !g_graph->ple_embedding ||
          ple.size() * sizeof(float) > ggml_nbytes(g_graph->ple_embedding))
        return false;
      std::memcpy(g_graph->ple_embedding->data, ple.data(),
                  ple.size() * sizeof(float));
    }
    return true;
  }

  if (g_graph && g_graph->ready && g_graph->decode_mode &&
      g_graph->engine == &engine) {
    if (!sync_decode_state_to_host(*g_graph)) {
      LOG_WARN("[GGML-QWEN-FULL-DECODE] state download failed while growing "
               "KV capacity; using layered decode\n");
      return false;
    }
  }
  // VRAM diagnostic (2026-09-15): two cheap cudaMemGetInfo reads bracketing
  // prefill's own g_graph teardown (before = prefill's buffers still
  // resident, after = immediately post-free, before decode allocates
  // anything of its own). Confirmed the ~0.33 GiB gap between PREFLIGHT's
  // startup free and this post-teardown free is generic CUDA context/
  // driver overhead (module/PTX loading, cuBLAS handle init) that persists
  // for the process lifetime -- NOT a leak in this file's own arena
  // teardown, which correctly frees essentially 100% of what it allocated
  // (confirmed: freed amount here matched the prefill arena's own tracked
  // size almost exactly). Kept as a permanent low-cost sanity check.
  {
    size_t free_before = 0, total_before = 0;
    cudaMemGetInfo(&free_before, &total_before);
    g_graph.reset();
    size_t free_after = 0, total_after = 0;
    cudaMemGetInfo(&free_after, &total_after);
    LOG_INFO("[GGML-QWEN-FULL-VRAM-TRACE] pre-reset free=%.3f GiB, "
             "post-reset free=%.3f GiB (freed %.3f GiB)\n",
             free_before / (1024.0 * 1024.0 * 1024.0),
             free_after / (1024.0 * 1024.0 * 1024.0),
             (free_after - free_before) / (1024.0 * 1024.0 * 1024.0));
  }

  auto g = std::make_unique<FullGraph>();
  g->engine = &engine;
  g->device = dev;
  g->S = 1;
  g->D = c.hidden;
  g->HC = c.hc_mult;
  g->n_layers = c.n_layers;
  g->H = c.n_heads;
  g->Hkv = std::max(1, c.n_head_kv);
  g->hd = c.head_dim;
  g->Hk_gdn = c.ssm_group_count;
  g->Hv_gdn = c.ssm_time_step_rank;
  g->gdn_hd = c.ssm_state_size;
  g->key_dim = g->Hk_gdn * g->gdn_hd;
  g->value_dim = c.ssm_inner_size;
  g->conv_dim = 2 * g->key_dim + g->value_dim;
  g->d_conv = c.ssm_conv_kernel;
  g->decode_mode = true;
  g->kv_capacity = env_decode_kv_capacity(engine.max_t, required);
  g->decode_last_pos = pos_base - 1;
  g->device_expert_layers =
      std::min(qwen4exp_full_graph_device_expert_layers(), c.n_layers);
  g->device_expert_trailing = env_device_expert_trailing();
  const char *partial_gate_up =
      std::getenv("MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_PARTIAL_GATE_UP");
  // partial_gate_up's boundary math (layer==device_expert_layers, the
  // first CPU layer just past a LEADING block) has no trailing-mode
  // equivalent yet -- force it off rather than apply leading-only math
  // to a trailing block. See device_expert_trailing's own comment.
  g->device_expert_partial_gate_up = g->device_expert_layers > 0 &&
                                     !g->device_expert_trailing &&
                                     partial_gate_up &&
                                     partial_gate_up[0] == '1';
  plan_expert_vram_split(*g); // must run before init_device_expert_arena
  g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(dev));
  g->cpu = ggml_backend_cpu_init();
  if (!g->cuda || !g->cpu)
    return false;
  g->cpu_threads = graph_cpu_threads();
  ggml_backend_cpu_set_n_threads(g->cpu, g->cpu_threads);
  ggml_threadpool_params cpu_tp_params =
      ggml_threadpool_params_default(g->cpu_threads);
  // 2026-09-15: tested ggml's own threadpool .poll (default 50 -- NOT
  // controlled by OMP_WAIT_POLICY/GOMP_SPINCOUNT, which only affect
  // libgomp's separate pool) as a candidate for the mixed CPU/GPU decode
  // regression via a temporary env override swept across 0/50/400/1000.
  // Ruled out via THROUGHPUT: all four values landed within noise of each
  // other (11.76-12.37 tok/s at device_expert_layers=20).
  // TEMPORARY re-test (2026-09-15, same day): now that the ELEVENTH ROUND
  // per-op gap instrumentation exists (MOESTREAM_QWEN4EXP_GRAPH_LAYER_
  // TIMING=1), re-checking whether .poll moves the GAP time specifically
  // (a real but small effect could have been lost in the earlier
  // throughput-only measurement's noise). Remove this override again if
  // inconclusive, per the original decision record's own discipline.
  if (const char *poll_env =
          std::getenv("MOESTREAM_QWEN4EXP_CPU_POLL_RETEST")) {
    char *end = nullptr;
    const long v = std::strtol(poll_env, &end, 10);
    if (end != poll_env && v >= 0 && v <= 100)
      cpu_tp_params.poll = static_cast<uint32_t>(v);
  }
  g->cpu_threadpool = ggml_threadpool_new(&cpu_tp_params);
  if (g->cpu_threadpool)
    ggml_backend_cpu_set_threadpool(g->cpu, g->cpu_threadpool);
  LOG_INFO("[GGML-QWEN-FULL-DECODE] CPU backend configured with %d threads%s\n",
           g->cpu_threads,
           g->cpu_threadpool ? " (persistent threadpool)" : "");

  // Every expert view aliases the already populated DDR graph arena, so this
  // mode has no graph-owned expert source admission or GGUF expert read.
  if (!init_device_dense_arena(*g) || !init_device_state_arena(*g) ||
      !init_device_expert_arena(*g))
    return false;
  ggml_backend_t backends[] = {g->cuda, g->cpu};
  ggml_backend_buffer_type_t sched_bufts[2] = {};
  ggml_backend_buffer_type_t *sched_bufts_ptr = nullptr;
  if (graph_cuda_host_buffers()) {
    sched_bufts[0] = ggml_backend_get_default_buffer_type(g->cuda);
    sched_bufts[1] = ggml_backend_dev_host_buffer_type(
        ggml_backend_get_device(g->cuda));
    if (sched_bufts[1]) {
      sched_bufts_ptr = sched_bufts;
      LOG_INFO(
          "[GGML-QWEN-FULL-DECODE] scheduler CPU buffer=%s (CUDA host buffer)\n",
          ggml_backend_buft_name(sched_bufts[1]));
    }
  }
  // Match the prefill scheduler mode above; event/four-copy mode is opt-in.
  g->sched =
      ggml_backend_sched_new(backends, sched_bufts_ptr, 2, kGraphSize,
                             graph_scheduler_parallel(), true);
  if (!g->sched)
    return false;
  LOG_INFO(
      "[GGML-QWEN-FULL-DECODE] scheduler pipeline_parallel=%s copies=%d\n",
      graph_scheduler_parallel() ? "true" : "false",
      ggml_backend_sched_get_n_copies(g->sched));
  g->layer_timing_enabled = env_layer_timing_enabled();
  if (g->layer_timing_enabled) {
    g->layer_timing_cpu_ms.assign(static_cast<size_t>(g->n_layers), 0.0);
    ggml_backend_sched_set_eval_callback(g->sched, layer_timing_eval_callback,
                                         g.get());
  }

  std::vector<float> ple;
  if (!prepare_ple_embedding(*g, ple, 1, pos_base, token_ids) &&
      has_ple_layer(c)) {
    LOG_WARN("[GGML-QWEN-FULL-DECODE] PLE embedding preparation declined\n");
    return false;
  }
  if (!build_graph(*g, has_ple_layer(c) ? &ple : nullptr)) {
    LOG_WARN("[GGML-QWEN-FULL-DECODE] graph construction declined; layered "
             "decode remains active\n");
    return false;
  }

  // Seed the device state once from the state/cache that the existing prefill
  // or layered path produced.  This is a one-time boundary transfer; later
  // decode submissions update these tensors in the graph itself.
  const size_t state_n = static_cast<size_t>(g->gdn_hd) * g->gdn_hd * g->Hv_gdn;
  const size_t hist_n = static_cast<size_t>(g->d_conv - 1) * g->conv_dim;
  for (size_t i = 0; i < g->gdn_layers.size(); ++i) {
    const int l = g->gdn_layers[i];
    if (l < 0 || l >= static_cast<int>(engine.qwen4exp_recur_state.size()) ||
        l >= static_cast<int>(engine.qwen4exp_conv_history.size()) ||
        engine.qwen4exp_recur_state[l].size() != state_n ||
        engine.qwen4exp_conv_history[l].size() != hist_n)
      return false;
    ggml_backend_tensor_set(g->gdn_state_storage[i],
                            engine.qwen4exp_recur_state[l].data(), 0,
                            state_n * sizeof(float));
    ggml_backend_tensor_set(g->gdn_hist_storage[i],
                            engine.qwen4exp_conv_history[l].data(), 0,
                            hist_n * sizeof(float));
  }
  const size_t kv_row = static_cast<size_t>(g->Hkv) * g->hd;
  const size_t kv_bytes = static_cast<size_t>(pos_base) * kv_row * sizeof(float);
  for (size_t i = 0; i < g->kv_layers.size(); ++i) {
    const int l = g->kv_layers[i];
    if (i >= g->kv_state_k.size() || i >= g->kv_state_v.size() || l < 0 ||
        l >= static_cast<int>(engine.Kc.size()) ||
        l >= static_cast<int>(engine.Vc.size()) || !engine.Kc[l] ||
        !engine.Vc[l])
      return false;
    if (kv_bytes > 0) {
      ggml_backend_tensor_set(g->kv_state_k[i], engine.Kc[l], 0, kv_bytes);
      ggml_backend_tensor_set(g->kv_state_v[i], engine.Vc[l], 0, kv_bytes);
    }
  }
  if (g->ple_layer >= 0 && g->ple_hist_storage) {
    const size_t ple_n = static_cast<size_t>(c.ple_conv_kernel - 1) *
                         c.ple_ngram_size * g->HC * g->D;
    if (g->ple_layer >=
            static_cast<int>(engine.qwen4exp_ple_conv_history.size()) ||
        engine.qwen4exp_ple_conv_history[g->ple_layer].size() != ple_n)
      return false;
    ggml_backend_tensor_set(
        g->ple_hist_storage,
        engine.qwen4exp_ple_conv_history[g->ple_layer].data(), 0,
        ple_n * sizeof(float));
  }

  LOG_INFO("[GGML-QWEN-FULL-DECODE] enabled: one scheduler submission per "
           "token, KV capacity=%d, device GDN/KV/PLE state, DDR expert "
           "arenas=%d\n",
           g->kv_capacity, c.n_layers);
  g_graph = std::move(g);
  return true;
}

bool qwen4exp_full_graph_decode_run(StreamingEngine &engine,
                                    float *hidden_states, int pos_base,
                                    const int *token_ids, float *logits,
                                    int vocab) {
  if (!qwen4exp_full_graph_decode_enabled() || !hidden_states || pos_base < 0 ||
      !g_graph || !g_graph->ready || !g_graph->decode_mode ||
      g_graph->engine != &engine || g_graph->S != 1 ||
      pos_base >= g_graph->kv_capacity ||
      ((logits == nullptr) != (vocab <= 0)))
    return false;
  FullGraph &g = *g_graph;
  if (engine.hc_stream_buf.size() < static_cast<size_t>(g.D) * g.HC)
    return false;
  if (has_ple_layer(engine.model_file.architecture_config) &&
      (!g.ple_embedding || !token_ids))
    return false;
  // Validate the optional output before submitting the graph.  If this
  // check were left until after compute, a malformed logits destination
  // could advance device KV/GDN state and then fall back to the layered path
  // with stale host state.
  if (logits && (!g.logits || ggml_nelements(g.logits) < vocab))
    return false;

  std::memcpy(g.input->data, engine.hc_stream_buf.data(),
              static_cast<size_t>(g.D) * g.HC * sizeof(float));
  static_cast<int32_t *>(g.positions->data)[0] = pos_base;
  auto *mask = static_cast<ggml_fp16_t *>(g.attn_mask->data);
  for (int k = 0; k < g.kv_capacity; ++k)
    mask[k] = ggml_fp32_to_fp16(k <= pos_base ? 0.0f : -INFINITY);
  auto *indices = static_cast<int32_t *>(g.decode_indices->data);
  for (int h = 0; h < g.Hkv; ++h)
    indices[h] = pos_base * g.Hkv + h;

  if (g.layer_timing_enabled) {
    std::fill(g.layer_timing_cpu_ms.begin(), g.layer_timing_cpu_ms.end(), 0.0);
    g.layer_timing_op_ms.clear();
    g.layer_timing_t0.clear();
    g.layer_timing_gap_ms_total = 0.0;
    g.layer_timing_gap_count = 0;
    g.layer_timing_have_last_ask_false = false;
  }
  const auto start = std::chrono::steady_clock::now();
  if (ggml_backend_sched_graph_compute_async(g.sched, g.graph) !=
      GGML_STATUS_SUCCESS)
    return false;
  ggml_backend_sched_synchronize(g.sched);
  ggml_backend_tensor_get(g.output, hidden_states, 0,
                          static_cast<size_t>(g.D) * sizeof(float));
  if (logits) {
    ggml_backend_tensor_get(g.logits, logits, 0,
                            static_cast<size_t>(vocab) * sizeof(float));
  }
  g.decode_last_pos = pos_base;

  // The PLE hash window is host metadata, not numerical recurrent state. Keep
  // it current so a later graph disable/capacity growth can seed the layered
  // path without changing the token-context semantics.
  if (has_ple_layer(engine.model_file.architecture_config)) {
    auto &hist = engine.qwen4exp_ple_tok_hist;
    hist.push_back(token_ids[0]);
    const int keep = std::max(
        0, engine.model_file.architecture_config.ple_ngram_size - 1);
    if (static_cast<int>(hist.size()) > keep)
      hist.erase(hist.begin(), hist.end() - keep);
    engine.qwen4exp_ple_next_pos = pos_base + 1;
  }
  engine.qwen4exp_full_graph_output_ready = true;
  engine.qwen4exp_full_graph_logits_ready = logits != nullptr;
  const double ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - start)
                         .count();
  LOG_INFO("[GGML-QWEN-FULL-DECODE-TIME] pos=%d total=%.3f ms\n", pos_base,
           ms);
  // Per-layer/per-op CPU timing (env: MOESTREAM_QWEN4EXP_GRAPH_LAYER_TIMING=1)
  // -- emitted every 10 tokens (not every token) to avoid log spam; see
  // memory: full-graph-per-layer-timing-instrumentation-plan.md for why
  // this exists and what it answers that nsys/DEBUG counters cannot.
  if (g.layer_timing_enabled &&
      ++g.layer_timing_tokens_since_emit >= 10) {
    g.layer_timing_tokens_since_emit = 0;
    double total_cpu_ms = 0.0;
    std::string per_layer;
    for (size_t l = 0; l < g.layer_timing_cpu_ms.size(); ++l) {
      const double lms = g.layer_timing_cpu_ms[l];
      total_cpu_ms += lms;
      if (lms > 0.01) {
        per_layer += " L" + std::to_string(l) + "=" +
                    std::to_string(static_cast<int>(lms * 100) / 100.0);
      }
    }
    LOG_INFO("[GGML-QWEN-FULL-LAYER-TIMING] pos=%d cpu_total=%.3fms "
             "gap_total=%.3fms (n=%ld gaps, dispatch/barrier/scheduling "
             "overhead BETWEEN CPU nodes, invisible to any op-bracket "
             "timing)%s\n",
             pos_base, total_cpu_ms, g.layer_timing_gap_ms_total,
             g.layer_timing_gap_count, per_layer.c_str());
    for (const auto &[stem, oms] : g.layer_timing_op_ms) {
      LOG_INFO("[GGML-QWEN-FULL-LAYER-TIMING]   op=%s cpu=%.3fms\n",
               stem.c_str(), oms);
    }
  }
  return true;
}

bool qwen4exp_full_graph_run(StreamingEngine &engine, float *hidden_states,
                             int S, int pos_base, const int *token_ids) {
  // Prefill-only while recurrent and dense KV state writeback is validated.
  if (!env_enabled() || !hidden_states || S <= 1 || pos_base != 0 || !g_graph ||
      !g_graph->ready || g_graph->engine != &engine || g_graph->S != S)
    return false;
  FullGraph &g = *g_graph;
  auto fail = [&](const char *why) {
    LOG_WARN("[GGML-QWEN-FULL] execution declined: %s\n", why);
    return false;
  };
  LOG_INFO(
      "[GGML-QWEN-FULL] execute candidate S=%d gdn=%zu kv=%zu ple_layer=%d\n",
      S, g.gdn_layers.size(), g.kv_layers.size(), g.ple_layer);

  // A fresh position-zero graph starts a new recurrent/KV stream.  This is
  // the same reset boundary used by the established Qwen attention path.
  for (size_t i = 0; i < engine.qwen4exp_conv_history.size(); ++i)
    std::fill(engine.qwen4exp_conv_history[i].begin(),
              engine.qwen4exp_conv_history[i].end(), 0.0f);
  for (size_t i = 0; i < engine.qwen4exp_recur_state.size(); ++i)
    std::fill(engine.qwen4exp_recur_state[i].begin(),
              engine.qwen4exp_recur_state[i].end(), 0.0f);
  std::fill(engine.qwen4exp_ple_tok_hist.begin(),
            engine.qwen4exp_ple_tok_hist.end(),
            static_cast<int32_t>(
                engine.model_file.architecture_config.ple_eos_token_id));
  engine.qwen4exp_ple_next_pos = 0;

  for (size_t i = 0; i < g.gdn_layers.size(); ++i) {
    const int l = g.gdn_layers[i];
    if (i >= g.gdn_state_inputs.size() || i >= g.gdn_hist_inputs.size())
      return fail("GDN input bookkeeping mismatch");
    const size_t state_n = static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn;
    const size_t hist_n = static_cast<size_t>(g.d_conv - 1) * g.conv_dim;
    if (l < 0 || l >= static_cast<int>(engine.qwen4exp_recur_state.size()) ||
        l >= static_cast<int>(engine.qwen4exp_conv_history.size()) ||
        engine.qwen4exp_recur_state[l].size() != state_n ||
        engine.qwen4exp_conv_history[l].size() != hist_n)
      return fail("GDN state/history size mismatch");
    std::memcpy(g.gdn_state_inputs[i], engine.qwen4exp_recur_state[l].data(),
                state_n * sizeof(float));
    std::memcpy(g.gdn_hist_inputs[i], engine.qwen4exp_conv_history[l].data(),
                hist_n * sizeof(float));
  }
  if (g.ple_layer >= 0 && g.ple_hist_input) {
    const auto &c = engine.model_file.architecture_config;
    // The engine keeps PLE history in the conv layout [HC*D, hist].  The
    // PLE embedding uses [n_heads, embd_per_layer], but those are not the
    // storage dimensions of the causal-convolution state.
    const size_t hist_n = static_cast<size_t>(c.ple_conv_kernel - 1) *
                          c.ple_ngram_size * g.HC * g.D;
    if (g.ple_layer >=
            static_cast<int>(engine.qwen4exp_ple_conv_history.size()) ||
        engine.qwen4exp_ple_conv_history[g.ple_layer].size() != hist_n)
      return fail("PLE history size mismatch");
    std::memcpy(g.ple_hist_input,
                engine.qwen4exp_ple_conv_history[g.ple_layer].data(),
                hist_n * sizeof(float));
  } else if (has_ple_layer(engine.model_file.architecture_config)) {
    return fail("PLE graph input is missing");
  }
  if (g.host_inputs.empty() ||
      engine.hc_stream_buf.size() < static_cast<size_t>(g.D) * g.HC * S)
    return fail("expanded HC input is missing or too small");
  std::memcpy(g.host_inputs[0], engine.hc_stream_buf.data(),
              static_cast<size_t>(g.D) * g.HC * S * sizeof(float));

  // Real-hardware incident (2026-09-09, GLM5NEXT side): a 480-token prompt
  // crashed the whole machine (hard reboot required, no surviving kernel/
  // OOM log -- this box has no persistent journald storage) with the last
  // surviving application-log line printed just before this exact kind of
  // compute call. Every OTHER host allocation on this path (arena prep,
  // expert source, graph context) goes through MemoryGuard::try_reserve_
  // host_bytes first; this compute call did not. Re-check REAL, fresh
  // MemAvailable (not the background watchdog's own possibly-500ms-stale
  // allocation_blocked() flag) immediately before committing to it -- a
  // large gap can open between prepare()'s own admission checks and this
  // moment (preload alone can take minutes). Not a confirmed fix for
  // whatever the real mechanism was (no kernel evidence survived to
  // confirm it), just a cheap, additional, defense-in-depth backstop,
  // applied identically to qwen4exp's own graph for the same reason.
  {
    const long avail_mb = MemoryGuard::read_available_mb();
    constexpr long kMinComputeHeadroomMb =
        10240; // 10 GiB, per explicit user instruction after the 2026-09-09
               // crash
    if (avail_mb >= 0 && avail_mb < kMinComputeHeadroomMb) {
      LOG_ERROR("[GGML-QWEN-FULL] refusing graph compute: MemAvailable=%ld MiB "
                "< %ld MiB safety floor -- falling back to the legacy path\n",
                avail_mb, kMinComputeHeadroomMb);
      return false;
    }
  }

  const auto t_compute_start = std::chrono::steady_clock::now();
  ggml_status st = ggml_backend_sched_graph_compute_async(g.sched, g.graph);
  const auto t_submit_end = std::chrono::steady_clock::now();
  if (st != GGML_STATUS_SUCCESS) {
    LOG_ERROR("[GGML-QWEN-FULL] graph compute failed: status=%d\n", (int)st);
    return false;
  }
  ggml_backend_sched_synchronize(g.sched);
  const auto t_sync_end = std::chrono::steady_clock::now();
  ggml_backend_tensor_get(g.output, hidden_states, 0,
                          static_cast<size_t>(g.D) * S * sizeof(float));
  const auto t_output_end = std::chrono::steady_clock::now();

  for (size_t i = 0; i < g.gdn_layers.size(); ++i) {
    const int l = g.gdn_layers[i];
    const size_t state_n = static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn;
    const size_t hist_n = static_cast<size_t>(g.d_conv - 1) * g.conv_dim;
    ggml_backend_tensor_get(g.gdn_state_out[i],
                            engine.qwen4exp_recur_state[l].data(), 0,
                            state_n * sizeof(float));
    ggml_backend_tensor_get(g.gdn_hist_out[i],
                            engine.qwen4exp_conv_history[l].data(), 0,
                            hist_n * sizeof(float));
  }
  for (size_t i = 0; i < g.kv_layers.size(); ++i) {
    const int l = g.kv_layers[i];
    if (l < 0 || l >= static_cast<int>(engine.Kc.size()) ||
        l >= static_cast<int>(engine.Vc.size()) || !engine.Kc[l] ||
        !engine.Vc[l])
      return fail("dense KV cache is not allocated");
    const size_t kv_n = static_cast<size_t>(g.S) * g.Hkv * g.hd;
    std::vector<float> k(kv_n), v(kv_n);
    ggml_backend_tensor_get(g.kv_k[i], k.data(), 0, kv_n * sizeof(float));
    ggml_backend_tensor_get(g.kv_v[i], v.data(), 0, kv_n * sizeof(float));
    const size_t row = static_cast<size_t>(g.Hkv) * g.hd;
    for (int s = 0; s < g.S && s + pos_base < engine.max_t; ++s) {
      std::memcpy(engine.Kc[l] + static_cast<size_t>(s + pos_base) * row,
                  k.data() + static_cast<size_t>(s) * row, row * sizeof(float));
      std::memcpy(engine.Vc[l] + static_cast<size_t>(s + pos_base) * row,
                  v.data() + static_cast<size_t>(s) * row, row * sizeof(float));
    }
  }
  if (g.ple_layer >= 0 && g.ple_hist_out) {
    const auto &c = engine.model_file.architecture_config;
    const size_t hist_n = static_cast<size_t>(c.ple_conv_kernel - 1) *
                          c.ple_ngram_size * g.HC * g.D;
    auto &dst = engine.qwen4exp_ple_conv_history[g.ple_layer];
    ggml_backend_tensor_get(g.ple_hist_out, dst.data(), 0,
                            hist_n * sizeof(float));
    engine.qwen4exp_ple_tok_hist.clear();
    const int keep = std::max(0, c.ple_ngram_size - 1);
    const int start = std::max(0, S - keep);
    for (int i = start; i < S; ++i)
      engine.qwen4exp_ple_tok_hist.push_back(token_ids[i]);
    engine.qwen4exp_ple_next_pos = S;
  }
  const auto t_state_end = std::chrono::steady_clock::now();
  const auto ms = [](const auto &a, const auto &b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  LOG_INFO(
      "[GGML-QWEN-FULL-TIME] submit=%.3f ms sync=%.3f ms output_get=%.3f ms "
      "state_cache_get=%.3f ms total=%.3f ms\n",
      ms(t_compute_start, t_submit_end), ms(t_submit_end, t_sync_end),
      ms(t_sync_end, t_output_end), ms(t_output_end, t_state_end),
      ms(t_compute_start, t_state_end));
  engine.qwen4exp_full_graph_output_ready = true;
  return true;
}

#else

bool qwen4exp_full_graph_enabled() { return false; }
bool qwen4exp_full_graph_decode_enabled() { return false; }
int qwen4exp_full_graph_device_expert_layers() { return 0; }
bool qwen4exp_full_graph_prepare(StreamingEngine &, int, int, const int *) {
  return false;
}
bool qwen4exp_full_graph_decode_prepare(StreamingEngine &, int, const int *) {
  return false;
}
bool qwen4exp_full_graph_run(StreamingEngine &, float *, int, int,
                             const int *) {
  return false;
}
bool qwen4exp_full_graph_decode_run(StreamingEngine &, float *, int,
                                    const int *, float *, int) {
  return false;
}
void qwen4exp_full_graph_shutdown() {}

#endif
