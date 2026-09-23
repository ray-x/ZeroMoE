#include "active_layer_upload_workers.hpp"
#include "async_prefetcher.hpp"
#include "cpu_expert_executor.hpp"
#include "gpu_expert_executor.hpp"
#include "dense_dispatch.hpp"
#include "engine.hpp"
#include "expert_activation.hpp"
#include "ggml_deepseek_v4_graph.hpp"
#include "ggml_gdn_proj_backend.hpp"
#include "ggml_glm5next_graph.hpp"
#include "ggml_hc_backend.hpp"
#include "ggml_moe_backend.hpp"
#include "ggml_qwen4exp_graph.hpp"
#include "kernels_cpu.hpp"
#include "kquant_dequant.hpp"
#include "logger.hpp"
#include "mtp_speculative.hpp"
#include "moe_pipeline.hpp"
#include "moe_router.hpp"
#include "perf_timer.hpp"
#include "prefill_expert_stream.hpp"
#include "ram_manager.hpp"
#include "scratch_buffers.hpp"
#include "tensor_trace.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

extern void mla_attention_absorbed_gguf(StreamingEngine &engine, int layer_idx,
                                        const float *Q_roped, int S,
                                        int pos_base, float *out);
extern void dense_attention_gguf(StreamingEngine &engine, int layer_idx,
                                 const float *hidden_states, int S,
                                 int pos_base, float *out);
extern bool qwen4exp_hc_device_live();
extern bool qwen4exp_hc_post_combine_device(StreamingEngine &engine,
                                            int layer_idx, int sub_block, int S,
                                            int D,
                                            const void *device_sub_block_out,
                                            const void *producer_event);
extern bool qwen4exp_hc_post_combine_device_raw(StreamingEngine &engine,
                                                int layer_idx, int sub_block,
                                                int S, int D,
                                                const void *device_data);
extern bool qwen4exp_hc_prepare_host_layer_input(StreamingEngine &engine,
                                                 int layer_idx, int S);
extern bool qwen4exp_hc_download_streams(StreamingEngine &engine, int S);
extern bool qwen4exp_hc_commit_host_layer_input(StreamingEngine &engine,
                                                int layer_idx, int S);
extern bool qwen4exp_copy_gdn_device_output_to_host(const void *device_output,
                                                    float *host, int S);

#include "moe_pipeline_internal.hpp"
#include "moe_router_service.hpp"
#include "shared_expert_executor.hpp"
#include "token_dispatch_pipeline.hpp"

void moe_pipeline_internal::qwen4exp_record_pilot_fetch_start(int layer_idx) {
  if (layer_idx < 0 || layer_idx >= PerfCounters::kMaxTrackedLayersForMiss)
    return;
  const int64_t fetch_start_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  const int64_t router_ready_ns =
      perf().qwen_pilot_router_ready_ns[layer_idx].exchange(
          0, std::memory_order_acq_rel);
  const int64_t prediction_ready_ns =
      perf().qwen_pilot_prediction_for_fetch_ns[layer_idx].exchange(
          0, std::memory_order_acq_rel);
  if (router_ready_ns > 0 && fetch_start_ns >= router_ready_ns) {
    PerfCounters::atomic_add(
        perf().t_qwen_pilot_router_to_fetch,
        (double)(fetch_start_ns - router_ready_ns) * 1e-9);
    perf().n_qwen_pilot_router_to_fetch.fetch_add(1,
                                                  std::memory_order_relaxed);
  }
  if (prediction_ready_ns > 0 && fetch_start_ns >= prediction_ready_ns) {
    PerfCounters::atomic_add(
        perf().t_qwen_pilot_ready_to_fetch,
        (double)(fetch_start_ns - prediction_ready_ns) * 1e-9);
    perf().n_qwen_pilot_ready_to_fetch.fetch_add(1,
                                                 std::memory_order_relaxed);
  }
}

static bool qwen4exp_device_route_enabled() {
  static const bool enabled = [] {
    const char *e = std::getenv("MOESTREAM_QWEN4EXP_DEVICE_ROUTE");
    return e && e[0] == '1';
  }();
  return enabled;
}

bool moe_pipeline_detail::qwen4exp_router_pilot_layer_allowed(
    const StreamingEngine &engine, int target_layer) {
  // A cpu_affinity layer is a standing CPU-only contract: its experts must
  // not be VRAM-pinned, asynchronously promoted, or used as a target for a
  // Qwen pilot graph.  The ordinary dispatch/preload paths already enforce
  // this through is_cpu_affinity_layer(); keep the Qwen-only advisory path
  // aligned so it does not spend prediction/PCIe work for a layer that can
  // never consume the predicted GPU experts.
  if (is_cpu_affinity_layer(target_layer))
    return false;
  if (!engine.qwen4exp_router_pilot_predict_layers.empty() &&
      std::find(engine.qwen4exp_router_pilot_predict_layers.begin(),
                engine.qwen4exp_router_pilot_predict_layers.end(),
                target_layer) ==
          engine.qwen4exp_router_pilot_predict_layers.end())
    return false;
  return true;
}

static std::array<std::atomic<uint8_t>,
                  PerfCounters::kMaxTrackedLayersForMiss>
    qwen4exp_decode_hot_streak{};

// Decode-only Qwen pilot policy: remember whether the exact route just used
// for a layer was already resident in either the temporary active area or the
// durable VRAM cache.  The old skip check required all 512 experts in the
// layer to be resident, which is almost never true when only a hot subset is
// pinned.  This deliberately observes the route AFTER exact routing, then
// lets the NEXT token bypass an otherwise redundant pilot after two stable
// observations.  A miss immediately resets the streak, so skipping the pilot
// can never change the authoritative route or its correctness fallback.
static void qwen4exp_note_decode_route_residency(
    const StreamingEngine &engine, const RAMSafetyManager &ram_mgr,
    int layer_idx, int S, int topk, const LayerScratch &scratch) {
  if (S != 1 || layer_idx < 0 ||
      layer_idx >= PerfCounters::kMaxTrackedLayersForMiss || topk <= 0 ||
      engine.family_ops.family != ModelFamily::Qwen4Exp ||
      is_cpu_affinity_layer(layer_idx) || !ram_mgr.vram_cache ||
      scratch.tok_topk_ids.size() < (size_t)topk)
    return;

  bool all_resident = true;
  for (int k = 0; k < topk; ++k) {
    const int expert_id = scratch.tok_topk_ids[(size_t)k];
    if (expert_id < 0 ||
        (!ram_mgr.vram_cache->active_layer_has(layer_idx, expert_id) &&
         !ram_mgr.vram_cache->main_layer_has(layer_idx, expert_id))) {
      all_resident = false;
      break;
    }
  }

  auto &p = perf();
  p.n_qwen_pilot_adaptive_residency_checks.fetch_add(
      1, std::memory_order_relaxed);
  if (!all_resident) {
    p.n_qwen_pilot_adaptive_residency_misses.fetch_add(
        1, std::memory_order_relaxed);
    qwen4exp_decode_hot_streak[layer_idx].store(0,
                                                std::memory_order_release);
    return;
  }

  auto &streak = qwen4exp_decode_hot_streak[layer_idx];
  uint8_t old = streak.load(std::memory_order_relaxed);
  while (old < 2 &&
         !streak.compare_exchange_weak(old, (uint8_t)(old + 1),
                                       std::memory_order_release,
                                       std::memory_order_relaxed)) {
  }
}

bool moe_pipeline_detail::qwen4exp_decode_hot_set_stable(int layer_idx) {
  return layer_idx >= 0 &&
         layer_idx < PerfCounters::kMaxTrackedLayersForMiss &&
         qwen4exp_decode_hot_streak[layer_idx].load(
             std::memory_order_acquire) >= 2;
}

// Queue the same prediction without waiting for its compact route table.
// The caller consumes it at the beginning of the predicted layer, after that
// layer's attention work has had a chance to run. This moves the pilot's
// unavoidable route-table synchronization out of the layer-N -> layer-N+1
// boundary instead of merely moving the GEMV between CPU and GPU.
bool moe_pipeline_detail::qwen4exp_router_pilot_gpu_async(
    StreamingEngine &engine, int target_layer, int D, int n_experts, int topk,
    const void **device_ids, const void **device_weights) {
  if (!device_ids || !device_weights || target_layer < 0 || n_experts <= 0 ||
      topk <= 0 || !moe_ggml_gpu_router() || !moe_ggml_mul_mat_id() ||
      !engine.qwen4exp_device_hc_ffn_input || !qwen4exp_hc_device_live() ||
      !moe_pipeline_detail::qwen4exp_router_pilot_layer_allowed(
          engine, target_layer))
    return false;
  const void *streams = GgmlHcRunner::instance().device_streams_tensor();
  if (!streams)
    return false;
  const std::string router_key =
      "blk." + std::to_string(target_layer) + ".ffn_gate_inp.weight";
  auto router_it = engine.resident_layers.find(router_key);
  if (router_it == engine.resident_layers.end() || !router_it->second.host_ptr)
    return false;
  GgmlRouterWeightRef ref;
  const auto &rw = router_it->second;
  ref.host_data = rw.host_ptr;
  ref.type = (uint32_t)rw.meta.type;
  ref.ne0 = rw.meta.shape.size() > 0 ? (int64_t)rw.meta.shape[0] : 0;
  ref.ne1 = rw.meta.shape.size() > 1 ? (int64_t)rw.meta.shape[1] : 0;
  ref.nbytes = rw.meta.size_bytes;
  const std::string bias_key =
      "blk." + std::to_string(target_layer) + ".exp_probs_b.bias";
  const float *bias = nullptr;
  auto bias_it = engine.resident_layers.find(bias_key);
  if (bias_it != engine.resident_layers.end() && bias_it->second.host_ptr &&
      bias_it->second.meta.type == GGUFType::F32)
    bias = reinterpret_cast<const float *>(bias_it->second.host_ptr);
  auto &router = GgmlRouterRunner::instance();
  const int device = moe_ggml_device() >= 0 ? moe_ggml_device() : 0;
  if (!router.ensure(
          device, D, n_experts, topk, /*n_tokens=*/1,
          engine.model_file.architecture_config.expert_weights_norm,
          engine.model_file.architecture_config.expert_weights_scale) ||
      !router.ensure_layer(target_layer, ref, bias))
    return false;
  if (!router.run_tensor_stream0_device(
          target_layer, streams, /*token_index=*/0, device_ids, device_weights))
    return false;
  // Start the tiny route-table D2H immediately after the prediction graph.
  // The consumer at the next layer polls its completion event and only waits
  // if the intervening attention work did not hide the copy.
  const bool queued = router.queue_route_to_host(target_layer);
  if (queued && target_layer < PerfCounters::kMaxTrackedLayersForMiss) {
    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    perf().qwen_pilot_prediction_ready_ns[target_layer].store(
        now_ns, std::memory_order_release);
  }
  return queued;
}

void moe_pipeline_detail::qwen4exp_process_pilot_predictions(
    StreamingEngine &engine, RAMSafetyManager &ram_mgr,
    LockFreePrefetcher &async_io_queue, LayerScratch &scratch, int target_layer,
    int qwen_k) {
  const auto &c = engine.model_file.architecture_config;
  if (target_layer < 0 || qwen_k <= 0 || qwen_k > LayerScratch::kQwenPilotMax ||
      is_cpu_affinity_layer(target_layer))
    return;
  const int64_t prediction_ready_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  if (target_layer < PerfCounters::kMaxTrackedLayersForMiss) {
    const int64_t queued_ns =
        perf().qwen_pilot_prediction_ready_ns[target_layer].exchange(
            0, std::memory_order_acq_rel);
    if (queued_ns > 0 && prediction_ready_ns >= queued_ns) {
      PerfCounters::atomic_add(
          perf().t_qwen_pilot_queue_to_ready,
          (double)(prediction_ready_ns - queued_ns) * 1e-9);
      perf().n_qwen_pilot_queue_to_ready.fetch_add(
          1, std::memory_order_relaxed);
    }
    // This timestamp is the point at which the predicted IDs are known on
    // the host and the prediction decision starts; it is distinct from the
    // earlier GPU route-graph submission above.
    perf().qwen_pilot_prediction_ready_ns[target_layer].store(
        prediction_ready_ns, std::memory_order_release);
  }
  const int qwen_prefetch_k =
      std::min({engine.qwen4exp_router_pilot_prefetch_k > 0
                    ? engine.qwen4exp_router_pilot_prefetch_k
                    : engine.qwen4exp_router_pilot_k,
                qwen_k, LayerScratch::kQwenPilotMax});
  const bool qwen_prefetch_layer_enabled =
      std::find(engine.qwen4exp_router_pilot_prefetch_exclude_layers.begin(),
                engine.qwen4exp_router_pilot_prefetch_exclude_layers.end(),
                target_layer) ==
      engine.qwen4exp_router_pilot_prefetch_exclude_layers.end();
  const bool qwen_prefetch_enabled =
      engine.qwen4exp_router_pilot_prefetch && qwen_prefetch_layer_enabled;
  perf().n_qwen_pilot_prefetch_k_observed.store(
      (uint32_t)(engine.qwen4exp_router_pilot_prefetch ? qwen_prefetch_k : 0),
      std::memory_order_relaxed);
  if (engine.qwen4exp_router_pilot_prefetch && !qwen_prefetch_layer_enabled) {
    perf().n_qwen_pilot_prefetch_excluded.fetch_add(1,
                                                    std::memory_order_relaxed);
  } else if (engine.qwen4exp_router_pilot_prefetch &&
             qwen_prefetch_k < qwen_k) {
    perf().n_qwen_pilot_prefetch_rank_skipped.fetch_add(
        (uint64_t)(qwen_k - qwen_prefetch_k), std::memory_order_relaxed);
  }
  scratch.qwen_pilot_pred_count = qwen_k;
  scratch.qwen_pilot_pred_layer = target_layer;
  scratch.qwen_pilot_prefetch_count =
      qwen_prefetch_enabled ? qwen_prefetch_k : 0;
  perf().n_qwen_pilot_predictions.fetch_add(1, std::memory_order_relaxed);

  for (int k = 0; k < qwen_k; ++k) {
    const int best = scratch.qwen_pilot_pred_ids[k];
    if (best < 0 || best >= c.n_experts)
      continue;
    // Classify without a mutating lookup. The active-layer pool is a real
    // VRAM source but is separate from the main LRU cache.
    bool already_vram =
        ram_mgr.vram_cache &&
        (ram_mgr.vram_cache->active_layer_has(target_layer, best) ||
         ram_mgr.vram_cache->main_layer_has(target_layer, best));
    const bool already_ram =
        !already_vram && ram_mgr.is_ram_resident(target_layer, best);
    const bool qwen_prefetch_this_rank =
        qwen_prefetch_enabled && k < qwen_prefetch_k;
    if (already_vram) {
      perf().n_qwen_pilot_pred_already_vram.fetch_add(
          1, std::memory_order_relaxed);
    } else if (already_ram) {
      perf().n_qwen_pilot_pred_already_ram.fetch_add(1,
                                                     std::memory_order_relaxed);
      if (qwen_prefetch_this_rank) {
        ExpertLocation *issued =
            ram_mgr.promote_ram_to_active_layer(
                engine, target_layer, best, -1, prediction_ready_ns);
        if (issued) {
          perf().n_qwen_pilot_active_prefetch_issued.fetch_add(
              1, std::memory_order_relaxed);
        } else {
          perf().n_qwen_pilot_active_prefetch_failed.fetch_add(
              1, std::memory_order_relaxed);
        }
      }
    } else {
      perf().n_qwen_pilot_pred_needs_fetch.fetch_add(1,
                                                     std::memory_order_relaxed);
      if (qwen_prefetch_this_rank) {
        async_io_queue.push_hint(target_layer, best);
        perf().n_qwen_pilot_hints_pushed.fetch_add(1,
                                                   std::memory_order_relaxed);
      }
    }
    if (qwen_prefetch_this_rank && already_vram) {
      perf().n_qwen_pilot_hints_skipped_resident.fetch_add(
          1, std::memory_order_relaxed);
    }
  }
}

// The grouped Qwen decode graph normally sources routed weights from the main
// VRAM cache/RAM tier, but its explicit allow_active_layer_sources argument
// also lets a completed prediction fill serve the graph through a D2D copy.
// Keep the older generic prewarm mechanisms family-gated so other model
// families retain their existing behavior.
bool moe_pipeline_detail::qwen4exp_decode_ggml_path(
    const StreamingEngine &engine) {
  return engine.qwen4exp_decode_ggml_moe &&
         engine.family_ops.family == ModelFamily::Qwen4Exp;
}
// Diagnostic/A-B only: route CPU-affinity Qwen decode layers with the local
// AVX-512 F32 matmul instead of the ordinary dense_proj GPU GEMV.  The
// production route remains unchanged unless this explicit Qwen-only switch is
// set.  Other model families, GPU-affinity Qwen layers, and prefill never take
// this branch.
static bool qwen4exp_decode_cpu_router_enabled() {
  static const bool enabled = [] {
    const char *e = std::getenv("MOESTREAM_QWEN_DECODE_CPU_ROUTER");
    return e && e[0] == '1';
  }();
  return enabled;
}

// Diagnostic/A-B only: use the already device-resident Qwen HC mixed row as
// the input to the ordinary CUDA router GEMV.  This is deliberately separate
// from qwen4exp_decode_gpu_router: it does not build a second GGML routing
// graph or change top-k/cache bookkeeping; it only removes the host->device
// activation upload from dense_proj().  Keep it opt-in until the router's
// device/host numerical result and real decode timing are confirmed.
static bool qwen4exp_decode_device_router_gemv_enabled() {
  static const bool enabled = [] {
    const char *e = std::getenv("MOESTREAM_QWEN_DECODE_DEVICE_ROUTER_GEMV");
    return e && e[0] == '1';
  }();
  return enabled;
}

// Bee-style whole-graph decode topology is deliberately Qwen-only and
// config-gated while it is being validated.  It is safe only for a layer
// backed by the immutable fixed-stride DDR arena: a scheduler graph cannot
// point one MUL_MAT_ID tensor at a changing set of lazy cache slots.  If the
// arena is absent, the caller remains on the existing lazy-loading path,
// which is the correct behavior for partially loaded Qwen layers.  GLM,
// DeepSeek, and other families never consult this switch; their lazy expert
// workflows are unchanged.  Do not add an environment override here: a
// decode full graph must be an explicit model/config decision, not an
// inherited process-wide experiment setting.
static bool qwen4exp_decode_scheduler_enabled(const StreamingEngine &engine) {
  return engine.qwen4exp_decode_scheduler;
}

// Predictive Partial Pinning empirical filter (2026-09-19): which layer to
// check, read directly from the same env var Phase 1/2 uses
// (MOESTREAM_QWEN4EXP_PARTIAL_LAYER) -- this measurement runs in LEGACY
// dispatch (where qwen4exp_router_pilot actually lives), so it cannot rely
// on engine.qwen4exp_partial_layer/plan_partial_layer(), which are only
// ever populated from the full-graph prepare path.
static int qwen4exp_ppp_target_layer() {
  static const int layer = [] {
    const char *v = std::getenv("MOESTREAM_QWEN4EXP_PARTIAL_LAYER");
    if (!v || !v[0])
      return -1;
    char *end = nullptr;
    const long n = std::strtol(v, &end, 10);
    return (end == v || *end != '\0' || n < 0) ? -1 : static_cast<int>(n);
  }();
  return layer;
}

// Mirrors plan_partial_layer's own ranking (ggml_qwen4exp_graph.cpp) --
// duplicated rather than shared across translation units because that
// function has internal linkage and engine.qwen4exp_partial_layer_gpu_ids
// is only ever populated by the full-graph prepare path, never by legacy
// dispatch. Computed once (function-local static cache), since the usage
// file and env vars are fixed for the life of the process.
static const std::unordered_set<int> &qwen4exp_ppp_pin_set(StreamingEngine &engine) {
  static const std::unordered_set<int> pin_set = [&engine] {
    std::unordered_set<int> result;
    const int layer = qwen4exp_ppp_target_layer();
    if (layer < 0)
      return result;
    const char *topn_env = std::getenv("MOESTREAM_QWEN4EXP_PARTIAL_LAYER_TOP_N");
    int top_n = 256;
    if (topn_env && topn_env[0]) {
      char *end = nullptr;
      const long n = std::strtol(topn_env, &end, 10);
      if (end != topn_env && *end == '\0' && n > 0)
        top_n = static_cast<int>(n);
    }
    const std::string path = engine.qwen4exp_graph_pin_file.empty()
                                 ? std::string(".moestream_usage")
                                 : engine.qwen4exp_graph_pin_file;
    auto [entries, maxc] = StreamingEngine::load_usage(path);
    (void)maxc;
    std::vector<StreamingEngine::UsageEntry> layer_entries;
    for (const auto &e : entries)
      if (e.layer == layer)
        layer_entries.push_back(e);
    std::stable_sort(layer_entries.begin(), layer_entries.end(),
                     [](const auto &a, const auto &b) {
                       return a.count > b.count;
                     });
    for (size_t i = 0; i < layer_entries.size() &&
                       static_cast<int>(i) < top_n;
         ++i)
      result.insert(layer_entries[i].expert);
    LOG_INFO("[QWEN-PILOT-PPP] layer=%d static pin set: %zu expert(s) from "
             "%s (top_n=%d)\n",
             layer, result.size(), path.c_str(), top_n);
    return result;
  }();
  return pin_set;
}

bool MoeRouterService::route(MoePipelineContext &context) {
  auto &engine = context.engine;
  auto &ram_mgr = context.ram_mgr;
  auto &async_io_queue = context.async_io_queue;
  const int layer_idx = context.layer_idx;
  const float *hidden_states = context.hidden_states;
  const int S = context.S;
  float *output = context.output;
  auto &scratch = context.scratch;
  const int *token_ids = context.token_ids;
  auto &c = context.c;
  const int D = context.D;
  const int topk = context.topk;
  const int n_experts = context.n_experts;
  const int moe_inter = context.moe_inter;
  const float clamp_exp = context.clamp_exp;
  const float clamp_shexp = context.clamp_shexp;
  auto &gpu_router_handled = context.gpu_router_handled;
  auto &gpu_route_device = context.gpu_route_device;
  auto &gpu_route_ids_device = context.gpu_route_ids_device;
  auto &gpu_route_weights_device = context.gpu_route_weights_device;
  auto &device_router = context.device_router;
  auto &gpu_route_ids_ks = context.gpu_route_ids_ks;
  auto &gpu_route_weights_ks = context.gpu_route_weights_ks;
  auto &batched_handled = context.batched_handled;
  // DeepSeek-V4 hash-routed layers (first c.hash_layer_count layers):
  // expert SELECTION is a direct token-id -> expert-id lookup via
  // ffn_gate_tid2eid.weight (one row of `topk` int32 expert ids per
  // vocab token, indexed by token id -- ggml's real ggml_get_rows(
  // tid2eid, token_id) semantics, no runtime hash computation despite
  // the tensor's name; verified against llama.cpp's real DeepSeek-V4
  // implementation, src/models/deepseek4.cpp + llama-graph.cpp's
  // build_moe_ffn). The router's own logits/probs (computed
  // unconditionally below, every layer -- see graph_initializer.cpp's
  // preload-manifest fix) still supply the WEIGHTS for whichever
  // experts this resolves to; only the selection step branches, in the
  // per-token loop further down.
  const int32_t *tid2eid = nullptr;
  // Real on-disk row count (ne[1]) of ffn_gate_tid2eid.weight for THIS
  // layer -- i.e. the number of vocab rows the table actually has.
  // Populated only when `tid2eid` above is populated. ROOT-CAUSE FIX:
  // the per-token lookup below used to bounds-check `tok` against
  // `c.vocab` (the engine's global vocab size, derived from
  // token_embd.weight -- see graph_initializer.cpp's own vocab_size
  // fallback history) under the assumption that it equals this
  // tensor's real ne[1]. Nothing actually enforces that the two agree:
  // if this GGUF (or a separate dense_model_path file, when
  // has_dense_model is set -- ffn_gate_tid2eid.weight loads from
  // whichever of the two `initialize_resident_memory` was given) ever
  // exported this specific table over a SMALLER vocab than c.vocab,
  // `tid2eid + tok*topk` for any real token id past that smaller
  // extent reads past the posix_memalign'd buffer entirely -- an
  // out-of-bounds heap read (UB), not a bug in the file's actual
  // content. That reads as EXACTLY this symptom: every hash-route
  // lookup "succeeds" past the type/shape check below (type IS I32,
  // shape[0] IS topk) yet every row comes back full of nonsense
  // expert ids, for every token, every hash layer -- because every
  // lookup this run happens to land past the table's real extent.
  // Bounding against the tensor's OWN shape[1] instead closes the UB
  // and turns a silent corruption into a clean, diagnosable fallback;
  // the mismatch check right after logs the smoking gun the moment it
  // fires.
  int64_t tid2eid_n_vocab = 0;
  if (layer_idx < c.hash_layer_count) {
    std::string tid2eid_key =
        "blk." + std::to_string(layer_idx) + ".ffn_gate_tid2eid.weight";
    auto tid2eid_it = engine.resident_layers.find(tid2eid_key);
    // Validate the on-disk type/shape BEFORE reinterpreting host_ptr as
    // int32_t* -- added after a real run showed EVERY hash-route lookup
    // failing its bounds check (100% of tokens, all hash layers), which
    // is far more consistent with reading the wrong bytes entirely
    // (e.g. the tensor's real ggml type isn't I32, so calculate_size()
    // undersized the read and left the tail as uninitialized heap
    // garbage -- see GGUFType::I64's own comment) than with a subtle
    // per-token routing quirk. Previously this reinterpret_cast was
    // unconditional on host_ptr alone, so a type/width mismatch would
    // silently produce garbage ids instead of a diagnosable error.
    const GGUFTensor *meta_ptr = (tid2eid_it != engine.resident_layers.end())
                                     ? &tid2eid_it->second.meta
                                     : nullptr;
    bool shape_ok = meta_ptr && !meta_ptr->shape.empty() &&
                    (int64_t)meta_ptr->shape[0] == (int64_t)topk;
    if (meta_ptr && tid2eid_it->second.host_ptr &&
        meta_ptr->type == GGUFType::I32 && shape_ok) {
      tid2eid = reinterpret_cast<const int32_t *>(tid2eid_it->second.host_ptr);
      tid2eid_n_vocab =
          (meta_ptr->shape.size() >= 2) ? (int64_t)meta_ptr->shape[1] : 0;
      // Loud, PER-LAYER, ONE-SHOT: fires only if the table's real
      // extent actually diverges from c.vocab -- the previous
      // silent-UB path never had a way to surface this at all.
      static std::vector<bool> tid2eid_vocab_mismatch_logged;
      if ((int)tid2eid_vocab_mismatch_logged.size() <= layer_idx)
        tid2eid_vocab_mismatch_logged.resize(layer_idx + 1, false);
      if (!tid2eid_vocab_mismatch_logged[layer_idx] && tid2eid_n_vocab > 0 &&
          tid2eid_n_vocab != (int64_t)c.vocab) {
        LOG_ERROR("[MOE-ROUTE] layer=%d ffn_gate_tid2eid.weight real n_vocab "
                  "(shape[1]=%lld) "
                  "!= engine c.vocab (%d) -- bounding token-id lookups to the "
                  "SMALLER of the "
                  "two from now on; any real token id in that gap can never "
                  "use the hash "
                  "route for this layer and falls back to the learned router "
                  "instead.\n",
                  layer_idx, (long long)tid2eid_n_vocab, c.vocab);
        tid2eid_vocab_mismatch_logged[layer_idx] = true;
      }
    } else if (meta_ptr && tid2eid_it->second.host_ptr) {
      // Tensor loaded, but its type or row width doesn't match what
      // this raw-int32 read path assumes -- loud, PER-LAYER (see
      // below), and named specifically so a real run pinpoints
      // exactly which assumption broke instead of just seeing every
      // token's row fail the downstream bounds check with no clue why.
      static std::vector<bool> tid2eid_shape_warned_per_layer;
      if ((int)tid2eid_shape_warned_per_layer.size() <= layer_idx)
        tid2eid_shape_warned_per_layer.resize(layer_idx + 1, false);
      if (!tid2eid_shape_warned_per_layer[layer_idx]) {
        LOG_ERROR(
            "[MOE-ROUTE] layer=%d ffn_gate_tid2eid.weight loaded but "
            "type/shape "
            "mismatch (type=%u expected I32=%u, shape[0]=%llu expected "
            "topk=%d) -- "
            "falling back to the normal learned-router top-k for this layer.\n",
            layer_idx, (unsigned)meta_ptr->type, (unsigned)GGUFType::I32,
            meta_ptr->shape.empty() ? 0ull
                                    : (unsigned long long)meta_ptr->shape[0],
            topk);
        tid2eid_shape_warned_per_layer[layer_idx] = true;
      }
    } else {
      // Loud, PER-LAYER warning -- FIX: the old version of this
      // warning was a single function-static bool SHARED across
      // every hash-routed layer, so only the very first one to hit
      // this path ever logged anything; every other hash-routed
      // layer's identical failure was completely silent. Falls back
      // to the normal argsort-over-learned-router-probs path below
      // for this layer -- a REAL, working fallback now that
      // ffn_gate_inp.weight is unconditionally loaded (see the
      // preload-manifest fix), unlike the previous bug where the
      // router itself was also missing and this whole layer's MoE
      // output was silently zero-filled instead of falling back to
      // anything.
      static std::vector<bool> tid2eid_warned_per_layer;
      if ((int)tid2eid_warned_per_layer.size() <= layer_idx)
        tid2eid_warned_per_layer.resize(layer_idx + 1, false);
      if (!tid2eid_warned_per_layer[layer_idx]) {
        LOG_ERROR(
            "[MOE-ROUTE] layer=%d is hash-routed (hash_layer_count=%d) but "
            "ffn_gate_tid2eid.weight failed to load (missing/short-read) -- "
            "falling back to the normal learned-router top-k for this layer, "
            "which will select DIFFERENT experts than the real hash table "
            "would.\n",
            layer_idx, c.hash_layer_count);
        tid2eid_warned_per_layer[layer_idx] = true;
      }
    }
  }

  // Bee-style decode topology experiment.  This is placed before the
  // ordinary host-router block so the CUDA router and the DDR-backed CPU
  // MUL_MAT_ID expert graph are submitted as one scheduler graph.  It is
  // intentionally conservative:
  //   * Qwen4Exp only; GLM/DeepSeek keep their existing lazy topology.
  //   * only S==1 and a complete immutable DDR arena; if an expert/layer is
  //     lazy, qwen4exp_ram_cpu_moe_layer() fails and the normal path below
  //     performs the existing lazy load.
  //   * only when the next layer is also CPU-affinity (or this is the last
  //     layer), because this first graph does not emit the existing Qwen
  //     pilot/prefetch handoff for a following GPU layer.
  //   * pilot/prefetch remains on the established path until route IDs from
  //     this graph are integrated into its recall/promotion accounting.
  //
  // The topology is opt-in during validation via the config key
  // qwen4exp_decode_scheduler=true.  A failed graph build or shared-GPU
  // handoff is a local fallback, never a model-wide failure.
  const bool scheduler_target_is_cpu_or_last =
      layer_idx + 1 >= c.n_layers || is_cpu_affinity_layer(layer_idx + 1);
  const bool scheduler_drops_active_pilot =
      !scheduler_target_is_cpu_or_last &&
      moe_pipeline_detail::qwen4exp_router_pilot_layer_allowed(
          engine, layer_idx + 1) &&
      (engine.qwen4exp_router_pilot ||
       engine.qwen4exp_router_pilot_prefetch);
  const bool qwen_decode_scheduler_candidate =
      qwen4exp_decode_scheduler_enabled(engine) && S == 1 &&
      engine.qwen4exp_cpu_ggml_moe &&
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      engine.family_ops.uses_hyper_connections &&
      engine.qwen_ram_preload_all_experts &&
      is_cpu_affinity_layer(layer_idx) && clamp_exp == 0.0f &&
      engine.graph_ram_layer_ready(layer_idx) &&
      !scheduler_drops_active_pilot && scheduler_target_is_cpu_or_last &&
      engine.qwen4exp_device_hc_ffn_input &&
      moe_ggml_shared_expert() && c.n_expert_shared > 0 &&
      // Projection-compact arenas have independent gate/up/down expert
      // strides.  The Bee-style scheduler graph has one common stride
      // contract, so declining here avoids repeated rebuild/reject attempts
      // and routes these layers through the parity-verified per-expert CPU
      // references in token_dispatch_pipeline.cpp.
      !engine.graph_ram_layer_projection_compact(layer_idx);
  if (qwen4exp_decode_scheduler_enabled(engine) && S == 1 &&
      layer_idx == 0) {
    static std::atomic<bool> logged_scheduler_gate{false};
    if (!logged_scheduler_gate.exchange(true, std::memory_order_relaxed)) {
      const std::string scheduler_router_key =
          "blk.0.ffn_gate_inp.weight";
      auto scheduler_router_it =
          engine.resident_layers.find(scheduler_router_key);
      LOG_INFO(
          "[GGML-QWEN-DECODE-SCHED] gate candidate=%d S=%d cpu=%d "
          "ram_preload=%d router_device=%d ram_arena=%d shared=%d hc=%d "
          "clamp=%.6g\n",
          qwen_decode_scheduler_candidate ? 1 : 0, S,
          is_cpu_affinity_layer(layer_idx) ? 1 : 0,
          engine.qwen_ram_preload_all_experts ? 1 : 0,
          scheduler_router_it != engine.resident_layers.end() &&
                  scheduler_router_it->second.device_ptr
              ? 1
              : 0,
          engine.graph_ram_layer_ready(layer_idx) ? 1 : 0,
          (moe_ggml_shared_expert() && c.n_expert_shared > 0) ? 1 : 0,
          engine.qwen4exp_device_hc_ffn_input ? 1 : 0, clamp_exp);
    }
  }
  if (qwen_decode_scheduler_candidate) {
    const std::string router_key =
        "blk." + std::to_string(layer_idx) + ".ffn_gate_inp.weight";
    auto router_it = engine.resident_layers.find(router_key);
    QwenRamCpuMoeLayer ram_layer;
    const bool scheduler_router_available =
        router_it != engine.resident_layers.end() &&
        (router_it->second.device_ptr || router_it->second.host_ptr);
    const bool scheduler_ram_available =
        scheduler_router_available &&
        qwen4exp_ram_cpu_moe_layer(engine, layer_idx, n_experts, ram_layer);
    if (scheduler_ram_available) {
      const auto &rw = router_it->second;
      GgmlQwenDecodeGraphRef ref;
      ref.router_device = rw.device_ptr;
      ref.router_device_bytes = rw.meta.size_bytes;
      ref.router_host = rw.host_ptr;
      ref.router_host_bytes = rw.meta.size_bytes;
      ref.router_type = static_cast<uint32_t>(rw.meta.type);
      ref.router_ne0 = rw.meta.shape.size() > 0 ? (int64_t)rw.meta.shape[0] : 0;
      ref.router_ne1 = rw.meta.shape.size() > 1 ? (int64_t)rw.meta.shape[1] : 0;
      ref.gate_plane = ram_layer.gate_plane;
      ref.up_plane = ram_layer.up_plane;
      ref.down_plane = ram_layer.down_plane;
      ref.gate_type = ram_layer.meta.gate_type;
      ref.up_type = ram_layer.meta.up_type;
      ref.down_type = ram_layer.meta.down_type;
      ref.gate_bytes = ram_layer.meta.gate_bytes;
      ref.up_bytes = ram_layer.meta.up_bytes;
      ref.down_bytes = ram_layer.meta.down_bytes;
      ref.expert_stride = ram_layer.expert_stride;
      const std::string bias_key =
          "blk." + std::to_string(layer_idx) + ".exp_probs_b.bias";
      auto bias_it = engine.resident_layers.find(bias_key);
      if (bias_it != engine.resident_layers.end() &&
          bias_it->second.host_ptr && bias_it->second.meta.type == GGUFType::F32)
        ref.selection_bias =
            reinterpret_cast<const float *>(bias_it->second.host_ptr);

      // If the persistent dense pool already owns the Qwen shared expert,
      // put that CUDA FFN in the same scheduler graph as the DDR-backed CPU
      // routed FFN.  A missing device copy deliberately declines this
      // optimization and falls through to run_qwen4exp_shared_device(), so
      // partially resident/lazy configurations retain their old behavior.
      const int scheduler_device =
          moe_ggml_device() >= 0 ? moe_ggml_device() : 0;
      const bool no_external_shared_weights = [] {
        const char *e = std::getenv("MOESTREAM_GGML_SHARED_NO_EXTERNAL");
        return e && e[0] == '1';
      }();
      const int shared_inter = moe_inter * c.n_expert_shared;
      auto make_decode_shared_ref =
          [&](const std::string &name, int64_t ne0,
              int64_t ne1) -> std::optional<GgmlSharedWeightRef> {
        auto it = engine.resident_layers.find(name);
        if (it == engine.resident_layers.end() || !it->second.host_ptr)
          return std::nullopt;
        const auto &w = it->second;
        if (w.meta.shape.size() < 2 ||
            (int64_t)w.meta.shape[0] != ne0 ||
            (int64_t)w.meta.shape[1] != ne1)
          return std::nullopt;
        GgmlSharedWeightRef out;
        out.host_data = w.host_ptr;
        out.type = static_cast<uint32_t>(w.meta.type);
        out.ne0 = ne0;
        out.ne1 = ne1;
        out.nbytes = w.meta.size_bytes;
        if (!no_external_shared_weights && engine.gpu &&
            engine.gpu->get_core_device() == scheduler_device &&
            w.device_ptr) {
          out.device_data = w.device_ptr;
          out.device_bytes = w.meta.size_bytes;
        }
        return out;
      };
      auto shared_gate = make_decode_shared_ref(
          "blk." + std::to_string(layer_idx) + ".ffn_gate_shexp.weight", D,
          shared_inter);
      auto shared_up = make_decode_shared_ref(
          "blk." + std::to_string(layer_idx) + ".ffn_up_shexp.weight", D,
          shared_inter);
      auto shared_down = make_decode_shared_ref(
          "blk." + std::to_string(layer_idx) + ".ffn_down_shexp.weight",
          shared_inter, D);
      std::optional<GgmlSharedWeightRef> shared_gate_inp;
      auto shared_gate_inp_it = engine.resident_layers.find(
          "blk." + std::to_string(layer_idx) + ".ffn_gate_inp_shexp.weight");
      if (shared_gate_inp_it != engine.resident_layers.end() &&
          shared_gate_inp_it->second.host_ptr &&
          !shared_gate_inp_it->second.meta.shape.empty()) {
        const auto &w = shared_gate_inp_it->second;
        GgmlSharedWeightRef out;
        out.host_data = w.host_ptr;
        out.type = static_cast<uint32_t>(w.meta.type);
        out.ne0 = (int64_t)w.meta.shape[0];
        out.ne1 = w.meta.shape.size() >= 2
                      ? (int64_t)w.meta.shape[1]
                      : 1;
        out.nbytes = w.meta.size_bytes;
        if (!no_external_shared_weights && engine.gpu &&
            engine.gpu->get_core_device() == scheduler_device &&
            w.device_ptr) {
          out.device_data = w.device_ptr;
          out.device_bytes = w.meta.size_bytes;
        }
        shared_gate_inp = out;
      }
      const bool shared_has_gate = shared_gate_inp.has_value();
      const bool fused_shared_ready =
          shared_gate && shared_up && shared_down &&
          shared_gate->device_data && shared_up->device_data &&
          shared_down->device_data &&
          (!shared_has_gate || shared_gate_inp->device_data);
      if (fused_shared_ready) {
        ref.shared_gate = *shared_gate;
        ref.shared_up = *shared_up;
        ref.shared_down = *shared_down;
        ref.shared_has_gate = shared_has_gate;
        if (shared_has_gate)
          ref.shared_gate_inp = *shared_gate_inp;
        ref.shared_clamp =
            layer_idx < (int)c.swiglu_clamp_shexp.size()
                ? c.swiglu_clamp_shexp[(size_t)layer_idx]
                : 0.0f;
      }

      auto &scheduler = GgmlQwenDecodeGraphRunner::instance_for_signature(
          ref.gate_type, ref.up_type, ref.down_type, ref.expert_stride);
      scratch.tok_topk_ids.resize((size_t)topk);
      scratch.tok_topk_weights.resize((size_t)topk);
      const bool fused_shared_graph =
          ref.shared_gate.device_data && ref.shared_up.device_data &&
          ref.shared_down.device_data &&
          (!ref.shared_has_gate || ref.shared_gate_inp.device_data);
      const void *scheduler_device_output = nullptr;
      const bool scheduler_ready =
          scheduler.ensure(scheduler_device, D, n_experts, topk, moe_inter,
                           c.expert_weights_norm, c.expert_weights_scale);
      const bool scheduler_scratch_ready =
          scheduler_ready &&
          scratch.qwen_cpu_moe_out.resize((size_t)D, "qwen_cpu_moe_out");
      const bool scheduler_run_ok =
          scheduler_scratch_ready &&
          scheduler.run(
              engine.qwen4exp_device_hc_ffn_input, ref,
              fused_shared_graph ? nullptr : scratch.qwen_cpu_moe_out.data(),
              fused_shared_graph ? &scheduler_device_output : nullptr,
              reinterpret_cast<int32_t *>(scratch.tok_topk_ids.data()),
              scratch.tok_topk_weights.data());
      if (S == 1 && layer_idx == 0) {
        static std::atomic<bool> logged_scheduler_attempt{false};
        if (!logged_scheduler_attempt.exchange(true, std::memory_order_relaxed))
            LOG_INFO(
              "[GGML-QWEN-DECODE-SCHED] decode attempt: router=%s ram=%d "
              "runner=%d scratch=%d run=%d fused_shared=%d\n",
              !scheduler_router_available
                  ? "missing"
                  : (router_it->second.device_ptr ? "device" : "host"),
              scheduler_ram_available ? 1 : 0, scheduler_ready ? 1 : 0,
              scheduler_scratch_ready ? 1 : 0, scheduler_run_ok ? 1 : 0,
              fused_shared_graph ? 1 : 0);
      }
      if (scheduler_run_ok) {
        bool route_valid = true;
        for (int k = 0; k < topk; ++k) {
          const int id = scratch.tok_topk_ids[(size_t)k];
          if (id < 0 || id >= n_experts) {
            route_valid = false;
            break;
          }
        }
        if (route_valid) {
          for (int k = 0; k < topk; ++k)
            engine.bump_route_freq(layer_idx,
                                   scratch.tok_topk_ids[(size_t)k]);
          const void *combined = scheduler_device_output;
          const bool fallback_shared_ok =
              !fused_shared_graph &&
              run_qwen4exp_shared_device(
                  engine, layer_idx, S, D, moe_inter,
                  engine.qwen4exp_device_hc_ffn_input, nullptr, c, &combined,
                  scratch.qwen_cpu_moe_out.data()) &&
              combined;
          if ((fused_shared_graph || fallback_shared_ok) && combined) {
            scratch.moe_device_output = combined;
            scratch.moe_shared_device_output = true;
            scratch.moe_device_output_ready = true;
            static std::atomic<bool> logged_scheduler{false};
            if (!logged_scheduler.exchange(true, std::memory_order_relaxed))
              LOG_INFO(
                  "[GGML-QWEN-DECODE-SCHED] Bee-style CUDA-router + "
                  "DDR-CPU-expert topology active; lazy layers fall back to "
                  "the existing cache path\n");
            return true;
          }
        }
      }
    } else if (S == 1 && layer_idx == 0) {
      static std::atomic<bool> logged_scheduler_unavailable{false};
      if (!logged_scheduler_unavailable.exchange(true,
                                                 std::memory_order_relaxed))
        LOG_INFO(
            "[GGML-QWEN-DECODE-SCHED] decode candidate declined before "
            "runner: router_available=%d ram_arena=%d\n",
            scheduler_router_available ? 1 : 0,
            engine.graph_ram_layer_ready(layer_idx) ? 1 : 0);
    }
  }

  // --- Router: compute expert probs and select top-k. Reference algorithm
  // ported from llama.cpp's build_moe_ffn (llama-graph.cpp):
  //   1) probs = gate_func(logits)     [sigmoid/softmax/sqrtsoftplus, per
  //   c.gating_func -- see moe_router.cpp] 2) selection_probs =
  //   probs + bias  [bias only shifts WHICH experts get
  //      picked — the DeepSeek-V3/GLM-5.2 e_score_correction_bias tensor]
  //   3) top-k selected BY selection_probs (biased)
  //   4) weights gathered from probs (UNBIASED) at the selected indices
  //   5) optionally normalize the top-k weights to sum to 1
  //   6) optionally scale by expert_weights_scale
  {
    PERF_SCOPE(t_moe_route);
    std::string norm_key =
        "blk." + std::to_string(layer_idx) + ".ffn_norm.weight";
    auto norm_it = engine.resident_layers.find(norm_key);
    // Qwen4Exp's hyper-connection pre-mix already applies hc_ffn_norm and
    // produces the exact [S,D] input consumed by both the router and the
    // routed experts.  Some GGUFs still expose an ordinary ffn_norm tensor,
    // but it is not part of this architecture.  Applying it here a second
    // time changes the route input (and therefore every expert projection)
    // even though the HC boundary itself is correct.
    const bool qwen_hc_route_input =
        engine.family_ops.family == ModelFamily::Qwen4Exp &&
        engine.family_ops.uses_hyper_connections;
    if (!qwen_hc_route_input &&
        norm_it != engine.resident_layers.end() && norm_it->second.host_ptr) {
      // This reinterprets host_ptr as float* unconditionally -- only
      // correct if ffn_norm.weight is actually F32 (see CLAUDE.md's
      // "quantized weight correctness" gotcha). Norm weights are
      // almost always F32 in practice, but if THIS one somehow isn't,
      // every element downstream (router logits AND every expert's
      // FFN input, since both consume scratch.normed) goes to garbage
      // simultaneously -- which would exactly explain gate_buf/up_buf
      // and router weight going NaN together on the very first expert
      // touched, with no GPU kernel or router-math bug required.
      if (norm_it->second.meta.type != GGUFType::F32) {
        LOG_ERROR("[MOE-ROUTE] layer=%d ffn_norm.weight type=%u (NOT F32) -- "
                  "rmsnorm_row "
                  "reinterprets host_ptr as float* unconditionally; this WILL "
                  "produce garbage.\n",
                  layer_idx, (unsigned)norm_it->second.meta.type);
      }
      // Check the RAW WEIGHT VALUES themselves, not just rmsnorm's
      // output -- the load-time [LOAD-CHECK] pass only verified the
      // TYPE tag is F32, never the actual bytes. This distinguishes
      // "the source data for this specific layer is corrupted" (a
      // loading/offset/file-corruption bug) from "rmsnorm_row's math
      // has an edge case" (e.g. a legitimately all-zero hidden_states
      // row combined with a zero norm weight -> Inf*0 = NaN even with
      // a nonzero eps).
      check_finite("ffn_norm.weight (raw, pre-rmsnorm)", layer_idx, -1,
                   reinterpret_cast<float *>(norm_it->second.host_ptr),
                   (size_t)D);
      // ROOT CAUSE (found by tracing per-expert output norms against
      // DMA): rmsnorm_row normalizes exactly ONE row (D floats).
      // This used to be called ONCE, unconditionally, regardless of S
      // -- correct only for S==1 (decode). During prefill (S>1) every
      // token past s=0 read whatever was already sitting in
      // scratch.normed (leftover from a PREVIOUS layer's rmsnorm,
      // since this is a reused thread_local buffer, never zeroed
      // between layers) as its router+expert input. That's why every
      // MoE layer's routed-expert output was exactly 0.0 for tokens
      // s>=1 throughout the whole prefill pass (dead/stale input ->
      // degenerate router logits -> saturated sigmoid, same
      // weight=0.3125 for every top-k expert -- exactly what the
      // trace showed), while single-token decode (S==1) never hit it.
      const float *norm_w = reinterpret_cast<float *>(norm_it->second.host_ptr);
      for (int s = 0; s < S; ++s) {
        rmsnorm_row(scratch.normed.data() + (size_t)s * D,
                    hidden_states + (size_t)s * D, norm_w, D, c.eps);
      }
      check_finite(
          "scratch.normed (post ffn_norm rmsnorm, router+expert input)",
          layer_idx, -1, scratch.normed.data(), (size_t)S * D);
    } else {
      if (qwen_hc_route_input) {
        static std::atomic<bool> logged_qwen_hc_route_input{false};
        if (!logged_qwen_hc_route_input.exchange(true,
                                                  std::memory_order_relaxed)) {
          LOG_INFO("[MOE-ROUTE] Qwen4Exp HC route input uses the HC pre-mix "
                   "directly; ordinary ffn_norm bypassed\n");
        }
      }
      bool copied_device_hc_input = false;
      if (qwen_hc_route_input && engine.qwen4exp_device_hc_ffn_input &&
          qwen4exp_hc_device_live()) {
        copied_device_hc_input = GgmlHcRunner::instance().copy_pre_mix_to_host(
            engine.qwen4exp_device_hc_ffn_input, scratch.normed.data(), S);
        if (!copied_device_hc_input) {
          LOG_WARN("[MOE-ROUTE] layer=%d failed to download device HC route "
                   "input; using host fallback\n", layer_idx);
        }
      }
      if (!copied_device_hc_input)
        std::memcpy(scratch.normed.data(), hidden_states,
                    S * D * sizeof(float));
      if (qwen_hc_route_input && layer_idx == 0 && S == 1 &&
          std::getenv("MOESTREAM_PREFILL_BOUNDARY_DUMP")) {
        uint64_t hash = 1469598103934665603ULL;
        const auto *bytes = reinterpret_cast<const uint8_t *>(scratch.normed.data());
        for (size_t i = 0; i < (size_t)D * sizeof(float); ++i) {
          hash ^= bytes[i];
          hash *= 1099511628211ULL;
        }
        LOG_INFO("[MOE-ROUTE] layer=0 HC route input fnv64=0x%016llx source=%s\n",
                 static_cast<unsigned long long>(hash),
                 copied_device_hc_input ? "device" : "host");
      }
    }

    std::string router_key =
        "blk." + std::to_string(layer_idx) + ".ffn_gate_inp.weight";
    auto router_it = engine.resident_layers.find(router_key);
    if (router_it == engine.resident_layers.end() ||
        !router_it->second.host_ptr) {
      std::fill(output, output + S * D, 0.0f);
      return true;
    }
    // Qwen4Exp's HC path has no ordinary FFN norm: the device HC result is
    // exactly the router input.  Do not key this on `norm_it == end` here.
    // The resident map may contain a metadata/cache entry for ffn_norm even
    // though Qwen's architecture does not consume it; using that incidental
    // map detail left the activation on CUDA while routing from stale host
    // data.  Keep this handoff explicitly Qwen+HC-only; other families and
    // any future norm-bearing variant remain on the established host route.
    // The graph also implements only the Qwen softmax semantics.
    // CPU-affinity controls expert execution, not necessarily route
    // selection.  BeeLlama's CPU expert split receives IDs/weights produced
    // by the CUDA graph.  Keep that option available for Qwen too: when
    // qwen4exp_decode_gpu_router is enabled during S==1 decode, a
    // CPU-affinity layer still executes all routed expert projections on the
    // CPU, but obtains the compact route table from the existing CUDA router.
    // The default flag is false, so the established host-router behavior is
    // unchanged unless a Qwen profile explicitly opts into this A/B.
    const bool qwen_decode_gpu_router =
        (S > 1 && !is_cpu_affinity_layer(layer_idx)) ||
        (S == 1 && engine.qwen4exp_decode_gpu_router);
    if (moe_ggml_gpu_router() && qwen_decode_gpu_router &&
        c.gating_func == GatingFunc::Softmax &&
        engine.family_ops.family == ModelFamily::Qwen4Exp &&
        engine.family_ops.uses_hyper_connections &&
        engine.qwen4exp_device_hc_ffn_input) {
      GgmlRouterWeightRef rref;
      const auto &rw = router_it->second;
      rref.host_data = rw.host_ptr;
      rref.type = (uint32_t)rw.meta.type;
      rref.ne0 = rw.meta.shape.size() > 0 ? (int64_t)rw.meta.shape[0] : 0;
      rref.ne1 = rw.meta.shape.size() > 1 ? (int64_t)rw.meta.shape[1] : 0;
      rref.nbytes = rw.meta.size_bytes;
      const float *route_bias = nullptr;
      std::string bias_key =
          "blk." + std::to_string(layer_idx) + ".exp_probs_b.bias";
      auto route_bias_it = engine.resident_layers.find(bias_key);
      if (route_bias_it != engine.resident_layers.end() &&
          route_bias_it->second.host_ptr &&
          route_bias_it->second.meta.type == GGUFType::F32)
        route_bias =
            reinterpret_cast<const float *>(route_bias_it->second.host_ptr);

      auto &router = GgmlRouterRunner::instance();
      if (router.ensure(/*device=*/moe_ggml_device() >= 0 ? moe_ggml_device()
                                                          : 0,
                        D, n_experts, topk, S, c.expert_weights_norm,
                        c.expert_weights_scale) &&
          router.ensure_layer(layer_idx, rref, route_bias)) {
        const int route_device = moe_ggml_device() >= 0 ? moe_ggml_device() : 0;
        bool all_experts_device_resident =
            qwen4exp_device_route_enabled() && moe_prefill_batched() && S > 1 &&
            !engine.layer_is_unpacked(layer_idx) && ram_mgr.vram_cache;
        if (all_experts_device_resident) {
          for (int e = 0; e < n_experts; ++e) {
            const ExpertLocation *loc =
                ram_mgr.vram_cache->find_expert(layer_idx, e);
            if (!loc || !loc->device_ptr || !loc->upload_ready ||
                loc->gpu_id != route_device) {
              all_experts_device_resident = false;
              break;
            }
          }
          cudaSetDevice(route_device);
        }
        if (all_experts_device_resident) {
          gpu_route_device = router.run_tensor_device(
              layer_idx, engine.qwen4exp_device_hc_ffn_input,
              &gpu_route_ids_device, &gpu_route_weights_device);
          gpu_router_handled = gpu_route_device;
          if (gpu_route_device)
            device_router = &router;
          if (gpu_route_device) {
            static bool logged_device = false;
            if (!logged_device) {
              logged_device = true;
              LOG_INFO("[GGML-ROUTER] Qwen4Exp all-resident device route "
                       "active: IDs/weights stay on CUDA\n");
            }
          }
        }
        if (!gpu_route_device) {
          gpu_route_ids_ks.resize((size_t)topk * S);
          gpu_route_weights_ks.resize((size_t)topk * S);
          gpu_router_handled = router.run_tensor(
              layer_idx, engine.qwen4exp_device_hc_ffn_input,
              gpu_route_ids_ks.data(), gpu_route_weights_ks.data());
        }
        if (gpu_router_handled) {
          static bool logged = false;
          if (!logged) {
            LOG_INFO("[GGML-ROUTER] Qwen4Exp device routing active: "
                     "GPU computes softmax/top-k; compact route tables %s\n",
                     gpu_route_device ? "stay on CUDA for all-resident layers"
                                      : "return to host");
            logged = true;
          }
        }

        // Opt-in correctness oracle for the Qwen GPU router.  The device
        // router is only allowed to replace the established host selection
        // once its IDs and normalized weights agree on the same HC input.
        // This deliberately downloads one [D] row and repeats the host
        // projection/top-k; it is never part of the production path.
        static const bool compare_qwen_router = [] {
          const char *e = std::getenv("MOESTREAM_QWEN_ROUTER_COMPARE");
          return e && e[0] == '1';
        }();
        if (compare_qwen_router && gpu_router_handled && S == 1 &&
            engine.family_ops.family == ModelFamily::Qwen4Exp &&
            engine.qwen4exp_device_hc_ffn_input &&
            gpu_route_ids_ks.size() >= (size_t)topk && topk >= 4) {
          std::vector<float> compare_x((size_t)D);
          if (GgmlHcRunner::instance().copy_pre_mix_to_host(
                  engine.qwen4exp_device_hc_ffn_input, compare_x.data(), 1)) {
            std::vector<float> compare_logits((size_t)n_experts);
            dense_proj(
                engine,
                "blk." + std::to_string(layer_idx) + ".ffn_gate_inp.weight",
                compare_x.data(), compare_logits.data(), 1, D, n_experts);
            std::vector<float> gpu_logits((size_t)n_experts);
            const bool gpu_logits_ok =
                router.copy_logits_to_host(layer_idx, gpu_logits.data());
            float max_raw_logit_abs = 0.0f;
            if (gpu_logits_ok) {
              for (int e = 0; e < n_experts; ++e)
                max_raw_logit_abs = std::max(
                    max_raw_logit_abs, std::abs(gpu_logits[(size_t)e] -
                                                compare_logits[(size_t)e]));
            }
            apply_gating_activation(c.gating_func, compare_logits.data(),
                                    n_experts);
            const std::string bias_key =
                "blk." + std::to_string(layer_idx) + ".exp_probs_b.bias";
            const float *compare_bias = nullptr;
            auto compare_bias_it = engine.resident_layers.find(bias_key);
            if (compare_bias_it != engine.resident_layers.end() &&
                compare_bias_it->second.host_ptr &&
                compare_bias_it->second.meta.type == GGUFType::F32)
              compare_bias = reinterpret_cast<const float *>(
                  compare_bias_it->second.host_ptr);
            std::vector<std::pair<float, int>> compare_scores;
            compare_scores.reserve((size_t)n_experts);
            for (int e = 0; e < n_experts; ++e)
              compare_scores.emplace_back(
                  compare_logits[(size_t)e] +
                      (compare_bias ? compare_bias[e] : 0.0f),
                  e);
            std::partial_sort(
                compare_scores.begin(), compare_scores.begin() + topk,
                compare_scores.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });
            std::vector<float> compare_weights((size_t)topk);
            float compare_sum = 0.0f;
            for (int k = 0; k < topk; ++k) {
              const int e = compare_scores[(size_t)k].second;
              compare_weights[(size_t)k] = compare_logits[(size_t)e];
              compare_sum += compare_weights[(size_t)k];
            }
            if (c.expert_weights_norm) {
              const float inv_sum =
                  1.0f / std::max(compare_sum, 6.103515625e-5f);
              for (float &w : compare_weights)
                w *= inv_sum;
            }
            if (c.expert_weights_scale != 0.0f)
              for (float &w : compare_weights)
                w *= c.expert_weights_scale;
            int id_mismatch = 0;
            float max_weight_abs = 0.0f;
            for (int k = 0; k < topk; ++k) {
              if (gpu_route_ids_ks[(size_t)k] !=
                  compare_scores[(size_t)k].second)
                ++id_mismatch;
              max_weight_abs = std::max(
                  max_weight_abs, std::abs(gpu_route_weights_ks[(size_t)k] -
                                           compare_weights[(size_t)k]));
            }
            LOG_WARN("[GGML-ROUTER-COMPARE] layer=%d id_mismatch=%d/%d "
                     "max_weight_abs=%g raw_logit_abs=%g gpu_ids=%d,%d,%d,%d "
                     "cpu_ids=%d,%d,%d,%d\n",
                     layer_idx, id_mismatch, topk, max_weight_abs,
                     max_raw_logit_abs, gpu_route_ids_ks[0],
                     gpu_route_ids_ks[1], gpu_route_ids_ks[2],
                     gpu_route_ids_ks[3], compare_scores[0].second,
                     compare_scores[1].second, compare_scores[2].second,
                     compare_scores[3].second);
          } else {
            LOG_WARN("[GGML-ROUTER-COMPARE] layer=%d could not download HC "
                     "input\n",
                     layer_idx);
          }
        }
      }
    }

    // GLM5NEXT native CUDA router. Unlike the optional GGML Qwen route above,
    // this keeps the route table host-visible so the existing RAM/VRAM cache,
    // prefetch, and pilot bookkeeping remain unchanged. The router weight is
    // already in the core GPU's dense pool; only the compact IDs/weights cross
    // back. Hash-routed layers must keep the CPU selection branch because
    // their expert IDs come from ffn_gate_tid2eid rather than learned top-k.
    if (!gpu_router_handled && !tid2eid && engine.glm5next_router_gpu &&
        engine.family_ops.family == ModelFamily::GLM5NEXT && engine.gpu &&
        router_it->second.device_ptr) {
      gpu_route_ids_ks.resize((size_t)topk * S);
      gpu_route_weights_ks.resize((size_t)topk * S);
      const float *route_bias = nullptr;
      std::string bias_key =
          "blk." + std::to_string(layer_idx) + ".exp_probs_b.bias";
      auto route_bias_it = engine.resident_layers.find(bias_key);
      if (route_bias_it != engine.resident_layers.end() &&
          route_bias_it->second.host_ptr &&
          route_bias_it->second.meta.type == GGUFType::F32)
        route_bias =
            reinterpret_cast<const float *>(route_bias_it->second.host_ptr);

      gpu_router_handled = engine.gpu->glm5next_router_gpu_host(
          router_it->second, scratch.normed.data(), route_bias, S, D, n_experts,
          topk, c.gating_func, c.expert_weights_norm, c.expert_weights_scale,
          gpu_route_ids_ks.data(), gpu_route_weights_ks.data());
      if (gpu_router_handled) {
        static bool logged_native_router = false;
        if (!logged_native_router) {
          logged_native_router = true;
          LOG_INFO("[GLM5NEXT-ROUTER] native CUDA routing active: "
                   "matmul/gating/top-k on core GPU, compact route table "
                   "returned to host cache bookkeeping\n");
        }
      } else {
        perf().n_glm5next_router_gpu_fallback_cpu.fetch_add(
            1, std::memory_order_relaxed);
      }
    }

    if (!gpu_router_handled) {
      bool cpu_router_done = false;
      const bool qwen_device_router_gemv =
          qwen4exp_decode_device_router_gemv_enabled() && S == 1 &&
          engine.family_ops.family == ModelFamily::Qwen4Exp &&
          engine.family_ops.uses_hyper_connections &&
          is_cpu_affinity_layer(layer_idx) && engine.gpu &&
          engine.qwen4exp_device_hc_ffn_input &&
          (router_it->second.device_ptr ||
           router_it->second.gpu_fast_ptr);
      if (qwen_device_router_gemv) {
        const void *device_normed =
            GgmlHcRunner::instance().device_data(
                engine.qwen4exp_device_hc_ffn_input);
        if (device_normed) {
          const auto &router_weight = router_it->second;
          const void *router_device_weight =
              router_weight.gpu_fast_ptr
                  ? static_cast<const void *>(router_weight.gpu_fast_ptr)
                  : static_cast<const void *>(router_weight.device_ptr);
          const GGUFType router_type =
              router_weight.gpu_fast_ptr ? GGUFType::BF16
                                         : router_weight.meta.type;
          cpu_router_done = engine.gpu->dense_matmul_device_host(
              router_type, router_device_weight,
              static_cast<const float *>(device_normed),
              scratch.router_logits.data(), S, D, n_experts);
          static std::atomic<bool> logged_device_router_gemv{false};
          if (cpu_router_done &&
              !logged_device_router_gemv.exchange(true,
                                                  std::memory_order_relaxed)) {
            LOG_INFO("[MOE-ROUTE] Qwen decode router uses device HC input "
                     "(CUDA GEMV; host top-k/CPU expert path retained)\n");
          }
        }
      }
      const bool qwen_cpu_router =
          S == 1 && qwen4exp_decode_cpu_router_enabled() &&
          engine.family_ops.family == ModelFamily::Qwen4Exp &&
          engine.family_ops.uses_hyper_connections &&
          is_cpu_affinity_layer(layer_idx) && router_it->second.host_ptr;
      if (qwen_cpu_router) {
        cpu_router_done = cpu_quant_matmul(
            router_it->second.meta.type, router_it->second.host_ptr,
            scratch.normed.data(), scratch.router_logits.data(), S, D,
            n_experts);
        static std::atomic<bool> logged_cpu_router{false};
        if (cpu_router_done &&
            !logged_cpu_router.exchange(true, std::memory_order_relaxed)) {
          LOG_INFO("[MOE-ROUTE] Qwen CPU-affinity decode router A/B active "
                   "(local CPU matmul; production GPU GEMV unchanged)\n");
        }
      }
      if (!cpu_router_done) {
        dense_proj(
            engine,
            "blk." + std::to_string(layer_idx) + ".ffn_gate_inp.weight",
            scratch.normed.data(), scratch.router_logits.data(), S, D,
            n_experts);
      }

      // Gating activation: turns raw logits into probs in-place. These are
      // the UNBIASED values later gathered as expert weights.
      for (int s = 0; s < S; ++s) {
        float *row = scratch.router_logits.data() + (size_t)s * n_experts;
        apply_gating_activation(c.gating_func, row, n_experts);
      }
      // No prior diagnostic covered this: expert_ffn's GPU-resident branch
      // only checks each expert's gate/up/down buffers AFTER routing has
      // already picked them and weighted their contribution -- if the
      // router's own dense_proj or the gating math above produces NaN,
      // nothing upstream would ever catch it.
      check_finite("router_logits (post gate_inp matmul + gating)", layer_idx,
                   -1, scratch.router_logits.data(), (size_t)S * n_experts);
    }
  }

  // Real, previously-untimed cost -- see t_moe_topk_select's own comment
  // (perf_timer.hpp) for why this got its own scope: this block (bias add
  // + argsort/partial_sort top-k selection + pilot recall bookkeeping) was
  // NOT inside t_moe_route's PERF_SCOPE above despite that counter's name
  // implying it covered "top-k selection" too. Explicit brace block (not a
  // bare statement) so this scope ends where Phase 1 actually ends, NOT
  // at the end of the whole function -- PERF_SCOPE is RAII, timed until
  // its ENCLOSING scope closes.
  if (!gpu_route_device) {
    {
      PERF_SCOPE(t_moe_topk_select);

      // Selection bias (DeepSeek-V3/GLM-5.2 style expert_selection_bias):
      // present only on some architectures, absent -> selection uses raw probs.
      // GGUF name confirmed against llama-arch.cpp: "blk.N.exp_probs_b.bias"
      // (previously looked up under a name —
      // "ffn_gate_inp.e_score_correction_bias" — that doesn't exist in real
      // GGUFs; see the matching fix in graph_initializer.cpp's preload manifest
      // for the full explanation).
      const float *bias_row = nullptr;
      {
        std::string bias_key =
            "blk." + std::to_string(layer_idx) + ".exp_probs_b.bias";
        auto bias_it = engine.resident_layers.find(bias_key);
        if (bias_it != engine.resident_layers.end() &&
            bias_it->second.host_ptr) {
          bias_row = reinterpret_cast<float *>(bias_it->second.host_ptr);
          check_finite("exp_probs_b.bias (raw)", layer_idx, -1, bias_row,
                       (size_t)n_experts);
        }
      }

      // Zero output accumulator
      std::memset(output, 0, S * D * sizeof(float));

      // --- Phase 1: batch-wide top-k selection (router already knows every
      // token's choice up front, since router_logits above is one batched
      // matmul over all S tokens) — compute once per token, cache the result,
      // and tally how often each expert is picked across the whole batch. ---
      std::fill(scratch.expert_freq.begin(), scratch.expert_freq.end(), 0);
      std::fill(scratch.first_needed_by_token.begin(),
                scratch.first_needed_by_token.end(), INT_MAX);
      for (int s = 0; s < S; ++s) {
        float *token_probs = scratch.router_logits.data() + s * n_experts;
        float sum_w = 0.0f;

        // Records one (expert_id, weight) pick into scratch's per-token
        // arrays and the routing-frequency histograms -- shared by both
        // the hash-routed and normal argsort selection branches below, so
        // the two paths can never diverge in what bookkeeping happens once
        // an expert is actually picked (weight is always the UNBIASED
        // token_probs[id], same convention either way).
        auto record_pick = [&](int k, int id, float w) {
          scratch.tok_topk_weights[s * topk + k] = w;
          scratch.tok_topk_ids[s * topk + k] = id;
          scratch.expert_freq[id]++;
          // EngineConfig::prefill_hint_order_by_earliest_token -- see
          // LayerScratch::first_needed_by_token's own comment. s only
          // ever increases across this loop, so the FIRST time an expert
          // is picked here IS its minimum -- no need to compare/take a
          // min, a plain unconditional check-and-set is correct and
          // cheaper.
          if (scratch.first_needed_by_token[id] == INT_MAX)
            scratch.first_needed_by_token[id] = s;
          // Phase A accumulate the cross-conversation
          // routing-frequency histogram (engine.route_freq), dumped to a
          // usage file at shutdown and read back at the NEXT process
          // startup to preload the measured hot set into VRAM. This is
          // the true demand signal (what the router actually picked),
          // not a cache-hit proxy
          engine.bump_route_freq(layer_idx, id);
          // Additionally track prefill-only selections in their own
          // histogram (route_freq_prefill) -- see that field's own
          // comment for why decode's higher routing volume would
          // otherwise crowd out prefill's signal in the combined one.
          // Never a replacement for the line above, purely additive.
          if (S > 1)
            engine.bump_route_freq_prefill(layer_idx, id);
          sum_w += w;
        };

        // Hash-routed SELECTION: direct token-id -> expert-id lookup,
        // bypassing argsort entirely for this token. Every guard here is
        // defensive (this is a brand-new raw-int32 read path with no prior
        // real-hardware mileage): token_ids may legitimately be null (a
        // caller that doesn't have per-token ids), an out-of-range token
        // id would read past the tensor's real extent, and a corrupt/
        // garbage row would hand a downstream expert-array access an
        // out-of-bounds id -- any of these falls through to the normal
        // argsort path below for just this token, never crashes.
        //
        // Bound against tid2eid_n_vocab (the table's OWN on-disk ne[1]),
        // not c.vocab: they're only guaranteed to agree if this file (or,
        // under has_dense_model, whichever separate file the table
        // actually loaded from) exported the table over the full engine
        // vocab. Using c.vocab here previously let a real token id past
        // the table's real, smaller extent read straight off the end of
        // the posix_memalign'd buffer -- silent heap OOB, not a bounds
        // check -- and come back with plausible-looking but meaningless
        // int32 garbage that just happened to look like an "out-of-range
        // expert id" downstream. tid2eid_n_vocab is 0 when the table
        // failed to load at all, so this also degrades cleanly.
        bool used_hash_route = false;
        if (gpu_router_handled) {
          // The GGML tensor is shaped [topk, tokens], but the 2-D export above
          // writes compact token rows to the host buffer.  MoEStream's routing
          // arrays are [tokens, topk], so consume that row-major export here.
          // The graph has already applied the unbiased probability gather,
          // top-k normalization, and scale.
          for (int k = 0; k < topk; ++k) {
            const size_t i = (size_t)s * topk + k;
            const int id = gpu_route_ids_ks[i];
            if (id < 0 || id >= n_experts) {
              LOG_ERROR(
                  "[GGML-ROUTER] layer=%d returned invalid expert id=%d\n",
                  layer_idx, id);
              return true;
            }
            record_pick(k, id, gpu_route_weights_ks[i]);
          }
          used_hash_route = true;
        } else if (tid2eid && token_ids) {
          int tok = token_ids[s];
          if (tok >= 0 && (int64_t)tok < tid2eid_n_vocab) {
            const int32_t *row = tid2eid + (size_t)tok * topk;
            bool ids_valid = true;
            for (int k = 0; k < topk; ++k) {
              if (row[k] < 0 || row[k] >= n_experts) {
                ids_valid = false;
                break;
              }
            }
            if (ids_valid) {
              for (int k = 0; k < topk; ++k)
                record_pick(k, row[k], token_probs[row[k]]);
              used_hash_route = true;
            } else {
              LOG_ERROR(
                  "[MOE-ROUTE] layer=%d token_id=%d: ffn_gate_tid2eid row "
                  "contains "
                  "an out-of-range expert id (n_experts=%d) -- falling "
                  "back to the "
                  "learned router's top-k for this token only.\n",
                  layer_idx, tok, n_experts);
            }
          }
        }
        if (!used_hash_route) {
          for (int e = 0; e < n_experts; ++e) {
            float sel = token_probs[e] + (bias_row ? bias_row[e] : 0.0f);
            scratch.scores[e] = {e, sel};
          }
          std::partial_sort(scratch.scores.begin(),
                            scratch.scores.begin() + topk,
                            scratch.scores.begin() + n_experts,
                            [](const ExpertScore &a, const ExpertScore &b) {
                              return a.score > b.score;
                            });
          for (int k = 0; k < topk; ++k)
            record_pick(k, scratch.scores[k].id,
                        token_probs[scratch.scores[k].id]);
        }
        if (!gpu_router_handled && c.expert_weights_norm) {
          float inv_sum = 1.0f / std::max(sum_w, 6.103515625e-5f);
          for (int k = 0; k < topk; ++k)
            scratch.tok_topk_weights[s * topk + k] *= inv_sum;
        }
        if (!gpu_router_handled && c.expert_weights_scale != 0.0f) {
          for (int k = 0; k < topk; ++k)
            scratch.tok_topk_weights[s * topk + k] *= c.expert_weights_scale;
        }

        // Router-selection trace: same VERBOSE gate as layers_forward_gguf's
        // [TRACE] hidden-norm dump, but for CONTENT (which experts, what
        // weight) rather than magnitude -- a routing bug (wrong experts
        // picked, e.g. from a bias-tensor or gating-function mixup) would
        // still produce numerically "healthy" norms downstream, since the
        // FFN math itself is correct, just applied to the wrong experts.
        // Only the LAST token (s==S-1) -- that's the one whose output
        // actually feeds the next-token prediction.
        if (s == S - 1 && trace_gate("TRACE-ROUTER")) {
          LogStream ls(LogLevel::VERBOSE);
          ls << "[TRACE l=" << layer_idx << "] router picks:";
          for (int k = 0; k < topk; ++k) {
            ls << " e" << scratch.tok_topk_ids[s * topk + k]
               << "(w=" << scratch.tok_topk_weights[s * topk + k] << ")";
          }
          ls << "\n";
        }

        // UNION coverage check (2026-08-25, measurement only) -- real
        // answer to "if DEPTH-1, DEPTH-2, and ATTN-PILOT are ALL enabled
        // and predicting this layer, is there still a real disk-cold miss
        // when we get here?" Reads all three pending predictions' CURRENT
        // state BEFORE any of their own individual checks below reset their
        // "am I pending" sentinel (pilot_pred_layer / pilot_pred2_slots[
        // layer_idx%2].layer / pilot_attn_pred_layer) -- this block must
        // run FIRST, or a later individual check's own reset would make an
        // earlier-pending prediction look absent here. Doesn't consume/
        // reset anything itself -- read-only, the three checks below are
        // still what actually resets each one.
        if (s == S - 1) {
          static thread_local std::array<int, 24> union_ids;
          int union_n = 0;
          auto add_unique = [&](int id) {
            for (int i = 0; i < union_n; ++i)
              if (union_ids[i] == id)
                return;
            if (union_n < (int)union_ids.size())
              union_ids[union_n++] = id;
          };
          bool any_pending = false;
          if (scratch.pilot_pred_layer == layer_idx) {
            any_pending = true;
            for (int pk = 0; pk < scratch.pilot_pred_count; ++pk)
              add_unique(scratch.pilot_pred_ids[pk]);
          }
          if (scratch.pilot_attn_pred_layer == layer_idx) {
            any_pending = true;
            for (int pk = 0; pk < scratch.pilot_attn_pred_count; ++pk)
              add_unique(scratch.pilot_attn_pred_ids[pk]);
          }
          const auto &union_slot2 = scratch.pilot_pred2_slots[layer_idx % 2];
          if (union_slot2.layer == layer_idx) {
            any_pending = true;
            for (int pk = 0; pk < union_slot2.count; ++pk)
              add_unique(union_slot2.ids[pk]);
          }
          if (any_pending) {
            int overlap = 0;
            for (int i = 0; i < union_n; ++i) {
              for (int k = 0; k < topk; ++k) {
                if (scratch.tok_topk_ids[s * topk + k] == union_ids[i]) {
                  ++overlap;
                  break;
                }
              }
            }
            perf().n_pilot_union_recall_hits.fetch_add(
                (uint64_t)overlap, std::memory_order_relaxed);
            perf().n_pilot_union_recall_checks.fetch_add(
                1, std::memory_order_relaxed);
            perf().n_pilot_union_recall_topk_sum.fetch_add(
                (uint64_t)topk, std::memory_order_relaxed);
            perf().n_pilot_union_pred_size_sum.fetch_add(
                (uint64_t)union_n, std::memory_order_relaxed);
          }
        }

        // EngineConfig::pilot_widen_all_tokens recall/COVERAGE check
        // (MEASUREMENT ONLY) -- two different comparisons against the
        // same widened prediction: (1) accuracy vs the last token's own
        // real picks, apples-to-apples with DEPTH-1's own recall number;
        // (2) coverage vs the WHOLE layer's real unique-expert demand
        // (scratch.expert_freq>0, valid here since every token's record_
        // pick has already run for this layer by s==S-1) -- the number
        // that actually answers "how much of what this layer needs did
        // widening actually address."
        if (s == S - 1 && scratch.pilot_wide_pred_layer == layer_idx) {
          int overlap_last = 0;
          for (int i = 0; i < scratch.pilot_wide_pred_count; ++i) {
            int pred_id = scratch.pilot_wide_pred_ids[i];
            for (int k = 0; k < topk; ++k) {
              if (scratch.tok_topk_ids[s * topk + k] == pred_id) {
                ++overlap_last;
                break;
              }
            }
          }
          int real_unique = 0;
          for (int e = 0; e < n_experts; ++e)
            if (scratch.expert_freq[e] > 0)
              ++real_unique;
          int coverage_hits = 0;
          for (int i = 0; i < scratch.pilot_wide_pred_count; ++i) {
            if (scratch.expert_freq[scratch.pilot_wide_pred_ids[i]] > 0)
              ++coverage_hits;
          }
          perf().n_pilot_wide_recall_hits_last.fetch_add(
              (uint64_t)overlap_last, std::memory_order_relaxed);
          perf().n_pilot_wide_recall_checks.fetch_add(
              1, std::memory_order_relaxed);
          perf().n_pilot_wide_recall_topk_sum.fetch_add(
              (uint64_t)topk, std::memory_order_relaxed);
          perf().n_pilot_wide_pred_size_sum.fetch_add(
              (uint64_t)scratch.pilot_wide_pred_count,
              std::memory_order_relaxed);
          perf().n_pilot_wide_coverage_hits.fetch_add(
              (uint64_t)coverage_hits, std::memory_order_relaxed);
          perf().n_pilot_wide_real_unique_sum.fetch_add(
              (uint64_t)real_unique, std::memory_order_relaxed);
          scratch.pilot_wide_pred_layer =
              -1; // consume -- never let a stale value linger
        }

        // Pilot RECALL check (always on, independent of log level -- a
        // perf counter, not a log line): if the pilot predicted THIS exact
        // layer's routing one iteration ago (LayerScratch::pilot_pred_ids),
        // compare its predicted set against what the router really just
        // picked for the same "last token" position, then consume/reset
        // the pending prediction either way so a stale value never lingers
        // into a later layer's unrelated comparison.
        if (s == S - 1 && scratch.pilot_pred_layer == layer_idx) {
          int overlap = 0;
          int hint_overlap = 0, hint_denom = 0;
          for (int pk = 0; pk < scratch.pilot_pred_count; ++pk) {
            int pred_id = scratch.pilot_pred_ids[pk];
            bool matched = false;
            for (int k = 0; k < topk; ++k) {
              if (scratch.tok_topk_ids[s * topk + k] == pred_id) {
                matched = true;
                break;
              }
            }
            if (matched)
              ++overlap;
            // Per-rank breakdown -- exact mirror of the DEPTH-2 version
            // further below (see n_pilot_recall_hits_by_rank's own
            // comment, perf_timer.hpp): pilot_pred_ids[k] is already
            // ranked by the pilot's own confidence (repeated argmax,
            // k=0 = highest), so this answers "does l+1's accuracy
            // concentrate in the high-confidence ranks" -- the number
            // that actually decides how many top ranks are safe to
            // promote RAM->VRAM speculatively.
            if (pk < 8) {
              perf().n_pilot_recall_checks_by_rank[pk].fetch_add(
                  1, std::memory_order_relaxed);
              if (matched)
                perf().n_pilot_recall_hits_by_rank[pk].fetch_add(
                    1, std::memory_order_relaxed);
              // TEMPORARY DIAGNOSTIC (2026-09-13): the genuine 2D
              // [layer][rank] cross table -- see n_pilot_recall_hits_by_
              // layer_rank's own comment (PerfCounters).
              if (layer_idx >= 0 &&
                  layer_idx < PerfCounters::kMaxTrackedLayersForMiss) {
                perf().n_pilot_recall_checks_by_layer_rank[layer_idx][pk]
                    .fetch_add(1, std::memory_order_relaxed);
                if (matched)
                  perf().n_pilot_recall_hits_by_layer_rank[layer_idx][pk]
                      .fetch_add(1, std::memory_order_relaxed);
              }
            }
            // Restricted tally: of the picks that were actually
            // fetch-worthy (not already cached, so a real hint got
            // pushed for them), how many turned out correct? This is
            // the decision-relevant number -- see
            // n_pilot_recall_hint_hits's own comment in perf_timer.hpp.
            if (scratch.pilot_pred_was_hint[pk]) {
              ++hint_denom;
              if (matched)
                ++hint_overlap;
              if (pk < 8) {
                perf().n_pilot_recall_hint_checks_by_rank[pk].fetch_add(
                    1, std::memory_order_relaxed);
                if (matched)
                  perf().n_pilot_recall_hint_hits_by_rank[pk].fetch_add(
                      1, std::memory_order_relaxed);
              }
            }
            // Third, disjoint tally: of the picks that were RAM-resident
            // but NOT yet VRAM-resident at prediction time (a completely
            // different population from the fetch-worthy one just above
            // -- see pilot_pred_was_ram_only's own comment), how many
            // matched the real router's later pick? This is the number
            // that actually calibrates a RAM->VRAM speculative-promotion
            // cutoff -- neither the raw per-rank recall above (mixes in
            // already-VRAM and needs-fetch picks too) nor the
            // fetch-worthy number (a disjoint population entirely) can
            // answer it.
            if (pk < 8 && scratch.pilot_pred_was_ram_only[pk]) {
              perf().n_pilot_ram_only_checks_by_rank[pk].fetch_add(
                  1, std::memory_order_relaxed);
              if (matched)
                perf().n_pilot_ram_only_hits_by_rank[pk].fetch_add(
                    1, std::memory_order_relaxed);
            }
          }
          perf().n_pilot_recall_hits.fetch_add((uint64_t)overlap,
                                               std::memory_order_relaxed);
          perf().n_pilot_recall_checks.fetch_add(1, std::memory_order_relaxed);
          perf().n_pilot_recall_topk_sum.fetch_add((uint64_t)topk,
                                                   std::memory_order_relaxed);
          perf().n_pilot_recall_hint_hits.fetch_add((uint64_t)hint_overlap,
                                                    std::memory_order_relaxed);
          perf().n_pilot_recall_hint_checks_denom.fetch_add(
              (uint64_t)hint_denom, std::memory_order_relaxed);
          // Per-layer breakdown (2026-08-25) -- see PerfCounters::
          // n_pilot_recall_hits_by_layer's own comment. Indexed by
          // layer_idx, the TARGET layer this prediction was FOR (not the
          // layer it fired from).
          if (layer_idx >= 0 &&
              layer_idx < PerfCounters::kMaxTrackedLayersForMiss) {
            perf().n_pilot_recall_hits_by_layer[layer_idx].fetch_add(
                (uint32_t)overlap, std::memory_order_relaxed);
            perf().n_pilot_recall_topk_sum_by_layer[layer_idx].fetch_add(
                (uint32_t)topk, std::memory_order_relaxed);
            perf().n_pilot_recall_checks_by_layer[layer_idx].fetch_add(
                1, std::memory_order_relaxed);
          }
          scratch.pilot_pred_layer = -1;
        }

        // ATTN-PILOT recall check (2026-08-25, measurement only) -- exact
        // mirror of the DEPTH-1 check above (same single-scalar survival
        // pattern), comparing scratch.pilot_attn_pred_ids against this same
        // real router pick. See the firing site's own comment (moe_forward_
        // gguf, right after the attention-residual check_finite call) for
        // the full "answers two questions with one signal" design and how
        // to read this against DEPTH-1's and DEPTH-2's own numbers.
        if (s == S - 1 && scratch.pilot_attn_pred_layer == layer_idx) {
          int overlap = 0;
          int hint_overlap = 0, hint_denom = 0;
          for (int pk = 0; pk < scratch.pilot_attn_pred_count; ++pk) {
            int pred_id = scratch.pilot_attn_pred_ids[pk];
            bool matched = false;
            for (int k = 0; k < topk; ++k) {
              if (scratch.tok_topk_ids[s * topk + k] == pred_id) {
                matched = true;
                break;
              }
            }
            if (matched)
              ++overlap;
            if (pk < 8) {
              perf().n_pilot_attn_recall_checks_by_rank[pk].fetch_add(
                  1, std::memory_order_relaxed);
              if (matched)
                perf().n_pilot_attn_recall_hits_by_rank[pk].fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (scratch.pilot_attn_pred_was_hint[pk]) {
              ++hint_denom;
              if (matched)
                ++hint_overlap;
              if (pk < 8) {
                perf().n_pilot_attn_recall_hint_checks_by_rank[pk].fetch_add(
                    1, std::memory_order_relaxed);
                if (matched)
                  perf().n_pilot_attn_recall_hint_hits_by_rank[pk].fetch_add(
                      1, std::memory_order_relaxed);
              }
            }
          }
          perf().n_pilot_attn_recall_hits.fetch_add((uint64_t)overlap,
                                                    std::memory_order_relaxed);
          perf().n_pilot_attn_recall_checks.fetch_add(
              1, std::memory_order_relaxed);
          perf().n_pilot_attn_recall_topk_sum.fetch_add(
              (uint64_t)topk, std::memory_order_relaxed);
          perf().n_pilot_attn_recall_hint_hits.fetch_add(
              (uint64_t)hint_overlap, std::memory_order_relaxed);
          perf().n_pilot_attn_recall_hint_checks_denom.fetch_add(
              (uint64_t)hint_denom, std::memory_order_relaxed);
          // Per-layer breakdown -- see DEPTH-1's own identical block above
          // and PerfCounters::n_pilot_attn_recall_hits_by_layer's comment.
          if (layer_idx >= 0 &&
              layer_idx < PerfCounters::kMaxTrackedLayersForMiss) {
            perf().n_pilot_attn_recall_hits_by_layer[layer_idx].fetch_add(
                (uint32_t)overlap, std::memory_order_relaxed);
            perf().n_pilot_attn_recall_topk_sum_by_layer[layer_idx].fetch_add(
                (uint32_t)topk, std::memory_order_relaxed);
            perf().n_pilot_attn_recall_checks_by_layer[layer_idx].fetch_add(
                1, std::memory_order_relaxed);
          }
          scratch.pilot_attn_pred_layer = -1;
        }

        // DEPTH-2 recall check (approach (a), measurement only) -- exact
        // mirror of the depth-1 check above, comparing the slot for THIS
        // parity (LayerScratch::pilot_pred2_slots[layer_idx % 2] -- see
        // that array's own comment for why it's two slots, not one)
        // against this same real router pick. Also buckets each pick's
        // outcome by ITS OWN RANK (pk, 0..7) into the *_by_rank counters
        // -- see n_pilot2_recall_hits_by_rank's own comment in
        // perf_timer.hpp for why: pilot_pred_ids/pilot_pred2_ids[k] is
        // already ranked by the pilot's own confidence (repeated argmax,
        // k=0 = highest), so this answers "does accuracy concentrate in
        // the high-confidence ranks" separately from the aggregate
        // overlap counters above, which only sum across the whole top-K
        // and can't distinguish "flat accuracy across all 8 ranks" from
        // "rank 0-2 are great, rank 5-7 are noise."
        {
          auto &slot2 = scratch.pilot_pred2_slots[layer_idx % 2];
          if (s == S - 1 && slot2.layer == layer_idx) {
            int overlap = 0;
            int hint_overlap = 0, hint_denom = 0;
            for (int pk = 0; pk < slot2.count; ++pk) {
              int pred_id = slot2.ids[pk];
              bool matched = false;
              for (int k = 0; k < topk; ++k) {
                if (scratch.tok_topk_ids[s * topk + k] == pred_id) {
                  matched = true;
                  break;
                }
              }
              if (matched)
                ++overlap;
              if (pk < 8) {
                perf().n_pilot2_recall_checks_by_rank[pk].fetch_add(
                    1, std::memory_order_relaxed);
                if (matched)
                  perf().n_pilot2_recall_hits_by_rank[pk].fetch_add(
                      1, std::memory_order_relaxed);
              }
              if (slot2.was_hint[pk]) {
                ++hint_denom;
                if (matched)
                  ++hint_overlap;
                if (pk < 8) {
                  perf().n_pilot2_recall_hint_checks_by_rank[pk].fetch_add(
                      1, std::memory_order_relaxed);
                  if (matched)
                    perf().n_pilot2_recall_hint_hits_by_rank[pk].fetch_add(
                        1, std::memory_order_relaxed);
                }
              }
            }
            perf().n_pilot2_recall_hits.fetch_add((uint64_t)overlap,
                                                  std::memory_order_relaxed);
            perf().n_pilot2_recall_checks.fetch_add(1,
                                                    std::memory_order_relaxed);
            perf().n_pilot2_recall_topk_sum.fetch_add(
                (uint64_t)topk, std::memory_order_relaxed);
            perf().n_pilot2_recall_hint_hits.fetch_add(
                (uint64_t)hint_overlap, std::memory_order_relaxed);
            perf().n_pilot2_recall_hint_checks_denom.fetch_add(
                (uint64_t)hint_denom, std::memory_order_relaxed);
            // Per-layer breakdown -- see DEPTH-1's own identical block
            // above and PerfCounters::n_pilot2_recall_hits_by_layer's
            // comment. layer_idx here IS the target layer (l+2) this
            // prediction was for, same convention as DEPTH-1/ATTN-PILOT
            // (this check only fires when the loop reaches slot2.layer).
            if (layer_idx >= 0 &&
                layer_idx < PerfCounters::kMaxTrackedLayersForMiss) {
              perf().n_pilot2_recall_hits_by_layer[layer_idx].fetch_add(
                  (uint32_t)overlap, std::memory_order_relaxed);
              perf().n_pilot2_recall_topk_sum_by_layer[layer_idx].fetch_add(
                  (uint32_t)topk, std::memory_order_relaxed);
              perf().n_pilot2_recall_checks_by_layer[layer_idx].fetch_add(
                  1, std::memory_order_relaxed);
            }
            slot2.layer = -1;
          }
        }
      }
    } // end t_moe_topk_select scope (Phase 1)
  }

  if (!gpu_route_device)
    qwen4exp_note_decode_route_residency(engine, ram_mgr, layer_idx, S, topk,
                                         scratch);

  // Qwen4Exp pilot recall is checked at the shared host route-table point,
  // after the real IDs have been populated and before any expert dispatch.
  // Decode and the normal Qwen GPU-router fallback both use this compact host
  // route. The all-resident device-only prefill route has no host IDs here and
  // therefore contributes no false measurement; its production behavior is
  // unchanged.
  if (!gpu_route_device && S == 1 &&
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      layer_idx >= 0 && layer_idx < PerfCounters::kMaxTrackedLayersForMiss) {
    const int64_t prediction_ready_ns =
        perf().qwen_pilot_prediction_ready_ns[layer_idx].exchange(
            0, std::memory_order_acq_rel);
    if (prediction_ready_ns > 0) {
      const int64_t now_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count();
      if (now_ns >= prediction_ready_ns) {
        PerfCounters::atomic_add(
            perf().t_qwen_pilot_ready_to_router,
            (double)(now_ns - prediction_ready_ns) * 1e-9);
        perf().n_qwen_pilot_ready_to_router.fetch_add(
            1, std::memory_order_relaxed);
      }
      // Keep both origins so the later get_experts_batch boundary can report
      // the complete prediction-ready -> DDR-fetch-start gap.
      perf().qwen_pilot_prediction_for_fetch_ns[layer_idx].store(
          prediction_ready_ns, std::memory_order_release);
      perf().qwen_pilot_router_ready_ns[layer_idx].store(
          now_ns, std::memory_order_release);
    }
  }
  if (!gpu_route_device && scratch.qwen_pilot_pred_layer == layer_idx &&
      S > 0) {
    const int s = S - 1;
    int overlap = 0;
    int prefetch_overlap = 0;
    const int prefetch_count = std::min(scratch.qwen_pilot_prefetch_count,
                                        scratch.qwen_pilot_pred_count);
    const bool track_qwen_layer =
        layer_idx >= 0 && layer_idx < PerfCounters::kMaxTrackedLayersForMiss;
    for (int pk = 0; pk < scratch.qwen_pilot_pred_count; ++pk) {
      const int predicted = scratch.qwen_pilot_pred_ids[pk];
      bool matched = false;
      for (int k = 0; k < topk; ++k) {
        if (scratch.tok_topk_ids[(size_t)s * topk + k] == predicted) {
          matched = true;
          break;
        }
      }
      if (matched)
        ++overlap;
      if (pk < prefetch_count) {
        perf().n_qwen_pilot_prefetch_rank_checks[pk].fetch_add(
            1, std::memory_order_relaxed);
        if (matched) {
          ++prefetch_overlap;
          perf().n_qwen_pilot_prefetch_rank_hits[pk].fetch_add(
              1, std::memory_order_relaxed);
        }
      }
      if (pk < PerfCounters::kQwenPilotMaxRanks) {
        perf().n_qwen_pilot_rank_checks[pk].fetch_add(
            1, std::memory_order_relaxed);
        if (matched)
          perf().n_qwen_pilot_rank_hits[pk].fetch_add(
              1, std::memory_order_relaxed);
        if (track_qwen_layer) {
          perf().n_qwen_pilot_rank_checks_by_layer[layer_idx][pk].fetch_add(
              1, std::memory_order_relaxed);
          if (matched)
            perf().n_qwen_pilot_rank_hits_by_layer[layer_idx][pk].fetch_add(
                1, std::memory_order_relaxed);
        }
      }
    }
    const uint64_t predicted_slots = (uint64_t)scratch.qwen_pilot_pred_count;
    perf().n_qwen_pilot_recall_hits.fetch_add((uint64_t)overlap,
                                              std::memory_order_relaxed);
    perf().n_qwen_pilot_recall_checks.fetch_add(1, std::memory_order_relaxed);
    perf().n_qwen_pilot_recall_pred_size_sum.fetch_add(
        predicted_slots, std::memory_order_relaxed);
    if (track_qwen_layer) {
      perf().n_qwen_pilot_recall_hits_by_layer[layer_idx].fetch_add(
          (uint32_t)overlap, std::memory_order_relaxed);
      perf().n_qwen_pilot_recall_pred_size_sum_by_layer[layer_idx].fetch_add(
          (uint32_t)predicted_slots, std::memory_order_relaxed);
      perf().n_qwen_pilot_recall_checks_by_layer[layer_idx].fetch_add(
          1, std::memory_order_relaxed);
      if (prefetch_count > 0) {
        perf().n_qwen_pilot_prefetch_recall_hits_by_layer[layer_idx].fetch_add(
            (uint32_t)prefetch_overlap, std::memory_order_relaxed);
        perf()
            .n_qwen_pilot_prefetch_recall_pred_size_sum_by_layer[layer_idx]
            .fetch_add((uint32_t)prefetch_count, std::memory_order_relaxed);
        perf()
            .n_qwen_pilot_prefetch_recall_checks_by_layer[layer_idx]
            .fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (prefetch_count > 0) {
      perf().n_qwen_pilot_prefetch_recall_hits.fetch_add(
          (uint64_t)prefetch_overlap, std::memory_order_relaxed);
      perf().n_qwen_pilot_prefetch_recall_checks.fetch_add(
          1, std::memory_order_relaxed);
      perf().n_qwen_pilot_prefetch_recall_pred_size_sum.fetch_add(
          (uint64_t)prefetch_count, std::memory_order_relaxed);
    }
    // Predictive Partial Pinning empirical filter (2026-09-19): does
    // (static pin set) UNION (pilot's staged predictions) cover this
    // token's REAL routed experts for the configured partial layer? Only
    // meaningful here, not in build_moe_partial_gpu's own dispatch --
    // this is a pure measurement, no actual staging/fetch is issued.
    if (layer_idx == qwen4exp_ppp_target_layer()) {
      const std::unordered_set<int> &pin_set = qwen4exp_ppp_pin_set(engine);
      if (!pin_set.empty()) {
        const int stage_k_scenarios[PerfCounters::kPPPStagingScenarios] = {
            0, prefetch_count, 5};
        for (int sc = 0; sc < PerfCounters::kPPPStagingScenarios; ++sc) {
          const int stage_k =
              std::min(stage_k_scenarios[sc], scratch.qwen_pilot_pred_count);
          int miss = 0;
          for (int k = 0; k < topk; ++k) {
            const int routed = scratch.tok_topk_ids[(size_t)s * topk + k];
            if (routed < 0 || pin_set.count(routed))
              continue;
            bool staged = false;
            for (int pk = 0; pk < stage_k; ++pk) {
              if (scratch.qwen_pilot_pred_ids[pk] == routed) {
                staged = true;
                break;
              }
            }
            if (!staged)
              ++miss;
          }
          if (miss == 0)
            perf().n_ppp_m0[sc].fetch_add(1, std::memory_order_relaxed);
          else if (miss == 1)
            perf().n_ppp_m1[sc].fetch_add(1, std::memory_order_relaxed);
          else
            perf().n_ppp_m2plus[sc].fetch_add(1, std::memory_order_relaxed);
            perf().n_ppp_m2plus_sum[sc].fetch_add(
                (uint64_t)miss, std::memory_order_relaxed);
        }
        perf().n_ppp_tokens_checked.fetch_add(1, std::memory_order_relaxed);
      }
    }
    scratch.qwen_pilot_pred_layer = -1;
    scratch.qwen_pilot_pred_count = 0;
  }

  return false;
}
