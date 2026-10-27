#include "active_layer_upload_workers.hpp"
#include "graph_config.hpp"
#include "async_prefetcher.hpp"
#include "cpu_expert_executor.hpp"
#include "gpu_expert_executor.hpp"
#include "dense_dispatch.hpp"
#include "engine.hpp"
#include "expert_activation.hpp"
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

#include "batched_dispatch_pipeline.hpp"
#include "moe_pipeline_internal.hpp"
#include "shared_expert_executor.hpp"

static int ggml_fixed_chunk_pair_capacity() {
  static const int value = [] {
    const char *env = zeromoe::graphcfg::env("ZEROMOE_GGML_FIXED_PAIR_CAPACITY");
    if (!env || !env[0])
      return 0;
    const int requested = std::atoi(env);
    if (requested <= 0)
      return 0;
    return ((requested + 255) / 256) * 256;
  }();
  return value;
}

// The reference GGML graph keeps the expert axis fixed at the model's full
// expert count.  The compact MoE adapter normally rounds the routed set to a
// 64-expert high-water mark, which still causes a graph/allocator rebuild
// whenever a layer exposes a new high-water mark.  Opt-in fixed slots let us
// measure the same stable graph topology without changing the default memory
// budget.  Unused slots are never addressed by ids and are initialized by the
// existing quantized-tail clear path.
static int ggml_fixed_expert_slots() {
  static const int value = [] {
    const char *env = zeromoe::graphcfg::env("ZEROMOE_GGML_FIXED_EXPERT_SLOTS");
    if (!env || !env[0])
      return 0;
    const int requested = std::atoi(env);
    if (requested <= 0)
      return 0;
    return ((requested + 63) / 64) * 64;
  }();
  return value;
}

// Qwen's `ram_preload_all_experts` is a source-ownership contract for layers
// assigned to the host-RAM tier, not for layers explicitly assigned to the
// full-graph VRAM block.  Those VRAM-only layers are excluded from the DDR
// preload manifest, so a cold GPU-cache miss there must retain the normal
// GGUF mmap staging source.  Applying the DDR-only rule globally made the
// first miss in a VRAM-only layer reject the grouped path and fall back to
// per-expert dispatch, even though that layer was never supposed to have a
// DDR copy.  Keep the strict no-mmap rule for host-owned layers and preserve
// existing behavior for other model families.
static bool qwen_ddr_only_expert_source(const StreamingEngine &engine,
                                        int layer_idx) {
  return engine.family_ops.family == ModelFamily::Qwen4Exp &&
         engine.qwen_ram_preload_all_experts &&
         !engine.qwen_ram_vram_only_layer(layer_idx);
}

// Qwen-only transfer A/B.  Packed VRAM-cache copies can be landed as a
// compact batch and unpacked with one scatter kernel at the D2D/H2D boundary.
// Keep the switch out of the generic runner so GLM/DeepSeek retain their exact
// existing per-expert transfer behavior.
static bool qwen_batch_d2d_unpack_enabled() {
  static const bool enabled = [] {
    const char *env = zeromoe::graphcfg::env("ZEROMOE_QWEN_BATCH_D2D_UNPACK");
    return env && env[0] == '1';
  }();
  return enabled;
}

// The Qwen decode fast path normally uses the grouped GGML runner, so it does
// not call RAMSafetyManager::get_experts_batch().  Record the first actual
// DDR-backed fetch boundary from both paths through this one helper.  The
// timestamp is consumed once per predicted target layer; active-layer/D2D
// reuse deliberately does not count because it never starts a DDR fetch.
// ============================================================
// ggml MUL_MAT_ID path for one prefill layer's routed experts (opt-in via
// moe_ggml_mul_mat_id). Returns true only if the ENTIRE layer's routed
// contribution has been computed and added into `output`; on false the caller
// runs its own per-expert path unchanged, so every rejection below is a
// fallback, never a dropped expert.
//
// Structurally this replaces Phase 3b's per-expert-bucket loop with three
// GGML_OP_MUL_MAT_ID nodes in one backend graph. The token-to-expert grouping
// Phase 3b does on the host (CSR build + OMP activation gather) is done inside
// ggml-cuda on the GPU instead, from `ids` alone -- which is why this function
// does not need offs[]/slot[]/act_arena at all, only the raw routing table.
// ============================================================
// Chunked variant of the device MoE path. The ordinary runner submits one
// graph after all compact expert weights have landed. This version keeps two
// compact weight ranges in flight: while GGML computes chunk N, the copy
// stream stages chunk N+1 into the other range. Each graph gathers only the
// token rows routed to that chunk and a CUDA scatter-add reduces its weighted
// output into one [S,D] device accumulator.
static bool run_ggml_moe_layer_chunked(
    StreamingEngine &engine, RAMSafetyManager &ram_mgr, GpuTopology *topo,
    int layer_idx, int S, int D, int moe_inter, int topk, int n_experts,
    const std::vector<int> &used, const int *tok_topk_ids,
    const float *tok_topk_weights, int chunk_experts, bool unpacked_layer,
    int chosen_device, size_t gb0, size_t ub0, size_t db0, const GGUFTensor *g0,
    const GGUFTensor *u0, const GGUFTensor *d0, const void *device_input,
    const void **device_output) {
  if (!device_input || !device_output || unpacked_layer || S <= 1 ||
      topk <= 0 || used.empty() || chunk_experts <= 0 || !topo)
    return false;

  // This graph requires one uniform 3-D weight geometry, just like the
  // non-chunked MUL_MAT_ID path. Reject mixed layers before allocating any
  // state so the caller can use its established fallback safely.
  for (int e : used) {
    size_t gb, ub, db;
    const GGUFTensor *gm, *um, *dm;
    if (!engine.expert_geometry_for(layer_idx, e, gb, ub, db, gm, um, dm) ||
        gb != gb0 || ub != ub0 || db != db0 || gm->type != g0->type ||
        um->type != u0->type || dm->type != d0->type)
      return false;
  }

  struct ChunkPairs {
    std::vector<int32_t> token;
    std::vector<int32_t> local;
    std::vector<float> weight;
  };
  const int chunk = std::min(chunk_experts, (int)used.size());
  const int n_chunks = ((int)used.size() + chunk - 1) / chunk;
  std::vector<ChunkPairs> pairs((size_t)n_chunks);
  std::vector<int> local((size_t)n_experts, -1);
  for (int i = 0; i < (int)used.size(); ++i)
    local[(size_t)used[(size_t)i]] = i;
  int max_pairs = 0;
  for (int s = 0; s < S; ++s) {
    for (int k = 0; k < topk; ++k) {
      const size_t flat = (size_t)s * topk + k;
      const int e = tok_topk_ids[flat];
      if (e < 0 || e >= n_experts || local[(size_t)e] < 0)
        return false;
      const int compact = local[(size_t)e];
      const int ci = compact / chunk;
      const int slot = compact % chunk;
      pairs[(size_t)ci].token.push_back((int32_t)s);
      pairs[(size_t)ci].local.push_back((int32_t)slot);
      pairs[(size_t)ci].weight.push_back(tok_topk_weights[flat]);
    }
  }
  for (const auto &p : pairs)
    max_pairs = std::max(max_pairs, (int)p.token.size());
  if (max_pairs <= 0)
    return false;
  // Keep the graph shape stable across nearby layers. Padding is cheap: the
  // padded route weights are zero, so the device scatter contributes nothing.
  int pair_capacity = ((max_pairs + 255) / 256) * 256;
  const int fixed_pair_capacity = ggml_fixed_chunk_pair_capacity();
  if (fixed_pair_capacity > 0) {
    if (max_pairs > fixed_pair_capacity) {
      LOG_WARN("[GGML-MOE] fixed pair capacity %d is below layer=%d "
               "maximum routed pairs=%d; keeping observed capacity\n",
               fixed_pair_capacity, layer_idx, max_pairs);
    } else {
      pair_capacity = fixed_pair_capacity;
    }
  }

  GgmlMoeRunner &runner = GgmlMoeRunner::instance_for(
      (uint32_t)g0->type, (uint32_t)u0->type, (uint32_t)d0->type);
  runner.set_batch_d2d_unpack(
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      qwen_batch_d2d_unpack_enabled());
  GgmlMoeLayerPlan plan;
  plan.device_id = chosen_device;
  plan.n_experts = 2 * chunk; // two alternating compact slot ranges
  plan.n_expert_slots = 2 * chunk;
  plan.n_tokens = pair_capacity; // padded routed-pair count
  plan.source_tokens = S;
  plan.topk = 1;
  plan.d_model = D;
  plan.d_ff = moe_inter;
  plan.gate_type = (uint32_t)g0->type;
  plan.up_type = (uint32_t)u0->type;
  plan.down_type = (uint32_t)d0->type;
  plan.chunk_experts = chunk;
  plan.chunked = true;
  if (!runner.begin_layer(plan))
    return false;
  // If this signature was used by an earlier layer, its copy stream may still
  // be reading the old VRAM-cache sources. Drain and release those holds only
  // when this runner is reused, allowing the other signature runner to make
  // progress between the two uses.
  runner.release_completed_vram_holds();
  if (runner.gate_expert_bytes() != gb0 || runner.up_expert_bytes() != ub0 ||
      runner.down_expert_bytes() != db0)
    return false;

  struct D2DHold {
    int expert_id;
    const uint8_t *device_ptr;
  };
  struct RamH2DHold {
    MoEExpertBlock *block;
  };
  std::vector<D2DHold> d2d_holds;
  std::vector<RamH2DHold> ram_h2d_holds;
  d2d_holds.reserve(used.size());
  ram_h2d_holds.reserve(used.size());
  int n_reused = 0;
  size_t mmap_staged_total = 0;

  auto abort_pipeline = [&]() {
    // A failed chunk may have queued work on both streams. Drain it before
    // releasing held slabs and reset the singleton so the caller's fallback
    // cannot reuse partially published graph state.
    cudaSetDevice(chosen_device);
    cudaDeviceSynchronize();
    for (const auto &h : d2d_holds)
      topo->release_hold(layer_idx, h.expert_id, h.device_ptr, false);
    for (const auto &h : ram_h2d_holds)
      ram_mgr.ram_unpin(*h.block);
    runner.release_completed_vram_holds();
    runner.shutdown();
    return false;
  };

  const size_t ram_cap =
      layer_idx >= 0 && layer_idx < (int)engine.layers_experts.size()
          ? std::min(engine.layers_experts[(size_t)layer_idx].size(),
                     (size_t)std::max(1, ram_mgr.cap_for_layer(layer_idx)))
          : 0;
  const bool qwen_ddr_only_source =
      qwen_ddr_only_expert_source(engine, layer_idx);

  for (int ci = 0; ci < n_chunks; ++ci) {
    const int first = ci * chunk;
    const int count = std::min(chunk, (int)used.size() - first);
    const int buf = ci % GgmlMoeRunner::NBUF;
    const int slot_base = buf * chunk;
    runner.wait_stage_free(buf);

    std::vector<unsigned char> staged((size_t)count, 1);
    if (!runner.begin_copy_batch(buf))
      return abort_pipeline();

    for (int i = 0; i < count; ++i) {
      const int e = used[(size_t)(first + i)];
      const ExpertLocation *rl = topo->find_expert(layer_idx, e);
      if (rl && rl->device_ptr && rl->upload_ready &&
          rl->gpu_id == chosen_device &&
          topo->hold_for_compute(layer_idx, e, rl->device_ptr, false)) {
        if (runner.copy_expert_d2d(buf, slot_base + i, rl->device_ptr)) {
          staged[(size_t)i] = 0;
          d2d_holds.push_back({e, rl->device_ptr});
          ++n_reused;
        } else {
          topo->release_hold(layer_idx, e, rl->device_ptr, false);
        }
      }
    }
    if (!runner.mark_copy_d2d_done(buf))
      return abort_pipeline();
    cudaSetDevice(chosen_device);

    std::vector<const uint8_t *> ram_sources((size_t)count, nullptr);
    std::vector<MoEExpertBlock *> ram_stage_holds;
    if (ram_cap > 0) {
      auto &ram_slots = engine.layers_experts[(size_t)layer_idx];
      for (int i = 0; i < count; ++i) {
        if (!staged[(size_t)i])
          continue;
        const uint8_t *slab = nullptr;
        size_t capacity = 0;
        MoEExpertBlock *rb = ram_mgr.find_ram_slot_if_ready_held(
            ram_slots, ram_cap, layer_idx, used[(size_t)(first + i)], &slab,
            &capacity);
        if (rb && slab && capacity >= gb0 + ub0 + db0) {
          ram_sources[(size_t)i] = slab;
          ram_stage_holds.push_back(rb);
        } else if (rb) {
          ram_mgr.ram_unpin(*rb);
        }
      }
    }

    std::vector<GgmlMoeStageSlot> slots((size_t)count);
    for (int i = 0; i < count; ++i)
      slots[(size_t)i] = runner.stage_slot(buf, i);
    std::atomic<bool> stage_ok{true};
    size_t staged_bytes = 0;
#pragma omp parallel for schedule(static) reduction(+ : staged_bytes)
    for (int i = 0; i < count; ++i) {
      if (!staged[(size_t)i])
        continue;
      const int e = used[(size_t)(first + i)];
      const GgmlMoeStageSlot &sl = slots[(size_t)i];
      const uint8_t *base = ram_sources[(size_t)i];
      const uint8_t *gs = nullptr;
      const uint8_t *us = nullptr;
      const uint8_t *ds = nullptr;
      if (base) {
        gs = base;
        us = base + gb0;
        ds = us + ub0;
      } else if (qwen_ddr_only_source) {
        // Do not let an incomplete RAM preload turn this optional pipeline
        // into a hidden file-backed expert read.  The caller will reject the
        // chunked path and use the normal DDR-aware fallback.
        stage_ok.store(false, std::memory_order_relaxed);
        continue;
      } else if (engine.model_file.mmap_active()) {
        const uint64_t goff = g0->offset + (uint64_t)e * gb0;
        const uint64_t uoff = u0->offset + (uint64_t)e * ub0;
        const uint64_t doff = d0->offset + (uint64_t)e * db0;
        gs = engine.model_file.mmap_bytes(*g0, goff, gb0);
        us = engine.model_file.mmap_bytes(*u0, uoff, ub0);
        ds = engine.model_file.mmap_bytes(*d0, doff, db0);
      }
      if (!sl.gate || !gs || !us || !ds) {
        stage_ok.store(false, std::memory_order_relaxed);
        continue;
      }
      std::memcpy(sl.gate, gs, gb0);
      std::memcpy(sl.up, us, ub0);
      std::memcpy(sl.down, ds, db0);
      staged_bytes += gb0 + ub0 + db0;
    }
    for (MoEExpertBlock *rb : ram_stage_holds)
      ram_mgr.ram_unpin(*rb);
    mmap_staged_total += staged_bytes;

    std::vector<unsigned char> mask = staged;
    if (!stage_ok.load(std::memory_order_relaxed) ||
        !runner.push_chunk(buf, slot_base, count, (const bool *)mask.data()))
      return abort_pipeline();

    std::vector<int32_t> token((size_t)pair_capacity, 0);
    std::vector<int32_t> ids((size_t)pair_capacity, slot_base);
    std::vector<float> weights((size_t)pair_capacity, 0.0f);
    const auto &p = pairs[(size_t)ci];
    for (size_t i = 0; i < p.token.size(); ++i) {
      token[i] = p.token[i];
      ids[i] = slot_base + p.local[i];
      weights[i] = p.weight[i];
    }
    const void *ignored = nullptr;
    if (!runner.run_chunk_device_input_tensor(device_input, token.data(),
                                              ids.data(), weights.data(), buf,
                                              &ignored))
      return abort_pipeline();
  }

  if (!runner.finish_chunked_output(device_output))
    return abort_pipeline();
  // Keep D2D source slots protected until this runner's copy stream reaches
  // its completion event. Reuse of this signature, or the request boundary,
  // performs the wait and releases the holds. The GGML graph and HC consumer
  // remain queued asynchronously on their own stream.
  for (const auto &h : d2d_holds)
    runner.defer_vram_hold((void *)topo, layer_idx, h.expert_id, h.device_ptr);
  for (const auto &h : ram_h2d_holds)
    ram_mgr.ram_unpin(*h.block);
  perf().n_prefill_stream_vram_hit.fetch_add((uint64_t)n_reused,
                                             std::memory_order_relaxed);
  perf().n_prefill_stream_vram_miss.fetch_add(
      (uint64_t)((int)used.size() - n_reused), std::memory_order_relaxed);
  perf().bytes_disk_read.fetch_add(mmap_staged_total,
                                   std::memory_order_relaxed);
  return true;
}

static bool run_ggml_moe_layer(
    StreamingEngine &engine, RAMSafetyManager &ram_mgr, GpuTopology *topo,
    int layer_idx, int S, int D, int moe_inter, int topk, int n_experts,
    const std::vector<int> &used, const int *tok_topk_ids,
    const float *tok_topk_weights, const float *normed, float *output,
    int chunk_experts, bool unpacked_layer,
    const void **device_output = nullptr, const void *device_input = nullptr,
    const void *device_route_ids = nullptr,
    const void *device_route_weights = nullptr,
    bool allow_active_layer_sources = false,
    GgmlMoeRunner **output_runner = nullptr) {
  if (device_output)
    *device_output = nullptr;
  if (output_runner)
    *output_runner = nullptr;
  if (!GgmlMoeRunner::compiled_in())
    return false;
  const int n_used = (int)used.size();
  if (n_used <= 0 || S < 1 || topk <= 0 || D <= 0 || moe_inter <= 0)
    return false;
  if (!topo)
    return false;
  static const bool trace_set =
      Logger::instance().would_log(LogLevel::DEBUG);

  // A single 3-D weight tensor per projection can only describe experts that
  // share a quant type AND a byte size. Packed layers satisfy this by
  // construction (every expert is a slice of one tensor); an unpacked layer
  // has a tensor per expert and can genuinely be mixed -- GLM-5.3's UD-IQ2_M
  // layers are. Checking rather than assuming is the difference between a
  // clean fallback and reading one expert's bytes as another's.
  size_t gb0 = 0, ub0 = 0, db0 = 0;
  const GGUFTensor *g0 = nullptr, *u0 = nullptr, *d0 = nullptr;
  if (!engine.expert_geometry_for(layer_idx, used[0], gb0, ub0, db0, g0, u0,
                                  d0))
    return false;
  if (unpacked_layer) {
    for (int i = 1; i < n_used; ++i) {
      size_t gb, ub, db;
      const GGUFTensor *gM, *uM, *dM;
      if (!engine.expert_geometry_for(layer_idx, used[(size_t)i], gb, ub, db,
                                      gM, uM, dM))
        return false;
      if (gb != gb0 || ub != ub0 || db != db0 || gM->type != g0->type ||
          uM->type != u0->type || dM->type != d0->type)
        return false;
    }
  }

  // Pick the vault GPU with the most FREE VRAM, not the biggest expert pool:
  // this path allocates its own ~1 GB weight buffer and never touches the
  // expert cache, so free space is the only thing that matters. Decided once
  // per process -- cudaMemGetInfo is not free, and the answer cannot change
  // meaningfully between layers.
  static int chosen_device = [&]() -> int {
    // ZEROMOE_GGML_MOE_DEVICE pins the graph to a specific GPU. The
    // automatic choice below maximises FREE VRAM, which on a box whose vault
    // GPU is deliberately filled with expert cache picks the OTHER card --
    // and on this hardware that is the PCIe x8 5060 Ti (6.8 GB/s H2D) rather
    // than the x16 3090 (12.3 GB/s), i.e. it optimises the wrong thing. There
    // is no way to weigh "more free VRAM" against "twice the link bandwidth"
    // without knowing how much of each the layer will need, so this is a
    // manual override rather than a heuristic.
    if (const char *envd = zeromoe::graphcfg::env("ZEROMOE_GGML_MOE_DEVICE")) {
      const int d = std::atoi(envd);
      LOG_INFO("[GGML-MOE] device forced to GPU %d by "
               "ZEROMOE_GGML_MOE_DEVICE\n",
               d);
      return d;
    }
    if (moe_ggml_device() >= 0) {
      LOG_INFO("[GGML-MOE] device set to GPU %d by config moe_ggml_device\n",
               moe_ggml_device());
      return moe_ggml_device();
    }
    int best = -1;
    size_t best_free = 0;
    for (int vi : topo->vault_gpus) {
      if ((size_t)vi >= topo->devices.size())
        continue;
      const int dev = topo->devices[(size_t)vi].id;
      size_t f = 0, t = 0;
      if (cudaSetDevice(dev) != cudaSuccess)
        continue;
      if (cudaMemGetInfo(&f, &t) != cudaSuccess)
        continue;
      if (f > best_free) {
        best_free = f;
        best = dev;
      }
    }
    LOG_INFO("[GGML-MOE] selected GPU %d (%.2f GB free)\n", best,
             (double)best_free / (1024.0 * 1024.0 * 1024.0));
    return best;
  }();
  if (chosen_device < 0)
    return false;

  static const bool chunk_pipeline = [] {
    const char *env = zeromoe::graphcfg::env("ZEROMOE_GGML_CHUNK_PIPELINE");
    return env && env[0] == '1';
  }();
  if (chunk_pipeline && !device_route_ids && !device_route_weights &&
      device_output && device_input && !unpacked_layer) {
    if (run_ggml_moe_layer_chunked(
            engine, ram_mgr, topo, layer_idx, S, D, moe_inter, topk, n_experts,
            used, tok_topk_ids, tok_topk_weights, chunk_experts, unpacked_layer,
            chosen_device, gb0, ub0, db0, g0, u0, d0, device_input,
            device_output)) {
      static bool logged = false;
      if (!logged) {
        LOG_INFO("[GGML-MOE] chunked device prefill pipeline active: "
                 "GPU gather + alternating weight chunks + device scatter\n");
        logged = true;
      }
      return true;
    }
    LOG_WARN("[GGML-MOE] chunked pipeline rejected layer=%d; using the "
             "single-graph path\n",
             layer_idx);
  }

  GgmlMoeRunner &runner = GgmlMoeRunner::instance_for(
      (uint32_t)g0->type, (uint32_t)u0->type, (uint32_t)d0->type);
  runner.set_batch_d2d_unpack(
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      qwen_batch_d2d_unpack_enabled());
  GgmlMoeLayerPlan plan;
  plan.device_id = chosen_device;
  plan.n_experts = n_used;
  // A GPU route table contains global expert IDs.  Keep it only for the
  // direct resident-layer view below.  If that view is unavailable, the
  // compact tensor uses a dense [0,n_used) axis and therefore needs the
  // ordinary host route table remapped to that compact axis.
  bool use_device_route_for_graph =
      device_route_ids && device_route_weights && device_input;
  // The runner normally rounds this request to a grow-only compact-slot
  // high-water mark (64-expert increments).  A fixed slot count is an opt-in
  // Full-graph shape experiment: it keeps the graph/allocator stable
  // across all layers of a quantization signature.  ids still address only
  // [0,n_used), so the unused tail is dead storage rather than extra work.
  const int fixed_expert_slots = ggml_fixed_expert_slots();
  plan.n_expert_slots =
      fixed_expert_slots > 0 ? std::max(fixed_expert_slots, n_used) : n_used;
  plan.n_tokens = S;
  plan.topk = topk;
  plan.d_model = D;
  plan.d_ff = moe_inter;
  plan.gate_type = (uint32_t)g0->type;
  plan.up_type = (uint32_t)u0->type;
  plan.down_type = (uint32_t)d0->type;
  plan.chunk_experts = std::min(chunk_experts > 0 ? chunk_experts : 32, n_used);

  // llama.cpp's resident-layer representation is a single expert tensor.  A
  // compact cache can provide the same view only when an explicitly selected
  // full-layer region was allocated and every expert is currently resident.
  // Host-routed Qwen decode is allowed to use this path too; other model
  // families retain the existing compact-repack behavior.
  const bool allow_qwen_host_external =
      engine.family_ops.family == ModelFamily::Qwen4Exp && S == 1 &&
      device_output && tok_topk_ids && tok_topk_weights;
  const bool compact_full_prefix_layer =
      topo->expert_compact_layout && layer_idx >= 0 &&
      layer_idx < topo->expert_full_layer_prefix;
  // GGML-CUDA's quantized MUL_MAT_ID path requires the full expert-axis
  // allocation to be contiguous. A compact cache slot may include alignment
  // padding, which is valid for ordinary repack/copy use but cannot be
  // exposed as a zero-repack tensor. Leave those layers on the safe compact
  // path until the backend has a stride-aware kernel.
  const bool compact_full_prefix_raw_contiguous =
      !compact_full_prefix_layer ||
      (layer_idx >= 0 &&
       (size_t)layer_idx < topo->expert_layer_slab_bytes.size() &&
       topo->expert_layer_slab_bytes[(size_t)layer_idx] == gb0 + ub0 + db0);
  const bool external_layout_possible =
      (topo->expert_slab_size > 0 || compact_full_prefix_layer) &&
      compact_full_prefix_raw_contiguous;
  if ((use_device_route_for_graph || allow_qwen_host_external) &&
      !unpacked_layer && external_layout_possible) {
    std::vector<std::pair<int, const uint8_t *>> external_holds;
    external_holds.reserve((size_t)n_experts);
    // Compact regions use the physical per-slot stride, which may include
    // GGUF/cache alignment padding. The routed payload itself is still only
    // gb0+ub0+db0 bytes; using that raw sum here would point expert 1 into the
    // padding/tail of expert 0 and correctly fail the contiguity check.
    const size_t external_stride =
        topo->expert_compact_layout
            ? ((layer_idx >= 0 &&
                (size_t)layer_idx < topo->expert_layer_slab_bytes.size())
                   ? topo->expert_layer_slab_bytes[(size_t)layer_idx]
                   : gb0 + ub0 + db0)
            : topo->expert_slab_size;
    const uint8_t *external_base = nullptr;
    const bool contiguous = topo->find_contiguous_resident_layer(
        layer_idx, n_experts, chosen_device, external_stride, &external_base,
        &external_holds);
    if (contiguous && external_base) {
      plan.external_weights = true;
      // The external tensor preserves the model's global expert axis.  The
      // compact `used` vector may contain only the experts routed by this
      // batch; that is fine because the device route IDs still index the full
      // resident layer.
      plan.n_experts = n_experts;
      plan.n_expert_slots = n_experts;
      plan.external_expert_stride = external_stride;
      plan.external_gate = external_base;
      plan.external_up = external_base + gb0;
      plan.external_down = external_base + gb0 + ub0;
      plan.chunk_experts = n_used;
      if (runner.begin_layer(plan)) {
        bool external_ok = false;
        if (use_device_route_for_graph) {
          external_ok = runner.run_layer_device_input_tensor_device_route(
              device_input, device_route_ids, device_route_weights,
              device_output);
        } else if (device_input) {
          external_ok = runner.run_layer_device_input_tensor(
              device_input, tok_topk_ids, tok_topk_weights, device_output);
        } else if (device_output) {
          external_ok = runner.run_layer_device(
              normed, tok_topk_ids, tok_topk_weights, device_output);
        } else {
          external_ok =
              runner.run_layer(normed, tok_topk_ids, tok_topk_weights, output);
        }
        if (external_ok) {
          if (output_runner)
            *output_runner = &runner;
          for (const auto &hold : external_holds)
            runner.defer_vram_hold((void *)topo, layer_idx, hold.first,
                                   hold.second);
          perf().n_prefill_stream_vram_hit.fetch_add((uint64_t)n_used,
                                                     std::memory_order_relaxed);
          static bool logged_external_layer = false;
          if (!logged_external_layer) {
            LOG_INFO("[GGML-MOE] layer=%d external resident weights: "
                     "%d slabs, stride=%zu, no repack/H2D (%s route)\n",
                     layer_idx, n_experts, external_stride,
                     use_device_route_for_graph ? "device" : "host");
            logged_external_layer = true;
          }
          return true;
        }
        // A failed graph submission is an exceptional path. Drain the CUDA
        // work before returning the cache holds so a later fallback cannot
        // race a partially submitted graph.
        cudaSetDevice(chosen_device);
        cudaDeviceSynchronize();
      }
      for (const auto &hold : external_holds)
        topo->release_hold(layer_idx, hold.first, hold.second, false);
      plan.external_weights = false;
      plan.external_expert_stride = 0;
      plan.external_gate = plan.external_up = plan.external_down = nullptr;
      use_device_route_for_graph = false;
    } else {
      for (const auto &hold : external_holds)
        topo->release_hold(layer_idx, hold.first, hold.second, false);
      static bool logged_scatter = false;
      if (!logged_scatter) {
        LOG_INFO("[GGML-MOE] resident layer view unavailable: cache slabs "
                 "are not complete/consecutive by expert ID; using compact "
                 "repack (layer=%d stride=%zu)\n",
                 layer_idx, external_stride);
        logged_scatter = true;
      }
      use_device_route_for_graph = false;
    }
  }

  if (!runner.begin_layer(plan))
    return false;

  // ggml's own per-expert byte counts, derived from the shapes rather than
  // from the GGUF metadata, must agree with what the file actually holds --
  // otherwise the memcpy below would either truncate or overrun.
  if (runner.gate_expert_bytes() != gb0 || runner.up_expert_bytes() != ub0 ||
      runner.down_expert_bytes() != db0) {
    LOG_ERROR("[GGML-MOE] layer=%d geometry mismatch (gate %zu vs %zu, up %zu "
              "vs %zu, down %zu vs %zu) -- falling back.\n",
              layer_idx, runner.gate_expert_bytes(), gb0,
              runner.up_expert_bytes(), ub0, runner.down_expert_bytes(), db0);
    return false;
  }

  const int chunk = plan.chunk_experts;
  const int n_chunks = (n_used + chunk - 1) / chunk;
  size_t mmap_staged_total = 0;
  // Experts held in the VRAM cache for the duration of their device-to-device
  // copy. Released only after sync_copies(), never per chunk: the copy is
  // asynchronous, and a hold dropped early lets an eviction reclaim the slab
  // the copy engine is still reading.
  struct D2DHold {
    int expert_id;
    const uint8_t *device_ptr;
    bool from_active_layer;
  };
  std::vector<D2DHold> d2d_holds;
  d2d_holds.reserve((size_t)n_used);
  struct RamH2DHold {
    int expert_id;
    MoEExpertBlock *block;
  };
  std::vector<RamH2DHold> ram_h2d_holds;
  ram_h2d_holds.reserve((size_t)n_used);
  int n_reused = 0;
  int n_active_reused = 0;
  int n_ram_direct = 0;
  int n_ram_direct_runs = 0;
  int n_ram_staged = 0;
  // Filled once per compact chunk by GpuTopology::find_expert_batch().
  // These pointers are only residency hints; hold_for_compute() below still
  // performs its independent identity check before the D2D enqueue.
  std::vector<ExpertLocation *> resident_locations((size_t)n_used, nullptr);
  std::vector<char> staged_flag((size_t)chunk, 1);
  // 0 = pending, 1 = copied from the selected GPU's VRAM cache,
  // 2 = copied directly from a pinned RAM slab, 3 = copied from a pageable
  // RAM slab through the pinned staging buffers, 4 = copied from mmap.
  std::vector<unsigned char> trace_source((size_t)n_used, 0);
  std::vector<int> trace_resident_any;
  std::vector<int> trace_resident_gpu;
  trace_resident_any.reserve((size_t)n_used);
  trace_resident_gpu.reserve((size_t)n_used);

  // Qwen's measured decode setting selects packed batch staging. Other
  // families keep the direct pinned-RAM transfer path.
  const bool qwen_decode_batch_stage =
      engine.family_ops.family == ModelFamily::Qwen4Exp && S == 1 &&
      engine.qwen4exp_decode_batch_stage;
  const bool force_batch_stage = qwen_decode_batch_stage;
  static bool logged_batch_stage = false;
  if (force_batch_stage && !logged_batch_stage) {
    LOG_INFO("[GGML-MOE] RAM sources use packed batch staging (%s)\n",
             "Qwen decode config");
    logged_batch_stage = true;
  }
  // D2D cache copies use a separate raw stream. The sync point below drains
  // only that stream in backend mode; H2D staging and graph work remain
  // continuously queued on GGML's shared stream.
  const auto *ram_arena = engine.graph_ram_arena(layer_idx);

  for (int c = 0; c < n_chunks; ++c) {
    const int c0 = c * chunk;
    const int c1 = std::min(n_used, c0 + chunk);
    const int buf = c % GgmlMoeRunner::NBUF;
    // Blocks until this buffer's PREVIOUS H2D drained, which is also what
    // overlaps chunk c's host memcpy with chunk c-1's transfer. Timed so the
    // [PREFILL] line's "CPU blocked on the double-buffer handoff" figure is
    // meaningful for this path too: a large value means the transfer is the
    // bottleneck and the staging memcpy is fully hidden; ~0 means the
    // opposite, and the two are NOT actually overlapping.
    PERF_START(gm_wait);
    runner.wait_stage_free(buf);
    PERF_END(gm_wait, t_prefill_stream_wait);

    // Start the copy-stream timing batch before VRAM reuse is queued.  This
    // lets the runner report the real D2D and H2D CUDA-event intervals
    // separately; the old placement started the timer only after D2D and
    // made the source breakdown ambiguous.
    if (!runner.begin_copy_batch(buf)) {
      runner.sync_copies();
      return false;
    }

    // Classify BEFORE staging: an expert already resident in this GPU's VRAM
    // cache is filled device-to-device (~600 GB/s) instead of being read out
    // of the mmap, memcpy'd into pinned host memory and shipped back over a
    // 12.3 GB/s link. Probe the whole compact set under one cache lock; the
    // per-expert hold below remains an independent race-safe check.
    // Only THIS device's residents qualify: a slab on the other GPU would be
    // a peer copy, which is not obviously cheaper than the H2D it replaces
    // and is not what this was measured against.
    topo->find_expert_batch(layer_idx, used.data() + c0, c1 - c0,
                            resident_locations.data() + c0);
    for (int i = c0; i < c1; ++i) {
      const int e = used[(size_t)i];
      staged_flag[(size_t)(i - c0)] = 1;
      // The residency lookup is shared across the compact batch, and the
      // per-expert hold below remains the race-safe ownership check.
      const ExpertLocation *rl = resident_locations[(size_t)i];
      if (trace_set && rl && rl->device_ptr && rl->upload_ready) {
        trace_resident_any.push_back(e);
        if (rl->gpu_id == chosen_device)
          trace_resident_gpu.push_back(e);
      }
      bool from_active_layer = false;
      if (!rl && allow_active_layer_sources) {
        bool promoted_to_main = false;
        rl = topo->find_active_expert_wait(
            layer_idx, e, [&](int, int) { promoted_to_main = true; });
        from_active_layer = rl != nullptr && !promoted_to_main;
      }
      if (rl && rl->device_ptr && rl->upload_ready &&
          rl->gpu_id == chosen_device &&
          topo->hold_for_compute(layer_idx, e, rl->device_ptr,
                                 from_active_layer)) {
        if (runner.copy_expert_d2d(buf, i, rl->device_ptr)) {
          staged_flag[(size_t)(i - c0)] = 0;
          trace_source[(size_t)i] = 1;
          d2d_holds.push_back({e, rl->device_ptr, from_active_layer});
          ++n_reused;
          if (from_active_layer)
            ++n_active_reused;
        } else {
          // Copy could not be issued -- give the hold straight back and let
          // this expert go through the normal staging path below.
          topo->release_hold(layer_idx, e, rl->device_ptr, from_active_layer);
        }
      }
    }
    if (!runner.mark_copy_d2d_done(buf)) {
      runner.sync_copies();
      return false;
    }
    // find_expert() -> finish_expert_hit() does cudaSetDevice(loc.gpu_id) and
    // does NOT restore it, so a hit on an expert resident on the other vault
    // GPU leaves that GPU current. Everything below assumes ours.
    cudaSetDevice(chosen_device);

    if (!runner.begin_copy_batch(buf)) {
      runner.sync_copies();
      for (const auto &h : d2d_holds)
        topo->release_hold(layer_idx, h.expert_id, h.device_ptr,
                           h.from_active_layer);
      for (const auto &h : ram_h2d_holds)
        ram_mgr.ram_unpin(*h.block);
      return false;
    }

    // Prefer a RAM-tier slab when it was persistently page-locked at
    // allocation time. Queue the three projection copies directly from the
    // [gate|up|down] slab and keep the RAM slot pinned until the layer's one
    // copy barrier below. Pageable RAM uses the same pinned staging buffers,
    // but the source is the resident RAM slab and the CPU memcpy completes
    // before the RAM hold is released. This avoids both mmap and CUDA's
    // hidden pageable-source staging copy.
    const size_t ram_cap =
        (layer_idx >= 0 && layer_idx < (int)engine.layers_experts.size())
            ? std::min(engine.layers_experts[(size_t)layer_idx].size(),
                       (size_t)std::max(1, ram_mgr.cap_for_layer(layer_idx)))
            : 0;
    const bool qwen_ddr_only_source =
        qwen_ddr_only_expert_source(engine, layer_idx);
    static std::atomic<bool> logged_qwen_ddr_miss{false};
    const bool can_coalesce_direct =
        !force_batch_stage && ram_arena && ram_arena->base &&
        ram_arena->stride >= gb0 + ub0 + db0;
    std::vector<const uint8_t *> direct_sources;
    std::vector<MoEExpertBlock *> direct_blocks;
    if (can_coalesce_direct) {
      direct_sources.assign((size_t)(c1 - c0), nullptr);
      direct_blocks.assign((size_t)(c1 - c0), nullptr);
    }
    if (ram_cap > 0) {
      auto &ram_slots = engine.layers_experts[(size_t)layer_idx];
      for (int i = c0; i < c1; ++i) {
        if (!staged_flag[(size_t)(i - c0)])
          continue;
        const uint8_t *ram_slab = nullptr;
        size_t ram_capacity = 0;
        MoEExpertBlock *rb = ram_mgr.find_ram_slot_if_ready_pinned(
            ram_slots, ram_cap, layer_idx, used[(size_t)i], &ram_slab,
            &ram_capacity);
        if (!rb)
          continue;
        if (!force_batch_stage && ram_slab &&
            ram_capacity >= gb0 + ub0 + db0) {
          if (can_coalesce_direct) {
            // Keep the source held while we classify the whole chunk. Direct
            // copies are issued below, where consecutive graph-arena slabs
            // can be coalesced into three pitched copies.
            direct_sources[(size_t)(i - c0)] = ram_slab;
            direct_blocks[(size_t)(i - c0)] = rb;
          } else if (runner.copy_expert_h2d(buf, i, ram_slab)) {
            // Preserve the established per-expert path for ordinary RAM
            // slabs; only a validated fixed-stride arena enters the new
            // batching path below.
            moe_pipeline_internal::qwen4exp_record_pilot_fetch_start(layer_idx);
            staged_flag[(size_t)(i - c0)] = 0;
            trace_source[(size_t)i] = 2;
            ram_h2d_holds.push_back({used[(size_t)i], rb});
            ++n_ram_direct;
            ++n_ram_direct_runs;
          } else {
            ram_mgr.ram_unpin(*rb);
          }
        } else {
          ram_mgr.ram_unpin(*rb);
        }
      }
    }

    const size_t direct_stride = can_coalesce_direct ? ram_arena->stride : 0;
    for (int i = c0; can_coalesce_direct && i < c1;) {
      const int local = i - c0;
      const uint8_t *src = direct_sources[(size_t)local];
      if (!src || !direct_blocks[(size_t)local]) {
        ++i;
        continue;
      }
      int j = i + 1;
      if (ram_arena && ram_arena->base && direct_stride >= gb0 + ub0 + db0 &&
          src == ram_arena->base + (size_t)used[(size_t)i] * direct_stride) {
        while (j < c1) {
          const int cur = j - c0;
          const int offset = j - i;
          if (!direct_sources[(size_t)cur] ||
              !direct_blocks[(size_t)cur] ||
              used[(size_t)j] != used[(size_t)i] + offset ||
              direct_sources[(size_t)cur] !=
                  src + (size_t)offset * direct_stride)
            break;
          ++j;
        }
      }
      const int run = j - i;
      const bool copied =
          run > 1
              ? runner.copy_expert_h2d_strided(buf, local, run, src,
                                                direct_stride)
              : runner.copy_expert_h2d(buf, local, src);
      if (!copied) {
        // A CUDA enqueue failure must not release a source slab while the
        // copy stream may still reference it. Drain the already-issued work,
        // release this run, and let the caller take its established fallback.
        runner.sync_copies();
        for (int k = i; k < j; ++k) {
          MoEExpertBlock *held = direct_blocks[(size_t)(k - c0)];
          if (held) ram_mgr.ram_unpin(*held);
        }
        return false;
      }
      // This is the first actual predicted-layer DDR->VRAM H2D issue when
      // the grouped runner cannot reuse a VRAM/active-layer source.
      moe_pipeline_internal::qwen4exp_record_pilot_fetch_start(layer_idx);
      for (int k = i; k < j; ++k) {
        staged_flag[(size_t)(k - c0)] = 0;
        trace_source[(size_t)k] = 2;
        ram_h2d_holds.push_back(
            {used[(size_t)k], direct_blocks[(size_t)(k - c0)]});
        ++n_ram_direct;
      }
      ++n_ram_direct_runs;
      i = j;
    }

    // Resident but pageable RAM is still the preferred source when the
    // aggregate cudaHostRegister ceiling has been reached. Hold each slab
    // through the synchronous CPU copy below; after that copy the queued H2D
    // reads only the separate pinned staging buffer.
    std::vector<const uint8_t *> ram_stage_sources((size_t)(c1 - c0), nullptr);
    std::vector<MoEExpertBlock *> ram_stage_holds;
    ram_stage_holds.reserve((size_t)(c1 - c0));
    if (ram_cap > 0) {
      auto &ram_slots = engine.layers_experts[(size_t)layer_idx];
      for (int i = c0; i < c1; ++i) {
        if (!staged_flag[(size_t)(i - c0)])
          continue;
        const uint8_t *ram_slab = nullptr;
        size_t ram_capacity = 0;
        MoEExpertBlock *rb = ram_mgr.find_ram_slot_if_ready_held(
            ram_slots, ram_cap, layer_idx, used[(size_t)i], &ram_slab,
            &ram_capacity);
        if (!rb)
          continue;
        if (ram_slab && ram_capacity >= gb0 + ub0 + db0) {
          ram_stage_sources[(size_t)(i - c0)] = ram_slab;
          trace_source[(size_t)i] = 3;
          ram_stage_holds.push_back(rb);
          ++n_ram_staged;
        } else {
          ram_mgr.ram_unpin(*rb);
        }
      }
    }

    // madvise(WILLNEED) ahead of the staging memcpy. Worth it only when the
    // pages are genuinely cold: at ~3 syscalls per expert this is ~22k calls
    // per prefill, which is pure overhead once the file is in page cache.
    if (!qwen_ddr_only_source && engine.model_file.mmap_active()) {
      for (int i = c0; i < c1; ++i) {
        if (!staged_flag[(size_t)(i - c0)] ||
            ram_stage_sources[(size_t)(i - c0)])
          continue;
        const int e = used[(size_t)i];
        const uint64_t goff =
            unpacked_layer ? g0->offset : g0->offset + (uint64_t)e * gb0;
        const uint64_t uoff =
            unpacked_layer ? u0->offset : u0->offset + (uint64_t)e * ub0;
        const uint64_t doff =
            unpacked_layer ? d0->offset : d0->offset + (uint64_t)e * db0;
        engine.model_file.mmap_willneed(*g0, goff, gb0);
        engine.model_file.mmap_willneed(*u0, uoff, ub0);
        engine.model_file.mmap_willneed(*d0, doff, db0);
      }
    }

    // Slot pointers are resolved OUTSIDE the parallel region on purpose. They
    // do not come from a thread_local here, but the same discipline that cost
    // two SIGSEGVs in the streaming path applies: nothing a worker touches
    // should depend on which thread is asking.
    std::vector<GgmlMoeStageSlot> slots((size_t)(c1 - c0));
    for (int i = c0; i < c1; ++i)
      slots[(size_t)(i - c0)] = runner.stage_slot(buf, i - c0);

    // Pageable RAM uses a synchronous DDR->pinned staging memcpy before the
    // asynchronous H2D.  Mark the boundary immediately before that work so
    // the reported gap is not confused with the later CUDA copy service time.
    if (n_ram_staged > 0)
      moe_pipeline_internal::qwen4exp_record_pilot_fetch_start(layer_idx);

    std::atomic<bool> stage_ok{true};
    size_t staged = 0;
    PERF_START(pf_stage);
#pragma omp parallel for schedule(static) reduction(+ : staged)
    for (int i = c0; i < c1; ++i) {
      if (!staged_flag[(size_t)(i - c0)])
        continue; // filled D2D already
      const int e = used[(size_t)i];
      const GgmlMoeStageSlot &sl = slots[(size_t)(i - c0)];
      if (!sl.gate) {
        stage_ok.store(false, std::memory_order_relaxed);
        continue;
      }
      const uint8_t *gs = nullptr;
      const uint8_t *us = nullptr;
      const uint8_t *ds = nullptr;
      const uint8_t *ram_base = ram_stage_sources[(size_t)(i - c0)];
      if (ram_base) {
        gs = ram_base;
        us = ram_base + gb0;
        ds = ram_base + gb0 + ub0;
      } else if (qwen_ddr_only_source) {
        bool expected = false;
        if (logged_qwen_ddr_miss.compare_exchange_strong(
                expected, true, std::memory_order_relaxed))
          LOG_ERROR("[GGML-MOE] Qwen ram_preload_all_experts source miss: "
                    "layer=%d expert=%d has no ready DDR slab; refusing GGUF "
                    "mmap fallback\n",
                    layer_idx, e);
        stage_ok.store(false, std::memory_order_relaxed);
        continue;
      } else if (engine.model_file.mmap_active()) {
        const uint64_t goff =
            unpacked_layer ? g0->offset : g0->offset + (uint64_t)e * gb0;
        const uint64_t uoff =
            unpacked_layer ? u0->offset : u0->offset + (uint64_t)e * ub0;
        const uint64_t doff =
            unpacked_layer ? d0->offset : d0->offset + (uint64_t)e * db0;
        gs = engine.model_file.mmap_bytes(*g0, goff, gb0);
        us = engine.model_file.mmap_bytes(*u0, uoff, ub0);
        ds = engine.model_file.mmap_bytes(*d0, doff, db0);
        trace_source[(size_t)i] = 4;
      }
      if (!gs || !us || !ds) {
        stage_ok.store(false, std::memory_order_relaxed);
        continue;
      }
      std::memcpy(sl.gate, gs, gb0);
      std::memcpy(sl.up, us, ub0);
      std::memcpy(sl.down, ds, db0);
      staged += gb0 + ub0 + db0;
    }
    PERF_END(pf_stage, t_disk_read);
    size_t ram_staged_bytes = 0;
    for (int i = c0; i < c1; ++i) {
      if (staged_flag[(size_t)(i - c0)] && ram_stage_sources[(size_t)(i - c0)])
        ram_staged_bytes += gb0 + ub0 + db0;
    }
    if (stage_ok.load(std::memory_order_relaxed))
      mmap_staged_total += staged - ram_staged_bytes;
    for (MoEExpertBlock *rb : ram_stage_holds)
      ram_mgr.ram_unpin(*rb);
    // std::vector<bool> is a bitset with no .data(), so the mask push_chunk
    // takes is built here as real bools rather than reusing staged_flag.
    std::vector<unsigned char> mask_store((size_t)(c1 - c0));
    for (int i = 0; i < c1 - c0; ++i)
      mask_store[(size_t)i] = staged_flag[(size_t)i] != 0;
    const bool *staged_mask = (const bool *)mask_store.data();
    static_assert(sizeof(bool) == sizeof(unsigned char),
                  "staged mask aliasing assumes 1-byte bool");
    if (!stage_ok.load(std::memory_order_relaxed) ||
        !runner.push_chunk(buf, c0, c1 - c0, staged_mask)) {
      // Anything already in flight has to drain before the caller reuses the
      // staging buffers on the next layer -- and before any held slab can be
      // released, since the copy engine may still be reading it.
      runner.sync_copies();
      if (runner.backend_stream_copies())
        runner.finish_h2d();
      for (const auto &h : d2d_holds)
        topo->release_hold(layer_idx, h.expert_id, h.device_ptr,
                           h.from_active_layer);
      for (const auto &h : ram_h2d_holds)
        ram_mgr.ram_unpin(*h.block);
      LOG_ERROR("[GGML-MOE] layer=%d chunk@%d: staging failed -- falling back "
                "to the per-expert path for this layer.\n",
                layer_idx, c0);
      return false;
    }
  }
  perf().bytes_disk_read.fetch_add(mmap_staged_total,
                                   std::memory_order_relaxed);

  // Every weight copy, H2D and D2D alike, must be complete before a held slab
  // can be handed back to the eviction machinery. Timed into the same bucket
  // as the per-chunk handoff wait: together they are the whole "CPU blocked
  // on the expert transfer" cost, and this one is the tail the pipeline
  // cannot hide (the last chunk's H2D only starts once its staging is done).
  PERF_START(gm_sync);
  if (!runner.backend_stream_copies())
    runner.sync_copies();
  PERF_END(gm_sync, t_prefill_stream_wait);
  if (!runner.backend_stream_copies()) {
    for (const auto &h : d2d_holds)
      topo->release_hold(layer_idx, h.expert_id, h.device_ptr,
                         h.from_active_layer);
    for (const auto &h : ram_h2d_holds)
      ram_mgr.ram_unpin(*h.block);
    ram_h2d_holds.clear();
  } else {
    // The backend-stream path has a device-side copy->graph dependency. Keep
    // both classes of source protected across the layer boundary; releasing a
    // RAM pin or VRAM hold here would let eviction overwrite bytes while the
    // queued copy stream is still reading them. finish_h2d() drains the
    // completion events once for the whole forward pass and runs these
    // releases then.
    for (const auto &h : d2d_holds)
      runner.defer_vram_hold((void *)topo, layer_idx, h.expert_id,
                             h.device_ptr);
    d2d_holds.clear();
    if (!ram_h2d_holds.empty()) {
      auto holds = std::move(ram_h2d_holds);
      runner.defer_host_release([&ram_mgr, holds = std::move(holds)]() mutable {
        for (const auto &h : holds)
          ram_mgr.ram_unpin(*h.block);
      });
    }
  }
  perf().n_prefill_stream_vram_hit.fetch_add((uint64_t)n_reused,
                                             std::memory_order_relaxed);
  perf().n_prefill_stream_vram_miss.fetch_add((uint64_t)(n_used - n_reused),
                                              std::memory_order_relaxed);

  // The ordinary graph uses compact host IDs.  The all-resident device-route
  // path deliberately keeps the router's global IDs on the device and makes
  // `used` the complete layer, so global and compact IDs are identical.
  std::vector<int32_t> ids;
  if (!use_device_route_for_graph) {
    if (!tok_topk_ids || !tok_topk_weights)
      return false;
    std::vector<int> local((size_t)n_experts, -1);
    for (int i = 0; i < n_used; ++i)
      local[(size_t)used[(size_t)i]] = i;
    ids.resize((size_t)S * topk);
    for (size_t i = 0, n = (size_t)S * topk; i < n; ++i) {
      const int e = tok_topk_ids[i];
      if (e < 0 || e >= n_experts || local[(size_t)e] < 0) {
        for (int b = 0; b < GgmlMoeRunner::NBUF; ++b)
          runner.wait_stage_free(b);
        for (const auto &h : ram_h2d_holds)
          ram_mgr.ram_unpin(*h.block);
        return false;
      }
      ids[i] = (int32_t)local[(size_t)e];
    }
  }

  PERF_START(ggml_run);
  const auto t_graph0 = std::chrono::steady_clock::now();
  const bool ok =
      use_device_route_for_graph
          ? (device_input ? runner.run_layer_device_input_tensor_device_route(
                                device_input, device_route_ids,
                                device_route_weights, device_output)
                          : false)
          : (device_output
                 ? (device_input ? runner.run_layer_device_input_tensor(
                                       device_input, ids.data(),
                                       tok_topk_weights, device_output)
                                 : runner.run_layer_device(normed, ids.data(),
                                                           tok_topk_weights,
                                                           device_output))
                 : runner.run_layer(normed, ids.data(), tok_topk_weights,
                                    output));
  PERF_END(ggml_run, t_gpu_compute);
  // Per-layer, not aggregated, DEBUG-only: this model's layers are NOT uniform (24 are
  // IQ2_XXS gate/up, 24 are IQ1_M -- see the [QUANT] startup report), and
  // ggml-cuda picks a completely different kernel family per type (MMQ for
  // one, the dequantise+cuBLAS fallback for the other). A single total would
  // average those together and hide which one actually costs.
  {
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t_graph0)
                          .count();
    int n_mmap_experts = 0;
    for (int i = 0; i < n_used; ++i)
      if (trace_source[(size_t)i] == 4)
        ++n_mmap_experts;
    perf().n_ggml_moe_d2d_experts.fetch_add((uint64_t)n_reused,
                                            std::memory_order_relaxed);
    perf().n_ggml_moe_active_d2d_experts.fetch_add((uint64_t)n_active_reused,
                                                   std::memory_order_relaxed);
    perf().n_ggml_moe_ram_direct_experts.fetch_add((uint64_t)n_ram_direct,
                                                   std::memory_order_relaxed);
    perf().n_ggml_moe_ram_staged_experts.fetch_add((uint64_t)n_ram_staged,
                                                   std::memory_order_relaxed);
    perf().n_ggml_moe_mmap_experts.fetch_add((uint64_t)n_mmap_experts,
                                             std::memory_order_relaxed);
    LOG_DEBUG("[GGML-MOE] layer=%d experts=%d gate=%u up=%u down=%u "
             "graph=%.2f ms d2d=%d active_d2d=%d ram_direct=%d "
             "direct_runs=%d ram_staged=%d mmap_staged=%.2f MB\n",
             layer_idx, n_used, (unsigned)g0->type, (unsigned)u0->type,
             (unsigned)d0->type, ms, n_reused, n_active_reused, n_ram_direct,
             n_ram_direct_runs, n_ram_staged,
             (double)mmap_staged_total / (1024.0 * 1024.0));
  }

  if (trace_set) {
    auto format_ids = [](const std::vector<int> &ids) {
      std::string result;
      for (size_t i = 0; i < ids.size(); ++i) {
        if (i != 0)
          result.push_back(',');
        result += std::to_string(ids[i]);
      }
      return result;
    };
    std::vector<int> d2d_ids;
    std::vector<int> ram_direct_ids;
    std::vector<int> ram_staged_ids;
    std::vector<int> mmap_ids;
    std::vector<int> h2d_ids;
    d2d_ids.reserve((size_t)n_used);
    ram_direct_ids.reserve((size_t)n_used);
    ram_staged_ids.reserve((size_t)n_used);
    mmap_ids.reserve((size_t)n_used);
    h2d_ids.reserve((size_t)n_used);
    for (int i = 0; i < n_used; ++i) {
      const int e = used[(size_t)i];
      switch (trace_source[(size_t)i]) {
      case 1:
        d2d_ids.push_back(e);
        break;
      case 2:
        ram_direct_ids.push_back(e);
        h2d_ids.push_back(e);
        break;
      case 3:
        ram_staged_ids.push_back(e);
        h2d_ids.push_back(e);
        break;
      case 4:
        mmap_ids.push_back(e);
        h2d_ids.push_back(e);
        break;
      default:
        break;
      }
    }
    const size_t expert_bytes = gb0 + ub0 + db0;
    LOG_INFO(
        "[MOE-SET] layer=%d required=%d resident_any=%zu "
        "resident_gpu=%zu d2d=%zu h2d=%zu ram_direct=%zu "
        "ram_staged=%zu mmap=%zu h2d_bytes=%zu required_ids=%s "
        "resident_any_ids=%s resident_gpu_ids=%s d2d_ids=%s "
        "ram_direct_ids=%s ram_staged_ids=%s mmap_ids=%s h2d_ids=%s\n",
        layer_idx, n_used, trace_resident_any.size(), trace_resident_gpu.size(),
        d2d_ids.size(), h2d_ids.size(), ram_direct_ids.size(),
        ram_staged_ids.size(), mmap_ids.size(), h2d_ids.size() * expert_bytes,
        format_ids(used).c_str(), format_ids(trace_resident_any).c_str(),
        format_ids(trace_resident_gpu).c_str(), format_ids(d2d_ids).c_str(),
        format_ids(ram_direct_ids).c_str(), format_ids(ram_staged_ids).c_str(),
        format_ids(mmap_ids).c_str(), format_ids(h2d_ids).c_str());
  }

  GgmlMoeTransferStats transfer;
  runner.collect_transfer_stats(transfer);
  if (transfer.h2d_bytes || transfer.d2d_bytes) {
    perf().t_ggml_moe_d2d.fetch_add(transfer.d2d_seconds,
                                    std::memory_order_relaxed);
    perf().t_ggml_moe_h2d.fetch_add(transfer.h2d_seconds,
                                    std::memory_order_relaxed);
    perf().t_ggml_moe_h2d_dma.fetch_add(transfer.h2d_dma_seconds,
                                        std::memory_order_relaxed);
    perf().bytes_ggml_moe_d2d.fetch_add(transfer.d2d_bytes,
                                        std::memory_order_relaxed);
    perf().bytes_ggml_moe_h2d.fetch_add(transfer.h2d_bytes,
                                        std::memory_order_relaxed);
    // Preserve the existing generic report for compatibility, but it now
    // contains only actual H2D bytes/time from this runner.
    if (transfer.h2d_bytes) {
      perf().t_prefill_stream_h2d.fetch_add(transfer.h2d_seconds,
                                            std::memory_order_relaxed);
      perf().bytes_prefill_stream_h2d.fetch_add(transfer.h2d_bytes,
                                                std::memory_order_relaxed);
    }
  }
  if (ok && output_runner && device_output && *device_output)
    *output_runner = &runner;
  return ok;
}

BatchedDispatchResult BatchedDispatchPipeline::run(
    MoePipelineContext &context) {
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
  // --- Phase 3b (OPT-IN): expert-major batched dispatch ------------------
  // Groups this batch's (token, expert-slot) pairs BY EXPERT and runs ONE
  // expert_ffn call per expert over all the tokens routed to it, instead of
  // the legacy per-token loop's S=1 call per (token, expert).
  //
  // This is the same restructuring llama.cpp's ggml_mul_mat_id performs; its
  // CUDA fallback (ggml-cuda.cu's ggml_cuda_mul_mat_id) builds exactly this
  // CSR-style layout -- tokens_per_expert + a sorted index list -- which is
  // what the code below mirrors, using Phase 1's already-computed
  // scratch.expert_freq[] as the counts so nothing has to be recounted.
  //
  // WHY: the legacy loop issues S*n_layers*topk dispatches and re-reads each
  // expert's ~3 MB weight slab once PER TOKEN. Measured on qwen4exp: 1.53M
  // dispatches for a 3,189-token prompt, and prefill throughput flat in
  // prompt length (5.4 tok/s at S=24 vs 6.02 at S=3189). See
  // dense_dispatch.hpp's moe_prefill_batched_flag() for the full writeup.
  //
  // The generic flag is shared by all model families and remains OFF by
  // default. GLM5NEXT reaches the same corrected grouped implementation only
  // through its separate family-gated flag below; with both flags off the
  // legacy loop remains byte-for-byte unchanged for every model.
  //
  // v1's legacy per-token path deliberately does NOT pipeline across experts
  // (issue-all/finish-all the way the old Pass A/Pass B did across one token's
  // topk). The GLM5NEXT grouped-prefill opt-in below is the narrow exception:
  // it keeps one chunk of device-accumulated expert work in flight, while the
  // generic path remains conservative.
  const bool glm5next_grouped_prefill =
      engine.glm5next_moe_prefill_gpu &&
      engine.family_ops.family == ModelFamily::GLM5NEXT;
  batched_handled = false;
  if (gpu_route_device && S > 1 && topk > 0 && n_experts > 0 &&
      !is_cpu_affinity_layer(layer_idx)) {
    // Keep the actual routed set for compact fallback.  The previous device
    // handoff made `used` the complete [0,n_experts) range so global IDs could
    // be passed directly to the device graph.  That was correct only for the
    // all-resident view: when a layer was scattered, the compact fallback
    // consequently repacked all 512 experts even though this batch touched
    // only a subset.  The direct resident path now restores the full axis
    // internally, while compact layers use this real routed subset and the
    // host route table for a local-ID remap.
    //
    // The device-router path does not populate expert_freq: it writes the
    // routed IDs and weights on the device, then copies those two tables back
    // for the host/device handoff.  Reconstruct the counts from that table
    // here.  Using expert_freq directly made it look as though no experts
    // were used, which disabled the batched graph and fell through to the
    // legacy per-token path for the early layers.
    std::fill(scratch.expert_freq.begin(), scratch.expert_freq.end(), 0);
    for (size_t i = 0; i < scratch.tok_topk_ids.size(); ++i) {
      const int e = scratch.tok_topk_ids[i];
      if (e >= 0 && e < n_experts)
        ++scratch.expert_freq[(size_t)e];
    }
    std::vector<int> used;
    used.reserve((size_t)n_experts);
    for (int e = 0; e < n_experts; ++e)
      if (scratch.expert_freq[(size_t)e] > 0)
        used.push_back(e);
    static const bool trace_required_set =
        Logger::instance().would_log(LogLevel::DEBUG);
    if (trace_required_set && gpu_route_device && device_router) {
      // Device routing deliberately leaves IDs/weights on CUDA in normal
      // execution. Export them only for the explicit set trace so its
      // required expert set is real rather than the untouched zero-filled
      // host scratch array.
      gpu_route_ids_ks.resize((size_t)topk * S);
      gpu_route_weights_ks.resize((size_t)topk * S);
      if (!device_router->copy_route_to_host(layer_idx, gpu_route_ids_ks.data(),
                                             gpu_route_weights_ks.data())) {
        LOG_ERROR("[MOE-REQUIRED] layer=%d failed to export device route\n",
                  layer_idx);
      } else {
        scratch.tok_topk_ids = gpu_route_ids_ks;
        scratch.tok_topk_weights = gpu_route_weights_ks;
        std::fill(scratch.expert_freq.begin(), scratch.expert_freq.end(), 0);
        for (int e : gpu_route_ids_ks) {
          if (e >= 0 && e < n_experts)
            ++scratch.expert_freq[(size_t)e];
        }
        used.clear();
        for (int e = 0; e < n_experts; ++e)
          if (scratch.expert_freq[(size_t)e] > 0)
            used.push_back(e);
      }
    }
    if (trace_required_set) {
      std::string required_ids;
      required_ids.reserve(used.size() * 4);
      for (size_t i = 0; i < used.size(); ++i) {
        if (i != 0)
          required_ids.push_back(',');
        required_ids += std::to_string(used[i]);
      }
      LOG_DEBUG("[MOE-REQUIRED] layer=%d required=%zu route=%s ids=%s\n",
               layer_idx, used.size(), gpu_route_device ? "device" : "host",
               required_ids.c_str());
    }
    const bool qwen_device_handoff =
        engine.family_ops.family == ModelFamily::Qwen4Exp &&
        engine.family_ops.uses_hyper_connections && qwen4exp_hc_device_live() &&
        !is_cpu_affinity_layer(layer_idx);
    bool ggml_handled =
        qwen_device_handoff && clamp_exp == 0.0f && ram_mgr.vram_cache &&
        run_ggml_moe_layer(
            engine, ram_mgr, ram_mgr.vram_cache, layer_idx, S, D, moe_inter,
            topk, n_experts, used, scratch.tok_topk_ids.data(),
            scratch.tok_topk_weights.data(), scratch.normed.data(), output,
            moe_prefill_stream_chunk(), /*unpacked_layer=*/false,
            &scratch.moe_device_output, engine.qwen4exp_device_hc_ffn_input,
            gpu_route_ids_device, gpu_route_weights_device,
            /*allow_active_layer_sources=*/false, &scratch.moe_device_runner);
    if (ggml_handled) {
      scratch.moe_device_output_ready = scratch.moe_device_output != nullptr;
      bool shared_device_handled = false;
      if (moe_ggml_shared_expert() && c.n_expert_shared > 0) {
        const void *combined = nullptr;
        shared_device_handled =
            run_qwen4exp_shared_device(engine, layer_idx, S, D, moe_inter,
                                       engine.qwen4exp_device_hc_ffn_input,
                                       scratch.moe_device_output, c, &combined);
        if (shared_device_handled) {
          scratch.moe_device_output = combined;
          scratch.moe_shared_device_output = true;
        }
      }
      if (!shared_device_handled) {
        // The routed graph is still correct; only the optional shared graph
        // failed.  Recover the routed result and use the established host
        // shared-expert implementation so this experimental path remains
        // numerically complete.
        if (engine.qwen4exp_device_hc_ffn_input &&
            !GgmlHcRunner::instance().copy_pre_mix_to_host(
                engine.qwen4exp_device_hc_ffn_input, scratch.normed.data(),
                S)) {
          LOG_ERROR("[GGML-SHARED] layer=%d: HC input recovery failed; "
                    "shared host fallback is unsafe\n",
                    layer_idx);
        }
        if (!scratch.moe_device_runner ||
            !scratch.moe_device_runner->copy_device_output_to_host(
                scratch.moe_device_output, output)) {
          LOG_ERROR("[GGML-MOE] layer=%d: all-resident routed output recovery "
                    "failed\n",
                    layer_idx);
          return BatchedDispatchResult::Abort;
        }
        run_qwen4exp_shared_host(engine, layer_idx, S, D, moe_inter,
                                 scratch.normed.data(), output, scratch, c);
        scratch.moe_device_output = nullptr;
        scratch.moe_shared_device_output = false;
        scratch.moe_device_output_ready = false;
      }
      batched_handled = true;
    } else {
      // A device-route graph failure must not drop the layer.  Re-run only
      // the tiny route table to host and let the existing validated batched
      // path continue below; the large HC activation remains device-resident.
      gpu_route_device = false;
      gpu_route_ids_device = nullptr;
      gpu_route_weights_device = nullptr;
      gpu_router_handled = false;
      if (device_router) {
        gpu_route_ids_ks.resize((size_t)topk * S);
        gpu_route_weights_ks.resize((size_t)topk * S);
        gpu_router_handled = device_router->run_tensor(
            layer_idx, engine.qwen4exp_device_hc_ffn_input,
            gpu_route_ids_ks.data(), gpu_route_weights_ks.data());
      }
    if (!gpu_router_handled) {
        LOG_ERROR("[GGML-MOE] layer=%d: device-route and host-route fallback "
                  "both failed\n",
                  layer_idx);
        return BatchedDispatchResult::Abort;
      }
    }
  }
  // The GLM5NEXT switch is deliberately separate from the generic flag.
  // GLM's mixed per-expert quant geometry is the reason the old shared flag
  // remains opt-in/off here; this path still uses the same corrected
  // per-expert metadata and routing-consistency checks, but its activation is
  // only reachable for an explicitly enabled GLM5NEXT configuration.
  const bool grouped_prefill =
      moe_prefill_batched() || glm5next_grouped_prefill;
  if (!batched_handled && grouped_prefill && S > 1 && topk > 0 &&
      n_experts > 0) {
    PERF_SCOPE(t_moe_phase3_dispatch);

    // CSR build: offs[e] .. offs[e+1] indexes into slot[], which holds the
    // flat (s*topk + k) id of every pair routed to expert e. expert_freq is
    // Phase 1's own count of how many times each expert was picked in this
    // batch -- reused directly rather than recomputed.
    std::vector<int> offs((size_t)n_experts + 1, 0);
    for (int e = 0; e < n_experts; ++e)
      offs[(size_t)e + 1] = offs[(size_t)e] + scratch.expert_freq[(size_t)e];
    const int total_pairs = offs[(size_t)n_experts];

    std::vector<int> cursor(offs.begin(), offs.end() - 1);
    std::vector<int> slot((size_t)std::max(1, total_pairs));
    bool routing_ok = (total_pairs == S * topk);
    if (routing_ok) {
      for (int s2 = 0; s2 < S && routing_ok; ++s2) {
        for (int k = 0; k < topk; ++k) {
          const int e = scratch.tok_topk_ids[(size_t)s2 * topk + k];
          if (e < 0 || e >= n_experts) {
            routing_ok = false;
            break;
          }
          slot[(size_t)cursor[(size_t)e]++] = s2 * topk + k;
        }
      }
    }
    if (!routing_ok) {
      // expert_freq disagrees with tok_topk_ids, or an id is out of range.
      // Fall through to the legacy loop rather than compute something wrong
      // -- this should be unreachable (both come from the same Phase 1) and
      // is here so a future change to either can never silently corrupt.
      LOG_ERROR("[MOE-BATCH] layer=%d: routing table inconsistent "
                "(sum(expert_freq)=%d vs S*topk=%d) -- falling back to the "
                "per-token path for this layer.\n",
                layer_idx, total_pairs, S * topk);
    } else {
      const bool ordered_prefill =
          engine.family_ops.family == ModelFamily::DeepSeek_V41;
      const bool cache_grouped_prefill = glm5next_grouped_prefill || ordered_prefill;
      // DeepSeek keeps one activation row per (token,k), so grouping by
      // expert or changing its GPU cannot change the weighted reduction.
      static thread_local PinnedFloatBuffer ordered_outputs;
      const size_t route_elements = (size_t)total_pairs * D;
      if (ordered_prefill &&
          !ordered_outputs.resize(route_elements, "dsv41_prefill_routes"))
        return BatchedDispatchResult::Abort;
      float *const ordered_base = ordered_prefill ? ordered_outputs.data() : nullptr;
      if (ordered_base)
        std::fill(ordered_base, ordered_base + route_elements, 0.0f);
      const int accum_rows = ordered_prefill ? total_pairs : S;
      // GPU-side accumulation (EngineConfig::moe_gpu_accum): per-pair
      // destination row and router weight, uploaded ONCE per layer, so each
      // expert's scatter kernel can read its own slice at offs[e]. Without
      // this every expert's [n,D] output has to come back to the host purely
      // so a CPU loop can do the weighted add -- ~16.9 GB/prefill at real
      // qwen4exp shape, measured as dn=16.81s against cmp=0.27s of actual
      // expert matmul.
      const bool accum_gpu = moe_gpu_accum() && ram_mgr.vram_cache;
      std::vector<int> tok_pair;
      std::vector<float> w_pair;
      if (accum_gpu) {
        tok_pair.resize((size_t)total_pairs);
        w_pair.resize((size_t)total_pairs);
        for (int i = 0; i < total_pairs; ++i) {
          const int flat = slot[(size_t)i];
          tok_pair[(size_t)i] = ordered_prefill ? flat : flat / topk;
          w_pair[(size_t)i] = ordered_prefill
                                    ? 1.0f : scratch.tok_topk_weights[(size_t)flat];
        }
      }
      bool accum_begun[8] = {false};
      std::vector<float> batch_out;
      // One PINNED activation arena per chunk, each expert owning its own
      // slice. That is what lets the per-expert H2D actually be async: the
      // previous code gathered into a pageable std::vector, so the driver
      // staged every copy synchronously AND the buffer had to be protected
      // by a blocking cudaStreamSynchronize before the next expert could
      // reuse it. MEASURED 2026-09-05: that sync alone was 4.469 s of an
      // 8.80 s expert bucket (9,982 calls @ 0.45 ms). With a private slice
      // per expert there is no reuse hazard, and stream ordering (the
      // upload is on compute_stream, the kernels on the legacy default
      // stream, which synchronize with each other implicitly) already
      // guarantees correctness without any host round trip.
      static thread_local PinnedFloatBuffer act_arena;
      std::vector<size_t> act_off;

      // Experts actually routed to this layer, ascending id. Ascending
      // matters for the streaming path below: within a packed tensor,
      // expert e lives at base + e*stride, so walking ids forward walks the
      // file forward, which is what the kernel's readahead wants.
      std::vector<int> used;
      used.reserve((size_t)n_experts);
      for (int e = 0; e < n_experts; ++e)
        if (offs[(size_t)e + 1] > offs[(size_t)e])
          used.push_back(e);
      const int n_used = (int)used.size();
      static const bool trace_required_set =
          Logger::instance().would_log(LogLevel::DEBUG);
      if (trace_required_set) {
        std::string required_ids;
        required_ids.reserve(used.size() * 4);
        for (size_t i = 0; i < used.size(); ++i) {
          if (i != 0)
            required_ids.push_back(',');
          required_ids += std::to_string(used[i]);
        }
        LOG_DEBUG("[MOE-REQUIRED] layer=%d required=%zu route=host ids=%s\n",
                 layer_idx, used.size(), required_ids.c_str());
      }

      // ---- mmap streaming setup (include/prefill_expert_stream.hpp) ----
      // Bypasses the RAM tier AND the VRAM LRU for the whole layer: during
      // prefill every expert is touched exactly once, so the cache reports
      // 0% hit by arithmetic and all of its machinery is pure overhead.
      static thread_local PrefillExpertStream pstream;
      const bool unpacked_layer = engine.layer_is_unpacked(layer_idx);
      const bool qwen_ddr_only_source =
          qwen_ddr_only_expert_source(engine, layer_idx);

      // ---- ggml MUL_MAT_ID path (opt-in) -----------------------------
      // Computes this layer's ENTIRE routed contribution in one backend
      // graph and adds it into `output`. On success everything below --
      // the PrefillExpertStream setup, the activation arena, the chunk
      // loop, the device accumulator -- is skipped (n_chunks becomes 0 and
      // accum_begun[] stays false), but `batched_handled` still ends up
      // true and the shared expert still runs, so the layer's FFN is
      // complete either way. On failure nothing has been written to
      // `output` and the normal path runs unchanged.
      bool ggml_handled = false;
      bool qwen_device_handoff = false;
      bool shared_device_handled = false;
      if (!ordered_prefill && moe_ggml_mul_mat_id() && clamp_exp == 0.0f && n_used > 0 &&
          !is_cpu_affinity_layer(layer_idx) && ram_mgr.vram_cache) {
        qwen_device_handoff =
            engine.family_ops.family == ModelFamily::Qwen4Exp &&
            engine.family_ops.uses_hyper_connections &&
            qwen4exp_hc_device_live();
        ggml_handled = run_ggml_moe_layer(
            engine, ram_mgr, ram_mgr.vram_cache, layer_idx, S, D, moe_inter,
            topk, n_experts, used, scratch.tok_topk_ids.data(),
            scratch.tok_topk_weights.data(), scratch.normed.data(), output,
            moe_prefill_stream_chunk(), unpacked_layer,
            qwen_device_handoff ? &scratch.moe_device_output : nullptr,
            qwen_device_handoff ? engine.qwen4exp_device_hc_ffn_input : nullptr,
            nullptr, nullptr, /*allow_active_layer_sources=*/false,
            qwen_device_handoff ? &scratch.moe_device_runner : nullptr);
        scratch.moe_device_output_ready = ggml_handled && qwen_device_handoff &&
                                          scratch.moe_device_output != nullptr;
      }

      // Qwen4Exp also has an always-on shared expert.  When the routed result
      // is already device-resident, run that FFN and add it to the routed
      // result on the same GGML stream before HC post-combine.  Keeping the
      // two results together is what lets us remove the shared expert's three
      // host dense projections and its CPU residual loop without changing the
      // HC state transition.  Any shape/type/graph failure falls back by
      // downloading only the routed result and using the existing host shared
      // path below.
      if (ggml_handled && qwen_device_handoff && moe_ggml_shared_expert() &&
          c.n_expert_shared > 0) {
        const void *combined = nullptr;
        shared_device_handled =
            run_qwen4exp_shared_device(engine, layer_idx, S, D, moe_inter,
                                       engine.qwen4exp_device_hc_ffn_input,
                                       scratch.moe_device_output, c, &combined);
        if (shared_device_handled) {
          scratch.moe_device_output = combined;
          scratch.moe_shared_device_output = true;
        }
        if (!shared_device_handled) {
          // The routed graph has no host output because it was handed to HC
          // as a device tensor.  Recover it before the existing shared CPU
          // block so the fallback remains an additive routed+shared result.
          if (!scratch.moe_device_runner ||
              !scratch.moe_device_runner->copy_device_output_to_host(
                  scratch.moe_device_output, output)) {
            LOG_ERROR("[GGML-SHARED] layer=%d: routed output recovery failed; "
                      "shared host fallback cannot be used safely\n",
                      layer_idx);
          }
          scratch.moe_device_output = nullptr;
          scratch.moe_shared_device_output = false;
          scratch.moe_device_output_ready = false;
        }
      }

      bool stream_enabled = false;
      int stream_chunk = n_used > 0 ? n_used : 1;
      if (!ggml_handled && !qwen_ddr_only_source && moe_prefill_stream() &&
          engine.model_file.mmap_active() && ram_mgr.vram_cache && n_used > 0) {
        size_t max_slab = 0;
        bool geom_ok = true;
        for (int e : used) {
          size_t gb, ub, db;
          const GGUFTensor *gM, *uM, *dM;
          if (!engine.expert_geometry_for(layer_idx, e, gb, ub, db, gM, uM,
                                          dM)) {
            geom_ok = false;
            break;
          }
          max_slab = std::max(max_slab, gb + ub + db);
        }
        // Prefer the vault GPU with the biggest expert pool -- on this box
        // that is the 3090 (measured H2D 12.30 GB/s) rather than the 5060 Ti
        // (6.81 GB/s), and this path is bandwidth-bound by construction.
        int vault = -1;
        size_t best = 0;
        for (int vi : ram_mgr.vram_cache->vault_gpus) {
          if ((size_t)vi >= ram_mgr.vram_cache->devices.size())
            continue;
          const auto &dv = ram_mgr.vram_cache->devices[(size_t)vi];
          if (dv.expert_cache_pool && dv.expert_cache_size > best) {
            best = dv.expert_cache_size;
            vault = dv.id;
          }
        }
        if (geom_ok && max_slab > 0 && vault >= 0) {
          stream_chunk = std::min(moe_prefill_stream_chunk(), n_used);
          if (stream_chunk < 1)
            stream_chunk = 1;
          stream_enabled = pstream.ensure(vault, max_slab, stream_chunk);
          // Ship only THIS layer's real expert bytes, not the model-wide
          // maximum the buffers had to be allocated for.
          if (stream_enabled)
            pstream.set_active_stride(max_slab);
        }
      }

      // Per-chunk classification, double-buffered alongside the transfer
      // buffers. An expert already resident in the VRAM cache needs NO
      // transfer at all -- the streaming path bypasses the cache, so
      // without this it re-ships bytes the GPU already holds. This box
      // keeps ~23 GB / 5792 expert slabs resident (see [VRAM-SLOTS]), and
      // routing is concentrated enough that a large share of any layer's
      // real picks are among them.
      std::vector<const ExpertLocation *> res_loc(
          (size_t)PrefillExpertStream::NBUF * (size_t)stream_chunk, nullptr);
      std::vector<int> slot_of(
          (size_t)PrefillExpertStream::NBUF * (size_t)stream_chunk, -1);

      // Stage a chunk's weights straight out of the mmap into the pinned
      // buffer and kick off its H2D. Returns false if anything about this
      // chunk cannot be streamed, in which case the caller runs it on the
      // legacy cached path rather than computing something wrong.
      auto stage_chunk = [&](int c) -> bool {
        const int sc0 = c * stream_chunk;
        const int sc1 = std::min(n_used, sc0 + stream_chunk);
        if (sc0 >= sc1)
          return false;
        const int buf = c % PrefillExpertStream::NBUF;

        // Wait for whatever this buffer held two chunks ago to finish
        // computing before refilling it -- the double-buffer handoff. Also
        // the point at which the PREVIOUS transfer through this buffer is
        // known complete, so its CUDA-event timing can be read back.
        PERF_START(pf_wait);
        pstream.wait_free(buf);
        PERF_END(pf_wait, t_prefill_stream_wait);
        double h2d_ms = 0.0;
        size_t h2d_bytes = 0;
        pstream.collect_h2d(buf, h2d_ms, h2d_bytes);
        if (h2d_bytes) {
          perf().t_prefill_stream_h2d.fetch_add(h2d_ms / 1000.0,
                                                std::memory_order_relaxed);
          perf().bytes_prefill_stream_h2d.fetch_add(h2d_bytes,
                                                    std::memory_order_relaxed);
        }

        // Classify first (serially -- find_expert takes cache_mutex, and at
        // ~10k calls/prefill this is ~0.07 s, far cheaper than the transfer
        // it avoids). Misses are COMPACTED into the low slots so the H2D
        // still moves one contiguous run with no holes.
        const int cbase = buf * stream_chunk;
        int n_miss = 0;
        for (int i = sc0; i < sc1; ++i) {
          const int e = used[(size_t)i];
          const ExpertLocation *rl =
              ram_mgr.vram_cache->find_expert(layer_idx, e);
          if (rl && rl->device_ptr && rl->upload_ready &&
              ram_mgr.vram_cache->hold_for_compute(layer_idx, e, rl->device_ptr,
                                                   false)) {
            res_loc[(size_t)cbase + (size_t)(i - sc0)] = rl;
            slot_of[(size_t)cbase + (size_t)(i - sc0)] = -1;
          } else {
            res_loc[(size_t)cbase + (size_t)(i - sc0)] = nullptr;
            slot_of[(size_t)cbase + (size_t)(i - sc0)] = n_miss++;
          }
        }
        perf().n_prefill_stream_vram_hit.fetch_add(
            (uint64_t)((sc1 - sc0) - n_miss), std::memory_order_relaxed);
        perf().n_prefill_stream_vram_miss.fetch_add((uint64_t)n_miss,
                                                    std::memory_order_relaxed);
        if (n_miss == 0)
          return true; // nothing to transfer for this chunk

        // The RAM tier can already contain nearly every routed expert, in
        // which case the prefetcher performs almost no real reads and cannot
        // warm the page cache for this mmap path. Give the kernel the exact
        // upcoming ranges before the OMP memcpy starts faulting them in.
        // This is advisory only; mmap_bytes() below remains the correctness
        // check and the existing staging failure path is unchanged.
        for (int i = sc0; i < sc1; ++i) {
          const int sl = slot_of[(size_t)cbase + (size_t)(i - sc0)];
          if (sl < 0)
            continue; // already resident in VRAM, no mmap bytes needed
          const int e = used[(size_t)i];
          size_t gb, ub, db;
          const GGUFTensor *gM, *uM, *dM;
          if (!engine.expert_geometry_for(layer_idx, e, gb, ub, db, gM, uM, dM))
            continue;
          const uint64_t goff =
              unpacked_layer ? gM->offset : gM->offset + (uint64_t)e * gb;
          const uint64_t uoff =
              unpacked_layer ? uM->offset : uM->offset + (uint64_t)e * ub;
          const uint64_t doff =
              unpacked_layer ? dM->offset : dM->offset + (uint64_t)e * db;
          engine.model_file.mmap_willneed(*gM, goff, gb);
          engine.model_file.mmap_willneed(*uM, uoff, ub);
          engine.model_file.mmap_willneed(*dM, doff, db);
        }

        std::atomic<bool> stage_ok{true};
        size_t staged_bytes = 0;
        // Hoisted OUT of the parallel region on purpose. pstream is
        // `static thread_local`, so calling pstream.host_slot() from an OMP
        // worker would resolve to that WORKER's own default-constructed
        // instance (h_stage=nullptr, slab_stride=0) rather than the one the
        // main thread just allocated -- a null memcpy destination. MEASURED
        // as a real SIGSEGV here 2026-09-05.
        uint8_t *const stage_base = pstream.h_stage[buf];
        const size_t stage_stride = pstream.stride();
        // Timed into t_disk_read/bytes_disk_read on purpose: this is the
        // direct replacement for load_expert_slab's pread, so the existing
        // `disk=` field in the [PREFILL] line stays an apples-to-apples
        // comparison against the pre-streaming baseline.
        PERF_START(pf_stage);
#pragma omp parallel for schedule(static) reduction(+ : staged_bytes)
        for (int i = sc0; i < sc1; ++i) {
          const int e = used[(size_t)i];
          size_t gb, ub, db;
          const GGUFTensor *gM, *uM, *dM;
          if (!engine.expert_geometry_for(layer_idx, e, gb, ub, db, gM, uM,
                                          dM)) {
            stage_ok.store(false, std::memory_order_relaxed);
            continue;
          }
          // Packed layers store every expert in one tensor at
          // base + e*stride; an unpacked layer's tensor IS the expert's own,
          // so it must NOT be scaled (getting this wrong silently reads a
          // different expert's bytes -- no error path is taken).
          const uint64_t goff =
              unpacked_layer ? gM->offset : gM->offset + (uint64_t)e * gb;
          const uint64_t uoff =
              unpacked_layer ? uM->offset : uM->offset + (uint64_t)e * ub;
          const uint64_t doff =
              unpacked_layer ? dM->offset : dM->offset + (uint64_t)e * db;
          const uint8_t *gs = engine.model_file.mmap_bytes(*gM, goff, gb);
          const uint8_t *us = engine.model_file.mmap_bytes(*uM, uoff, ub);
          const uint8_t *ds = engine.model_file.mmap_bytes(*dM, doff, db);
          if (!gs || !us || !ds || gb + ub + db > stage_stride) {
            stage_ok.store(false, std::memory_order_relaxed);
            continue;
          }
          const int sl = slot_of[(size_t)cbase + (size_t)(i - sc0)];
          if (sl < 0)
            continue; // already VRAM-resident, nothing to ship
          uint8_t *dst = stage_base + (size_t)sl * stage_stride;
          std::memcpy(dst, gs, gb);
          std::memcpy(dst + gb, us, ub);
          std::memcpy(dst + gb + ub, ds, db);
          staged_bytes += gb + ub + db;
        }
        PERF_END(pf_stage, t_disk_read);
        perf().bytes_disk_read.fetch_add(staged_bytes,
                                         std::memory_order_relaxed);
        if (!stage_ok.load(std::memory_order_relaxed) ||
            !pstream.push(buf, n_miss)) {
          LOG_ERROR("[PREFILL-STREAM] layer=%d chunk@%d: staging failed -- "
                    "falling back to the cached path for this chunk.\n",
                    layer_idx, sc0);
          return false;
        }
        return true;
      };

      // Size the activation arena ONCE for the whole layer: sum(n) over any
      // chunk can never exceed total_pairs, so this is a hard upper bound
      // and the arena cannot grow (and therefore cannot reallocate under an
      // in-flight DMA) part-way through the chunk loop.
      if (!ggml_handled && !act_arena.resize((size_t)total_pairs * (size_t)D,
                                             "moe_prefill_activation_arena")) {
        LOG_WARN("[MEM-GUARD] grouped MoE activation arena refused at layer=%d "
                 "(%zu bytes); aborting this layer before any grouped output "
                 "is committed\n",
                 layer_idx, (size_t)total_pairs * (size_t)D * sizeof(float));
        return BatchedDispatchResult::Abort;
      }

      // The GLM-specific path also chunks the grouped GPU work when the
      // legacy mmap streaming path is off. This bounds the number of live
      // per-expert CUDA scratch buffers and lets the accumulator synchronize
      // once per small group, while retaining GLM's normal RAM-tier source.
      if (!stream_enabled && cache_grouped_prefill && n_used > 0) {
        stream_chunk =
            std::min(std::max(1, moe_prefill_stream_chunk()), n_used);
      }
      const int n_chunks = (!ggml_handled && n_used > 0)
                               ? (n_used + stream_chunk - 1) / stream_chunk
                               : 0;
      std::vector<char> chunk_ok((size_t)std::max(1, n_chunks), 0);
      // With device SwiGLU and device accumulation enabled, each expert
      // bucket can keep its own device scratch until the whole streaming
      // chunk has issued. That lets the gate/up, activation, down, and
      // scatter work remain ordered on CUDA streams without a host sync
      // after every bucket. The old path remains available whenever either
      // fast path is disabled.
      const bool grouped_async_enabled = accum_gpu && expert_gpu_activation();
      if (grouped_async_enabled &&
          (int)scratch.gpu_pool.size() < std::max(1, stream_chunk))
        scratch.gpu_pool.resize(std::max(1, stream_chunk));
      struct AsyncChunkHold {
        const ExpertLocation *loc = nullptr;
        MoEExpertBlock *exp = nullptr;
        int expert_id = -1;
        bool needs_release = false;
        bool from_active_layer = false;
      };
      std::vector<AsyncChunkHold> async_holds;
      std::array<bool, 8> async_gpu_used{};
      std::array<bool, 8> async_gpu_synced{};
      // Prime the pipeline so chunk 0's transfer is already in flight before
      // the loop body runs.
      if (stream_enabled && n_chunks > 0)
        chunk_ok[0] = stage_chunk(0) ? 1 : 0;

      for (int c = 0; c < n_chunks; ++c) {
        const int c0 = c * stream_chunk;
        const int c1 = std::min(n_used, c0 + stream_chunk);
        const int buf = c % PrefillExpertStream::NBUF;
        const bool grouped_async =
            grouped_async_enabled &&
            ((stream_enabled && chunk_ok[(size_t)c] != 0) ||
             (cache_grouped_prefill && !stream_enabled));
        async_holds.clear();
        async_holds.reserve((size_t)(c1 - c0));
        async_gpu_used.fill(false);
        async_gpu_synced.fill(false);

        // Kick the NEXT chunk's staging + H2D off BEFORE computing this one.
        // That is the entire point of the double buffer: c+1's transfer (and
        // the CPU memcpy feeding it) overlap c's GEMMs, instead of stage /
        // H2D / compute simply ADDING the way the legacy per-expert path
        // made them.
        if (stream_enabled && c + 1 < n_chunks)
          chunk_ok[(size_t)c + 1] = stage_chunk(c + 1) ? 1 : 0;

        const bool chunk_streamed = stream_enabled && chunk_ok[(size_t)c] != 0;
        if (chunk_streamed)
          pstream.gate_compute(buf);

        // Gather every expert's token rows for this chunk in one parallel
        // pass, into per-expert slices of the pinned arena.
        act_off.assign((size_t)(c1 - c0) + 1, 0);
        for (int i = c0; i < c1; ++i) {
          const int ee = used[(size_t)i];
          act_off[(size_t)(i - c0) + 1] =
              act_off[(size_t)(i - c0)] +
              (size_t)(offs[(size_t)ee + 1] - offs[(size_t)ee]);
        }
        if (!act_arena.resize(act_off.back() * (size_t)D,
                              "moe_prefill_activation_arena")) {
          LOG_WARN("[MEM-GUARD] grouped MoE activation arena grow refused at "
                   "layer=%d chunk=%d (%zu bytes); stopping grouped dispatch\n",
                   layer_idx, c, act_off.back() * (size_t)D * sizeof(float));
          return BatchedDispatchResult::Abort;
        }
        // Hoisted for the same reason as stage_base above: act_arena is
        // `static thread_local`, so an OMP worker calling act_arena.data()
        // would get its OWN empty instance, not this one. Cost me a second
        // SIGSEGV in the same session -- if you add another thread_local
        // buffer here, hoist its pointer too.
        float *const arena_base = act_arena.data();
#pragma omp parallel for schedule(static)
        for (int i = c0; i < c1; ++i) {
          const int ee = used[(size_t)i];
          const int *mine = slot.data() + offs[(size_t)ee];
          const int nn = offs[(size_t)ee + 1] - offs[(size_t)ee];
          float *dst = arena_base + act_off[(size_t)(i - c0)] * (size_t)D;
          for (int j = 0; j < nn; ++j)
            std::memcpy(dst + (size_t)j * D,
                        scratch.normed.data() + (size_t)(mine[j] / topk) * D,
                        (size_t)D * sizeof(float));
        }

        for (int i = c0; i < c1; ++i) {
          const int e = used[(size_t)i];
          const int n = offs[(size_t)e + 1] - offs[(size_t)e];
          if (n <= 0)
            continue;
          const int *my = slot.data() + offs[(size_t)e];

          // exp stays null on the streaming path -- there is no RAM slot at
          // all, which is the entire point (no claim_or_wait, no LRU scan, no
          // eviction, no temp-pin). Every later use is guarded on it.
          MoEExpertBlock *exp = nullptr;
          MoEExpertBlock shadow;
          ExpertLocation sloc;
          const ExpertLocation *loc = nullptr;
          bool needs_release = false, from_active = false;
          const GGUFTensor *gmp = nullptr, *ump = nullptr, *dmp = nullptr;

          bool vram_shadowed = false;
          if (chunk_streamed) {
            // Metadata only: type + the packed slab's slab-RELATIVE offsets
            // (gate=0, up=gb, down=gb+ub). That is exactly the layout this
            // chunk slot was staged in, so expert_ffn_gpu_issue's existing
            // `base + (up_meta.offset - gate_meta.offset)` pointer math works
            // against it unchanged.
            if (!engine.populate_vram_shadow(shadow, layer_idx, e)) {
              LOG_ERROR("[PREFILL-STREAM] layer=%d expert=%d: no geometry (%d "
                        "token(s) dropped).\n",
                        layer_idx, e, n);
              continue;
            }
            gmp = &shadow.gate_proj.meta;
            ump = &shadow.up_proj.meta;
            dmp = &shadow.down_proj.meta;
            const size_t cls = (size_t)(buf * stream_chunk) + (size_t)(i - c0);
            const ExpertLocation *rl = res_loc[cls];
            if (rl) {
              // Already in VRAM -- use it directly. hold_for_compute was taken
              // at classification time and is released after this expert's
              // kernels are issued, below.
              loc = rl;
              needs_release = true;
              from_active = false;
              gmp = &shadow.gate_proj.meta;
              ump = &shadow.up_proj.meta;
              dmp = &shadow.down_proj.meta;
              goto have_loc;
            }
            sloc.gpu_id = pstream.device_id;
            sloc.device_ptr = pstream.dev_slot(buf, slot_of[cls]);
            sloc.size = gmp->size_bytes + ump->size_bytes + dmp->size_bytes;
            sloc.layer_idx = layer_idx;
            sloc.expert_id = e;
            sloc.upload_ready = true;
            loc = &sloc;
          have_loc:;
          } else {
            // GLM5NEXT's grouped path may reach this branch with an expert
            // already resident in VRAM but with its RAM-tier copy absent.  A
            // get_expert() before the VRAM check would materialize a redundant
            // host copy and can issue an SSD read just to obtain metadata that
            // the GPU path never reads.  Check the VRAM shadow first.
            //
            // The shadow contains only projection types and packed-slab-
            // relative offsets.  expert_ffn_gpu_issue reads the actual bytes
            // from loc->device_ptr.  Unsupported types intentionally continue
            // to get_expert(), because their CPU fallback needs host bytes.
            if (cache_grouped_prefill && ram_mgr.vram_cache) {
              const ExpertLocation *resident =
                  ram_mgr.vram_cache->find_expert(layer_idx, e);
              if (resident && resident->device_ptr &&
                  engine.populate_vram_shadow(shadow, layer_idx, e) &&
                  gpu_type_supported(shadow.gate_proj.meta.type) &&
                  gpu_type_supported(shadow.up_proj.meta.type) &&
                  gpu_type_supported(shadow.down_proj.meta.type)) {
                exp = &shadow;
                loc = resident;
                gmp = &shadow.gate_proj.meta;
                ump = &shadow.up_proj.meta;
                dmp = &shadow.down_proj.meta;
                vram_shadowed = true;
                if (glm5next_grouped_prefill)
                  perf().n_glm_grouped_vram_shadow.fetch_add(
                      1, std::memory_order_relaxed);
              }
            }

            if (!vram_shadowed) {
              if (glm5next_grouped_prefill)
                perf().n_glm_grouped_host_lookup.fetch_add(
                    1, std::memory_order_relaxed);
              exp = ram_mgr.get_expert(engine, layer_idx, e);
              if (!exp || !exp->gate_proj.host_ptr || !exp->up_proj.host_ptr ||
                  !exp->down_proj.host_ptr) {
                LOG_ERROR(
                    "[MOE-BATCH] layer=%d expert=%d: no RAM block (%d token(s) "
                    "dropped this layer).\n",
                    layer_idx, e, n);
                continue;
              }

              // Pin for the WHOLE lifetime of this expert's use, not just the
              // CPU branch. The GPU cold-batch route (upload_expert /
              // upload_expert_active) DMAs straight out of exp->slab_buffer,
              // so a prefetch worker's victim scan reclaiming or realloc'ing
              // this slot mid-upload yields "cudaMemcpy H2D failed ... invalid
              // argument" on a host_src that was valid microseconds earlier.
              // Passing n (not 1) as S makes resolve_gpu_loc take the
              // cold-batch upload route far more often than the per-token path
              // ever does, which is why only the batched path shows this.
              ram_mgr.ram_pin(*exp);
              gmp = &exp->gate_proj.meta;
              ump = &exp->up_proj.meta;
              dmp = &exp->down_proj.meta;
            }
          }

          const GGUFTensor &gm = *gmp;
          const GGUFTensor &um = *ump;
          const GGUFTensor &dm = *dmp;
          // Hoisted out of the legacy branch's ExpertGpuContext, which no
          // longer exists on the streaming path -- same value either way.
          GpuTopology *const topo = ram_mgr.vram_cache;

          // This expert's slice of the chunk-wide pinned arena, gathered above.
          const float *const my_in =
              act_arena.data() + act_off[(size_t)(i - c0)] * (size_t)D;
          batch_out.assign((size_t)n * D, 0.0f);

          if (!chunk_streamed) {
            ExpertGpuContext gpu_ctx;
            gpu_ctx.topo = ram_mgr.vram_cache;
            gpu_ctx.loc = loc;
            gpu_ctx.layer_idx = layer_idx;
            gpu_ctx.expert_id = e;
            gpu_ctx.slab_buffer = exp->slab_buffer;
            // Real total, NOT exp->slab_capacity. This mirrors the per-token
            // path's own identical fix (see dispatch_expert_k below): a RAM
            // slot's slab_capacity is its GROW-ONLY allocation, which after an
            // LRU reuse can be larger than whatever smaller expert now occupies
            // it -- and this field is the byte count every synchronous
            // cold-miss GPU upload below reads. Getting it wrong uploads the
            // wrong number of bytes with no error path taken, i.e. silent
            // garbage weights. MEASURED: this was the cause of
            // moe_prefill_batched producing incoherent output on GLM-5.3
            // (2026-09-04), whose UD-IQ2_M layers have genuinely mixed
            // per-expert sizes (IQ2_XXS/IQ2_S gate, IQ3_XXS/IQ4_XS down) so the
            // two values routinely disagree. qwen4exp's UD-Q4_K_XL is
            // near-uniform, which is why it never surfaced there.
            gpu_ctx.slab_capacity = exp->gate_proj.meta.size_bytes +
                                    exp->up_proj.meta.size_bytes +
                                    exp->down_proj.meta.size_bytes;
            gpu_ctx.exp = exp;

            loc = expert_ffn_resolve_gpu_loc(
                &gpu_ctx, n, gm, um, dm, &needs_release, &from_active, ram_mgr,
                engine);
          }

          bool did_gpu = false;
          bool gpu_accumulated_on_device = false;
          if (loc) {
            // Same three helpers the per-token path uses -- every one of them
            // already takes S, so the ONLY difference here is passing n
            // instead of 1.
            GpuExpertScratch &gs =
                scratch.gpu_pool[grouped_async ? (size_t)(i - c0) : 0];
            if (expert_ffn_gpu_issue(gs, loc, my_in, gm, um, n, D, moe_inter,
                                     topo)) {
              if (!scratch.gate_buf.resize((size_t)n * moe_inter,
                                           "grouped_gate_buf") ||
                  !scratch.up_buf.resize((size_t)n * moe_inter,
                                         "grouped_up_buf")) {
                LOG_WARN("[MEM-GUARD] grouped MoE expert scratch refused at "
                         "layer=%d expert=%d; stopping grouped dispatch\n",
                         layer_idx, e);
                if (needs_release && topo)
                  topo->release_hold(layer_idx, e, loc->device_ptr,
                                     from_active);
                return BatchedDispatchResult::Abort;
              }
              expert_ffn_gpu_finish_stage1(
                  gs, loc, scratch.gate_buf.data(), scratch.up_buf.data(), gm,
                  dm, layer_idx, e, n, D, moe_inter, clamp_exp, topo);
              const int gid = loc->gpu_id;
              // Sync ONLY when the host is about to read the result. With
              // moe_gpu_accum the accumulate kernel launches on the legacy
              // default stream, ordered behind this expert's own down-proj
              // kernel, so nothing crosses back to the host and a sync here
              // would just stall the CPU for the chunk's in-flight H2D.
              // gs's device buffers are shared across experts, but the
              // upload (compute_stream) and the kernels (default stream)
              // implicitly synchronize with each other, so reuse is ordered
              // by the streams rather than by this call.
              if (grouped_async && gid >= 0 && gid < 8)
                async_gpu_used[(size_t)gid] = true;
              const bool skip_sync = grouped_async;
              if (!skip_sync) {
                PERF_START(bsync);
                cudaStreamSynchronize(0);
                PERF_END(bsync, t_gpu_batch_sync);
                perf().n_gpu_batch_sync_calls.fetch_add(
                    1, std::memory_order_relaxed);
              }
              bool accumulated = false;
              if (accum_gpu && gid >= 0 && gid < 8) {
                if (!accum_begun[gid])
                  accum_begun[gid] = engine.gpu->expert_accum_begin(
                      gid, accum_rows, D, tok_pair.data(), w_pair.data(), total_pairs);
                if (accum_begun[gid]) {
                  accumulated = engine.gpu->expert_accum_add(
                      gid, gs.d_down, offs[(size_t)e], n, D, accum_rows);
                  if (accumulated && grouped_async) {
                    // Keep both the VRAM hold and the RAM pin until the
                    // chunk-level synchronization below confirms that this
                    // expert's down/scatter work is complete.
                    async_holds.push_back({needs_release ? loc : nullptr,
                                           vram_shadowed ? nullptr : exp,
                                           e, needs_release, from_active});
                  } else if (accumulated) {
                    // Non-grouped path preserves the old per-expert release
                    // behavior because its stream has already been waited on.
                    if (needs_release && topo)
                      topo->release_hold(layer_idx, e, loc->device_ptr,
                                         from_active);
                  } else {
                    perf().n_moe_gpu_accum_fallback.fetch_add(
                        1, std::memory_order_relaxed);
                  }
                }
              }
              if (!accumulated) {
                if (grouped_async) {
                  // A fallback download needs the result now; this is rare
                  // (accumulator allocation or launch failure), so isolate its
                  // synchronization rather than compromising the normal
                  // grouped path.
                  cudaSetDevice(loc->gpu_id);
                  cudaStreamSynchronize(0);
                }
                expert_ffn_gpu_finish_stage2(
                    loc, batch_out.data(), layer_idx, e, n, D, topo,
                    needs_release, from_active, gs);
              }
              did_gpu = true;
              gpu_accumulated_on_device = accumulated;
            } else if (needs_release && topo) {
              topo->release_hold(layer_idx, e, loc->device_ptr, from_active);
            }
          }
          if (!did_gpu) {
            // The streaming path has no host copy of the weights (that is the
            // point), so a GPU-dispatch failure has to materialise one before
            // it can fall back. Pulling the RAM slot here rather than dropping
            // the expert keeps this a slow path, not a silently-wrong one --
            // an expert whose contribution is skipped corrupts the residual
            // stream with no error the output would reveal.
            // A VRAM shadow contains metadata only.  It deliberately has no
            // host_ptr fields, so it must never be accepted as the CPU
            // fallback source when its GPU dispatch loses the race.  Force a
            // fresh RAM-tier materialization for that case; otherwise this
            // branch would log a dropped expert and silently corrupt GLM's
            // grouped-prefill residual.
            MoEExpertBlock *fb = vram_shadowed ? nullptr : exp;
            bool fb_pinned = false;
            if (!fb) {
              fb = ram_mgr.get_expert(engine, layer_idx, e);
              if (fb) {
                ram_mgr.ram_pin(*fb);
                fb_pinned = true;
              }
            }
            if (fb && fb->gate_proj.host_ptr && fb->up_proj.host_ptr &&
                fb->down_proj.host_ptr) {
              if (!scratch.gate_buf.resize((size_t)n * moe_inter,
                                           "grouped_cpu_gate_buf") ||
                  !scratch.up_buf.resize((size_t)n * moe_inter,
                                         "grouped_cpu_up_buf")) {
                LOG_WARN(
                    "[MEM-GUARD] grouped MoE CPU-fallback scratch refused at "
                    "layer=%d expert=%d; preserving prior output\n",
                    layer_idx, e);
                if (fb_pinned)
                  ram_mgr.ram_unpin(*fb);
                return BatchedDispatchResult::Abort;
              }
              expert_ffn_cpu(
                  my_in, batch_out.data(), fb->gate_proj.host_ptr,
                  fb->gate_proj.meta, fb->up_proj.host_ptr, fb->up_proj.meta,
                  fb->down_proj.host_ptr, fb->down_proj.meta,
                  scratch.gate_buf.data(), scratch.up_buf.data(), n, D,
                  moe_inter, clamp_exp, layer_idx, e);
            } else {
              LOG_ERROR(
                  "[MOE-BATCH] layer=%d expert=%d: GPU dispatch failed and "
                  "no host fallback available (%d token(s) dropped).\n",
                  layer_idx, e, n);
            }
            if (fb_pinned)
              ram_mgr.ram_unpin(*fb);
          }

          // Scatter-accumulate: output[s] += w[s,k] * expert_out[i]
          // (skipped when the GPU already did it into the device accumulator)
          for (int i2 = 0; !gpu_accumulated_on_device && i2 < n; ++i2) {
            const int flat = my[i2];
            const int s2 = flat / topk;
            const float w = scratch.tok_topk_weights[(size_t)flat];
            float *dst = output + (size_t)s2 * D;
            const float *src = batch_out.data() + (size_t)i2 * D;
            if (ordered_prefill)
              std::memcpy(ordered_base + (size_t)flat * D, src,
                          (size_t)D * sizeof(float));
            else
              for (int d = 0; d < D; ++d)
                dst[d] += w * src[d];
          }

          const bool deferred_unpin =
              grouped_async && gpu_accumulated_on_device;
          if (exp && !vram_shadowed && !deferred_unpin)
            ram_mgr.ram_unpin(*exp);
        } // per-expert

        if (grouped_async) {
          // One wait per vault GPU for the whole chunk, instead of one wait
          // per expert bucket. The default-stream kernels are already
          // ordered behind each bucket's activation/down/scatter work.
          for (int gid = 0; gid < 8; ++gid) {
            if (!async_gpu_used[(size_t)gid])
              continue;
            cudaSetDevice(gid);
            PERF_START(bsync_chunk);
            cudaStreamSynchronize(0);
            PERF_END(bsync_chunk, t_gpu_batch_sync);
            perf().n_gpu_batch_sync_calls.fetch_add(1,
                                                    std::memory_order_relaxed);
            async_gpu_synced[(size_t)gid] = true;
          }
          for (const AsyncChunkHold &hold : async_holds) {
            if (hold.needs_release && hold.loc && ram_mgr.vram_cache)
              ram_mgr.vram_cache->release_hold(layer_idx, hold.expert_id,
                                               hold.loc->device_ptr,
                                               hold.from_active_layer);
            if (hold.exp)
              ram_mgr.ram_unpin(*hold.exp);
          }
        }

        // Signal that this chunk's compute is issued, so the buffer can be
        // recycled two chunks from now. Recorded on the legacy default
        // stream, which is where every cuda/ matmul actually launched.
        if (chunk_streamed)
          pstream.mark_computed(buf);
      } // per-chunk
      // One [S,D] download per GPU for this whole layer, ADDED onto whatever
      // CPU-dispatched experts already wrote into `output`.
      for (int g = 0; g < 8; ++g)
        if (accum_begun[g] && !engine.gpu->expert_accum_end(
                g, ordered_prefill ? ordered_base : output, accum_rows, D))
          perf().n_moe_gpu_accum_fallback.fetch_add(1,
                                                    std::memory_order_relaxed);
      if (ordered_prefill) {
        const float *const route_weights = scratch.tok_topk_weights.data();
#pragma omp parallel for schedule(static)
        for (int token = 0; token < S; ++token) {
          float *dst = output + (size_t)token * D;
          for (int k = 0; k < topk; ++k) {
            const size_t flat = (size_t)token * topk + k;
            const float *src = ordered_base + flat * D;
            const __m512 w = _mm512_set1_ps(route_weights[flat]);
            int d = 0;
            for (; d + 16 <= D; d += 16)
              _mm512_storeu_ps(dst + d, _mm512_fmadd_ps(
                  w, _mm512_loadu_ps(src + d), _mm512_loadu_ps(dst + d)));
            for (; d < D; ++d) dst[d] += route_weights[flat] * src[d];
          }
        }
        LOG_INFO("[DSV41-GROUPED-PREFILL] layer=%d tokens=%d experts=%d "
                 "chunk=%d ordered_routes=1 graph-owned-MoE-bytes=0\n",
                 layer_idx, S, n_used, stream_chunk);
      }

      // The shared expert is not part of the routed expert buckets above.
      // The legacy path executes it inside its per-token loop, but the
      // batched path skips that loop entirely once batched_handled becomes
      // true. Run the shared expert over all S normalized token rows here so
      // enabling moe_prefill_batched preserves the model's full FFN:
      // routed top-k contribution PLUS the always-active shared expert.
      const std::string shared_prefix =
          "blk." + std::to_string(layer_idx) + ".";
      if (!shared_device_handled && c.n_expert_shared > 0 &&
          engine.resident_layers.count(shared_prefix + "ffn_gate_shexp.weight"))
        run_qwen4exp_shared_host(engine, layer_idx, S, D, moe_inter,
                                 scratch.normed.data(), output, scratch, c);
      batched_handled = true;
    }
  }

  // Decode-only grouped routed FFN. The ordinary grouped path above
  // is intentionally prefill-only because its expert-major buckets are built
  // for many source rows. For S==1, the same GGML MUL_MAT_ID graph is still
  // valuable: it collapses this token's up-to-topk expert cache lookup,
  // upload, dispatch, and host accumulation into one graph submission.
  //
  // Device-HC is preferred. If exact device-HC is disabled because its larger
  // Q8 HC copies do not fit beside the VRAM cache, use the same grouped graph
  // with the already-computed host HC activation. That costs one small input
  // upload and result readback per layer, but avoids the old per-expert GPU
  // dispatch/synchronization sequence. This remains explicitly Qwen-only;
  // MiMo uses the host-input form directly: it has no HC or shared expert,
  // so the runner copies the final routed sum back into `output` itself.
  const bool qwen_decode_grouped_device =
      engine.qwen4exp_decode_ggml_moe &&
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      engine.family_ops.uses_hyper_connections && qwen4exp_hc_device_live() &&
      engine.qwen4exp_device_hc_ffn_input && moe_ggml_mul_mat_id();
  const bool qwen_decode_grouped_host =
      engine.qwen4exp_decode_ggml_moe &&
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      engine.family_ops.uses_hyper_connections && !qwen_decode_grouped_device &&
      moe_ggml_mul_mat_id();
  const bool mimo_decode_grouped_host =
      engine.family_ops.family == ModelFamily::MiMo2 &&
      moe_ggml_mul_mat_id();
  if (!batched_handled && S == 1 && topk > 0 && n_experts > 0 &&
      (qwen_decode_grouped_device || qwen_decode_grouped_host ||
       mimo_decode_grouped_host) &&
      clamp_exp == 0.0f && !is_cpu_affinity_layer(layer_idx) &&
      ram_mgr.vram_cache &&
      !engine.qwen4exp_option_b_layer_enabled(layer_idx) &&
      !engine.qwen4exp_cpu_offload_layer_enabled(layer_idx)) {
    std::vector<int> used;
    used.reserve((size_t)n_experts);
    for (int e = 0; e < n_experts; ++e)
      if (scratch.expert_freq[(size_t)e] > 0)
        used.push_back(e);

    if (!used.empty()) {
      const void *grouped_device_input =
          qwen_decode_grouped_device ? engine.qwen4exp_device_hc_ffn_input
                                     : nullptr;
      const void *grouped_moe_output = nullptr;
      const bool handled = run_ggml_moe_layer(
          engine, ram_mgr, ram_mgr.vram_cache, layer_idx, S, D, moe_inter, topk,
          n_experts, used, scratch.tok_topk_ids.data(),
          scratch.tok_topk_weights.data(), scratch.normed.data(), output,
          moe_prefill_stream_chunk(), engine.layer_is_unpacked(layer_idx),
          (qwen_decode_grouped_device || qwen_decode_grouped_host)
              ? &grouped_moe_output
              : nullptr,
          grouped_device_input, nullptr, nullptr,
          /*allow_active_layer_sources=*/true,
          (qwen_decode_grouped_device || qwen_decode_grouped_host)
              ? &scratch.moe_device_runner
              : nullptr);
      if (handled) {
        bool shared_device_handled = false;
        if (qwen_decode_grouped_device || qwen_decode_grouped_host) {
          // Keep the routed runner's output separate from the optional shared
          // graph result.  In host-HC mode the shared graph can successfully
          // queue a combined device result but still fail its host readback;
          // the fallback must then recover the original routed result through
          // the routed runner, not pass the shared runner's pointer to it.
          const void *routed_device_output = grouped_moe_output;
          scratch.moe_device_output = grouped_moe_output;
          scratch.moe_device_output_ready = grouped_moe_output != nullptr;
          if (moe_ggml_shared_expert() && c.n_expert_shared > 0) {
            const void *shared_input =
                qwen_decode_grouped_device
                    ? engine.qwen4exp_device_hc_ffn_input
                    : (scratch.moe_device_runner
                           ? scratch.moe_device_runner->device_input_tensor()
                           : nullptr);
            const void *combined = nullptr;
            shared_device_handled = run_qwen4exp_shared_device(
                engine, layer_idx, S, D, moe_inter, shared_input,
                scratch.moe_device_output, c, &combined);
            if (shared_device_handled) {
              scratch.moe_device_output = combined;
              scratch.moe_shared_device_output = true;
              // Even when the HC pre-mix was also exported to the host for
              // routing/cache bookkeeping, keep this routed+shared result on
              // the device. The layer loop below passes it to
              // qwen4exp_hc_post_combine_device(), which can update the
              // device-authoritative HC state without a per-layer D2H wait.
              // If that device post-combine fails, the existing fallback
              // recovers the routed result and runs the host shared path.
            }
          }
          if (qwen_decode_grouped_host) {
            // A successful shared graph already copied the combined result to
            // `output`; do not fall through and interpret that shared-runner
            // pointer as a routed-runner output.  Only recover the routed
            // result when the shared graph was unavailable or its combined
            // host copy failed.
            if (!shared_device_handled) {
              if (!scratch.moe_device_runner ||
                  !scratch.moe_device_runner->copy_device_output_to_host(
                      scratch.moe_device_output, output)) {
                LOG_ERROR("[GGML-MOE] Qwen decode layer=%d: routed output "
                          "recovery failed\n",
                          layer_idx);
              }
              if (moe_ggml_shared_expert() && c.n_expert_shared > 0)
                run_qwen4exp_shared_host(engine, layer_idx, S, D, moe_inter,
                                         scratch.normed.data(), output, scratch,
                                         c);
            }
          } else if (qwen_decode_grouped_device && !shared_device_handled &&
                     c.n_expert_shared > 0) {
            // Device-HC fallback must recover the HC input before using the
            // established host shared-expert path.
            if (!GgmlHcRunner::instance().copy_pre_mix_to_host(
                    engine.qwen4exp_device_hc_ffn_input, scratch.normed.data(),
                    S)) {
              LOG_ERROR("[GGML-SHARED] decode layer=%d: HC input recovery "
                        "failed; shared host fallback is unsafe\n",
                        layer_idx);
            }
            if (!scratch.moe_device_runner ||
                !scratch.moe_device_runner->copy_device_output_to_host(
                    scratch.moe_device_output, output)) {
              LOG_ERROR("[GGML-MOE] Qwen decode layer=%d: routed output "
                        "recovery failed\n",
                        layer_idx);
            }
            run_qwen4exp_shared_host(engine, layer_idx, S, D, moe_inter,
                                     scratch.normed.data(), output, scratch, c);
          }
          // In the device-HC case the routed+shared graph result is the
          // actual input to qwen4exp_hc_post_combine_device(), which runs
          // later in the layer loop.  Do not clear the handoff here: doing
          // so silently skipped the MoE contribution to the device HC state
          // and left the next layer's router reading stale state.  The host
          // grouped path has already recovered/copy-combined its result and
          // can keep the old cleanup behavior.
          if (!(qwen_decode_grouped_device && shared_device_handled)) {
            scratch.moe_device_output = nullptr;
            scratch.moe_device_runner = nullptr;
            scratch.moe_shared_device_output = false;
            scratch.moe_device_output_ready = false;
          }
        }
        batched_handled = true;
        static bool logged_qwen_decode_graph_device = false;
        static bool logged_qwen_decode_graph_host = false;
        static bool logged_mimo_decode_graph_host = false;
        if (mimo_decode_grouped_host && !logged_mimo_decode_graph_host) {
          LOG_INFO("[GGML-MOE] MiMo2 grouped decode graph active (S=1, "
                   "host input/output)\n");
          logged_mimo_decode_graph_host = true;
        }
        bool &logged = qwen_decode_grouped_device
                           ? logged_qwen_decode_graph_device
                           : logged_qwen_decode_graph_host;
        if (!mimo_decode_grouped_host && !logged) {
          LOG_INFO("[GGML-MOE] Qwen4Exp grouped decode graph active (S=1, "
                   "%s HC input)\n",
                   qwen_decode_grouped_device ? "device" : "host");
          logged = true;
        }
      }
    }
  }


  return batched_handled ? BatchedDispatchResult::Handled
                         : BatchedDispatchResult::Continue;
}
