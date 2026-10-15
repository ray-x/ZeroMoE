// GLM5Next full graph: FFN (cold lane, hot/cold split) and whole-model graph construction.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_glm5next_graph_impl.hpp.

#include <future>
#include "ggml_glm5next_graph_impl.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

using namespace glm5next_graph;

namespace glm5next_graph {

// Builds the lane for one layer's cold experts and the graph nodes around it:
// the fork (activation snapshot + route readback) is added to the graph here;
// the caller consumes lane->gpu_input as the hot branches' input and lane->gpu_output
// as the cold branch's per-route weighted output, and must expand lane->cpu_marker
// after the layer's GPU work and before the merge. Returns nullptr when not applicable.
EarlyReadyLane *make_cold_lane(FullGraph &g, ggml_context *ctx, int layer,
                               ggml_tensor *cur, ggml_tensor *ids,
                               ggml_tensor *weights, ggml_tensor *cold_map,
                               ggml_tensor *cold_mask,
                               const moe_routed::ExpertWeights &cold_w,
                               float clamp)  {
  if (!g.decode_mode || g.S != 1 || !g.sched || !g.cuda || !g.cpu ||
      !cold_w.valid() || !cold_map || !cold_mask || cur->type != GGML_TYPE_F32)
    return nullptr;
  if (!ggml_backend_sched_early_cpu_supported ||
      ggml_backend_sched_early_cpu_supported() != 1) {
    LOG_WARN("[GGML-GLM5NEXT-EARLY-READY] layer %d: linked ggml has no early-CPU scheduler hook\n", layer);
    return nullptr;
  }
  const auto &c = g.engine->model_file.architecture_config;
  auto lane = std::make_unique<EarlyReadyLane>();
  lane->swiglu_clamp = clamp;
  lane->swiglu_clamp_fused = swiglu_clamp_fused();
  // The lane reads its routing tables on the host: copy them out of the graph's
  // (device-resident) constant tensors once, at build time.
  const auto host_table = [&](ggml_tensor *src) -> ggml_tensor * {
    ggml_tensor *t = ggml_new_tensor_3d(ctx, src->type, src->ne[0], src->ne[1], src->ne[2]);
    if (!t)
      return nullptr;
    ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(g.cpu, ggml_nbytes(t));
    if (!buf)
      return nullptr;
    lane->extra_buffers.push_back(buf);
    if (ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf)) != GGML_STATUS_SUCCESS)
      return nullptr;
    std::vector<uint8_t> tmp(ggml_nbytes(t));
    ggml_backend_tensor_get(src, tmp.data(), 0, tmp.size());
    ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size());
    return t;
  };
  ggml_tensor *remap = host_table(cold_map);
  ggml_tensor *mask = host_table(cold_mask);
  if (!remap || !mask)
    return nullptr;
  if (!lane->prepare(g.cpu, g.device, g.D, c.topk, cold_w, remap, mask))
    return nullptr;
  lane->trace_layer = layer;

  ggml_tensor *dst = ggml_dup_tensor(ctx, cur);
  lane->device_buffer = ggml_backend_alloc_buffer(g.cuda, ggml_nbytes(dst));
  if (!lane->device_buffer ||
      ggml_backend_tensor_alloc(lane->device_buffer, dst,
                                ggml_backend_buffer_get_base(lane->device_buffer)) != GGML_STATUS_SUCCESS)
    return nullptr;
  ggml_tensor *fork = ggml_cpy(ctx, cur, dst);
  fork->src[2] = ggml_cont(ctx, ids);
  fork->src[3] = ggml_cont(ctx, weights);
  fork->src[4] = nullptr;  // no positions tensor: skip the staleness check
  fork->extra = lane.get();
  ggml_set_name(fork, ("glm5next_early_ready_fork_l" + std::to_string(layer)).c_str());
  ggml_backend_sched_set_tensor_backend(g.sched, dst, g.cuda);
  ggml_backend_sched_set_tensor_backend(g.sched, fork, g.cuda);
  lane->gpu_input = fork;
  ggml_build_forward_expand(g.graph, fork);

  ggml_tensor *cold_leaf = ggml_dup_tensor(ctx, lane->output);
  cold_leaf->buffer = lane->output->buffer;
  cold_leaf->data = lane->output->data;
  ggml_tensor *marker = ggml_map_custom1_inplace(ctx, cold_leaf, EarlyReadyLane::fallback, 1, lane.get());
  marker->extra = lane.get();
  ggml_set_name(marker, ("glm5next_early_ready_cold_l" + std::to_string(layer)).c_str());
  ggml_backend_sched_set_tensor_backend(g.sched, cold_leaf, g.cpu);
  ggml_backend_sched_set_tensor_backend(g.sched, marker, g.cpu);
  if (!lane->async_output_copy)
    return nullptr;  // the GLM merge consumes the uploaded copy
  lane->cpu_marker = marker;
  if (!lane->prepare_gpu_output(g.cuda, ctx))
    return nullptr;
  ggml_set_name(lane->gpu_output, ("glm5next_early_ready_output_l" + std::to_string(layer)).c_str());
  ggml_backend_sched_set_tensor_backend(g.sched, lane->gpu_output, g.cuda);

  if (offload_layer(layer)) {
    OffloadState *o = get_offload(*g.engine);
    if (!o)
      return nullptr;
    if (split_load_wanted() && o->store->async_loads()) {
      // One in-flight load per lane: the lane runs its layer once per token.
      struct InFlight {
        moestream::offload::ExpertResidency::Plan plan;
        std::future<moestream::offload::ExpertResidency::Ensure> result;
        std::chrono::steady_clock::time_point t0;
        int n = 0;
      };
      auto flight = std::make_shared<InFlight>();
      lane->begin_load = [o, layer, flight](const int32_t *experts, int n, std::vector<int32_t> &pending) {
        flight->t0 = std::chrono::steady_clock::now();
        flight->n = n;
        flight->plan = o->store->begin_ensure(layer, experts, n, moestream::offload::ExpertCache::Kind::Demand);
        pending = flight->plan.to_load;
        pending.insert(pending.end(), flight->plan.to_wait.begin(), flight->plan.to_wait.end());
        if (pending.empty() && flight->plan.r.errors == 0) {
          // Nothing to read: account for the hits the same way finish would.
          flight->result = std::async(std::launch::deferred, [o, flight] {
            return o->store->finish_ensure(flight->plan);
          });
          return true;
        }
        flight->result = std::async(std::launch::async, [o, flight] {
          return o->store->finish_ensure(flight->plan);
        });
        return true;
      };
      lane->finish_load = [o, layer, flight]() {
        const auto r = flight->result.get();
        auto &ct = o->store->counters();
        ct.calls.fetch_add(1, std::memory_order_relaxed);
        ct.routes.fetch_add((uint64_t)flight->n, std::memory_order_relaxed);
        ct.cold_routes.fetch_add((uint64_t)flight->n, std::memory_order_relaxed);
        ct.misses.fetch_add((uint64_t)r.misses, std::memory_order_relaxed);
        ct.loads.fetch_add((uint64_t)r.loads, std::memory_order_relaxed);
        ct.evictions.fetch_add((uint64_t)r.evictions, std::memory_order_relaxed);
        ct.load_us.fetch_add((uint64_t)(r.load_ms * 1000.0), std::memory_order_relaxed);
        ct.cb_us.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - flight->t0).count(),
                           std::memory_order_relaxed);
        if (r.errors > 0) {
          ct.errors.fetch_add((uint64_t)r.errors, std::memory_order_relaxed);
          LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] layer %d: %d expert load error(s); aborting rather than computing with "
                    "missing experts\n", layer, r.errors);
          o->store->report("fatal");
          std::abort();
        }
        return true;
      };
    } else
    lane->before_compute = [o, layer](const int32_t *experts, int n) {
      const auto t0 = std::chrono::steady_clock::now();
      const auto r = o->store->ensure(layer, experts, n, moestream::offload::ExpertCache::Kind::Demand);
      auto &ct = o->store->counters();
      ct.calls.fetch_add(1, std::memory_order_relaxed);
      ct.routes.fetch_add((uint64_t)n, std::memory_order_relaxed);
      ct.cold_routes.fetch_add((uint64_t)n, std::memory_order_relaxed);
      ct.misses.fetch_add((uint64_t)r.misses, std::memory_order_relaxed);
      ct.loads.fetch_add((uint64_t)r.loads, std::memory_order_relaxed);
      ct.evictions.fetch_add((uint64_t)r.evictions, std::memory_order_relaxed);
      ct.load_us.fetch_add((uint64_t)(r.load_ms * 1000.0), std::memory_order_relaxed);
      ct.cb_us.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count(),
                         std::memory_order_relaxed);
      if (r.errors > 0) {
        ct.errors.fetch_add((uint64_t)r.errors, std::memory_order_relaxed);
        LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] layer %d: %d expert load error(s); aborting rather than computing with "
                  "missing experts\n", layer, r.errors);
        o->store->report("fatal");
        std::abort();
      }
      return true;
    };
  }
  if (gpu_assist_slots() > 0 && offload_layer(layer) && gpu_assist_layer_ok(g, layer)) {
    if (OffloadState *o = get_offload(*g.engine)) {
      // One backend + one ring per assist device, shared by every lane of this graph; freed with the last lane.
      // Device 0: MOESTREAM_GLM5NEXT_ASSIST_DEVICE (default the trunk's device), GPU_ASSIST slots. Optional device 1:
      // MOESTREAM_GLM5NEXT_ASSIST2_DEVICE / _ASSIST2_SLOTS -- its own PCIe link, so it adds to device 0's copy bandwidth.
      static std::weak_ptr<GpuAssistShared> shared_weak[2];
      const auto make_ring = [&](std::weak_ptr<GpuAssistShared> &weak, int dev,
                                 int slots) -> std::shared_ptr<GpuAssistShared> {
        if (std::shared_ptr<GpuAssistShared> have = weak.lock())
          return have;
        auto s = std::make_shared<GpuAssistShared>();
        s->be = ggml_backend_cuda_init(dev);
        s->slots = slots;
        bool ok = s->be != nullptr;
        for (int p = 0; ok && p < 3; ++p) {
          for (int l = o->first; l < o->first + o->n_layers; ++l)
            if (gpu_assist_layer_ok(g, l))  // the ring only has to fit layers that use it
              s->plane_cap[p] = std::max<size_t>(s->plane_cap[p], (size_t)o->arena->plane_bytes(l, p));
          s->ring[p] = ggml_backend_alloc_buffer(s->be, s->plane_cap[p] * (size_t)s->slots + 4096);
          ok = s->ring[p] != nullptr;
        }
        if (!ok)
          return nullptr;
        weak = s;
        return s;
      };
      const int assist_dev = moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_ASSIST_DEVICE", g.device);
      std::shared_ptr<GpuAssistShared> shared = make_ring(shared_weak[0], assist_dev, gpu_assist_slots());
      std::shared_ptr<GpuAssistShared> shared2;
      const int assist2_dev = gpu_assist2_device();
      if (assist2_dev >= 0 && assist2_dev != assist_dev && gpu_assist2_slots() > 0)
        shared2 = make_ring(shared_weak[1], assist2_dev, gpu_assist2_slots());
      if (shared && lane->prepare_gpu_assist(cold_w, shared, 0)) {
        if (shared2 && !lane->prepare_gpu_assist(cold_w, shared2, 1)) {
          lane->assists[1].graph = nullptr;
          LOG_WARN("[GGML-GLM5NEXT-EARLY-READY] layer=%d second GPU assist unavailable\n", layer);
        }
        lane->expert_planes = [o, layer](int expert, const void *planes[3]) {
          if (!o->residency->ready({layer, expert}) || !o->store->is_pinned(layer, expert))
            return false;  // an unpinned slot would make the async copy a blocking staged one
          for (int k = 0; k < 3; ++k)
            planes[k] = o->arena->ptr(layer, expert, k);
          return planes[0] && planes[1] && planes[2];
        };
      } else {
        lane->assist.graph = nullptr;
        LOG_WARN("[GGML-GLM5NEXT-EARLY-READY] layer=%d GPU assist unavailable\n", layer);
      }
    }
  }
  EarlyReadyLane *raw = lane.get();
  g.early_ready.emplace(layer, std::move(lane));
  LOG_INFO("[GGML-GLM5NEXT-EARLY-READY] layer=%d private CPU lane enabled\n", layer);
  return raw;
}

ggml_tensor *build_ffn(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *forced_ids)  {
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
  if (partial_weights && moestream::graphcfg::env("MOESTREAM_GLM5NEXT_DEBUG_PARTIAL")) {
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

  const moe_routed::RouterSpec rspec =
      family_router_spec(g.engine->family_ops.family, c, g.S);
  ggml_tensor *probs = moe_routed::gate_probs(ctx, rspec, router, cur);
  // DeepSeek-V4's first hash-routed layers use a token-id -> expert table
  // for selection, while the learned router still supplies the unbiased
  // weights.  The table is prepared as a host input by the V4 graph builder;
  // every other layer keeps GGML's normal biased top-k node.
  ggml_tensor *ids =
      forced_ids ? forced_ids : moe_routed::select_ids(ctx, rspec, probs, bias);
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
  ggml_tensor *weights = moe_routed::route_weights(ctx, rspec, probs, ids);
  ggml_build_forward_expand(g.graph, weights);

  float clamp = 0.0f;
  if (layer >= 0 && layer < (int)c.swiglu_clamp_exp.size())
    clamp = c.swiglu_clamp_exp[(size_t)layer];
  ggml_tensor *input3 = ggml_reshape_3d(ctx, cur, g.D, 1, g.S);
  if (g.partial_expert_residency && !prepare_partial_hot(g, ctx, layer))
    return nullptr;
  if (g.partial_expert_residency && !prepare_partial_hot1(g, ctx, layer))
    return nullptr;
  if (partial_weights && moestream::graphcfg::env("MOESTREAM_GLM5NEXT_DEBUG_PARTIAL")) {
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

  // First expert id of the second GPU's share of the cold experts (0 = not split). MOESTREAM_GLM5NEXT_PREFILL_SPLIT is the
  // main GPU's share of each layer's cold experts (0.63 ~ the 3090 x16 / 5060 Ti x8 PCIe Gen3 bandwidth ratio; default 1 = off: both
  // links do copy concurrently, but per layer attention + the miss node (~60 ms) still run before the copies, so the
  // wall time did not drop -- it needs the expert copies prefetched ahead of the router).
  int split_at = 0;
  if (!g.decode_mode && g.S >= 64 && g.gpu1 && offload_layer(layer) && !gate_up && ge && ue && de) {
    static const double ratio = [] {
      const char *e = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PREFILL_SPLIT");
      return e ? std::atof(e) : 1.0;
    }();
    if (const PartialExpertLayer *pl = partial_layer(g, layer); pl && ratio > 0.0 && ratio < 1.0 && pl->cold.size() >= 16) {
      std::vector<int> sorted(pl->cold.begin(), pl->cold.end());
      std::sort(sorted.begin(), sorted.end());
      split_at = sorted[(size_t)(ratio * (double)sorted.size())];
      if (split_at <= 0 || split_at >= (int)ge->ne[2])
        split_at = 0;
    }
  }
  const auto cold_split_applicable = [&](FullGraph &, int) { return true; };

  const auto run_branch = [&](ggml_tensor *branch_gate_up,
                              ggml_tensor *branch_gate, ggml_tensor *branch_up,
                              ggml_tensor *branch_down, ggml_tensor *branch_ids,
                              ggml_tensor *branch_weights,
                              ggml_backend_t pin_be = nullptr) -> ggml_tensor * {
    if (!branch_down || !branch_ids || !branch_weights ||
        (!branch_gate_up && (!branch_gate || !branch_up)))
      return nullptr;
    // Merged gate_up matches llama.cpp's graph: one routed projection, then
    // zero-copy views feed SwiGLU.
    moe_routed::ExpertWeights ew;
    ew.gate_up = branch_gate_up;
    ew.gate = branch_gate;
    ew.up = branch_up;
    ew.down = branch_down;
    const moe_routed::FfnNodes ffn = moe_routed::expert_ffn(
        ctx, ew, input3, branch_ids, clamp, swiglu_clamp_fused());
    ggml_tensor *experts = ggml_mul(ctx, ffn.down, branch_weights);
    ggml_tensor *summed = moe_routed::sum_topk(ctx, experts, g.D, g.S, c.topk);
    // The weighting and the sum over the routed experts stay on the GPU that ran the experts: left to the scheduler they ran
    // on the main GPU, so the unreduced [D, topk, S] result (105 MB for 805 tokens) crossed between the GPUs through the host
    // (56 ms per layer) instead of the [D, S] sum (13 MB).
    static const bool pin_on = moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_PIN_BRANCH_SUM", 1) != 0;
    if (pin_be && g.sched && pin_on && !g.decode_mode) {
      ggml_backend_sched_set_tensor_backend(g.sched, experts, pin_be);
      ggml_backend_sched_set_tensor_backend(g.sched, summed, pin_be);
    }
    return summed;
  };

  // Shared expert FFN output (nullptr when the layer has none). With a CPU lane it is built and
  // scheduled before the lane's marker so the GPU runs it while the CPU computes the cold experts.
  const auto build_shared = [&]() -> ggml_tensor * {
    ggml_tensor *sg = resident_tensor(g, ctx, p + "ffn_gate_shexp.weight");
    ggml_tensor *su = resident_tensor(g, ctx, p + "ffn_up_shexp.weight");
    ggml_tensor *sd = resident_tensor(g, ctx, p + "ffn_down_shexp.weight");
    if (!(sg && su && sd))
      return nullptr;
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
    return ggml_mul_mat(ctx, sd, a);
  };
  ggml_tensor *shared_out = nullptr;

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
    ggml_tensor *stage_slots = nullptr, *stage_m = nullptr;
    if (has_cold) {
      ggml_tensor *cold_map = partial_map(g, ctx, layer, false, false);
      ggml_tensor *cold_mask = partial_map(g, ctx, layer, false, true);
      if (!cold_map || !cold_mask)
        return nullptr;
      cold_in = moe_partial::make_branch_inputs(ctx, cold_map, cold_mask, route_ids,
                                                weights, c.topk, g.S);
      if (offload_layer(layer)) {
        if (stage) {
          // Packed result [cold ids | slots | staged flags]: views feed the CPU branch (cold ids, keep-weights) and the ring branch.
          ggml_tensor *packed = stage_miss_node(g, ctx, *stage, layer, cold_in.ids, route_ids);
          if (!packed)
            return nullptr;
          const int64_t k = c.topk;
          const size_t row = (size_t)(3 * k) * sizeof(int32_t);
          cold_in.ids = ggml_view_2d(ctx, packed, k, 1, row, 0);
          stage_slots = ggml_view_2d(ctx, packed, k, 1, row, (size_t)k * sizeof(int32_t));
          ggml_tensor *flags = ggml_view_2d(ctx, packed, k, 1, row, (size_t)(2 * k) * sizeof(int32_t));
          stage_m = ggml_reshape_3d(ctx, ggml_cast(ctx, flags, GGML_TYPE_F32), 1, k, 1);
          // Zero the weights of routes the ring serves. This runs on the GPU, after the CPU cold chain, so it adds no scheduler split.
          cold_in.weights = ggml_mul(ctx, cold_in.weights, ggml_scale_bias(ctx, stage_m, -1.0f, 1.0f));
        } else {
          cold_in.ids = offload_miss_node(g, ctx, layer, cold_in.ids);
          if (!cold_in.ids)
            return nullptr;
        }
      }
    }
    EarlyReadyLane *lane = nullptr;
    if (has_cold && early_ready_wanted() && !stage && (partial->hot1.empty() || lane_with_gpu1_wanted()) &&
        family_supports_cpu_lane(g.engine->family_ops.family)) {
      moe_routed::ExpertWeights cold_w;
      cold_w.gate_up = gate_up;
      cold_w.gate = ge;
      cold_w.up = ue;
      cold_w.down = de;
      lane = make_cold_lane(g, ctx, layer, cur, ids, weights,
                            partial_map(g, ctx, layer, false, false),
                            partial_map(g, ctx, layer, false, true), cold_w, clamp);
    }
    if (lane) {
      // fork -> hot GPU work -> shared expert -> CPU lane marker -> merge. The marker
      // is expanded after everything the GPU can do without the cold result.
      input3 = ggml_reshape_3d(ctx, lane->gpu_input, g.D, 1, g.S);
      ggml_tensor *hot_out = nullptr;
      if (has_hot) {
        hot_out = run_branch(partial->hot_gate_up, partial->hot_gate,
                             partial->hot_up, partial->hot_down, hot_in.ids,
                             hot_in.weights);
        if (!hot_out)
          return nullptr;
        ggml_build_forward_expand(g.graph, hot_out);
      }
      shared_out = build_shared();
      if (shared_out)
        ggml_build_forward_expand(g.graph, shared_out);
      ggml_build_forward_expand(g.graph, lane->cpu_marker);
      ggml_tensor *cold_out =
          moe_routed::sum_topk(ctx, lane->gpu_output, g.D, g.S, c.topk);
      out = hot_out ? ggml_add(ctx, hot_out, cold_out) : cold_out;
    } else {
      out = moe_partial::combine_hot_cold(
          ctx, has_hot, has_cold, hot_in, cold_in,
          [&](const moe_partial::BranchInputs &in) {
            return run_branch(partial->hot_gate_up, partial->hot_gate,
                              partial->hot_up, partial->hot_down, in.ids,
                              in.weights);
          },
          [&](const moe_partial::BranchInputs &in) -> ggml_tensor * {
            // Batched prefill: split the cold experts of the layer between the two GPUs, so both PCIe links copy at
            // once (the second GPU's share runs first so its weight copies are issued before the main GPU's launch).
            if (split_at > 0 && cold_split_applicable(g, layer)) {
              const int n_exp = (int)ge->ne[2];
              // Every host node of the split is created and scheduled before either GPU branch, so the scheduler runs
              // them as one CPU split and does not synchronize the first GPU branch before launching the second.
              ggml_tensor *split_ids[2], *split_mask[2];
              for (int h = 0; h < 2; ++h) {
                split_ids[h] = cold_split_node(ctx, in.ids, split_at, h);
                split_mask[h] = ggml_cast(ctx, cold_split_node(ctx, in.ids, split_at, h + 2), GGML_TYPE_F32);
                if (!split_ids[h] || !split_mask[h])
                  return nullptr;
                ggml_build_forward_expand(g.graph, split_ids[h]);
                ggml_build_forward_expand(g.graph, split_mask[h]);
              }
              const auto weights_of = [&](int mode) {
                return ggml_mul(ctx, in.weights, ggml_reshape_3d(ctx, split_mask[mode - 2], 1, c.topk, g.S));
              };
              // The weight views are host-backend nodes too: schedule them with the other host nodes (see above).
              ggml_tensor *views[2][3];
              for (int h = 0; h < 2; ++h) {
                const int lo = h ? split_at : 0, cnt = h ? n_exp - split_at : split_at;
                int k = 0;
                for (ggml_tensor *t : {ge, ue, de}) {
                  views[h][k] = ggml_view_3d(ctx, t, t->ne[0], t->ne[1], cnt, t->nb[1], t->nb[2], (size_t)lo * t->nb[2]);
                  ggml_build_forward_expand(g.graph, views[h][k++]);
                }
              }
              const auto part = [&](int lo, int cnt, int id_mode, ggml_backend_t be) -> ggml_tensor * {
                (void)lo;
                (void)cnt;
                moe_routed::ExpertWeights ew;
                ew.gate = views[id_mode][0];
                ew.up = views[id_mode][1];
                ew.down = views[id_mode][2];
                ggml_tensor *pid = split_ids[id_mode];
                const moe_routed::FfnNodes ffn =
                    moe_routed::expert_ffn(ctx, ew, input3, pid, clamp, swiglu_clamp_fused());
                if (be && g.sched) {
                  ggml_backend_sched_set_tensor_backend(g.sched, ffn.gate, be);
                  ggml_backend_sched_set_tensor_backend(g.sched, ffn.up, be);
                  ggml_backend_sched_set_tensor_backend(g.sched, ffn.down, be);
                }
                ggml_tensor *experts = ggml_mul(ctx, ffn.down, weights_of(id_mode + 2));
                ggml_tensor *summed = moe_routed::sum_topk(ctx, experts, g.D, g.S, c.topk);
                if (be && g.sched && !g.decode_mode) {
                  ggml_backend_sched_set_tensor_backend(g.sched, experts, be);
                  ggml_backend_sched_set_tensor_backend(g.sched, summed, be);
                }
                return summed;
              };
              ggml_tensor *hi = part(split_at, n_exp - split_at, 1, g.gpu1);
              ggml_tensor *lo = part(0, split_at, 0, nullptr);
              return hi && lo ? ggml_add(ctx, hi, lo) : nullptr;
            }
            if (prefetch_layer_active(layer)) {
              // Prefetched compact weights; in.ids already hold compact slots (the miss node rewrote them). The slots
              // are split between the two GPUs: part 1 (second GPU) first, so its copies are already in flight.
              int split = 0;
              const int n_slots = prefetch_layer_slots(layer, &split);
              ggml_tensor *out_part[2] = {nullptr, nullptr};
              ggml_tensor *pid[2], *pmask[2];
              for (int h = 0; h < 2; ++h) {
                pid[h] = pmask[h] = nullptr;
                if (h == 1 && split >= n_slots)
                  continue;
                if (split >= n_slots) {
                  pid[h] = in.ids;
                  pmask[h] = nullptr;
                } else {
                  pid[h] = cold_split_node(ctx, in.ids, split, h);
                  pmask[h] = ggml_cast(ctx, cold_split_node(ctx, in.ids, split, h + 2), GGML_TYPE_F32);
                  if (!pid[h] || !pmask[h])
                    return nullptr;
                  ggml_build_forward_expand(g.graph, pid[h]);
                  ggml_build_forward_expand(g.graph, pmask[h]);
                }
              }
              for (int h = 1; h >= 0; --h) {
                if (!pid[h])
                  continue;
                ggml_tensor *pg = prefetch_weight(g, ctx, layer, "ffn_gate_exps.weight", h);
                ggml_tensor *pu = prefetch_weight(g, ctx, layer, "ffn_up_exps.weight", h);
                ggml_tensor *pd = prefetch_weight(g, ctx, layer, "ffn_down_exps.weight", h);
                if (!pg || !pu || !pd)
                  return nullptr;
                ggml_tensor *w = pmask[h] ? ggml_mul(ctx, in.weights, ggml_reshape_3d(ctx, pmask[h], 1, c.topk, g.S))
                                          : in.weights;
                out_part[h] = run_branch(nullptr, pg, pu, pd, pid[h], w, h == 1 ? g.gpu1 : nullptr);
                if (!out_part[h])
                  return nullptr;
              }
              if (out_part[0] && out_part[1])
                return ggml_add(ctx, out_part[1], out_part[0]);
              return out_part[0] ? out_part[0] : out_part[1];
            }
            return run_branch(gate_up, ge, ue, de, in.ids, in.weights);
          });
    }
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
                                   partial->hot1_down, in1.ids, in1.weights, g.gpu1);
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

  if (!shared_out)
    shared_out = build_shared();
  if (shared_out)
    out = ggml_add(ctx, out, shared_out);
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
                                : (((int64_t)g.pos_base + g.S + 255) & ~((int64_t)255));
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
                                          : (k < g.pos_base + g.S && k <= g.pos_base + q ? 0.0f
                                                                                         : -INFINITY));

  if (!g.decode_mode && g.sparse_prefill) {
    const int kpool = g.indexer_kpool;
    const int total_pools = (g.pos_base + g.S) / kpool;
    if (kpool <= 1 || total_pools <= 0)
      return false;
    g.prefill_pool_valid = host_input(g, g.ctx, GGML_TYPE_F32, {total_pools, g.S},
                                      (size_t)total_pools * g.S * sizeof(float), "glm5next_prefill_pool_valid");
    g.prefill_tail01 = host_input(g, g.ctx, GGML_TYPE_F32, {kv_tokens, g.S},
                                  (size_t)kv_tokens * g.S * sizeof(float), "glm5next_prefill_tail01");
    if (!g.prefill_pool_valid || !g.prefill_tail01)
      return false;
    auto *valid = static_cast<float *>(g.prefill_pool_valid->data);
    auto *tail = static_cast<float *>(g.prefill_tail01->data);
    for (int q = 0; q < g.S; ++q) {
      const int t = g.pos_base + q;
      const int complete = (t + 1) / kpool;
      for (int j = 0; j < total_pools; ++j)
        valid[(size_t)q * total_pools + j] = j < complete ? 1.0f : 0.0f;
      const int tail_start = complete * kpool;
      for (int k = 0; k < kv_tokens; ++k)
        tail[(size_t)q * kv_tokens + k] =
            (g.indexer_select_tail && k >= tail_start && k <= t) ? 1.0f : 0.0f;
    }
  }

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
    if (g.decode_mode && moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PROBE_DUMP")) {
      ggml_set_output(cur_pre);
      g.probe_x.push_back({l, cur_pre});
    }
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
  if (g.decode_mode && !moestream::graphcfg::env("MOESTREAM_GLM5NEXT_CPU_LM_HEAD")) {
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
  if (const char *dl = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_SPLIT_DUMP")) {
    // Debug: print the backend of every node of one MoE layer (from the previous layer's end to this layer's end).
    const std::string tag = std::string("_") + dl; // node names of the layer end with _<layer>
    int first = -1, last = -1;
    for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
      const char *nm = ggml_get_name(ggml_graph_node(g.graph, i));
      if (std::strstr(nm, ("stage_miss" + tag).c_str()) || std::strstr(nm, ("offload_stage_ids" + tag).c_str()) || std::strstr(nm, ("stage_pred" + tag).c_str()))
        if (first < 0) first = i;
    }
    if (first >= 0) {
      const int lo = std::max(0, first - 60), hi = std::min(ggml_graph_n_nodes(g.graph), first + 160);
      std::string prev;
      for (int i = lo; i < hi; ++i) {
        ggml_tensor *t = ggml_graph_node(g.graph, i);
        ggml_backend_t b = ggml_backend_sched_get_tensor_backend(g.sched, t);
        const std::string bn = b ? ggml_backend_name(b) : "none";
        LOG_INFO("[SPLIT-DUMP] %5d %s %-14s %s\n", i, bn == prev ? "  " : "->", ggml_op_name(t->op), ggml_get_name(t));
        prev = bn;
        (void)last;
      }
    }
  }
  if (moestream::graphcfg::env("MOESTREAM_GLM5NEXT_WS_DUMP")) {
    // Debug: intermediate-tensor liveness over graph order; prints the live-set peak and its largest members.
    const int n = ggml_graph_n_nodes(g.graph);
    std::unordered_map<const ggml_tensor *, int> idx;
    for (int i = 0; i < n; ++i) idx[ggml_graph_node(g.graph, i)] = i;
    std::vector<int> last(n, -1);
    for (int i = 0; i < n; ++i) {
      ggml_tensor *t = ggml_graph_node(g.graph, i);
      for (int s = 0; s < GGML_MAX_SRC; ++s) {
        const ggml_tensor *src = t->src[s];
        while (src && src->view_src) src = src->view_src;
        if (!src) continue;
        auto it = idx.find(src);
        if (it != idx.end()) last[it->second] = std::max(last[it->second], i);
      }
      const ggml_tensor *v = t;
      while (v->view_src) v = v->view_src;
      if (v != t) { auto it = idx.find(v); if (it != idx.end()) last[it->second] = std::max(last[it->second], i); }
    }
    size_t live = 0, peak = 0; int peak_i = 0;
    std::vector<std::pair<int, size_t>> ends;
    for (int i = 0; i < n; ++i) {
      ggml_tensor *t = ggml_graph_node(g.graph, i);
      if (!t->view_src) { live += ggml_nbytes(t); }
      if (live > peak) { peak = live; peak_i = i; }
      for (int j = 0; j <= i; ++j) {}
      // release tensors whose last use is i
      (void)ends;
    }
    // second pass with releases (kept separate for clarity)
    live = 0; peak = 0; peak_i = 0;
    std::vector<std::vector<int>> rel(n + 1);
    for (int i = 0; i < n; ++i) rel[std::max(last[i], i)].push_back(i);
    for (int i = 0; i < n; ++i) {
      ggml_tensor *t = ggml_graph_node(g.graph, i);
      if (!t->view_src) live += ggml_nbytes(t);
      if (live > peak) { peak = live; peak_i = i; }
      for (int r : rel[i]) { const ggml_tensor *rt = ggml_graph_node(g.graph, r); if (!rt->view_src) live -= ggml_nbytes(rt); }
    }
    std::vector<std::pair<size_t, int>> members;
    for (int i = 0; i <= peak_i; ++i) {
      const ggml_tensor *t = ggml_graph_node(g.graph, i);
      if (!t->view_src && std::max(last[i], i) >= peak_i) members.push_back({ggml_nbytes(t), i});
    }
    std::sort(members.rbegin(), members.rend());
    LOG_INFO("[WS-DUMP] nodes=%d live-set peak=%.1f MiB at node %d (%s)\n", n, peak / 1048576.0, peak_i,
             ggml_get_name(ggml_graph_node(g.graph, peak_i)));
    for (size_t k = 0; k < members.size() && k < 15; ++k) {
      const ggml_tensor *t = ggml_graph_node(g.graph, members[k].second);
      LOG_INFO("[WS-DUMP]   %8.1f MiB  %-12s %-5s ne=%lld,%lld,%lld,%lld %s (node %d..%d)\n", members[k].first / 1048576.0,
               ggml_op_name(t->op), ggml_type_name(t->type), (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
               (long long)t->ne[3], ggml_get_name(t), members[k].second, last[members[k].second]);
    }
    // per-op totals of the largest single tensors across the whole graph
    std::vector<std::pair<size_t, int>> all;
    for (int i = 0; i < n; ++i) if (!ggml_graph_node(g.graph, i)->view_src) all.push_back({ggml_nbytes(ggml_graph_node(g.graph, i)), i});
    std::sort(all.rbegin(), all.rend());
    for (size_t k = 0; k < all.size() && k < 12; ++k) {
      const ggml_tensor *t = ggml_graph_node(g.graph, all[k].second);
      LOG_INFO("[WS-DUMP] biggest %8.1f MiB %-12s %-5s %s\n", all[k].first / 1048576.0, ggml_op_name(t->op), ggml_type_name(t->type), ggml_get_name(t));
    }
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
      (double)g.dense.used / (1024.0 * 1024.0 * 1024.0),
      (double)g.dense.capacity / (1024.0 * 1024.0 * 1024.0), g.dense.tensors,
      (double)g.expert.used / (1024.0 * 1024.0 * 1024.0),
      (double)g.expert.capacity / (1024.0 * 1024.0 * 1024.0),
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

} // namespace glm5next_graph

#endif // MOESTREAM_USE_GGML_CUDA
