// Qwen4Exp full graph: Routed-MoE layer builders: host node, GPU stage node, expert split tables,
// and the per-layer MoE graph.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_qwen4exp_graph_internal.hpp.

#include "ggml_qwen4exp_graph_internal.hpp"

#include <cstdlib>

#if defined(ZEROMOE_USE_GGML_CUDA)

using namespace qwen4exp_graph;

namespace qwen4exp_graph {

// Replace one complete routed MoE FFN with a CUDA-captured host marker. The
// marker carries route tensors as hidden GGML dependencies, while the bridge
// uses fixed pinned buffers and an immutable DDR arena. Both the legacy
// interleaved arena and the newer projection-compact arena are supported;
// the latter is passed to GGML through three independent expert-axis strides.
// This is intentionally decode-only: the CPU runner's strided interface is a
// single-token contract and the scheduler-split prefill path remains unchanged.
ggml_tensor *build_qwen4exp_host_node(
    FullGraph &g, ggml_context *ctx, ggml_tensor *cur, ggml_tensor *ids,
    ggml_tensor *weights, int layer, bool partial_cold_only,
    ggml_tensor *gate_exps_for_cpu,
    ggml_tensor *up_exps_for_cpu,
    ggml_tensor *down_exps_for_cpu,
    ggml_tensor *remap_cpu_for_cpu,
    ggml_tensor *mask_cpu_for_cpu)  {
  if (!g.decode_mode || g.S != 1 || !g.engine ||
      (!partial_cold_only &&
       !g.engine->qwen4exp_cpu_offload_layer_enabled(layer)))
    return nullptr;

  const auto &c = g.engine->model_file.architecture_config;
  const auto *arena = g.engine->graph_ram_arena(layer);
  const bool arena_valid =
      arena && g.engine->graph_ram_layer_ready(layer) && arena->base &&
      arena->gate_bytes != 0 && arena->up_bytes != 0 &&
      arena->down_bytes != 0 &&
      (arena->projection_compact
           ? (arena->gate_stride >= arena->gate_bytes &&
              arena->up_stride >= arena->up_bytes &&
              arena->down_stride >= arena->down_bytes)
           : (arena->stride != 0 &&
              arena->stride >= arena->gate_bytes + arena->up_bytes +
                                    arena->down_bytes));
  if (!arena_valid ||
      layer < 0 || layer >= static_cast<int>(g.engine->layers_experts.size()) ||
      g.engine->layers_experts[static_cast<size_t>(layer)].empty()) {
    LOG_WARN(
        "[QWEN4EXP-HOST-NODE] layer=%d declined: complete compatible "
        "DDR arena is required\n",
        layer);
    return nullptr;
  }

  const auto &first = g.engine->layers_experts[static_cast<size_t>(layer)][0];
  if (!first.gate_proj.host_ptr || !first.up_proj.host_ptr ||
      !first.down_proj.host_ptr)
    return nullptr;

  auto node_ctx = std::make_unique<Qwen4ExpCudaHostNodeContext>();
  node_ctx->layer_index = layer;
  node_ctx->partial_cold_only = partial_cold_only;
  if (partial_cold_only) {
    if (!gate_exps_for_cpu || !up_exps_for_cpu || !down_exps_for_cpu ||
        !remap_cpu_for_cpu || !mask_cpu_for_cpu)
      return nullptr;
    node_ctx->partial_cpu_graph = std::make_unique<EarlyReadyLane>();
    if (!node_ctx->partial_cpu_graph->prepare(
            g.cpu, g.device, g.D, c.topk, gate_exps_for_cpu,
            up_exps_for_cpu, down_exps_for_cpu, remap_cpu_for_cpu,
            mask_cpu_for_cpu))
      return nullptr;
    node_ctx->resident_expert_index.assign(
        static_cast<size_t>(c.n_experts), -1);
    node_ctx->staged_expert_index.assign(
        static_cast<size_t>(c.n_experts), -1);
    node_ctx->route_classification.reserve(static_cast<size_t>(c.topk));
  }
  const auto split_resident_it = g.split_gpu_expert_ids.find(layer);
  if (split_resident_it != g.split_gpu_expert_ids.end()) {
    node_ctx->gpu_resident_experts = split_resident_it->second;
    if (partial_cold_only) {
      for (size_t slot = 0; slot < node_ctx->gpu_resident_experts.size();
           ++slot) {
        const int expert = node_ctx->gpu_resident_experts[slot];
        if (expert < 0 || expert >= c.n_experts ||
            node_ctx->resident_expert_index[static_cast<size_t>(expert)] >= 0) {
          LOG_ERROR("[QWEN4EXP-PARTIAL-CLASSIFY] layer=%d has invalid or "
                    "duplicate resident expert id=%d; refusing graph build\n",
                    layer, expert);
          return nullptr;
        }
        node_ctx->resident_expert_index[static_cast<size_t>(expert)] =
            static_cast<int32_t>(slot);
      }
    }
  } else if (partial_cold_only) {
    LOG_ERROR("[QWEN4EXP-PARTIAL-CLASSIFY] layer=%d has no resident expert "
              "table; refusing partial graph build\n",
              layer);
    return nullptr;
  }
  // Reuse the persistent CPU pool already owned by this FullGraph.  The
  // callback runner borrows it; FullGraph remains responsible for teardown.
  node_ctx->shared_cpu_threadpool = g.cpu_threadpool;
  node_ctx->shared_cpu_threads = g.cpu_threads;
  node_ctx->transport_diag = env_host_node_transport_diag_enabled(layer);
  node_ctx->projection_compact = arena->projection_compact;
  LOG_INFO("[QWEN4EXP-HOST-NODE] layer=%d allocating callback context "
           "arena=%p bytes=%zu layout=%s gate_stride=%zu up_stride=%zu "
           "down_stride=%zu\n",
           layer, static_cast<void *>(arena->base), arena->bytes,
           arena->projection_compact ? "projection-compact" : "interleaved",
           arena->gate_stride, arena->up_stride, arena->down_stride);
  node_ctx->gate_plane = arena->base + arena->gate_offset;
  node_ctx->up_plane = arena->base + arena->up_offset;
  node_ctx->down_plane = arena->base + arena->down_offset;
  node_ctx->layer = {node_ctx->gate_plane,
                     node_ctx->up_plane,
                     node_ctx->down_plane,
                     arena->gate_bytes,
                     arena->up_bytes,
                     arena->down_bytes,
                     static_cast<uint32_t>(first.gate_proj.meta.type),
                     static_cast<uint32_t>(first.up_proj.meta.type),
                     static_cast<uint32_t>(first.down_proj.meta.type)};
  node_ctx->n_experts = c.n_experts;
  node_ctx->d_model = g.D;
  node_ctx->d_ff = c.moe_inter;
  node_ctx->topk = c.topk;
  node_ctx->expert_stride = arena->stride;
  node_ctx->gate_stride = arena->gate_stride;
  node_ctx->up_stride = arena->up_stride;
  node_ctx->down_stride = arena->down_stride;
  node_ctx->input_bytes = static_cast<size_t>(g.D) * sizeof(float);
  node_ctx->gpu_reduce = partial_cold_only ? true : env_host_node_gpu_reduce();
  node_ctx->output_bytes =
      node_ctx->input_bytes *
      (node_ctx->gpu_reduce ? static_cast<size_t>(c.topk) : 1);
  node_ctx->ids_bytes = static_cast<size_t>(c.topk) * sizeof(int32_t);
  node_ctx->weights_bytes = static_cast<size_t>(c.topk) * sizeof(float);
  node_ctx->host_input = static_cast<float *>(
      alloc_registered_host(g, node_ctx->input_bytes));
  node_ctx->host_output = static_cast<float *>(
      alloc_registered_host(g, node_ctx->output_bytes));
  if (node_ctx->transport_diag) {
    node_ctx->host_output_gpu_check = static_cast<float *>(
        alloc_registered_host(g, node_ctx->output_bytes));
  }
  node_ctx->host_ids = static_cast<int32_t *>(
      alloc_registered_host(g, node_ctx->ids_bytes));
  node_ctx->host_weights = static_cast<float *>(
      alloc_registered_host(g, node_ctx->weights_bytes));
  LOG_INFO("[QWEN4EXP-HOST-NODE] layer=%d callback buffers input=%p output=%p "
           "ids=%p weights=%p\n",
           layer, static_cast<void *>(node_ctx->host_input),
           static_cast<void *>(node_ctx->host_output),
           static_cast<void *>(node_ctx->host_ids),
           static_cast<void *>(node_ctx->host_weights));
  if (!node_ctx->host_input || !node_ctx->host_output ||
      !node_ctx->host_ids || !node_ctx->host_weights ||
      (node_ctx->transport_diag && !node_ctx->host_output_gpu_check))
    return nullptr;

  cudaSetDevice(g.device);
  LOG_INFO("[QWEN4EXP-HOST-NODE] layer=%d creating timing events\n", layer);
  if (cudaEventCreate(&node_ctx->d2h_start) != cudaSuccess ||
      cudaEventCreate(&node_ctx->d2h_end) != cudaSuccess ||
      cudaEventCreate(&node_ctx->h2d_start) != cudaSuccess ||
      cudaEventCreate(&node_ctx->h2d_end) != cudaSuccess ||
      (node_ctx->transport_diag &&
       cudaEventCreate(&node_ctx->post_h2d_check_end) != cudaSuccess))
    return nullptr;

  // Warm the persistent CPU GGML plan before capture. The graph callback must
  // not be the first call that allocates/builds the CPU plan.
  std::fill_n(node_ctx->host_input, g.D, 0.0f);
  std::fill_n(node_ctx->host_output, node_ctx->output_bytes / sizeof(float),
              0.0f);
  std::fill_n(node_ctx->host_ids, c.topk, 0);
  std::fill_n(node_ctx->host_weights, c.topk,
              1.0f / static_cast<float>(std::max(1, c.topk)));
  LOG_INFO("[QWEN4EXP-HOST-NODE] layer=%d warming CPU plan type=(%u,%u,%u) "
           "dims=(%d,%d,%d) layout=%s planes=(%p,%p,%p) strides=(%zu,%zu,%zu)\n",
           layer, node_ctx->layer.gate_type, node_ctx->layer.up_type,
           node_ctx->layer.down_type, node_ctx->d_model, node_ctx->d_ff,
           node_ctx->topk,
           node_ctx->projection_compact ? "projection-compact" : "interleaved",
           node_ctx->gate_plane, node_ctx->up_plane, node_ctx->down_plane,
           node_ctx->gate_stride, node_ctx->up_stride,
           node_ctx->down_stride);
  bool warm_ok = false;
  GgmlQwenCpuMoeRunner *warm_runner = nullptr;
  if (node_ctx->projection_compact) {
    warm_runner = &GgmlQwenCpuMoeRunner::instance_for_strided_planes(
            node_ctx->layer, node_ctx->n_experts, node_ctx->d_model,
            node_ctx->d_ff, node_ctx->topk, node_ctx->gate_stride,
            node_ctx->up_stride, node_ctx->down_stride,
            node_ctx->gpu_reduce);
    if (node_ctx->gpu_reduce)
      warm_ok = warm_runner->run_strided_planes_down(
          node_ctx->host_input, node_ctx->host_output, node_ctx->layer,
          node_ctx->gate_plane, node_ctx->up_plane, node_ctx->down_plane,
          node_ctx->host_ids, node_ctx->n_experts, node_ctx->d_model,
          node_ctx->d_ff, node_ctx->topk, node_ctx->gate_stride,
          node_ctx->up_stride, node_ctx->down_stride);
    else
    warm_ok = warm_runner->run_strided_planes(
        node_ctx->host_input, node_ctx->host_output, node_ctx->layer,
        node_ctx->gate_plane, node_ctx->up_plane, node_ctx->down_plane,
        node_ctx->host_ids, node_ctx->n_experts, node_ctx->host_weights,
        node_ctx->d_model, node_ctx->d_ff, node_ctx->topk,
        node_ctx->gate_stride, node_ctx->up_stride, node_ctx->down_stride);
  } else {
    warm_runner = &GgmlQwenCpuMoeRunner::instance_for_strided(
            node_ctx->layer, node_ctx->n_experts, node_ctx->d_model,
            node_ctx->d_ff, node_ctx->topk, node_ctx->expert_stride,
            node_ctx->gpu_reduce);
    if (node_ctx->gpu_reduce)
      warm_ok = warm_runner->run_strided_down(
          node_ctx->host_input, node_ctx->host_output, node_ctx->layer,
          node_ctx->gate_plane, node_ctx->up_plane, node_ctx->down_plane,
          node_ctx->host_ids, node_ctx->n_experts, node_ctx->d_model,
          node_ctx->d_ff, node_ctx->topk, node_ctx->expert_stride);
    else
    warm_ok = warm_runner->run_strided(
        node_ctx->host_input, node_ctx->host_output, node_ctx->layer,
        node_ctx->gate_plane, node_ctx->up_plane, node_ctx->down_plane,
        node_ctx->host_ids, node_ctx->n_experts, node_ctx->host_weights,
        node_ctx->d_model, node_ctx->d_ff, node_ctx->topk,
        node_ctx->expert_stride);
  }
  if (!warm_ok) {
    LOG_WARN("[QWEN4EXP-HOST-NODE] layer=%d CPU plan warm-up failed\n", layer);
    return nullptr;
  }
  const char *direct_pool_env = zeromoe::graphcfg::env("ZEROMOE_QWEN_CPU_DIRECT_POOL");
  const char *direct_fused_env =
      zeromoe::graphcfg::env("ZEROMOE_QWEN_CPU_DIRECT_FUSED");
  const bool direct_fused_requested =
      direct_fused_env && direct_fused_env[0] == '1';
  const bool direct_pool_requested =
      direct_pool_env && direct_pool_env[0] == '1' &&
      !direct_fused_requested;
  if (warm_runner && direct_pool_requested) {
    // The raw direct executor owns its pre-sized worker team.  Borrowing the
    // FullGraph GGML pool here would tear down that direct signature during
    // warm-up and recreate it from inside the first replay callback, exactly
    // where we need deterministic wake-up behavior.  Prepare the team now,
    // before capture; replay only arms/disarms it.
    warm_runner->prepare_direct_workers();
    node_ctx->cpu_runner = warm_runner;
    LOG_INFO("[QWEN4EXP-HOST-NODE] layer=%d prepared direct CPU team "
             "threads=%d before graph capture\n",
             layer, warm_runner->thread_count());
  } else if (warm_runner && !direct_fused_requested &&
             node_ctx->shared_cpu_threadpool &&
             node_ctx->shared_cpu_threads > 0) {
    warm_runner->bind_external_threadpool(node_ctx->shared_cpu_threadpool,
                                          node_ctx->shared_cpu_threads);
    node_ctx->cpu_runner = warm_runner;
    LOG_INFO("[QWEN4EXP-HOST-NODE] layer=%d borrowing FullGraph CPU pool "
             "%p threads=%d\n",
             layer, static_cast<void *>(node_ctx->shared_cpu_threadpool),
             node_ctx->shared_cpu_threads);
  } else if (warm_runner) {
    node_ctx->cpu_runner = warm_runner;
  }
  // An owned runner may suspend its private native ggml pool between
  // callbacks.  A borrowed FullGraph pool must remain available to ordinary
  // scheduler CPU splits, so it is deliberately never paused here.
  if (warm_runner && !warm_runner->uses_external_threadpool())
    warm_runner->pause_workers();
  LOG_INFO("[QWEN4EXP-HOST-NODE] layer=%d CPU plan warm-up complete\n", layer);

  ggml_tensor *input3 = ggml_reshape_3d(ctx, cur, g.D, 1, g.S);
  if (!input3)
    return nullptr;
  if (input3->type != GGML_TYPE_F32)
    input3 = ggml_cast(ctx, input3, GGML_TYPE_F32);
  input3 = ggml_cont(ctx, input3);
  ggml_tensor *route_ids = ggml_cont(ctx, ids);
  ggml_tensor *route_weights = ggml_cont(ctx, weights);
  if (!input3 || !route_ids || !route_weights)
    return nullptr;

  const bool gpu_reduce = node_ctx->gpu_reduce;
  ggml_tensor *destination = ggml_new_tensor_2d(
      ctx, GGML_TYPE_F32, g.D, gpu_reduce ? c.topk : g.S);
  if (!destination)
    return nullptr;
  ggml_set_name(destination,
                ("qwen4exp_host_moe_output_l" + std::to_string(layer))
                    .c_str());
  // In GPU-reduce mode only row 0 is a real copy target; the bridge H2D
  // fills all topk rows of the destination from the callback's output.
  ggml_tensor *copy_dst =
      (gpu_reduce && !partial_cold_only)
          ? ggml_view_2d(ctx, destination, g.D, 1, destination->nb[1], 0)
          : destination;
  ggml_tensor *marker_source =
      partial_cold_only ? ggml_dup_tensor(ctx, destination) : input3;
  ggml_tensor *marker = ggml_cpy(ctx, marker_source, copy_dst);
  if (!marker)
    return nullptr;
  marker->extra = node_ctx.get();
  // These are dependency-only sources. The CUDA bridge copies them to the
  // pinned route buffers before the callback; the ordinary CPY backend never
  // sees this marker because the bridge consumes it first.
  marker->src[2] = route_ids;
  if (!gpu_reduce || partial_cold_only)
    marker->src[3] = route_weights;
  if (partial_cold_only)
    marker->src[4] = input3;
  ggml_set_name(marker,
                ("qwen4exp_host_moe_marker_l" + std::to_string(layer))
                    .c_str());
  ggml_tensor *result = marker;
  if (gpu_reduce && !partial_cold_only) {
    // [D, topk] unweighted expert outputs -> weight and sum over topk.
    ggml_tensor *per_expert =
        ggml_view_3d(ctx, marker, g.D, c.topk, 1, destination->nb[1],
                     destination->nb[1] * c.topk, 0);
    ggml_tensor *prod = ggml_mul(ctx, per_expert, route_weights);
    ggml_tensor *t = ggml_cont(ctx, ggml_permute(ctx, prod, 1, 0, 2, 3));
    ggml_tensor *sum = ggml_sum_rows(ctx, t);
    result = ggml_reshape_2d(ctx, sum, g.D, g.S);
    if (!per_expert || !prod || !t || !sum || !result)
      return nullptr;
    if (g.sched) {
      ggml_backend_sched_set_tensor_backend(g.sched, per_expert, g.cuda);
      ggml_backend_sched_set_tensor_backend(g.sched, prod, g.cuda);
      ggml_backend_sched_set_tensor_backend(g.sched, t, g.cuda);
      ggml_backend_sched_set_tensor_backend(g.sched, sum, g.cuda);
    }
  }
  if (g.sched) {
    ggml_backend_sched_set_tensor_backend(g.sched, destination, g.cuda);
    ggml_backend_sched_set_tensor_backend(g.sched, marker, g.cuda);
  }
  g.host_node_contexts.push_back(std::move(node_ctx));
  LOG_INFO("[QWEN4EXP-HOST-NODE] captured CPU MoE marker for layer=%d "
           "(D=%d topk=%d stride=%zu)\n",
           layer, g.D, c.topk, arena->stride);
  return result;
}

// Generalizes Option-B's layer-13 staging proof
// (qwen4exp_stage_layer13_before_graph/StageBinding) into a mechanism
// covering every host-resident layer listed in
// engine.qwen4exp_gpu_stage_layers, for real per-token routing instead of
// a forced test set. Where the layer-13 proof issues its
// upload_expert_active_projections_async() call as ordinary host code
// immediately before graph replay -- which only works because it forces a
// fixed route (build_moe's g.stage_route_ids branch) -- this uses the SAME
// CUDA-graph host-node-callback technique qwen4exp_cuda_host_node_callback/
// zeromoe_qwen4exp_cuda_host_node_enqueue already prove works mid-replay
// with REAL per-token data: a captured D2H(ids)->host-callback(gather)->
// H2D(compact planes) sequence that re-executes on every cudaGraphLaunch
// replay with that token's real routed ids.
//
// Correctness invariant (see also Qwen4ExpGpuStageContext's own comment):
// the callback writes routed expert ids[k]'s bytes into destination slot
// k, for k in [0,topk), in the SAME order `ids` itself lists them. Because
// build_moe's combine weights (`weights = ggml_get_rows(ctx, probs3,
// ids)`) are ALSO indexed by that same order, a MUL_MAT_ID call against
// these compact destinations can use a CONSTANT [0..topk-1] ids tensor --
// slot k IS real expert ids[k] by construction -- with no per-token id
// remap needed. Do NOT reassign `ids` itself here the way the layer-13
// proof does (build_moe's `ids = g.stage_route_ids` line): that would
// desync the real combine weights from the real routing, which the proof
// gets away with only because it forces both consistently to the same
// fake set.
//
// Decode (S==1) only -- see engine.qwen4exp_gpu_stage_enabled's own
// comment for why prefill needs a different, batch-proportional design.
// Returns the index into g.gpu_stage_bindings on success, -1 on any
// failure (caller falls through to the next dispatch strategy for this
// layer -- CPU-offload if enabled, otherwise today's plain CPU path;
// never a hard error for the whole graph build).
int build_gpu_stage_node(FullGraph &g, ggml_context *ctx, ggml_tensor *ids,
                         int layer) {
  if (!g.decode_mode || g.S != 1 || !g.engine ||
      !g.engine->qwen4exp_gpu_stage_layer_enabled(layer))
    return -1;

  const auto &c = g.engine->model_file.architecture_config;
  const auto *arena = g.engine->graph_ram_arena(layer);
  const bool arena_valid =
      arena && g.engine->graph_ram_layer_ready(layer) && arena->base &&
      arena->gate_bytes != 0 && arena->up_bytes != 0 &&
      arena->down_bytes != 0 &&
      (arena->projection_compact
           ? (arena->gate_stride >= arena->gate_bytes &&
              arena->up_stride >= arena->up_bytes &&
              arena->down_stride >= arena->down_bytes)
           : (arena->stride != 0 &&
              arena->stride >= arena->gate_bytes + arena->up_bytes +
                                    arena->down_bytes));
  if (!arena_valid || layer < 0 ||
      layer >= static_cast<int>(g.engine->layers_experts.size()) ||
      g.engine->layers_experts[static_cast<size_t>(layer)].empty()) {
    LOG_WARN("[QWEN4EXP-GPU-STAGE] layer=%d declined: complete compatible "
             "DDR arena is required\n",
             layer);
    return -1;
  }
  const auto &first = g.engine->layers_experts[static_cast<size_t>(layer)][0];
  if (!first.gate_proj.host_ptr || !first.up_proj.host_ptr ||
      !first.down_proj.host_ptr)
    return -1;

  // Compact destination tensors: [cols, rows, topk] -- same shape
  // convention build_split_gpu_tensor already uses for its own compact GPU
  // tensors, sized for `topk` (constant per token) instead of a static
  // hot-set count. Bump-allocated from the SAME device_expert_buffer arena
  // device-resident layers use (see device_expert_bytes()'s staging-bytes
  // block for the matching capacity accounting) -- never copied into here
  // directly; only the per-token host-node callback below ever writes
  // their bytes.
  auto make_dst = [&](const char *stem,
                      ggml_type type) -> ggml_tensor * {
    const GGUFTensor *m = model_tensor(
        *g.engine, "blk." + std::to_string(layer) + "." + stem);
    if (!m || m->shape.size() < 3)
      return nullptr;
    std::vector<int64_t> shape = {static_cast<int64_t>(m->shape[0]),
                                  static_cast<int64_t>(m->shape[1]),
                                  static_cast<int64_t>(c.topk)};
    ggml_tensor *t = new_tensor(ctx, type, shape);
    if (!t)
      return nullptr;
    void *addr = nullptr;
    bool new_allocation = false;
    const std::string key =
        "gpu_stage:" + std::to_string(layer) + ":" + stem;
    if (!bind_or_allocate_expert(g, t, key, &addr, &new_allocation))
      return nullptr;
    ggml_set_name(t, (std::string("blk.") + std::to_string(layer) + "." +
                      stem + ".gpu_stage")
                         .c_str());
    return t;
  };

  FullGraph::GpuStageBinding binding;
  binding.gate_dst = make_dst(
      "ffn_gate_exps.weight", static_cast<ggml_type>(first.gate_proj.meta.type));
  binding.up_dst = make_dst(
      "ffn_up_exps.weight", static_cast<ggml_type>(first.up_proj.meta.type));
  binding.down_dst = make_dst(
      "ffn_down_exps.weight", static_cast<ggml_type>(first.down_proj.meta.type));
  if (!binding.gate_dst || !binding.up_dst || !binding.down_dst) {
    LOG_WARN("[QWEN4EXP-GPU-STAGE] layer=%d declined: compact device "
             "destination allocation failed (VRAM budget?)\n",
             layer);
    return -1;
  }

  // Bounded overhead fix: gate_dst/up_dst/down_dst were just bump-allocated
  // back-to-back (nothing else allocates from device_expert_buffer between
  // the three make_dst() calls above), so they form one
  // contiguous-modulo-alignment span -- confirm that here rather than
  // assume it, so a future allocator change fails closed (falls back to a
  // separate allocation) instead of silently corrupting an H2D copy.
  const auto *gate_base = static_cast<const uint8_t *>(binding.gate_dst->data);
  const auto *up_base = static_cast<const uint8_t *>(binding.up_dst->data);
  const auto *down_base = static_cast<const uint8_t *>(binding.down_dst->data);
  const size_t up_offset = static_cast<size_t>(up_base - gate_base);
  const size_t down_offset = static_cast<size_t>(down_base - gate_base);
  const size_t combined_bytes = down_offset + ggml_nbytes(binding.down_dst);
  if (up_base <= gate_base || down_base <= up_base ||
      up_offset < ggml_nbytes(binding.gate_dst) ||
      down_offset < up_offset + ggml_nbytes(binding.up_dst)) {
    LOG_WARN("[QWEN4EXP-GPU-STAGE] layer=%d declined: gate/up/down device "
             "destinations are not contiguous (bump-allocator layout "
             "changed?) -- combined single-copy staging requires it\n",
             layer);
    return -1;
  }

  binding.ids_dispatch_const = host_input(
      g, ctx, GGML_TYPE_I32, {c.topk, g.S},
      static_cast<size_t>(c.topk) * static_cast<size_t>(g.S) * sizeof(int32_t),
      (std::string("qwen_gpu_stage_ids_l") + std::to_string(layer)).c_str());
  if (!binding.ids_dispatch_const)
    return -1;
  {
    auto *ids_data = static_cast<int32_t *>(binding.ids_dispatch_const->data);
    for (int s = 0; s < g.S; ++s)
      for (int k = 0; k < c.topk; ++k)
        ids_data[s * c.topk + k] = k;
  }

  auto node_ctx = std::make_unique<Qwen4ExpGpuStageContext>();
  node_ctx->layer_index = layer;
  node_ctx->n_experts = c.n_experts;
  node_ctx->topk = c.topk;
  node_ctx->gate_plane = arena->base + arena->gate_offset;
  node_ctx->up_plane = arena->base + arena->up_offset;
  node_ctx->down_plane = arena->base + arena->down_offset;
  node_ctx->gate_stride = arena->gate_stride;
  node_ctx->up_stride = arena->up_stride;
  node_ctx->down_stride = arena->down_stride;
  node_ctx->gate_bytes = arena->gate_bytes;
  node_ctx->up_bytes = arena->up_bytes;
  node_ctx->down_bytes = arena->down_bytes;
  node_ctx->ids_bytes = static_cast<size_t>(c.topk) * sizeof(int32_t);
  node_ctx->host_ids = static_cast<int32_t *>(
      alloc_registered_host(g, node_ctx->ids_bytes));
  // Per-expert slot size within the combined buffer stays gate_bytes/
  // up_bytes/down_bytes (topk slots each, same as before) -- only the
  // BASE offset of each projection's run of slots changed, from three
  // independent allocations to up_offset/down_offset within one span.
  node_ctx->combined_up_offset = up_offset;
  node_ctx->combined_down_offset = down_offset;
  node_ctx->combined_bytes = combined_bytes;
  node_ctx->host_combined_bounce = static_cast<uint8_t *>(
      alloc_registered_host(g, node_ctx->combined_bytes));
  if (!node_ctx->host_ids || !node_ctx->host_combined_bounce)
    return -1;
  node_ctx->device_combined_dst = binding.gate_dst->data;

  cudaSetDevice(g.device);
  if (cudaEventCreate(&node_ctx->d2h_start) != cudaSuccess ||
      cudaEventCreate(&node_ctx->d2h_end) != cudaSuccess ||
      cudaEventCreate(&node_ctx->h2d_start) != cudaSuccess ||
      cudaEventCreate(&node_ctx->h2d_end) != cudaSuccess)
    return -1;
  // Warm the pinned buffers before capture -- same discipline
  // build_qwen4exp_host_node follows, so capture never observes
  // first-touch page faults.
  std::fill_n(node_ctx->host_ids, c.topk, 0);
  std::memset(node_ctx->host_combined_bounce, 0, node_ctx->combined_bytes);

  // A tiny dummy leaf (same type/shape as gate_dst) satisfies ggml_cpy's
  // own shape/type validation for the marker's formal "source" -- the real
  // work happens entirely inside the enqueue hijack
  // (qwen4exp_cuda_gpu_stage_node_enqueue), never via ordinary CPY
  // execution. `ids` (real routing, ggml_cont'd) is attached as an EXTRA
  // dependency source purely for D2H/ordering purposes, matching the
  // existing host-node marker's own src[2]/src[3] convention.
  ggml_tensor *dummy_src = ggml_dup_tensor(ctx, binding.gate_dst);
  if (!dummy_src)
    return -1;
  ggml_tensor *route_ids = ggml_cont(ctx, ids);
  if (!route_ids)
    return -1;
  ggml_tensor *marker = ggml_cpy(ctx, dummy_src, binding.gate_dst);
  if (!marker)
    return -1;
  marker->extra = node_ctx.get();
  marker->src[2] = route_ids;
  ggml_set_name(marker, (std::string("qwen4exp_gpu_stage_marker_l") +
                        std::to_string(layer))
                           .c_str());
  if (g.sched)
    ggml_backend_sched_set_tensor_backend(g.sched, marker, g.cuda);

  // Force gate/up/down's own MUL_MAT_ID consumers to depend on the marker,
  // so ggml's topological sort schedules the D2H/callback/H2D sequence
  // strictly before anything reads gate_dst/up_dst/down_dst's contents --
  // dst/up/down_dst are plain bump-allocated leaves with no other src[]
  // edge to the marker, so this dependency must be explicit (ggml only
  // tracks producer/consumer ordering through node src[] chains, never
  // through incidental construction order or shared backing memory).
  binding.gpu_stage_marker_dep = marker;

  binding.layer = layer;
  g.gpu_stage_contexts.push_back(std::move(node_ctx));
  g.gpu_stage_bindings.push_back(binding);
  LOG_INFO("[QWEN4EXP-GPU-STAGE] captured GPU staging marker for layer=%d "
           "(D=%d topk=%d)\n",
           layer, g.D, c.topk);
  return static_cast<int>(g.gpu_stage_bindings.size()) - 1;
}

// `safe_dummy_fill`: a literal -1 ("skip this route") id is only honored on
// stock ggml when the ZeroMoE negative-skip patch is applied
// (patches/ggml-mul-mat-id-negative-skip.patch, CPU op + CUDA MMVQ decode
// kernels). Unpatched, ggml-cuda's mul_mat_vec_q reads `ids[channel_dst]` as
// an unsigned offset (illegal memory access, reproduced 2026-09-29) and the
// CPU op's `assert(i02 >= 0 ...)` is compiled out by NDEBUG. Even patched,
// prefill's mmq/mmf/sorted paths are not covered. When `safe_dummy_fill` is
// true, both remap tables use a valid dummy index (0) instead of -1 and the
// mask zeroes the contribution, at the price of every branch computing all
// topk routes (~2x expert compute). Default false is used by
// build_moe_partial_gpu's Phase 1/2 caller, which ignores the masks.
ExpertSplitTables build_expert_split_tables(FullGraph &g, ggml_context *ctx,
                                            int n_experts,
                                            const std::vector<int> &gpu_ids,
                                            bool safe_dummy_fill)  {
  ExpertSplitTables r;
  if (n_experts <= 0)
    return r;
  // Decode-only builder (S == 1).  The inactive-route id, the 0/1 masks and the
  // table layout come from moe_partial::ExpertBranchMap
  // (include/moe_partial_route.hpp), the same code every graph uses: -1 only
  // when the linked ggml advertises the negative-id skip, else the dummy id
  // + mask, so a build against a stock ggml degrades safely instead of faulting.
  const size_t n = static_cast<size_t>(n_experts);
  std::vector<int32_t> gpu_index(n, -1), cpu_index(n, -1);
  for (int e = 0; e < n_experts; ++e)
    cpu_index[static_cast<size_t>(e)] = e;
  for (size_t k = 0; k < gpu_ids.size(); ++k) {
    const int e = gpu_ids[k];
    if (e < 0 || e >= n_experts)
      continue;
    gpu_index[static_cast<size_t>(e)] = static_cast<int32_t>(k);
    cpu_index[static_cast<size_t>(e)] = -1;
  }
  const moe_partial::SentinelPolicy policy =
      safe_dummy_fill ? moe_partial::SentinelPolicy::ForceDummy
                      : moe_partial::SentinelPolicy::Auto;
  const moe_partial::ExpertBranchMap gpu_map =
      moe_partial::ExpertBranchMap::build(gpu_index, 1, true, {}, policy);
  const moe_partial::ExpertBranchMap cpu_map =
      moe_partial::ExpertBranchMap::build(cpu_index, 1, true, {}, policy);
  const std::vector<int32_t> &remap_gpu = gpu_map.route_ids();
  const std::vector<int32_t> &remap_cpu = cpu_map.route_ids();
  const std::vector<float> &mgpu = gpu_map.route_mask();
  const std::vector<float> &mcpu = cpu_map.route_mask();
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

bool exl3_ggml_bridge_requested(const FullGraph &g, int layer,
                                std::string &model_path) {
  if (g.engine && !g.engine->exl3_ggml_bridge_model_path.empty() &&
      (g.engine->exl3_ggml_bridge_layer == layer ||
       std::find(g.engine->exl3_ggml_bridge_layers.begin(),
                 g.engine->exl3_ggml_bridge_layers.end(), layer) !=
           g.engine->exl3_ggml_bridge_layers.end())) {
    model_path = g.engine->exl3_ggml_bridge_model_path;
    return true;
  }
  const char *path = zeromoe::graphcfg::env("ZEROMOE_EXL3_GGML_BRIDGE_MODEL_PATH");
  const char *layer_text = zeromoe::graphcfg::env("ZEROMOE_EXL3_GGML_BRIDGE_LAYER");
  if (!path || !path[0] || !layer_text || !layer_text[0])
    return false;
  char *end = nullptr;
  const long selected = std::strtol(layer_text, &end, 10);
  if (end == layer_text || *end != '\0' || selected != layer)
    return false;
  model_path = path;
  return true;
}

// The early-ready CPU lane for multi-token decode windows is opt-in
// (ZEROMOE_QWEN4EXP_WINDOW_LANE=1): with it on, the text of a window run diverged from plain
// decoding on the test prompt (investigation open), and it saved only ~2 ms per window.
// The cold branch's ggml node chain is replaced by one fused CPU node (qwen4exp_cold::build) whenever the
// expert tensors are Q2_0; ZEROMOE_QWEN4EXP_COLD_FUSED=0 turns it off. Bit-identical to the chain
// (tests/test_cold_ffn_fused.cpp); 5-23% faster MTP windows (docs/design/mtp-window-timeline.md).
static bool cold_fused_enabled() {
  const char *v = zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_COLD_FUSED");
  return !(v && v[0] == '0');
}

static bool window_lane_enabled() {
  const char *v = zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_WINDOW_LANE");
  return v && v[0] == '1';
}

ggml_tensor *build_moe(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *router,
                       ggml_tensor *selection_bias, ggml_tensor *gate_exps,
                       ggml_tensor *up_exps, ggml_tensor *down_exps,
                       ggml_tensor *gate_exps_gpu,
                       ggml_tensor *up_exps_gpu,
                       ggml_tensor *down_exps_gpu,
                       const std::vector<int> *split_gpu_ids,
                       ggml_tensor *gate_up_exps,
                       // Phase 1 partial-layer instrumentation only: when
                       // non-null, receives this call's routing `ids`
                       // tensor so the caller can mark it as a prefill
                       // graph output for a real per-prompt routing
                       // readback. Never used for anything decode does.
                       ggml_tensor **ids_out,
                       // Same idea, but captures the raw post-softmax
                       // router distribution (BEFORE selection_bias) so a
                       // diagnostic can check how peaked/confident real
                       // routing actually is -- see the "near-uniform =>
                       // noise" precedent already used for the LM head in
                       // generation_loop.hpp.
                       ggml_tensor **probs_out,
                       // Phase 2 prefix builds only: compute the router
                       // (logits/probs/ids/weights, already added to the
                       // graph via the ggml_build_forward_expand below)
                       // and return nullptr immediately, WITHOUT
                       // constructing any dispatch/FFN nodes at all. Lets
                       // the prefix graph capture ids_out for the host
                       // decision without paying for (or even
                       // constructing) the big expert matmuls it will
                       // never use -- the real dispatch happens once,
                       // for real, in whichever suffix graph gets chosen.
                       bool routing_only,
                       ggml_tensor *fixed_route_ids,
                       ggml_tensor *fixed_route_weights,
                       ggml_tensor **route_weights_out)  {
  const auto &c = g.engine->model_file.architecture_config;
  const moe_routed::RouterSpec rspec =
      family_router_spec(g.engine->family_ops.family, c, g.S);
  ggml_tensor *probs = nullptr;
  if (!fixed_route_ids || !fixed_route_weights || probs_out)
    probs = moe_routed::gate_probs(ctx, rspec, router, cur);
  if (probs_out)
    *probs_out = probs;
  ggml_tensor *ids = fixed_route_ids;
  if (!ids)
    ids = moe_routed::select_ids(ctx, rspec, probs, selection_bias);
  if (!ids)
    return nullptr;
  if (!fixed_route_ids && g.decode_mode && g.stage_enabled &&
      layer == g.stage_layer) {
    if (!g.stage_route_ids) {
      g.stage_route_ids = host_input(
          g, ctx, GGML_TYPE_I32, {c.topk, g.S},
          static_cast<size_t>(c.topk) * g.S * sizeof(int32_t),
          "qwen_layer13_stage_route_ids");
      if (g.stage_route_ids) {
        auto *forced = static_cast<int32_t *>(g.stage_route_ids->data);
        for (int s = 0; s < g.S; ++s) {
          for (int k = 0; k < c.topk; ++k) {
            const int index = s * c.topk + k;
            forced[index] = g.stage_experts.empty()
                                ? g.stage_expert
                                : g.stage_experts[k % g.stage_experts.size()];
          }
        }
      }
    }
    if (g.stage_route_ids) {
      ids = g.stage_route_ids;
      if (!g.stage_route_logged) {
        std::ostringstream experts;
        for (size_t i = 0; i < g.stage_experts.size(); ++i)
          experts << (i == 0 ? "" : ",") << g.stage_experts[i];
        LOG_INFO("[OPTION-B-STAGE] forcing Layer %d route IDs to cold "
                 "experts [%s] for the %zu-slot proof\n",
                 g.stage_layer, experts.str().c_str(),
                 g.stage_experts.size());
        g.stage_route_logged = true;
      }
    }
  }
  if (ids_out) {
    // ggml_argsort_top_k is argsort (full n_experts-wide) + a VIEW that
    // only shrinks ne[0] to topk -- the view's row stride (nb[1]) stays
    // at the ORIGINAL n_experts-wide stride, not topk. A single flat
    // ggml_backend_tensor_get over S>1 rows therefore does not read S
    // independent per-token top-k slices; it reads a contiguous run of
    // (topk*S) values out of THIS VIEW's ne[0]=topk logical columns
    // reinterpreted at the wrong stride, which for S*topk <= n_experts
    // (true for every case this repo exercises, e.g. topk=10, S<=51)
    // silently degenerates into reading token 0's OWN top-(topk*S)
    // ranking sliced into S fake "per-token" chunks -- found via real
    // hardware testing, 2026-09-18 (every reported PREFILL coverage/
    // routing-divergence number computed before this fix is invalid;
    // see [[qwen4exp-legacy-vs-fullgraph-routing-divergence]] and
    // [[qwen4exp-partial-layer-phase2-implemented]]). DECODE (S==1) was
    // never affected -- there is only one row, so the flat read already
    // lands on real data with no other row to conflate it with.
    // ggml_cont() forces a real, densely-strided [topk, S] copy so any
    // host-side flat readback of *ids_out is safe.
    *ids_out = ggml_cont(ctx, ids);
  }
  ggml_tensor *weights = fixed_route_weights;
  if (!weights)
    weights = moe_routed::route_weights(ctx, rspec, probs, ids);
  if (!weights)
    return nullptr;
  if (route_weights_out)
    *route_weights_out = weights;
  ggml_build_forward_expand(g.graph, weights);
  if (routing_only) {
    // Phase 2 prefix graph: caller only wanted ids_out/probs_out for the
    // per-token host dispatch decision. Stop here -- no FFN/dispatch
    // nodes constructed, so this layer's real expert compute happens
    // exactly once, in whichever suffix graph actually gets chosen.
    return nullptr;
  }

  std::string exl3_bridge_path;
  if (g.exl3_cuda_adapter && g.engine && g.engine->gpu &&
      exl3_ggml_bridge_requested(g, layer, exl3_bridge_path)) {
    auto bridge = std::make_unique<Exl3GgmlBridgeLayer>();
    if (bridge->initialize(exl3_bridge_path, ctx, g.cuda,
                           g.engine->exl3_gpu_executor, layer, c.n_experts,
                           c.topk, g.S, g.device, true,
                           g.engine->exl3_cpu_matrix_cache,
                           &g.engine->gpu->topology, g.engine)) {
      ggml_tensor *bridge_output =
          bridge->build(ctx, cur, ids, weights, g.sched,
                        g.exl3_cuda_adapter);
      if (bridge_output) {
        LOG_INFO("[EXL3-GGML-BRIDGE] layer=%d routed custom op references "
                 "the existing GPU expert cache\n",
                 layer);
        g.exl3_bridge_layers.push_back(std::move(bridge));
        return bridge_output;
      }
      LOG_ERROR("[EXL3-GGML-BRIDGE] layer=%d custom node construction failed; "
                "refusing to substitute GGUF expert weights\n",
                layer);
      g.exl3_bridge_failed = true;
      return nullptr;
    }
    LOG_ERROR("[EXL3-GGML-BRIDGE] layer=%d initialization failed; refusing "
              "to substitute GGUF expert weights\n",
              layer);
    g.exl3_bridge_failed = true;
    return nullptr;
  }

  // Generalized GPU staging for host-resident decode layers (see
  // engine.qwen4exp_gpu_stage_enabled/_layers, build_gpu_stage_node).
  // Highest precedence among host-resident dispatch strategies: takes over
  // before the CPU-offload host-node marker and the static split_gpu_ids
  // branch get a chance, since staging fully replaces what either would do
  // for this layer's compute. `weights` above was already computed from
  // the real, unmodified `ids` -- staging must NOT reassign `ids` itself
  // (see build_gpu_stage_node's own comment on why that would desync the
  // real combine weights from real routing).
  if (g.decode_mode && g.S == 1 && g.engine &&
      g.engine->family_ops.family == ModelFamily::Qwen4Exp &&
      g.engine->qwen4exp_gpu_stage_layer_enabled(layer)) {
    const int stage_idx = build_gpu_stage_node(g, ctx, ids, layer);
    if (stage_idx >= 0) {
      const FullGraph::GpuStageBinding &sb = g.gpu_stage_bindings[
          static_cast<size_t>(stage_idx)];
      ggml_tensor *input3_stage = ggml_reshape_3d(ctx, cur, g.D, 1, g.S);
      ggml_tensor *up_s = ggml_mul_mat_id(ctx, sb.up_dst, input3_stage,
                                          sb.ids_dispatch_const);
      ggml_tensor *gate_s = ggml_mul_mat_id(ctx, sb.gate_dst, input3_stage,
                                            sb.ids_dispatch_const);
      if (up_s && gate_s && sb.gpu_stage_marker_dep) {
        // Force ordering: the marker's D2H/callback/H2D sequence must
        // complete before either matmul reads gate_dst/up_dst -- see
        // GpuStageBinding::gpu_stage_marker_dep's own comment.
        up_s->src[3] = sb.gpu_stage_marker_dep;
        gate_s->src[3] = sb.gpu_stage_marker_dep;
      }
      ggml_tensor *act_s = (up_s && gate_s)
                              ? ggml_swiglu_split(ctx, gate_s, up_s)
                              : nullptr;
      ggml_tensor *down_s =
          act_s ? ggml_mul_mat_id(ctx, sb.down_dst, act_s,
                                  sb.ids_dispatch_const)
                : nullptr;
      if (down_s && sb.gpu_stage_marker_dep)
        down_s->src[3] = sb.gpu_stage_marker_dep;
      if (down_s) {
        // Same weighted-sum-over-topk reduction as the shared tail below
        // (out = view+add loop over `experts`) -- duplicated here rather
        // than falling through to it, matching build_qwen4exp_host_node's
        // own self-contained-early-return convention for a dispatch
        // strategy that fully replaces the ordinary mul_mat_id chain.
        ggml_tensor *weighted_s = ggml_mul(ctx, down_s, weights);
        ggml_tensor *out_s =
            moe_routed::sum_topk(ctx, weighted_s, g.D, g.S, c.topk);
        (void)layer;
        return out_s;
      }
    }
    LOG_WARN("[QWEN4EXP-GPU-STAGE] layer=%d falling through to the next "
             "dispatch strategy (CPU-offload if enabled, else plain CPU)\n",
             layer);
  }

  // The host callback is a single-token strided CPU contract.  Prefill must
  // keep the ordinary batched CPU/GGML MoE path so it can seed recurrent/KV
  // state; entering this branch for S>1 returns a null marker and leaves the
  // graph builder with no valid routed result.
  // A static per-expert split has a GPU branch and a CPU branch constructed
  // below.  Do not let the whole-layer host marker intercept it: doing so
  // silently discards the GPU hot set and runs every routed expert in the
  // callback, which is both slower and contrary to Option B's contract.
  const bool split_planned_for_layer =
      split_gpu_ids && !split_gpu_ids->empty() && gate_exps_gpu &&
      up_exps_gpu && down_exps_gpu;
  if (g.decode_mode &&
      g.engine->family_ops.family == ModelFamily::Qwen4Exp &&
      env_qwen4exp_host_node_enabled() &&
      g.engine->qwen4exp_cpu_offload_layer_enabled(layer) &&
      !split_planned_for_layer) {
    // The marker returns the already weighted/reduced routed result, so it
    // intentionally bypasses the ordinary three MUL_MAT_ID + SwiGLU chain.
    // A failure is fatal to this opt-in graph build; falling through to the
    // scheduler CPU path here would make a supposedly unified capture
    // silently contain the old scpu work again.
    ggml_tensor *host_node =
        build_qwen4exp_host_node(g, ctx, cur, ids, weights, layer);
    if (!host_node)
      return nullptr;
    return host_node;
  }
  if (g.decode_mode &&
      g.engine->family_ops.family == ModelFamily::Qwen4Exp &&
      env_qwen4exp_host_node_enabled() &&
      g.engine->qwen4exp_cpu_offload_layer_enabled(layer) &&
      split_planned_for_layer && qwen4exp_host_node_timing_enabled()) {
    LOG_DEBUG("[QWEN4EXP-HOST-NODE] layer=%d bypassed whole-layer marker "
             "because a static GPU hot/cold split is planned\n",
             layer);
  }

  ggml_tensor *input3 = ggml_reshape_3d(ctx, cur, g.D, 1, g.S);
  ggml_tensor *experts = nullptr;

  // Genuine per-expert split path: route IDs are remapped independently for
  // each backend, over the FULL topk width. See the S==1 note below for how
  // foreign routes are neutralized; the mask_gpu/mask_cpu multiply zeroes
  // their contribution either way. Every other layer/config takes the
  // unchanged single-mul_mat_id path below.
  if (split_gpu_ids && !split_gpu_ids->empty() && gate_exps_gpu &&
      up_exps_gpu && down_exps_gpu) {
    // Decode (S==1): a real -1 id makes each backend skip the foreign route
    // and write zeros (patches/ggml-mul-mat-id-negative-skip.patch covers the
    // CPU op and the CUDA MMVQ decode kernels), so each branch does only its
    // own share of the expert compute. Prefill (S>1) goes through mmq/mmf/
    // sorted paths the patch does not cover, so it keeps the valid dummy id
    // and relies on the mask multiply to zero the foreign contribution.
    ExpertSplitTables tab = build_expert_split_tables(
        g, ctx, c.n_experts, *split_gpu_ids,
        /*safe_dummy_fill=*/g.S != 1 && !g.decode_mode);
    if (tab.remap_gpu && tab.remap_cpu && tab.mask_gpu && tab.mask_cpu) {
      if (g.decode_mode) {
        auto &pl = g.promo_layers[layer];
        pl.remap_gpu = tab.remap_gpu; pl.remap_cpu = tab.remap_cpu;
        pl.mask_gpu = tab.mask_gpu;   pl.mask_cpu = tab.mask_cpu;
      }
      const char *early_env = zeromoe::graphcfg::env("ZEROMOE_QWEN_EARLY_READY_LAYER13");
      const char *all_early_env = zeromoe::graphcfg::env("ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT");
      const bool early_requested =
          ((early_env && early_env[0] == '1' && layer == 13) ||
           (!all_early_env || all_early_env[0] != '0')) &&
          g.decode_mode && g.S >= 1 &&
          (g.S == 1 || window_lane_enabled());
      const bool partial_requested =
          g.decode_mode && g.S == 1 && env_partial_graph_enabled(layer);
      if (partial_requested && early_requested) {
        LOG_WARN("[QWEN4EXP-PARTIAL-GRAPH] layer=%d cannot combine "
                 "partial host callback with early-ready in one A/B\n",
                 layer);
        return nullptr;
      }
      std::unique_ptr<EarlyReadyLane> early;
      ExpertSplitTables route_tab = tab;
      if (early_requested) {
        GGML_ASSERT(ggml_backend_sched_early_cpu_supported &&
                    ggml_backend_sched_early_cpu_supported() == 1);
        GGML_ASSERT(!g.layer_timing_enabled && g.early_ready.count(layer) == 0);
        early = std::make_unique<EarlyReadyLane>();
        GGML_ASSERT(early->prepare(g.cpu, g.device, g.D, c.topk,
                    gate_exps, up_exps, down_exps, tab.remap_cpu, tab.mask_cpu, g.S));
        if (early->resident_tables) {
          GGML_ASSERT(early->prepare_gpu_tables(g.cuda, ctx,
                      tab.remap_gpu, tab.remap_cpu, tab.mask_gpu, tab.mask_cpu));
          route_tab = {early->gpu_tables[0], early->gpu_tables[1],
                       early->gpu_tables[2], early->gpu_tables[3]};
          for (auto *table : early->gpu_tables)
            ggml_backend_sched_set_tensor_backend(g.sched, table, g.cuda);
        }
        const char *stage = g.S == 1 ? zeromoe::graphcfg::env("ZEROMOE_QWEN_EARLY_READY_STAGE_LAYER") : nullptr;
        if (stage && g.split_gpu_expert_ids.count(layer - 1) &&
            (std::strcmp(stage, "all") == 0 || std::atoi(stage) == layer)) {
          if (!g.early_stage_shapes)
            g.early_stage_shapes = std::make_unique<Qwen4ExpStageShapeClassPool>();
          Qwen4ExpStageShapeClassPool::Key shape_key;
          shape_key.topk = c.topk;
          shape_key.shape = {gate_exps->ne[0], gate_exps->ne[1],
                             up_exps->ne[0], up_exps->ne[1],
                             down_exps->ne[0], down_exps->ne[1]};
          shape_key.type = {static_cast<uint32_t>(gate_exps->type),
                            static_cast<uint32_t>(up_exps->type),
                            static_cast<uint32_t>(down_exps->type)};
          auto *shape_service = g.early_stage_shapes->service_for(shape_key);
          auto pool = std::make_unique<Qwen4ExpStagePool>();
          GGML_ASSERT(layer > 0 && g.split_gpu_expert_ids.count(layer - 1));
          GGML_ASSERT(pool->prepare(g.cuda, ctx, g.device, c.topk,
                      gate_exps, up_exps, down_exps, early->cold_mask,
                      shape_service));
          early->stage_target = pool.get();
          GGML_ASSERT(early->prepare_stage_routes(g.cuda, ctx));
          for (auto *weight : pool->weights)
            ggml_backend_sched_set_tensor_backend(g.sched, weight, g.cuda);
          ggml_backend_sched_set_tensor_backend(g.sched, early->stage_gpu_ids, g.cuda);
          ggml_backend_sched_set_tensor_backend(g.sched, early->stage_gpu_slots, g.cuda);
          ggml_backend_sched_set_tensor_backend(g.sched, early->stage_gpu_mask, g.cuda);
          g.early_stage_pools.push_back(std::move(pool));
        }
      }
      // The split tables are [1, n_experts, 1]; ggml_get_rows wants one row
      // index per element, so a window (S > 1) looks up the flattened ids and
      // the results are reshaped back to the [topk, S] / weights layout.
      ggml_tensor *ids_flat =
          g.S == 1 ? ids
                   : ggml_reshape_1d(ctx, ggml_cont(ctx, ids),
                                     static_cast<int64_t>(c.topk) * g.S);
      const auto lookup_like_weights = [&](ggml_tensor *table) {
        ggml_tensor *r = ggml_get_rows(ctx, table, ids_flat);
        return g.S == 1 ? r
                        : ggml_reshape_4d(ctx, r, weights->ne[0], weights->ne[1],
                                          weights->ne[2], weights->ne[3]);
      };
      ggml_tensor *ids_gpu_raw = ggml_get_rows(ctx, route_tab.remap_gpu, ids_flat);
      ggml_tensor *ids_gpu = ggml_reshape_2d(ctx, ids_gpu_raw, c.topk, g.S);
      ggml_tensor *ids_cpu_raw = ggml_get_rows(ctx, route_tab.remap_cpu, ids_flat);
      ggml_tensor *ids_cpu = ggml_reshape_2d(ctx, ids_cpu_raw, c.topk, g.S);
      ggml_tensor *weights_gpu =
          ggml_mul(ctx, weights, lookup_like_weights(route_tab.mask_gpu));
      ggml_tensor *weights_cpu =
          ggml_mul(ctx, weights, lookup_like_weights(route_tab.mask_cpu));

      ggml_tensor *cold_weighted = nullptr;
      if (partial_requested) {
        // DDR-only partial milestone: the callback reduces the cold
        // complement while the resident CUDA branch below computes hot
        // routes. SSD service and staged publication are intentionally absent
        // from this first graph; a missing DDR arena declines the build.
        cold_weighted = build_qwen4exp_host_node(
            g, ctx, cur, ids, weights, layer, /*partial_cold_only=*/true,
            gate_exps, up_exps, down_exps, tab.remap_cpu, tab.mask_cpu);
        if (!cold_weighted) {
          LOG_WARN("[QWEN4EXP-PARTIAL-GRAPH] layer=%d declined: DDR "
                   "cold callback could not be built\n",
                   layer);
          return nullptr;
        }
        LOG_INFO("[QWEN4EXP-PARTIAL-GRAPH] layer=%d resident CUDA + "
                 "cold DDR host callback\n",
                 layer);
      }
      if (early) {
        auto *dst = ggml_dup_tensor(ctx, cur);
        early->device_buffer = ggml_backend_alloc_buffer(g.cuda, ggml_nbytes(dst));
        GGML_ASSERT(early->device_buffer);
        GGML_ASSERT(ggml_backend_tensor_alloc(early->device_buffer, dst,
                    ggml_backend_buffer_get_base(early->device_buffer)) == GGML_STATUS_SUCCESS);
        auto *fork = ggml_cpy(ctx, cur, dst);
        fork->src[2] = ggml_cont(ctx, ids);
        fork->src[3] = ggml_cont(ctx, weights);
        fork->src[4] = g.positions;
        fork->extra = early.get();
        ggml_set_name(fork, ("qwen_early_ready_fork_l" + std::to_string(layer)).c_str());
        ggml_backend_sched_set_tensor_backend(g.sched, dst, g.cuda);
        ggml_backend_sched_set_tensor_backend(g.sched, fork, g.cuda);
        early->gpu_input = fork;
        input3 = ggml_reshape_3d(ctx, fork, g.D, 1, g.S);
        ggml_build_forward_expand(g.graph, fork);
        const char *predict = zeromoe::graphcfg::env("ZEROMOE_QWEN_EARLY_READY_PREDICT");  // also for windows (lane on)
        const char *target = zeromoe::graphcfg::env("ZEROMOE_QWEN_EARLY_READY_PREDICT_LAYER");
        const char *stage = g.S == 1 ? zeromoe::graphcfg::env("ZEROMOE_QWEN_EARLY_READY_STAGE_LAYER") : nullptr;
        const bool stage_next = stage &&
            (std::strcmp(stage, "all") == 0 || std::atoi(stage) == layer + 1);
        const bool predict_this_layer = !target || std::atoi(target) == layer + 1;
        if (((predict && predict[0] == '1' && predict_this_layer) || stage_next) &&
            g.split_gpu_expert_ids.count(layer + 1)) {
          const std::string next = "blk." + std::to_string(layer + 1) + ".";
          auto *next_router = resident_tensor(g, ctx, next + "ffn_gate_inp.weight");
          auto *next_bias = resident_tensor(g, ctx, next + "exp_probs_b.bias");
          GGML_ASSERT(next_router && early->prepare_prediction(g.cuda, ctx,
                      next_router, next_bias, layer + 1));
          ggml_backend_sched_set_tensor_backend(g.sched, early->prediction, g.cuda);
          if (stage_next) {
            GGML_ASSERT(early->prepare_prediction_readback(g.cuda, ctx));
            ggml_backend_sched_set_tensor_backend(g.sched, early->prediction_marker, g.cuda);
            ggml_build_forward_expand(g.graph, early->prediction_marker);
          } else {
            ggml_build_forward_expand(g.graph, early->prediction);
          }
        }
        auto *cold_leaf = ggml_dup_tensor(ctx, early->output);
        cold_leaf->buffer = early->output->buffer;
        cold_leaf->data = early->output->data;
        cold_weighted = ggml_map_custom1_inplace(ctx, cold_leaf,
                         EarlyReadyLane::fallback, 1, early.get());
        cold_weighted->extra = early.get();
        ggml_set_name(cold_weighted, ("qwen_early_ready_cold_l" + std::to_string(layer)).c_str());
        ggml_backend_sched_set_tensor_backend(g.sched, cold_leaf, g.cpu);
        ggml_backend_sched_set_tensor_backend(g.sched, cold_weighted, g.cpu);
        if (early->async_output_copy) {
          early->cpu_marker = cold_weighted;
          GGML_ASSERT(early->prepare_gpu_output(g.cuda, ctx));
          ggml_set_name(early->gpu_output,
                        ("qwen_early_ready_output_l" + std::to_string(layer)).c_str());
          ggml_backend_sched_set_tensor_backend(g.sched, early->gpu_output, g.cuda);
          cold_weighted = early->gpu_output;
        }
        g.early_ready.emplace(layer, std::move(early));
        LOG_INFO("[OPTION-B-EARLY-READY] layer=%d private CPU graph enabled\n", layer);
      }

      moe_routed::ExpertWeights hot_w;
      hot_w.gate = gate_exps_gpu;
      hot_w.up = up_exps_gpu;
      hot_w.down = down_exps_gpu;
      const moe_routed::FfnNodes hot_ffn =
          moe_routed::expert_ffn(ctx, hot_w, input3, ids_gpu);
      ggml_tensor *experts_gpu = hot_ffn.down;

      ggml_tensor *hot_weighted = ggml_mul(ctx, experts_gpu, weights_gpu);
      const auto stage = g.early_ready.find(layer);
      if (stage != g.early_ready.end() && stage->second->stage_target) {
        auto &early = *stage->second;
        auto &pool = *early.stage_target;
        moe_routed::ExpertWeights stage_w;
        stage_w.gate = pool.weights[0];
        stage_w.up = pool.weights[1];
        stage_w.down = pool.weights[2];
        const moe_routed::FfnNodes stage_ffn =
            moe_routed::expert_ffn(ctx, stage_w, input3, early.stage_gpu_ids);
        auto *staged_weighted = ggml_mul(ctx, stage_ffn.down, ggml_mul(ctx, weights, early.stage_gpu_mask));
        if (pool.verify_enabled)
          GGML_ASSERT(pool.prepare_verification(staged_weighted, g.early_stage_pools.front().get()));
        hot_weighted = ggml_add(ctx, hot_weighted, staged_weighted);
      }
      if (cold_weighted) {
        // Submit hot work before the independent private CPU marker.
        ggml_build_forward_expand(g.graph, hot_weighted);
      } else {
        // Opt-in fused cold branch (ZEROMOE_QWEN4EXP_COLD_FUSED=1): one CPU node for the whole branch.
        // Resident-expert routes and the recorded masks are read from the same host tables as below.
        ggml_tensor *fused = nullptr;
        if (cold_fused_enabled() && g.decode_mode && !g.stage_route_ids &&
            qwen4exp_cold::supported(gate_exps, up_exps, down_exps, c.topk, g.S)) {
          std::shared_ptr<qwen4exp_cold::Context> owner;
          fused = qwen4exp_cold::build(
              ctx, owner, input3, ggml_cont(ctx, ids),
              ggml_cont(ctx, weights), tab.remap_cpu, tab.mask_cpu, gate_exps, up_exps,
              down_exps, g.cpu_threads);
          if (fused) {
            g.cold_fused.push_back(std::move(owner));
            ggml_backend_sched_set_tensor_backend(g.sched, fused, g.cpu);
            cold_weighted = fused;
          }
        }
        if (!fused) {
          moe_routed::ExpertWeights cold_w;
          cold_w.gate = gate_exps;
          cold_w.up = up_exps;
          cold_w.down = down_exps;
          const moe_routed::FfnNodes cold_ffn =
              moe_routed::expert_ffn(ctx, cold_w, input3, ids_cpu);
          ggml_tensor *experts_cpu = cold_ffn.down;
          cold_weighted = ggml_mul(ctx, experts_cpu, weights_cpu);
        }
      }
      experts = ggml_add(ctx, hot_weighted, cold_weighted);
    }
  }

  if (!experts) {
    // llama.cpp's Qwen graph and the CPU runner use one routed matmul for the
    // adjacent gate+up planes (a zero-copy view over the shared DDR arena for
    // CPU layers); layers whose source arena cannot present adjacent,
    // type-matched planes fall back to separate gate and up matmuls.
    moe_routed::ExpertWeights ew;
    ew.gate_up = gate_up_exps;
    ew.gate = gate_exps;
    ew.up = up_exps;
    ew.down = down_exps;
    const moe_routed::FfnNodes ffn =
        moe_routed::expert_ffn(ctx, ew, input3, ids);
    const char *probe_dump =
        zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_PROBE_DUMP");
    const char *probe_layer_text =
        zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_PROBE_LAYER");
    const int probe_layer = probe_layer_text ? std::atoi(probe_layer_text) : 0;
    if (g.engine->family_ops.family == ModelFamily::Qwen &&
        layer == probe_layer && probe_dump) {
      auto probe_expert = [&](const char *label, ggml_tensor *value) {
        if (!value || !g.graph)
          return;
        const std::string name = "qwen35_expert_" + std::string(label) + "_" +
                                 std::to_string(layer);
        ggml_set_name(value, name.c_str());
        ggml_set_output(value);
        ggml_build_forward_expand(g.graph, value);
        g.gdn_probe_taps.emplace_back(layer, "expert_" + std::string(label),
                                      value);
      };
      // Select one layer so the diagnostic exposes projection arithmetic
      // without retaining every expert intermediate across the full graph.
      probe_expert("gate", ffn.gate);
      probe_expert("up", ffn.up);
      probe_expert("act", ffn.act);
      probe_expert("down", ffn.down);
    }
    ggml_tensor *up = ffn.up;
    ggml_tensor *gate = ffn.gate;
    ggml_tensor *gate_up = ffn.gate_up; // only set on the fused path
    ggml_tensor *act = ffn.act;
    ggml_tensor *down = ffn.down;
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

  ggml_tensor *out = moe_routed::sum_topk(ctx, experts, g.D, g.S, c.topk);
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

} // namespace qwen4exp_graph

#endif // ZEROMOE_USE_GGML_CUDA
