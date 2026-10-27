// Qwen4Exp full graph: Prefill and decode execution of the built graph.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_qwen4exp_graph_internal.hpp.

#include "ggml_qwen4exp_graph_internal.hpp"
#include "tensor_trace.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)

using namespace qwen4exp_graph;

bool qwen4exp_full_graph_decode_run(StreamingEngine &engine,
                                    float *hidden_states, int pos_base,
                                    const int *token_ids, float *logits,
                                    int vocab) {
  if (!full_graph_decode_enabled_for_engine(engine) || !hidden_states ||
      pos_base < 0 ||
      !g_graph || !g_graph->ready || !g_graph->decode_mode ||
      g_graph->engine != &engine || g_graph->S != 1 ||
      pos_base >= g_graph->kv_capacity ||
      ((logits == nullptr) != (vocab <= 0)))
    return false;
  FullGraph &g = *g_graph;
  const bool standard_qwen = engine.family_ops.family == ModelFamily::Qwen;
  if (!standard_qwen &&
      engine.hc_stream_buf.size() < static_cast<size_t>(g.D) * g.HC)
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

  const float *graph_input = standard_qwen
                                 ? hidden_states
                                 : engine.hc_stream_buf.data();
  std::memcpy(g.input->data, graph_input,
              static_cast<size_t>(g.D) * g.HC * sizeof(float));
  auto *positions = static_cast<int32_t *>(g.positions->data);
  if (g.position_ids_per_token == 4) {
    positions[0] = pos_base;
    positions[1] = pos_base;
    positions[2] = pos_base;
    positions[3] = 0;
  } else {
    positions[0] = pos_base;
  }
  auto *mask = static_cast<ggml_fp16_t *>(g.attn_mask->data);
  for (int k = 0; k < g.kv_capacity; ++k)
    mask[k] = ggml_fp32_to_fp16(k <= pos_base ? 0.0f : -INFINITY);
  auto *indices = static_cast<int32_t *>(g.decode_indices->data);
  // See build_graph()'s initial decode_indices fill for why this must be
  // h * kv_capacity + pos_base, not pos_base * Hkv + h.
  for (int h = 0; h < g.Hkv; ++h)
    indices[h] = h * g.kv_capacity + pos_base;

  if (g.layer_timing_enabled) {
    std::fill(g.layer_timing_cpu_ms.begin(), g.layer_timing_cpu_ms.end(), 0.0);
    g.layer_timing_op_ms.clear();
    g.layer_timing_t0.clear();
    g.layer_timing_gap_ms_total = 0.0;
    g.layer_timing_gap_count = 0;
    g.layer_timing_have_last_ask_false = false;
  }
  if (!qwen4exp_stage_layer13_before_graph(g, pos_base)) {
    LOG_WARN("[OPTION-B-STAGE] declining graph replay because the Layer-13 "
             "staging proof could not be queued\n");
    return false;
  }
  for (auto &entry : g.early_ready) {
    entry.second->expected_pos = pos_base;
    entry.second->residency_ready = false;
    entry.second->completed = false;
  }
  const auto start = std::chrono::steady_clock::now();
  const auto replay_start = std::chrono::steady_clock::now();
  if (ggml_backend_sched_graph_compute_async(g.sched, g.graph) !=
      GGML_STATUS_SUCCESS)
    return false;
  const auto replay_submit_end = std::chrono::steady_clock::now();
  if (g.stage_enabled && g.stage_compute_end_event) {
    cudaStream_t graph_stream =
        zeromoe_ggml_backend_cuda_get_stream(g.cuda_raw);
    if (!graph_stream ||
        cudaEventRecord(g.stage_compute_end_event, graph_stream) !=
            cudaSuccess)
      return false;
  }
  ggml_backend_sched_synchronize(g.sched);
  if (g.engine && g.engine->exl3_gpu_executor)
    (void)g.engine->exl3_gpu_executor->stats();
  qwen_probe_dump(g);
  const auto replay_sync_end = std::chrono::steady_clock::now();
  for (const auto &entry : g.early_ready) {
    if (entry.second->stage_source)
      entry.second->stage_source->token_complete(entry.second->calls);
    if (entry.second->stage_target) {
      auto &early = *entry.second;
      early.stage_target->verify(early.input, early.weights, early.ids, early.stage_ids);
    }
  }
  const char *route_trace = zeromoe::graphcfg::env("ZEROMOE_QWEN_EARLY_READY_ROUTE_TRACE");
  if (route_trace && route_trace[0] && !g.early_ready.empty()) {
    if (!g.early_route_trace) {
      g.early_route_trace = std::make_unique<std::ofstream>(route_trace);
      GGML_ASSERT(g.early_route_trace->is_open());
      for (int layer = 0; layer < g.n_layers; ++layer) {
        const auto entry = g.early_ready.find(layer);
        if (entry == g.early_ready.end()) continue;
        auto &trace = *g.early_route_trace;
        const auto &mask = entry->second->cold_mask;
        trace << "{\"kind\":\"layer\",\"layer\":" << layer
              << ",\"expert_bytes\":" << entry->second->expert_bytes << ",\"cold_mask\":[";
        for (size_t e = 0; e < mask.size(); ++e)
          trace << (e ? "," : "") << int(mask[e]);
        trace << "]}\n";
      }
    }
    auto &trace = *g.early_route_trace;
    for (int layer = 0; layer < g.n_layers; ++layer) {
      const auto entry = g.early_ready.find(layer);
      if (entry == g.early_ready.end()) continue;
      const auto &early = *entry->second;
      trace << "{\"position\":" << pos_base << ",\"layer\":" << layer << ",\"ids\":[";
      for (int k = 0; k < early.k; ++k)
        trace << (k ? "," : "") << early.ids[k];
      trace << "],\"cold\":[";
      for (int k = 0; k < early.k; ++k)
        trace << (k ? "," : "") << int(early.cold_mask[early.ids[k]]);
      trace << "],\"epoch\":" << early.calls << ",\"stage_ids\":[";
      if (early.stage_target)
        for (int k = 0; k < early.k; ++k)
          trace << (k ? "," : "") << early.stage_ids[k];
      trace << "],\"prediction_layer\":" << early.prediction_layer << ",\"prediction\":[";
      if (early.prediction) {
        std::vector<int32_t> ids(early.k);
        ggml_backend_tensor_get(early.prediction, ids.data(), 0, ids.size() * sizeof(int32_t));
        for (int k = 0; k < early.k; ++k)
          trace << (k ? "," : "") << ids[k];
      }
      trace << "]}\n";
    }
    trace.flush();
    GGML_ASSERT(trace.good());
  }
  for (const auto &entry : g.early_ready) {
    const auto &early = *entry.second;
    GGML_ASSERT(early.completed);
    if (early.calls % 32 == 0)
      LOG_INFO("[OPTION-B-EARLY-READY] calls=%llu early=%llu gpu_pending=%llu wait=%.2fus cpu=%.2fus layer=%d all_hot=%llu skipped=%llu uploads=%llu staged=%llu\n",
               (unsigned long long)early.calls,
               (unsigned long long)early.early_calls,
               (unsigned long long)early.gpu_pending,
               double(early.wait_us) / early.calls,
               double(early.compute_us) / early.calls, entry.first,
               (unsigned long long)early.all_hot_visits,
               (unsigned long long)early.all_hot_skips,
               (unsigned long long)early.output_uploads,
               (unsigned long long)early.staged_routes);
    if (early.calls % 32 == 0 && early.stage_target) {
      const auto &pool = *early.stage_target;
      const auto pt = pool.timing();
      const auto wt = pool.service
                          ? pool.service->stats()
                          : Qwen4ExpStageWorkService::Stats{};
      LOG_INFO("[OPTION-B-STAGE-PREDICT] layer=%d requested=%llu dropped=%llu "
               "predicted=%llu uploaded=%llu late=%llu verified=%llu max_delta=%.9g "
               "expected_max=%.9g actual_max=%.9g dma_avg=%.3fms "
               "worker_queue_avg=%.3fms worker_run_avg=%.3fms\n", entry.first,
               (unsigned long long)pool.requests.load(),
               (unsigned long long)pool.dropped.load(),
               (unsigned long long)pool.predicted.load(),
               (unsigned long long)pool.uploaded.load(),
               (unsigned long long)pool.late.load(),
               (unsigned long long)pool.verified, pool.verify_max_delta,
               pool.verify_expected_max_abs, pool.verify_actual_max_abs,
               pt.dma_completed ? pt.dma_ms_total / pt.dma_completed : 0.0,
               wt.completed ? wt.queue_ms_total / wt.completed : 0.0,
               wt.completed ? wt.run_ms_total / wt.completed : 0.0);
    }
  }
  double host_node_d2h_ms = 0.0;
  double host_node_h2d_ms = 0.0;
  double host_node_callback_span_ms = 0.0;
  double host_node_cpu_ms = 0.0;
  double host_node_callback_total_ms = 0.0;
  double host_node_callback_guard_wait_ms = 0.0;
  double host_node_callback_body_ms = 0.0;
  bool host_node_all_ok = true;
  if (!g.host_node_contexts.empty()) {
    float d2h_ms = 0.0f;
    float h2d_ms = 0.0f;
    double cpu_ms = 0.0;
    bool all_ok = true;
    for (const auto &host_node : g.host_node_contexts) {
      if (!host_node)
        continue;
      float node_d2h = 0.0f;
      float node_h2d = 0.0f;
      float node_callback_span = 0.0f;
      if (cudaEventElapsedTime(&node_d2h, host_node->d2h_start,
                               host_node->d2h_end) != cudaSuccess ||
          cudaEventElapsedTime(&node_h2d, host_node->h2d_start,
                               host_node->h2d_end) != cudaSuccess) {
        LOG_WARN("[QWEN4EXP-HOST-NODE] event timing unavailable at pos=%d\n",
                 pos_base);
      } else {
        d2h_ms += node_d2h;
        h2d_ms += node_h2d;
      }
      if (cudaEventElapsedTime(&node_callback_span, host_node->d2h_end,
                               host_node->h2d_start) == cudaSuccess)
        host_node_callback_span_ms += node_callback_span;
      cpu_ms += host_node->last_cpu_ms.load(std::memory_order_acquire);
      host_node_callback_total_ms +=
          host_node->last_callback_total_ms.load(std::memory_order_acquire);
      host_node_callback_guard_wait_ms +=
          host_node->last_callback_guard_wait_ms.load(
              std::memory_order_acquire);
      host_node_callback_body_ms +=
          host_node->last_callback_body_ms.load(std::memory_order_acquire);
      all_ok = all_ok && host_node->last_ok.load(std::memory_order_acquire);
      if (host_node->transport_diag && host_node->host_output_gpu_check) {
        const size_t n = host_node->output_bytes / sizeof(float);
        const uint64_t cpu_hash =
            host_node->last_cpu_hash.load(std::memory_order_acquire);
        const uint64_t gpu_hash =
            qwen4exp_fnv64(host_node->host_output_gpu_check,
                           host_node->output_bytes);
        host_node->last_gpu_hash.store(gpu_hash, std::memory_order_release);
        float max_abs_delta = 0.0f;
        double sum_delta = 0.0;
        for (size_t i = 0; i < n; ++i) {
          max_abs_delta = std::max(
              max_abs_delta,
              std::abs(host_node->host_output[i] -
                        host_node->host_output_gpu_check[i]));
          sum_delta += static_cast<double>(host_node->host_output_gpu_check[i]) -
                       static_cast<double>(host_node->host_output[i]);
        }
        LOG_INFO("[QWEN4EXP-HOST-NODE-DIAG] layer=%d "
                 "cpu_pre_h2d_fnv64=0x%016llx "
                 "gpu_post_h2d_fnv64=0x%016llx hash_match=%d "
                 "max_abs_delta=%.9g sum_delta=%.9g bytes=%zu\n",
                 host_node->layer_index,
                 static_cast<unsigned long long>(cpu_hash),
                 static_cast<unsigned long long>(gpu_hash),
                 host_node->last_cpu_hash_valid.load(
                         std::memory_order_acquire) &&
                         cpu_hash == gpu_hash
                     ? 1
                     : 0,
                 max_abs_delta, sum_delta, host_node->output_bytes);
      }
    }
    const double scheduler_gate_wait_ms =
        qwen4exp_host_node_gate_wait_ms_since_last();
    const double ordinary_cpu_wait_ms =
        qwen4exp_host_node_ordinary_cpu_wait_ms_since_last();
    const double cpu_split_enter_wait_ms =
        qwen4exp_cpu_split_enter_wait_ms_since_last();
    LOG_INFO("[QWEN4EXP-HOST-NODE-TIME] pos=%d nodes=%zu D2H=%.3fms "
             "CPU=%.3fms H2D=%.3fms callback_total=%.3fms "
             "callback_span=%.3fms callback_guard_wait=%.3fms "
             "callback_body=%.3fms "
             "scheduler_gate_wait=%.3fms ordinary_cpu_wait=%.3fms "
             "cpu_split_enter_wait=%.3fms "
             "callback_ok=%d\n",
             pos_base, g.host_node_contexts.size(), d2h_ms, cpu_ms, h2d_ms,
             host_node_callback_total_ms, host_node_callback_span_ms,
             host_node_callback_guard_wait_ms, host_node_callback_body_ms,
             scheduler_gate_wait_ms,
             ordinary_cpu_wait_ms, cpu_split_enter_wait_ms, all_ok ? 1 : 0);
    host_node_d2h_ms = d2h_ms;
    host_node_h2d_ms = h2d_ms;
    host_node_cpu_ms = cpu_ms;
    host_node_all_ok = all_ok;
    if (!all_ok)
      return false;
  }
  // GPU-staging D2H/gather/H2D breakdown -- the events were already created
  // and recorded at enqueue time (qwen4exp_cuda_gpu_stage_node_enqueue) but
  // never queried anywhere; add that here, mirroring the host-node ledger
  // immediately above, so a real per-op breakdown exists instead of only
  // the callback body's own wall-clock (ZEROMOE_QWEN4EXP_GPU_STAGE_TIMING).
  if (!g.gpu_stage_contexts.empty()) {
    float d2h_ms = 0.0f;
    float h2d_ms = 0.0f;
    float callback_span_ms = 0.0f;
    double callback_body_ms = 0.0;
    bool all_ok = true;
    for (const auto &stage_ctx : g.gpu_stage_contexts) {
      if (!stage_ctx)
        continue;
      float node_d2h = 0.0f;
      float node_h2d = 0.0f;
      float node_callback_span = 0.0f;
      // NOTE: cudaEventElapsedTime reliably fails with cudaErrorInvalidValue
      // here on every REPLAY token (never on the first/capture token, and
      // forcing cudaEventSynchronize first does not help) -- a real CUDA
      // Graphs limitation, not a race: events recorded as NODES inside a
      // captured graph are not reliably re-queryable for elapsed time
      // across multiple cudaGraphLaunch replays via this API. D2H/H2D stay
      // unmeasured; callback_body (host-side chrono, not CUDA-event-based)
      // is unaffected and remains the reliable number.
      const cudaError_t err_d2h = cudaEventElapsedTime(
          &node_d2h, stage_ctx->d2h_start, stage_ctx->d2h_end);
      const cudaError_t err_h2d = cudaEventElapsedTime(
          &node_h2d, stage_ctx->h2d_start, stage_ctx->h2d_end);
      if (err_d2h != cudaSuccess || err_h2d != cudaSuccess) {
        LOG_WARN("[QWEN4EXP-GPU-STAGE] event timing unavailable at pos=%d "
                 "d2h_err=%s h2d_err=%s\n",
                 pos_base, cudaGetErrorString(err_d2h),
                 cudaGetErrorString(err_h2d));
      } else {
        d2h_ms += node_d2h;
        h2d_ms += node_h2d;
      }
      if (cudaEventElapsedTime(&node_callback_span, stage_ctx->d2h_end,
                               stage_ctx->h2d_start) == cudaSuccess)
        callback_span_ms += node_callback_span;
      callback_body_ms +=
          stage_ctx->last_callback_ms.load(std::memory_order_acquire);
      all_ok = all_ok && stage_ctx->last_ok.load(std::memory_order_acquire);
    }
    LOG_INFO("[QWEN4EXP-GPU-STAGE-TIME] pos=%d layers=%zu D2H=%.3fms "
             "callback_span=%.3fms callback_body=%.3fms H2D=%.3fms "
             "total=%.3fms ok=%d\n",
             pos_base, g.gpu_stage_contexts.size(), d2h_ms, callback_span_ms,
             callback_body_ms, h2d_ms, d2h_ms + callback_span_ms + h2d_ms,
             all_ok ? 1 : 0);
  }
  if (g.stage_enabled && !g.stage_slots.empty()) {
    float h2d_ms = 0.0f, wait_ms = 0.0f, d2d_ms = 0.0f, compute_ms = 0.0f;
    cudaSetDevice(g.device);
    const ExpertLocation *first_slot = g.stage_slots.front();
    const ExpertLocation *last_slot = g.stage_slots.back();
    cudaEventElapsedTime(&h2d_ms, first_slot->upload_start_event,
                         last_slot->upload_event);
    // This brackets the actual device-side stream fence.  The previous
    // single-slot probe measured upload completion -> wait completion, which
    // excluded the stall when the compute stream arrived before the DMA.
    cudaEventElapsedTime(&wait_ms, g.stage_wait_start_event,
                         g.stage_wait_done_event);
    cudaEventElapsedTime(&d2d_ms, g.stage_d2d_start_event,
                         g.stage_d2d_end_event);
    cudaEventElapsedTime(&compute_ms, g.stage_compute_start_event,
                         g.stage_compute_end_event);
    LOG_INFO("[OPTION-B-STAGE-TIME] pos=%d layer=%d staged=%zu H2D=%.3fms "
             "wait=%.3fms D2D=%.3fms graph_cuda=%.3fms\n",
             pos_base, g.stage_layer, g.stage_slots.size(), h2d_ms, wait_ms,
             d2d_ms, compute_ms);
    for (ExpertLocation *slot : g.stage_slots)
      g.engine->gpu->topology.complete_active_layer_upload_after_compute(slot);
    g.stage_slots.clear();
    g.stage_slot = nullptr;
  }
  // Standard Qwen's full decode graph also computes and returns next-token
  // logits. The host hidden row is not consumed by the next step (the next
  // token embedding replaces it), and reading it back adds a synchronous
  // device-to-host transfer every token. Preserve it for graph modes whose
  // host path consumes hidden states, speculative decoding, or diagnostics.
  const bool need_host_hidden =
      !standard_qwen || !logits || engine.mtp_enabled ||
      trace_gate("HIDDEN") ||
      zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_PROBE_DUMP");
  if (need_host_hidden)
    ggml_backend_tensor_get(g.output, hidden_states, 0,
                            static_cast<size_t>(g.D) * sizeof(float));
  if (g.stage_enabled) {
    double l1 = 0.0;
    bool finite = true;
    for (int i = 0; i < g.D; ++i) {
      finite = finite && std::isfinite(hidden_states[i]);
      l1 += std::abs(static_cast<double>(hidden_states[i]));
    }
    LOG_INFO("[OPTION-B-STAGE-VERIFY] pos=%d output_finite=%d output_l1=%.6e\n",
             pos_base, finite ? 1 : 0, l1);
    if (!finite)
      return false;
  }
  if (logits) {
    ggml_backend_tensor_get(g.logits, logits, 0,
                            static_cast<size_t>(vocab) * sizeof(float));
  }
  if (engine.qwen4exp_wide_export && !standard_qwen) {
    const size_t wide_n = static_cast<size_t>(g.D) * g.HC;
    if (!g.final_streams || !ggml_is_contiguous(g.final_streams) ||
        static_cast<size_t>(ggml_nelements(g.final_streams)) != wide_n) {
      LOG_ERROR("[QWEN-MTP-EXPORT] decode graph has no contiguous wide "
                "residual of %zu floats\n", wide_n);
      return false;
    }
    engine.qwen4exp_wide_out.resize(wide_n);
    ggml_backend_tensor_get(g.final_streams, engine.qwen4exp_wide_out.data(), 0,
                            wide_n * sizeof(float));
  }
  g.decode_last_pos = pos_base;
  g.decode_host_state_dirty = env_prefill_state_handoff_enabled();

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
  // Avoid a synchronous INFO log for every generated token during normal
  // inference. The detailed replay ledger below remains available when its
  // profiling switch is enabled; layer profiling also retains this total.
  if (env_graph_replay_timing_enabled() || env_layer_timing_enabled()) {
    LOG_INFO("[GGML-QWEN-FULL-DECODE-TIME] pos=%d total=%.3f ms\n", pos_base,
             ms);
  }
  if (env_graph_replay_timing_enabled()) {
    double ordinary_cpu_compute_ms = 0.0;
    for (const double layer_ms : g.layer_timing_cpu_ms)
      ordinary_cpu_compute_ms += layer_ms;
    const double ordinary_cpu_gap_ms = g.layer_timing_gap_ms_total;
    const double replay_submit_ms = std::chrono::duration<double, std::milli>(
        replay_submit_end - replay_start).count();
    const double replay_sync_ms = std::chrono::duration<double, std::milli>(
        replay_sync_end - replay_submit_end).count();
    const double replay_total_ms = std::chrono::duration<double, std::milli>(
        replay_sync_end - replay_start).count();
    LOG_INFO(
        "[QWEN4EXP-GRAPH-REPLAY-LEDGER] pos=%d splits=%d host_nodes=%zu "
        "replay_submit=%.3fms replay_sync=%.3fms replay_total=%.3fms "
        "host_node_d2h=%.3fms host_node_cpu=%.3fms host_node_h2d=%.3fms "
        "host_node_ok=%d ordinary_cpu_compute=%.3fms "
        "ordinary_cpu_between_node_gap=%.3fms ordinary_cpu_ledger=%.3fms\n",
        pos_base, ggml_backend_sched_get_n_splits(g.sched),
        g.host_node_contexts.size(), replay_submit_ms, replay_sync_ms,
        replay_total_ms, host_node_d2h_ms, host_node_cpu_ms,
        host_node_h2d_ms, host_node_all_ok ? 1 : 0, ordinary_cpu_compute_ms,
        ordinary_cpu_gap_ms, ordinary_cpu_compute_ms + ordinary_cpu_gap_ms);
  }
  // Per-layer/per-op CPU timing (env: ZEROMOE_QWEN4EXP_GRAPH_LAYER_TIMING=1)
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
  if (!full_graph_enabled_for_engine(engine) || !hidden_states || S <= 1 ||
      pos_base < 0 || !g_graph ||
      !g_graph->ready || g_graph->engine != &engine || g_graph->S != S ||
      g_graph->pos_base != pos_base || g_graph->decode_mode)
    return false;
  FullGraph &g = *g_graph;
  const bool standard_qwen = engine.family_ops.family == ModelFamily::Qwen;
  const auto reset_host_recurrent_state = [&]() {
    for (auto &history : engine.qwen4exp_conv_history)
      std::fill(history.begin(), history.end(), 0.0f);
    for (auto &state : engine.qwen4exp_recur_state)
      std::fill(state.begin(), state.end(), 0.0f);
  };
  auto fail = [&](const char *why) {
    if (g.prefill_device_state_inputs)
      reset_host_recurrent_state();
    LOG_WARN("[GGML-QWEN-FULL] execution declined: %s\n", why);
    return false;
  };
  if (engine.qwen4exp_full_graph_logits_target && g.logits &&
      engine.qwen4exp_full_graph_logits_target_size < g.logits->ne[0]) {
    LOG_ERROR("[GGML-QWEN-FULL] prefill logits shape mismatch: target=%d "
              "graph=[%lld,%lld] model_vocab=%d\n",
              engine.qwen4exp_full_graph_logits_target_size,
              static_cast<long long>(g.logits->ne[0]),
              static_cast<long long>(g.logits->ne[1]),
              engine.model_file.architecture_config.vocab);
    return fail("prefill last-token logits destination too small");
  }
  LOG_INFO(
      "[GGML-QWEN-FULL] execute candidate S=%d gdn=%zu kv=%zu ple_layer=%d\n",
      S, g.gdn_layers.size(), g.kv_layers.size(), g.ple_layer);

  // A fresh position-zero graph starts a new recurrent/KV stream.  This is
  // the same reset boundary used by the established Qwen attention path.
  // A later batch of a chunked prompt continues the recurrent state, the dense KV rows and the PLE window.
  if (pos_base == 0) {
    if (!g.prefill_device_state_inputs)
      reset_host_recurrent_state();
    std::fill(engine.qwen4exp_ple_tok_hist.begin(),
              engine.qwen4exp_ple_tok_hist.end(),
              static_cast<int32_t>(
                  engine.model_file.architecture_config.ple_eos_token_id));
    engine.qwen4exp_ple_next_pos = 0;
  }
  for (const auto &pk : g.prev_kv_inputs) {
    if (pk.layer < 0 || pk.layer >= static_cast<int>(engine.Kc.size()) || !engine.Kc[pk.layer] ||
        !engine.Vc[pk.layer])
      return fail("dense KV cache of the earlier batches is not allocated");
    const size_t n = static_cast<size_t>(pos_base) * g.Hkv * g.hd * sizeof(float);
    std::memcpy(pk.k, engine.Kc[pk.layer], n);
    std::memcpy(pk.v, engine.Vc[pk.layer], n);
  }

  for (size_t i = 0; i < g.gdn_layers.size(); ++i) {
    const int l = g.gdn_layers[i];
    if (!g.prefill_device_state_inputs &&
        (i >= g.gdn_state_inputs.size() || i >= g.gdn_hist_inputs.size()))
      return fail("GDN input bookkeeping mismatch");
    const size_t state_n = static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn;
    const size_t hist_n = static_cast<size_t>(g.d_conv - 1) * g.conv_dim;
    if (l < 0 || l >= static_cast<int>(engine.qwen4exp_recur_state.size()) ||
        l >= static_cast<int>(engine.qwen4exp_conv_history.size()) ||
        engine.qwen4exp_recur_state[l].size() != state_n ||
        engine.qwen4exp_conv_history[l].size() != hist_n)
      return fail("GDN state/history size mismatch");
    if (!g.prefill_device_state_inputs) {
      // The host state is in the legacy layout; the scan op takes the native (transposed) one. A first batch starts from
      // zeros, where the layout does not matter; a later batch of a chunked prompt needs the transposition.
      gdn_state_transpose_host(g.gdn_state_inputs[i], engine.qwen4exp_recur_state[l].data(), g.gdn_hd, g.Hv_gdn);
      std::memcpy(g.gdn_hist_inputs[i], engine.qwen4exp_conv_history[l].data(),
                  hist_n * sizeof(float));
    }
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
  const float *prefill_input =
      standard_qwen ? hidden_states : engine.hc_stream_buf.data();
  if (g.host_inputs.empty() ||
      (!standard_qwen &&
       engine.hc_stream_buf.size() < static_cast<size_t>(g.D) * g.HC * S))
    return fail(standard_qwen ? "standard Qwen prefill input is missing"
                              : "expanded HC input is missing or too small");
  std::memcpy(g.host_inputs[0], prefill_input,
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
      return fail("host memory safety floor");
    }
  }

  const auto t_compute_start = std::chrono::steady_clock::now();
  ggml_status st = ggml_backend_sched_graph_compute_async(g.sched, g.graph);
  const auto t_submit_end = std::chrono::steady_clock::now();
  if (st != GGML_STATUS_SUCCESS) {
    LOG_ERROR("[GGML-QWEN-FULL] graph compute failed: status=%d\n", (int)st);
    return fail("scheduler compute failed");
  }
  ggml_backend_sched_synchronize(g.sched);
  if (g.engine && g.engine->exl3_gpu_executor)
    (void)g.engine->exl3_gpu_executor->stats();
  qwen_probe_dump(g);
  const auto t_sync_end = std::chrono::steady_clock::now();
  ggml_backend_tensor_get(g.output, hidden_states, 0,
                          static_cast<size_t>(g.D) * S * sizeof(float));
  if (engine.qwen4exp_wide_export &&
      engine.family_ops.family != ModelFamily::Qwen) {
    const size_t wide_n = static_cast<size_t>(g.D) * g.HC * S;
    if (!g.final_streams || !ggml_is_contiguous(g.final_streams) ||
        static_cast<size_t>(ggml_nelements(g.final_streams)) != wide_n) {
      LOG_ERROR("[QWEN-MTP-EXPORT] prefill graph has no contiguous wide "
                "residual of %zu floats\n", wide_n);
      return false;
    }
    if (pos_base == 0) engine.qwen4exp_wide_prompt.clear();
    const size_t before = engine.qwen4exp_wide_prompt.size();
    if (before != static_cast<size_t>(pos_base) * g.D * g.HC) {
      LOG_ERROR("[QWEN-MTP-EXPORT] prefill batch at pos %d does not continue "
                "the exported rows (%zu floats held)\n", pos_base, before);
      return false;
    }
    engine.qwen4exp_wide_prompt.resize(before + wide_n);
    ggml_backend_tensor_get(g.final_streams,
                            engine.qwen4exp_wide_prompt.data() + before, 0,
                            wide_n * sizeof(float));
  }
  if (g.logits && engine.qwen4exp_full_graph_logits_target) {
    // Standard Qwen builds logits for every prompt row; Qwen4Exp builds them
    // for the last row only. Read the last column that actually exists.
    const size_t last_column =
        static_cast<size_t>(std::max<int64_t>(g.logits->ne[1], 1) - 1);
    ggml_backend_tensor_get(g.logits,
                            engine.qwen4exp_full_graph_logits_target,
                            last_column * g.logits->nb[1],
                            static_cast<size_t>(g.logits->ne[0]) * sizeof(float));
    engine.qwen4exp_full_graph_logits_ready = true;
  }
  const auto t_output_end = std::chrono::steady_clock::now();


  g.prefill_host_state_deferred =
      env_prefill_state_handoff_enabled() &&
      qwen4exp_full_graph_decode_enabled() &&
      env_decode_borrow_prefill_arenas() && !engine.prompt_cache &&
      !engine.mtp_enabled && engine.qwen4exp_partial_layer < 0 &&
      engine.qwen4exp_gpu_stage_layers.empty() &&
      prepare_prefill_state_handoff(g);
  if (g.prefill_host_state_deferred) {
    LOG_INFO("[GGML-QWEN-FULL-STATE-HANDOFF] deferred GDN host readback; "
             "retaining %.3f GiB prefill workspace for decode\n",
             static_cast<double>(g.graph_bytes) / (1024.0 * 1024.0 * 1024.0));
  } else if (!sync_prefill_recurrent_state_to_host(g)) {
    return fail("GDN host state readback failed");
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
    // Prefill is the only place where the full S>1 PLE graph writes the
    // recurrent convolution history.  Keep a diagnostic copy of that exact
    // post-compute state so it can be compared with llama.cpp's prefill cache
    // without disturbing the state owner or adding a graph node.
    if (!g.decode_mode) {
    }
    engine.qwen4exp_ple_tok_hist.clear();
    const int keep = std::max(0, c.ple_ngram_size - 1);
    const int start = std::max(0, S - keep);
    for (int i = start; i < S; ++i)
      engine.qwen4exp_ple_tok_hist.push_back(token_ids[i]);
    engine.qwen4exp_ple_next_pos = pos_base + S;
  }
  if (g.partial_layer_ids_out) {
    // Phase 1 all-or-nothing partial-layer dispatch: this prompt's real
    // layer-N routing, read back once per prefill. Coverage = fraction of
    // this prompt's DISTINCT selected experts that fall inside the
    // usage-ranked top-N pinned set (plan_partial_layer). Committing
    // decode to the GPU-only path only when coverage clears the
    // (default: 100%) threshold is what keeps build_moe_partial_gpu's
    // silent-miss risk rare in practice, not just in theory -- see that
    // function's own comment for what happens on the rare miss anyway.
    const auto &c = engine.model_file.architecture_config;
    const size_t n = static_cast<size_t>(c.topk) * static_cast<size_t>(S);
    std::vector<int32_t> ids_host(n);
    ggml_backend_tensor_get(g.partial_layer_ids_out, ids_host.data(), 0,
                            n * sizeof(int32_t));
    std::unordered_set<int32_t> used(ids_host.begin(), ids_host.end());
    std::unordered_set<int> gpu_set(
        engine.qwen4exp_partial_layer_gpu_ids.begin(),
        engine.qwen4exp_partial_layer_gpu_ids.end());
    size_t covered = 0;
    for (int32_t id : used)
      if (id >= 0 && gpu_set.count(id))
        ++covered;
    const double coverage =
        used.empty() ? 1.0
                     : static_cast<double>(covered) / static_cast<double>(used.size());
    engine.qwen4exp_partial_layer_committed =
        coverage >= engine.qwen4exp_partial_layer_min_coverage;
    // Root-cause diagnostic (2026-09-18): last-prefill-token router picks,
    // in the SAME format as legacy dispatch's own TRACE-ROUTER line
    // (moe_router_service.cpp, "[TRACE l=<layer>] router picks: e<id> ...")
    // -- for a fixed prompt, the last prefill token's input is IDENTICAL
    // regardless of dispatch path (no generated-token drift can have
    // contaminated it yet), so this is the one directly comparable,
    // uncontaminated data point between the two paths. Diff this line
    // against a legacy run's TRACE-ROUTER output (log_level=VERBOSE,
    // trace_filter=TRACE-ROUTER) for the same prompt/layer to test
    // [[qwen4exp-legacy-vs-fullgraph-routing-divergence]]'s BF16-router-
    // precision hypothesis.
    if (S > 0) {
      // Cross-check: what raw probability does THIS SAME build_moe
      // invocation's own probs_out assign to each id its own ids_out
      // selected? If this doesn't match a plausible top-k-by-probability
      // ranking, the divergence is internal to full-graph's own
      // ids/probs pairing, not (only) a legacy-vs-full-graph difference.
      std::vector<float> probs_host2;
      if (g.partial_layer_probs_out) {
        const size_t np2 =
            static_cast<size_t>(c.n_experts) * static_cast<size_t>(S);
        probs_host2.assign(np2, 0.0f);
        ggml_backend_tensor_get(g.partial_layer_probs_out, probs_host2.data(),
                                0, np2 * sizeof(float));
      }
      std::string picks;
      for (int k = 0; k < c.topk; ++k) {
        int32_t eid = ids_host[(S - 1) * c.topk + k];
        picks += " e" + std::to_string(eid);
        if (!probs_host2.empty()) {
          const float *col2 =
              probs_host2.data() + static_cast<size_t>(S - 1) * c.n_experts;
          const float p = (eid >= 0 && eid < c.n_experts) ? col2[eid] : -1.0f;
          picks += "(p=" + std::to_string(p) + ")";
        }
      }
      LOG_INFO("[TRACE-ROUTER-FULLGRAPH l=%d] router picks (last prefill "
               "token):%s\n",
               engine.qwen4exp_partial_layer, picks.c_str());
      // Root-cause diagnostic (2026-09-18), continued: dump full-graph's
      // own top-20 by raw post-softmax probability for the same token, so
      // a legacy pick NOT in full-graph's top-10 can be checked for
      // "just outside, nearly tied" (rank ~11-20, close probability) vs
      // "nowhere close" (rank in the hundreds) -- distinguishes a
      // precision-flips-a-near-tie explanation from a genuinely different
      // computation on one side.
      if (g.partial_layer_probs_out) {
        const size_t np = static_cast<size_t>(c.n_experts) * static_cast<size_t>(S);
        std::vector<float> probs_host(np);
        ggml_backend_tensor_get(g.partial_layer_probs_out, probs_host.data(), 0,
                                np * sizeof(float));
        const float *col = probs_host.data() +
                           static_cast<size_t>(S - 1) * c.n_experts;
        std::vector<int> order(c.n_experts);
        for (int e = 0; e < c.n_experts; ++e)
          order[e] = e;
        std::partial_sort(order.begin(), order.begin() + std::min(20, c.n_experts),
                          order.end(),
                          [&](int a, int b) { return col[a] > col[b]; });
        std::string top20;
        for (int k = 0; k < std::min(20, c.n_experts); ++k)
          top20 += " e" + std::to_string(order[k]) + "(p=" +
                   std::to_string(col[order[k]]) + ")";
        LOG_INFO("[TRACE-ROUTER-FULLGRAPH-TOP20 l=%d] last prefill token:%s\n",
                 engine.qwen4exp_partial_layer, top20.c_str());
      }
      if (g.partial_layer_bias_out) {
        std::vector<float> bias_host(static_cast<size_t>(c.n_experts));
        ggml_backend_tensor_get(g.partial_layer_bias_out, bias_host.data(), 0,
                                bias_host.size() * sizeof(float));
        float bmin = bias_host[0], bmax = bias_host[0], bsum = 0.0f;
        for (float v : bias_host) {
          bmin = std::min(bmin, v);
          bmax = std::max(bmax, v);
          bsum += v;
        }
        LOG_INFO("[TRACE-ROUTER-FULLGRAPH-BIAS l=%d] exp_probs_b.bias: "
                 "min=%.6f max=%.6f mean=%.6f (n=%d)\n",
                 engine.qwen4exp_partial_layer, bmin, bmax,
                 bsum / static_cast<float>(c.n_experts), c.n_experts);
      }
    }
    // Router-confidence diagnostic: is this layer's routing genuinely
    // peaked (real signal) or effectively near-uniform noise? A uniform
    // distribution over n_experts would put EVERY token's top-1
    // probability at 1/n_experts; a working router should sit well above
    // that. Directly tests the "near-zero-collision selection" anomaly
    // this whole check exists because of -- see
    // qwen4exp-full-graph-decode-benchmark-baseline memory's follow-on
    // note for the statistics that motivated it.
    if (g.partial_layer_probs_out) {
      const size_t np = static_cast<size_t>(c.n_experts) * static_cast<size_t>(S);
      std::vector<float> probs_host(np);
      ggml_backend_tensor_get(g.partial_layer_probs_out, probs_host.data(), 0,
                              np * sizeof(float));
      double top1_min = 1.0, top1_max = 0.0, top1_sum = 0.0;
      for (int s = 0; s < S; ++s) {
        float m = 0.0f;
        const float *col = probs_host.data() + static_cast<size_t>(s) * c.n_experts;
        for (int e = 0; e < c.n_experts; ++e)
          m = std::max(m, col[e]);
        top1_min = std::min(top1_min, static_cast<double>(m));
        top1_max = std::max(top1_max, static_cast<double>(m));
        top1_sum += m;
      }
      const double top1_mean = S > 0 ? top1_sum / S : 0.0;
      const double uniform_baseline = 1.0 / static_cast<double>(c.n_experts);
      LOG_INFO(
          "[PARTIAL-LAYER-DIAG] layer=%d router top-1 confidence over %d "
          "token(s): min=%.4f mean=%.4f max=%.4f (uniform/no-signal "
          "baseline would be %.4f) -- %s\n",
          engine.qwen4exp_partial_layer, S, top1_min, top1_mean, top1_max,
          uniform_baseline,
          top1_mean > uniform_baseline * 3.0
              ? "routing looks peaked/real"
              : "routing looks close to uniform -- possible router issue");
    }
    LOG_INFO(
        "[PARTIAL-LAYER] layer=%d prefill routing: %zu distinct expert(s) "
        "selected this prompt, %zu covered by the top-%d pinned set "
        "(%.1f%% coverage, threshold %.1f%%) -- decode will run %s for "
        "this layer\n",
        engine.qwen4exp_partial_layer, used.size(), covered,
        engine.qwen4exp_partial_layer_top_n, coverage * 100.0,
        engine.qwen4exp_partial_layer_min_coverage * 100.0,
        engine.qwen4exp_partial_layer_committed
            ? "GPU-ONLY (committed)"
            : "the normal CPU/GPU path (insufficient coverage)");
  } else if (engine.qwen4exp_partial_layer >= 0 &&
             !partial_prefill_capture_enabled()) {
    LOG_INFO("[PARTIAL-LAYER] layer=%d prefill routing capture skipped "
             "(Phase 2 routes per token; the capture would perturb prefill "
             "numerics -- set ZEROMOE_QWEN4EXP_PARTIAL_LAYER_PREFILL_CAPTURE=1 "
             "to force it)\n",
             engine.qwen4exp_partial_layer);
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
  g_qwen4exp_cont_engine = &engine;
  g_qwen4exp_cont_total = pos_base + S;
  return true;
}

#endif // ZEROMOE_USE_GGML_CUDA
