// Qwen4Exp full graph: multi-token decode window (MTP verification).
//
// Runs `S` consecutive tokens through the decode graph in one scheduler
// submission. The graph was built with g.S == S (engine.qwen4exp_decode_window)
// and shares the ordinary decode state arena, so after the call the live
// recurrent/KV/PLE state stands after the last token of the window. A caller
// that must keep only a prefix restores a checkpoint taken before the window
// (ggml_qwen4exp_graph_checkpoint.cpp) and replays the prefix.

#include "ggml_qwen4exp_graph_internal.hpp"
#include "nvtx_range.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)

using namespace qwen4exp_graph;

bool qwen4exp_full_graph_window_run(StreamingEngine &engine, int S, int pos_base,
                                    const int *token_ids, float *logits,
                                    float *wide) {
  if (!full_graph_decode_enabled_for_engine(engine) || !token_ids || !logits ||
      !wide || pos_base < 0 || !g_graph || !g_graph->ready ||
      !g_graph->decode_mode || g_graph->engine != &engine || g_graph->S != S ||
      S < 1 || pos_base + S > g_graph->kv_capacity ||
      g_graph->position_ids_per_token != 1 ||
      engine.family_ops.family != ModelFamily::Qwen4Exp)
    return false;
  FullGraph &g = *g_graph;
  const size_t W = static_cast<size_t>(g.D) * g.HC;
  if (engine.hc_stream_buf.size() < W * static_cast<size_t>(S) || !g.input ||
      !g.logits || ggml_nelements(g.logits) <
          static_cast<int64_t>(S) * engine.model_file.architecture_config.vocab ||
      !g.final_streams ||
      static_cast<size_t>(ggml_nelements(g.final_streams)) != W * S)
    return false;
  if (has_ple_layer(engine.model_file.architecture_config) && !g.ple_embedding)
    return false;

  ZE_NVTX("win.run");
  std::memcpy(g.input->data, engine.hc_stream_buf.data(),
              W * static_cast<size_t>(S) * sizeof(float));
  auto *positions = static_cast<int32_t *>(g.positions->data);
  for (int i = 0; i < S; ++i)
    positions[i] = pos_base + i;
  auto *mask = static_cast<ggml_fp16_t *>(g.attn_mask->data);
  for (int i = 0; i < S; ++i)
    for (int k = 0; k < g.kv_capacity; ++k)
      mask[static_cast<size_t>(i) * g.kv_capacity + k] =
          ggml_fp32_to_fp16(k <= pos_base + i ? 0.0f : -INFINITY);
  auto *indices = static_cast<int32_t *>(g.decode_indices->data);
  for (int i = 0; i < S; ++i)
    for (int h = 0; h < g.Hkv; ++h)
      indices[i * g.Hkv + h] = h * g.kv_capacity + pos_base + i;

  // Same per-submission preamble as qwen4exp_full_graph_decode_run.
  if (!qwen4exp_stage_layer13_before_graph(g, pos_base))
    return false;
  for (auto &entry : g.early_ready) {
    entry.second->expected_pos = pos_base;
    entry.second->residency_ready = false;
    entry.second->completed = false;
  }
  const auto start = std::chrono::steady_clock::now();
  {
    ZE_NVTX("win.submit");  // includes every CPU-split compute the scheduler runs inline
    if (ggml_backend_sched_graph_compute_async(g.sched, g.graph) != GGML_STATUS_SUCCESS)
      return false;
  }
  {
    ZE_NVTX("win.sync");
    ggml_backend_sched_synchronize(g.sched);
  }
  const int vocab = engine.model_file.architecture_config.vocab;
  {
    ZE_NVTX("win.readback");
    ggml_backend_tensor_get(g.logits, logits, 0,
                            static_cast<size_t>(S) * vocab * sizeof(float));
    ggml_backend_tensor_get(g.final_streams, wide, 0, W * static_cast<size_t>(S) * sizeof(float));
  }

  // The graph has advanced the device state past the whole window. Host
  // metadata (PLE token window, positions) is applied by the commit below, so
  // a caller keeping only a prefix never sees the rejected tokens.
  g.window_tokens.assign(token_ids, token_ids + S);
  g.window_pos = pos_base;
  engine.qwen4exp_full_graph_output_ready = true;
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count();
  if (Logger::instance().would_log(LogLevel::DEBUG))
    LOG_DEBUG("[QWEN4EXP-WINDOW] pos=%d S=%d compute+readback=%.3f ms\n", pos_base, S, ms);
  return true;
}

// Keep the first `m` tokens of the window last run: make the device state the
// state after exactly those tokens and advance the host metadata to match.
// m == S is free (the graph already left the full-window state live).
bool qwen4exp_full_graph_window_commit(StreamingEngine &engine, int m) {
  ZE_NVTX("win.commit");
  if (!g_graph || !g_graph->ready || !g_graph->decode_mode ||
      g_graph->engine != &engine)
    return false;
  FullGraph &g = *g_graph;
  const int S = g.S;
  if (m < 1 || m > S || g.window_pos < 0 ||
      static_cast<int>(g.window_tokens.size()) != S)
    return false;
  if (m < S) {
    if (g.window_gdn_states.size() != g.gdn_state_storage.size() ||
        g.window_conv_in.size() != g.gdn_hist_storage.size() ||
        cudaSetDevice(g.device) != cudaSuccess)
      return false;
    cudaStream_t stream = zeromoe_ggml_backend_cuda_get_stream(g.cuda_raw);
    const size_t state_bytes = static_cast<size_t>(g.gdn_hd) * g.gdn_hd *
                               g.Hv_gdn * sizeof(float);
    const size_t hist_w = static_cast<size_t>(g.d_conv - 1) * sizeof(float);
    for (size_t j = 0; j < g.gdn_state_storage.size(); ++j) {
      const ggml_tensor *snap = g.window_gdn_states[j][static_cast<size_t>(m - 1)];
      const ggml_tensor *conv_in = g.window_conv_in[j];
      if (!snap || !snap->data || !conv_in || !conv_in->data ||
          cudaMemcpyAsync(g.gdn_state_storage[j]->data, snap->data, state_bytes,
                          cudaMemcpyDefault, stream) != cudaSuccess ||
          cudaMemcpy2DAsync(g.gdn_hist_storage[j]->data, hist_w,
                            static_cast<const char *>(conv_in->data) +
                                static_cast<size_t>(m) * sizeof(float),
                            conv_in->nb[1], hist_w, g.conv_dim,
                            cudaMemcpyDefault, stream) != cudaSuccess)
        return false;
    }
    if (g.window_ple_padded && g.ple_hist_storage) {
      const size_t ple_w = static_cast<size_t>(
          (engine.model_file.architecture_config.ple_conv_kernel - 1) *
          engine.model_file.architecture_config.ple_ngram_size) * sizeof(float);
      if (cudaMemcpy2DAsync(g.ple_hist_storage->data, ple_w,
                            static_cast<const char *>(g.window_ple_padded->data) +
                                static_cast<size_t>(m) * sizeof(float),
                            g.window_ple_padded->nb[1], ple_w,
                            static_cast<size_t>(g.D) * g.HC, cudaMemcpyDefault,
                            stream) != cudaSuccess)
        return false;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess)
      return false;
  }
  g.decode_last_pos = g.window_pos + m - 1;
  g.decode_host_state_dirty = env_prefill_state_handoff_enabled();
  if (has_ple_layer(engine.model_file.architecture_config)) {
    auto &hist = engine.qwen4exp_ple_tok_hist;
    for (int i = 0; i < m; ++i)
      hist.push_back(g.window_tokens[static_cast<size_t>(i)]);
    const int keep = std::max(0, engine.model_file.architecture_config.ple_ngram_size - 1);
    if (static_cast<int>(hist.size()) > keep)
      hist.erase(hist.begin(), hist.end() - keep);
    engine.qwen4exp_ple_next_pos = g.window_pos + m;
  }
  g.window_pos = -1;
  qwen4exp_promotion_step(g, m);
  return true;
}

#endif // ZEROMOE_USE_GGML_CUDA
