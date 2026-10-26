// Qwen4Exp full graph: multi-token decode window (MTP verification).
//
// Runs `S` consecutive tokens through the decode graph in one scheduler
// submission. The graph was built with g.S == S (engine.qwen4exp_decode_window)
// and shares the ordinary decode state arena, so after the call the live
// recurrent/KV/PLE state stands after the last token of the window. A caller
// that must keep only a prefix restores a checkpoint taken before the window
// (ggml_qwen4exp_graph_checkpoint.cpp) and replays the prefix.

#include "ggml_qwen4exp_graph_internal.hpp"

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
  if (ggml_backend_sched_graph_compute_async(g.sched, g.graph) != GGML_STATUS_SUCCESS)
    return false;
  ggml_backend_sched_synchronize(g.sched);
  const int vocab = engine.model_file.architecture_config.vocab;
  ggml_backend_tensor_get(g.logits, logits, 0,
                          static_cast<size_t>(S) * vocab * sizeof(float));
  ggml_backend_tensor_get(g.final_streams, wide, 0, W * static_cast<size_t>(S) * sizeof(float));

  g.decode_last_pos = pos_base + S - 1;
  g.decode_host_state_dirty = env_prefill_state_handoff_enabled();
  if (has_ple_layer(engine.model_file.architecture_config)) {
    auto &hist = engine.qwen4exp_ple_tok_hist;
    for (int i = 0; i < S; ++i)
      hist.push_back(token_ids[i]);
    const int keep = std::max(0, engine.model_file.architecture_config.ple_ngram_size - 1);
    if (static_cast<int>(hist.size()) > keep)
      hist.erase(hist.begin(), hist.end() - keep);
    engine.qwen4exp_ple_next_pos = pos_base + S;
  }
  engine.qwen4exp_full_graph_output_ready = true;
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count();
  if (zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_WINDOW_TIMING"))
    LOG_INFO("[QWEN4EXP-WINDOW] pos=%d S=%d compute+readback=%.3f ms\n", pos_base, S, ms);
  return true;
}

#endif // ZEROMOE_USE_GGML_CUDA
