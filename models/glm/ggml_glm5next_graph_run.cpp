// GLM5Next full graph: Prefill execution of the built graph.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_glm5next_graph_impl.hpp.

#include "ggml_glm5next_graph_impl.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

using namespace glm5next_graph;

bool glm5next_full_graph_run(StreamingEngine &engine, float *hidden_states,
                             int S, int pos_base, const int *) {
  if (!graph_enabled_for(engine) || !hidden_states || S <= 1 || pos_base < 0 || !g_graph ||
      !g_graph->ready || g_graph->engine != &engine || g_graph->S != S ||
      g_graph->pos_base != pos_base || g_graph->decode_mode)
    return false;
  FullGraph &g = *g_graph;
  const auto &c = engine.model_file.architecture_config;
  const size_t state_n = (size_t)g.hd * g.hd * g.H;
  const size_t hist_n = (size_t)(g.d_conv - 1) * 3 * g.d_inner;

  for (size_t i = 0; i < g.kda_layers.size(); ++i) {
    const int l = g.kda_layers[i];
    if (l < 0 || l >= (int)engine.glm5next_recur_state.size() ||
        l >= (int)engine.glm5next_conv_history.size() ||
        engine.glm5next_recur_state[l].size() != state_n ||
        engine.glm5next_conv_history[l].size() != hist_n)
      return false;
    if (pos_base == 0) {
      std::fill(g.kda_state_inputs[i], g.kda_state_inputs[i] + state_n, 0.0f);
      std::fill(g.kda_hist_inputs[i], g.kda_hist_inputs[i] + hist_n, 0.0f);
    } else {
      // The previous chunk exported its state in the engine's layout; the scan kernel takes it transposed.
      transpose_glm5next_state(g.kda_state_inputs[i], engine.glm5next_recur_state[l].data(), g.hd, g.H);
      std::memcpy(g.kda_hist_inputs[i], engine.glm5next_conv_history[l].data(), hist_n * sizeof(float));
    }
  }
  for (const auto &in : g.indexer_prev_pool_inputs) {
    const Glm5nextIndexerSeed *seed = nullptr;
    for (const Glm5nextIndexerSeed &p : g.prev_indexer_seeds)
      if (p.layer == in.layer)
        seed = &p;
    const size_t n = (size_t)in.pools * g.indexer_key_length;
    if (!seed || seed->pooled_keys.size() < n)
      return false;
    std::memcpy(in.data, seed->pooled_keys.data(), n * sizeof(float));
  }
  for (const auto &[layer, dst] : g.dsa_prev_inputs) {
    if (layer < 0 || layer >= (int)engine.Lc.size() || !engine.Lc[layer] ||
        (g.qk_rope > 0 &&
         (layer >= (int)engine.Rc.size() || !engine.Rc[layer])))
      return false;
    for (int t = 0; t < pos_base; ++t) {
      std::memcpy(dst + (size_t)t * g.kv_cache_width,
                  engine.Lc[layer] + (size_t)t * g.kv_lora,
                  (size_t)g.kv_lora * sizeof(float));
      if (g.qk_rope > 0)
        std::memcpy(dst + (size_t)t * g.kv_cache_width + g.kv_lora,
                    engine.Rc[layer] + (size_t)t * g.qk_rope,
                    (size_t)g.qk_rope * sizeof(float));
    }
  }
  const bool plain_residual = engine.family_ops.family == ModelFamily::BailingMoE3;
  if ((!plain_residual &&
       engine.hc_stream_buf.size() < (size_t)g.D * g.HC * S) ||
      g.host_inputs.empty())
    return false;
  std::memcpy(g.host_inputs[0],
              plain_residual ? hidden_states : engine.hc_stream_buf.data(),
              (size_t)g.D * g.HC * S * sizeof(float));

  // Real-hardware incident (2026-09-09): a 480-token prompt crashed the
  // whole machine (hard reboot required, no surviving kernel/OOM log --
  // this box has no persistent journald storage) with the last surviving
  // application-log line being "graph prepared for S=...", i.e. somewhere
  // in the gap between prepare() returning and this compute call. Every
  // OTHER host allocation on this path (arena prep, expert source, graph
  // context) goes through MemoryGuard::try_reserve_host_bytes first; this
  // compute call did not, and it is the last real memory-relevant, cheaply
  // avoidable step before the model correctness this session verified
  // could otherwise be undone by a repeat crash. Re-check REAL, fresh
  // MemAvailable (not the background watchdog's own possibly-500ms-stale
  // allocation_blocked() flag) immediately before committing to it -- a
  // large gap can open between the admission checks earlier in prepare()
  // (preload alone took minutes) and this moment. Not a fix for whatever
  // the real mechanism was (unconfirmed -- no kernel evidence survived),
  // just a cheap, additional, defense-in-depth backstop.
  {
    const long avail_mb = MemoryGuard::read_available_mb();
    constexpr long kMinComputeHeadroomMb =
        10240; // 10 GiB, per explicit user instruction after the 2026-09-09
               // crash
    if (avail_mb >= 0 && avail_mb < kMinComputeHeadroomMb) {
      LOG_ERROR(
          "[GGML-GLM5NEXT-FULL] refusing graph compute: MemAvailable=%ld MiB "
          "< %ld MiB safety floor -- falling back to the legacy path rather "
          "than risk the host-memory incident this check was added for "
          "(2026-09-09, see this call site's own comment)\n",
          avail_mb, kMinComputeHeadroomMb);
      return false;
    }
  }

  if (!hold_graph_vram_experts(g))
    return false;
  for (auto &entry : g.early_ready)
    entry.second->completed = false;
  struct OpProf {
    std::chrono::steady_clock::time_point last;
    std::map<std::string, std::pair<double, int>> agg;
    ggml_backend_sched_t sched = nullptr;
  } op;
  const bool op_prof_on = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_OP_PROFILE") != nullptr;
  if (op_prof_on) {
    // Debug: per-op wall time, attributed to the op that just finished (the callback also splits the graph per node).
    ggml_backend_sched_set_eval_callback(
        g.sched,
        [](ggml_tensor *t, bool ask, void *ud) -> bool {
          if (ask) return true;
          auto *o = static_cast<OpProf *>(ud);
          const auto now = std::chrono::steady_clock::now();
          const double ms = std::chrono::duration<double, std::milli>(now - o->last).count();
          o->last = now;
          std::string key = ggml_op_name(t->op);
          if (t->op == GGML_OP_MUL_MAT_ID || t->op == GGML_OP_MUL_MAT)
            key += std::string("(") + ggml_type_name(t->src[0]->type) + ")/" +
                   ggml_backend_name(ggml_backend_sched_get_tensor_backend(o->sched, t));
          auto &e = o->agg[key];
          e.first += ms;
          ++e.second;
          return true;
        },
        &op);
    op.sched = g.sched;
    op.last = std::chrono::steady_clock::now();
  }
  const auto start = std::chrono::steady_clock::now();
  if (ggml_backend_sched_graph_compute_async(g.sched, g.graph) !=
      GGML_STATUS_SUCCESS) {
    ggml_backend_sched_synchronize(g.sched);
    release_graph_vram_experts(g);
    return false;
  }
  ggml_backend_sched_synchronize(g.sched);
  release_graph_vram_experts(g);
  if (op_prof_on) {
    ggml_backend_sched_set_eval_callback(g.sched, nullptr, nullptr);
    std::vector<std::pair<double, std::string>> rows;
    for (auto &[k, v] : op.agg) rows.push_back({v.first, k + " n=" + std::to_string(v.second)});
    std::sort(rows.rbegin(), rows.rend());
    for (size_t i = 0; i < rows.size() && i < 30; ++i)
      LOG_INFO("[GLM-PREFILL-OPPROF] %9.1f ms  %s\n", rows[i].first, rows[i].second.c_str());
  }
  const auto synced = std::chrono::steady_clock::now();
  ggml_backend_tensor_get(g.output, hidden_states, 0,
                          (size_t)g.D * S * sizeof(float));

  for (size_t i = 0; i < g.kda_layers.size(); ++i) {
    const int l = g.kda_layers[i];
    ggml_backend_tensor_get(g.kda_state_out[i],
                            engine.glm5next_recur_state[l].data(), 0,
                            state_n * sizeof(float));
    ggml_backend_tensor_get(g.kda_hist_out[i],
                            engine.glm5next_conv_history[l].data(), 0,
                            hist_n * sizeof(float));
  }
  const size_t kv_n = (size_t)g.S * g.kv_cache_width;
  std::vector<float> kv(kv_n);
  for (size_t i = 0; i < g.dsa_layers.size(); ++i) {
    const int l = g.dsa_layers[i];
    if (l < 0 || l >= (int)engine.Lc.size() || !engine.Lc[l] ||
        (g.qk_rope > 0 &&
         (l >= (int)engine.Rc.size() || !engine.Rc[l])))
      return false;
    ggml_backend_tensor_get(g.dsa_kv_out[i], kv.data(), 0,
                            kv_n * sizeof(float));
    for (int t = 0; t < g.S; ++t) {
      std::memcpy(engine.Lc[l] + (size_t)(pos_base + t) * g.kv_lora,
                  kv.data() + (size_t)t * g.kv_cache_width,
                  (size_t)g.kv_lora * sizeof(float));
      if (g.qk_rope > 0)
        std::memcpy(engine.Rc[l] + (size_t)(pos_base + t) * g.qk_rope,
                    kv.data() + (size_t)t * g.kv_cache_width + g.kv_lora,
                    (size_t)g.qk_rope * sizeof(float));
    }
  }
  g.glm5next_indexer_seeds.clear();
  g.glm5next_indexer_seeds.reserve(g.glm5next_indexer_layers.size());
  bool indexer_seeds_ok = true;
  for (const Glm5nextIndexerLayer &state : g.glm5next_indexer_layers) {
    if (!state.key_gate_out) {
      indexer_seeds_ok = false;
      break;
    }
    // Earlier chunks' rows come first (their pools are complete: chunk boundaries are multiples of kpool).
    const Glm5nextIndexerSeed *prev = nullptr;
    if (pos_base > 0) {
      for (const Glm5nextIndexerSeed &p : g.prev_indexer_seeds)
        if (p.layer == state.layer)
          prev = &p;
      if (!prev || prev->token_count != pos_base) {
        indexer_seeds_ok = false;
        break;
      }
    }
    const size_t chunk_rows = (size_t)2 * g.indexer_key_length * g.S;
    const int chunk_pools = g.indexer_kpool > 1 ? g.S / g.indexer_kpool : 0;
    Glm5nextIndexerSeed seed;
    seed.layer = state.layer;
    seed.token_count = pos_base + g.S;
    seed.pool_count = (g.indexer_kpool > 1 ? pos_base / g.indexer_kpool : 0) + chunk_pools;
    const size_t prev_rows = prev ? prev->key_gate.size() : 0;
    const size_t prev_pooled = prev ? prev->pooled_keys.size() : 0;
    seed.key_gate.resize(prev_rows + chunk_rows);
    seed.pooled_keys.resize(prev_pooled + (size_t)g.indexer_key_length * chunk_pools);
    if (prev) {
      std::memcpy(seed.key_gate.data(), prev->key_gate.data(), prev_rows * sizeof(float));
      std::memcpy(seed.pooled_keys.data(), prev->pooled_keys.data(), prev_pooled * sizeof(float));
    }
    ggml_backend_tensor_get(state.key_gate_out, seed.key_gate.data() + prev_rows, 0, chunk_rows * sizeof(float));
    if (chunk_pools > 0) {
      if (!state.pooled_out) {
        indexer_seeds_ok = false;
        break;
      }
      ggml_backend_tensor_get(state.pooled_out, seed.pooled_keys.data() + prev_pooled, 0,
                              (size_t)g.indexer_key_length * chunk_pools * sizeof(float));
    }
    g.glm5next_indexer_seeds.push_back(std::move(seed));
  }
  if (!indexer_seeds_ok) {
    g.glm5next_indexer_seeds.clear();
    LOG_WARN("[GGML-GLM5NEXT-FULL] indexer seed capture unavailable; "
             "decode will retain dense attention\n");
  }
  const auto done = std::chrono::steady_clock::now();
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  LOG_INFO("[GGML-GLM5NEXT-FULL-TIME] sync=%.3f ms state_get=%.3f ms "
           "total=%.3f ms\n",
           ms(start, synced), ms(synced, done), ms(start, done));
  if (!g.glm5next_indexer_seeds.empty())
    LOG_INFO("[GGML-GLM5NEXT-FULL] captured KPool prompt seeds: layers=%zu "
             "tokens=%d pools=%d\n",
             g.glm5next_indexer_seeds.size(), pos_base + g.S,
             (pos_base + g.S) / std::max(1, g.indexer_kpool));
  g.prev_indexer_seeds.clear();
  g.ran = true;
  glm5next_publish_continuation(engine, g);
  engine.glm5next_full_graph_output_ready = true;
  (void)c;
  return true;
}

#endif // MOESTREAM_USE_GGML_CUDA
