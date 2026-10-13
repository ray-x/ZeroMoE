// GLM5Next full graph: Decode preparation and decode execution.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_glm5next_graph_impl.hpp.

#include "ggml_glm5next_graph_impl.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

using namespace glm5next_graph;

bool glm5next_full_graph_decode_prepare(StreamingEngine &engine, int pos_base,
                                        const int *) {
  if (!env_enabled() || engine.mtp_enabled || pos_base < 0 ||
      engine.family_ops.family != ModelFamily::GLM5NEXT || engine.max_t <= 0 ||
      pos_base >= engine.max_t)
    return false;
  if (g_glm5next_decode_disabled_engine == &engine)
    return false;
  const auto &c = engine.model_file.architecture_config;
  const int64_t rounded_capacity =
      ((int64_t)engine.max_t + 255) & ~((int64_t)255);
  if (rounded_capacity <= 0 || rounded_capacity > INT_MAX)
    return false;
  const int capacity = static_cast<int>(rounded_capacity);

  if (g_graph && g_graph->ready && g_graph->engine == &engine &&
      g_graph->decode_mode && g_graph->kv_capacity >= pos_base + 1 &&
      g_graph->decode_last_pos + 1 == pos_base)
    return true;

  if (g_graph && g_graph->ready && g_graph->decode_mode &&
      g_graph->engine == &engine) {
    // Copying the device decode state back to the host is only valid while this is still the SAME request (the graph
    // needs a rebuild, e.g. more KV capacity). When the position does not continue the previous decode, a new
    // request has been prefilled since: the host state now belongs to that prompt, and writing the previous
    // request's recurrent state / latent KV over it makes every later turn of a session answer the earlier prompt.
    if (g_graph->decode_last_pos + 1 == pos_base) {
      if (!sync_glm5next_decode_state_to_host(*g_graph)) {
        g_glm5next_decode_disabled_engine = &engine;
        return false;
      }
    } else {
      LOG_INFO("[GGML-GLM5NEXT-FULL-DECODE] new request (position %d, previous decode ended at %d): dropping the "
               "previous decode state without a host sync\n",
               pos_base, g_graph->decode_last_pos);
    }
  }

  auto g = std::make_unique<FullGraph>();
  g->engine = &engine;
  g->topology = g_glm5next_vram_cache;
  g->device = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  g->S = 1;
  g->decode_mode = true;
  g->kv_capacity = capacity;
  g->decode_last_pos = pos_base - 1;
  g->D = c.hidden;
  g->HC = std::max(1, c.hc_mult);
  g->n_layers = c.n_layers;
  g->H = c.n_heads;
  g->hd = c.kda_head_dim;
  g->d_inner = g->H * g->hd;
  g->d_conv = c.ssm_conv_kernel;
  g->qk_nope = c.qk_nope;
  g->kv_lora = c.kv_lora;
  g->indexer_heads = c.glm5next_indexer_head_count;
  g->indexer_key_length = c.glm5next_indexer_key_length;
  g->indexer_top_k = c.glm5next_indexer_top_k;
  g->indexer_kpool = c.glm5next_indexer_kpool;
  g->indexer_select_tail =
      engine.model_file.get_config_int("attention.indexer.kpool_select_tail", 0) != 0;
  g->v_head = c.v_head;
  g->first_dense = c.first_dense;
  g->device_expert_layers = std::min(
      glm5next_full_graph_device_expert_layers(),
      std::max(0, c.n_layers - c.first_dense));
  if (configure_partial_experts(*g))
    g->device_expert_layers = 0;

  // Sparse decode requires prompt-computed indexer keys and complete KPool
  // summaries. Carry those across the prompt->decode graph replacement only
  // when the prompt graph actually produced a matching seed for every owner.
  std::vector<Glm5nextIndexerSeed> indexer_seeds;
  int expected_indexer_owners = 0;
  if (g->indexer_kpool > 1 && g->indexer_top_k > 0 &&
      g->indexer_top_k % g->indexer_kpool == 0 &&
      g_graph && g_graph->ready && !g_graph->decode_mode &&
      g_graph->engine == &engine && g_graph->S == pos_base &&
      g_graph->indexer_key_length == g->indexer_key_length &&
      g_graph->indexer_kpool == g->indexer_kpool) {
    for (int l = 0; l < g->n_layers; ++l) {
      if (!glm5next_is_kda_layer(c, l) && glm5next_indexer_owner(*g, l))
        ++expected_indexer_owners;
    }
    bool seeds_match = expected_indexer_owners > 0 &&
                       (int)g_graph->glm5next_indexer_seeds.size() ==
                           expected_indexer_owners;
    const int expected_pools = pos_base / g->indexer_kpool;
    for (const Glm5nextIndexerSeed &seed : g_graph->glm5next_indexer_seeds) {
      seeds_match = seeds_match && seed.token_count == pos_base &&
                    seed.pool_count == expected_pools &&
                    seed.key_gate.size() ==
                        (size_t)2 * g->indexer_key_length * pos_base &&
                    seed.pooled_keys.size() ==
                        (size_t)g->indexer_key_length * expected_pools;
    }
    if (seeds_match) {
      g->sparse_decode = true;
      indexer_seeds = std::move(g_graph->glm5next_indexer_seeds);
      LOG_INFO("[GGML-GLM5NEXT-FULL-DECODE] enabling sparse KPool decode: "
               "owners=%d top_k=%d kpool=%d prompt=%d\n",
               expected_indexer_owners, g->indexer_top_k,
               g->indexer_kpool, pos_base);
    } else {
      LOG_WARN("[GGML-GLM5NEXT-FULL-DECODE] prompt indexer seeds missing or "
               "incompatible; keeping dense DSA decode\n");
    }
  }

  // Preserve the large, reusable DDR expert source while replacing the
  // prompt-shaped graph with the fixed S=1 decode topology.
  bool source_transferred = false;
  if (g_graph && g_graph->has_expert_source() &&
      source_configuration_matches(*g_graph, *g)) {
    if (!g_graph->move_expert_source_to(*g))
      return false;
    source_transferred = true;
  } else if (g_glm5next_source &&
             source_configuration_matches(*g_glm5next_source, *g)) {
    if (!g_glm5next_source->move_expert_source_to(*g))
      return false;
    source_transferred = true;
  }
  g_graph.reset();
  if (g_glm5next_source && !g_glm5next_source->has_expert_source())
    g_glm5next_source.reset();

  auto retain_source_on_failure = [&]() {
    if (!g->has_expert_source())
      return;
    if (g_glm5next_source && g_glm5next_source->has_expert_source())
      g_glm5next_source.reset();
    if (!g_glm5next_source)
      g_glm5next_source = std::make_unique<FullGraph>();
    if (!g->move_expert_source_to(*g_glm5next_source))
      LOG_ERROR("[GGML-GLM5NEXT-FULL-DECODE] failed to retain expert "
                "source after graph decline\n");
  };
  auto decline_decode_graph = [&]() {
    // Design policy: a GLM5NEXT request is served by the full graph or it fails. MOESTREAM_GLM5NEXT_GRAPH_STRICT=1
    // turns the legacy-decode fallback into a hard error (default stays permissive until prefill is graph-only too).
    if (const char *strict = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_GRAPH_STRICT");
        strict && strict[0] == '1') {
      LOG_ERROR("[GGML-GLM5NEXT-FULL-DECODE] decode graph declined and "
                "MOESTREAM_GLM5NEXT_GRAPH_STRICT=1: failing the request "
                "instead of falling back to the layered path\n");
      std::fflush(nullptr);
      std::abort();
    }
    retain_source_on_failure();
    g_glm5next_decode_disabled_engine = &engine;
    return false;
  };

  g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(g->device));
  g->cpu = ggml_backend_cpu_init();
  if (g->cpu) {
    const char *e = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_CPU_THREADS");
    ggml_backend_cpu_set_n_threads(g->cpu, e ? std::max(1, std::atoi(e)) : 6);
  }
  if (!g->cuda || !g->cpu)
    return decline_decode_graph();
  if (!source_transferred && g_glm5next_source &&
      source_configuration_matches(*g_glm5next_source, *g)) {
    if (!g_glm5next_source->move_expert_source_to(*g))
      return decline_decode_graph();
    source_transferred = true;
  }
  if (!source_transferred &&
      !prepare_expert_source_only(*g, "GGML-GLM5NEXT-FULL-DECODE"))
    return decline_decode_graph();
  if (!init_expert_arena(*g) || !init_dense_arena(*g) ||
      !init_decode_state_arena(*g)) {
    return decline_decode_graph();
  }
  if (const char *e = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_GPU1_LAYERS")) {
    int a = -1, b = -2;
    if (std::sscanf(e, "%d-%d", &a, &b) == 2 && a >= 0 && b >= a) {
      g->gpu1 = static_cast<ggml_backend_t>(ggml_shared_backend(1));
      if (g->gpu1) {
        g->gpu1_first = a;
        g->gpu1_last = b;
        LOG_INFO("[GGML-GLM5NEXT-FULL-DECODE] GPU1 expert layers %d-%d\n", a, b);
      }
    }
  }
  if (!g->gpu1) {
    // GPU1 hot experts (MOESTREAM_GLM5NEXT_GPU1_HOT_GIB) need the second backend even without whole GPU1 layers.
    for (const auto &pl : g->partial_experts)
      if (!pl.hot1.empty()) {
        g->gpu1 = static_cast<ggml_backend_t>(ggml_shared_backend(1));
        break;
      }
  }
  ggml_backend_t backends[] = {g->cuda, g->gpu1, g->cpu};
  if (!g->gpu1)
    backends[1] = g->cpu;
  g->sched = ggml_backend_sched_new(backends, nullptr, g->gpu1 ? 3 : 2,
                                    kGraphSize, false, false);
  if (!g->sched || !build_graph(*g)) {
    LOG_WARN("[GGML-GLM5NEXT-FULL-DECODE] graph construction declined; "
             "using legacy decode\n");
    return decline_decode_graph();
  }

  const size_t state_n = (size_t)g->hd * g->hd * g->H;
  const size_t hist_n = (size_t)(g->d_conv - 1) * 3 * g->d_inner;
  for (size_t i = 0; i < g->kda_layers.size(); ++i) {
    const int layer = g->kda_layers[i];
    if (layer < 0 || layer >= (int)engine.glm5next_recur_state.size() ||
        layer >= (int)engine.glm5next_conv_history.size() ||
        engine.glm5next_recur_state[layer].size() != state_n ||
        engine.glm5next_conv_history[layer].size() != hist_n ||
        i >= g->kda_state_storage.size() || i >= g->kda_hist_storage.size()) {
      return decline_decode_graph();
    }
    std::vector<float> native_state(state_n);
    transpose_glm5next_state(native_state.data(),
                             engine.glm5next_recur_state[layer].data(), g->hd,
                             g->H);
    ggml_backend_tensor_set(g->kda_state_storage[i], native_state.data(), 0,
                            state_n * sizeof(float));
    ggml_backend_tensor_set(g->kda_hist_storage[i],
                            engine.glm5next_conv_history[layer].data(), 0,
                            hist_n * sizeof(float));
  }

  for (size_t i = 0; i < g->dsa_layers.size(); ++i) {
    const int layer = g->dsa_layers[i];
    if (layer < 0 || layer >= (int)engine.Lc.size() || !engine.Lc[layer] ||
        i >= g->dsa_kv_storage.size()) {
      return decline_decode_graph();
    }
    ggml_tensor *cache = g->dsa_kv_storage[i];
    if (cudaMemset(cache->data, 0, ggml_nbytes(cache)) != cudaSuccess) {
      return decline_decode_graph();
    }
    const size_t seed_rows = (size_t)pos_base * g->kv_lora;
    if (seed_rows > 0) {
      std::vector<ggml_fp16_t> half_cache(seed_rows);
      const float *src = engine.Lc[layer];
      for (size_t j = 0; j < seed_rows; ++j)
        half_cache[j] = ggml_fp32_to_fp16(src[j]);
      ggml_backend_tensor_set(cache, half_cache.data(), 0,
                              seed_rows * sizeof(ggml_fp16_t));
    }
  }

  for (Glm5nextIndexerLayer &state : g->glm5next_indexer_layers) {
    if (!state.key_gate_cache || !state.pooled_cache ||
        cudaMemset(state.key_gate_cache->data, 0,
                   ggml_nbytes(state.key_gate_cache)) != cudaSuccess ||
        cudaMemset(state.pooled_cache->data, 0,
                   ggml_nbytes(state.pooled_cache)) != cudaSuccess)
      return decline_decode_graph();
    const auto seed = std::find_if(
        indexer_seeds.begin(), indexer_seeds.end(),
        [&state](const Glm5nextIndexerSeed &candidate) {
          return candidate.layer == state.layer;
        });
    if (seed == indexer_seeds.end() ||
        seed->key_gate.size() !=
            (size_t)2 * g->indexer_key_length * pos_base ||
        seed->pooled_keys.size() !=
            (size_t)g->indexer_key_length * (pos_base / g->indexer_kpool))
      return decline_decode_graph();
    if (!seed->key_gate.empty())
      ggml_backend_tensor_set(state.key_gate_cache, seed->key_gate.data(), 0,
                              seed->key_gate.size() * sizeof(float));
    if (!seed->pooled_keys.empty())
      ggml_backend_tensor_set(state.pooled_cache, seed->pooled_keys.data(), 0,
                              seed->pooled_keys.size() * sizeof(float));
  }

  g->ready = true;
  g_graph = std::move(g);
  LOG_INFO("[GGML-GLM5NEXT-FULL-DECODE] reusable graph prepared: "
           "position=%d capacity=%d KDA=%zu DSA=%zu\n",
           pos_base, capacity, g_graph->kda_layers.size(),
           g_graph->dsa_layers.size());
  return true;
}

bool glm5next_full_graph_decode_run(StreamingEngine &engine,
                                    float *hidden_states, int pos_base,
                                    const int *) {
  if (!env_enabled() || !hidden_states || pos_base < 0 || !g_graph ||
      !g_graph->ready || !g_graph->decode_mode || g_graph->engine != &engine ||
      g_graph->S != 1 || pos_base >= g_graph->kv_capacity ||
      g_graph->decode_last_pos + 1 != pos_base)
    return false;
  FullGraph &g = *g_graph;
  struct NanProbe {
    int reported = 0;
    int seen = 0;
  };
  static NanProbe probe;
  const bool nan_probe_on =
      moestream::graphcfg::env("MOESTREAM_GLM5NEXT_DEBUG_NAN_NODES") != nullptr &&
      probe.reported == 0;
  if (nan_probe_on) {
    probe.seen = 0;
    ggml_backend_sched_set_eval_callback(
        g.sched,
        [](ggml_tensor *t, bool ask, void *ud) -> bool {
          auto *p = static_cast<NanProbe *>(ud);
          if (ask)
            return p->reported < 6 && (t->type == GGML_TYPE_F32 ||
                                       t->type == GGML_TYPE_F16);
          const int idx = p->seen++;
          if (p->reported >= 6)
            return true;
          const size_t n = ggml_nelements(t);
          std::vector<uint8_t> raw(ggml_nbytes(t));
          ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
          size_t bad = 0;
          if (t->type == GGML_TYPE_F32) {
            for (size_t i = 0; i < n; ++i)
              if (!std::isfinite(reinterpret_cast<const float *>(raw.data())[i]))
                ++bad;
          } else {
            for (size_t i = 0; i < n; ++i)
              if (!std::isfinite(ggml_fp16_to_fp32(
                      reinterpret_cast<const ggml_fp16_t *>(raw.data())[i])))
                ++bad;
          }
          if (bad) {
            ++p->reported;
            if (t->op == GGML_OP_MUL_MAT_ID && t->src[2] &&
                t->src[2]->type == GGML_TYPE_I32 && t->type == GGML_TYPE_F32) {
              const size_t nid = ggml_nelements(t->src[2]);
              std::vector<int32_t> ids(nid);
              ggml_backend_tensor_get(t->src[2], ids.data(), 0,
                                      nid * sizeof(int32_t));
              std::string line;
              const size_t row_len = (size_t)t->ne[0];
              for (size_t r = 0; r < (size_t)t->ne[1] && r < nid; ++r) {
                size_t rb = 0;
                for (size_t i = 0; i < row_len; ++i)
                  if (!std::isfinite(reinterpret_cast<const float *>(
                          raw.data())[r * row_len + i]))
                    ++rb;
                line += " [row" + std::to_string(r) + " id=" +
                        std::to_string(ids[r]) + " bad=" + std::to_string(rb) +
                        "]";
              }
              LOG_WARN("[NAN-NODE-IDS] w_ne2=%lld w_type=%s%s\n",
                       (long long)t->src[0]->ne[2],
                       ggml_type_name(t->src[0]->type), line.c_str());
              const ggml_tensor *w = t->src[0];
              const size_t blk_bytes = ggml_type_size(w->type);
              const size_t blk_elems = (size_t)ggml_blck_size(w->type);
              const size_t slot_blocks =
                  (size_t)w->ne[0] * (size_t)w->ne[1] / blk_elems;
              for (size_t r = 0; r < 8 && r < nid; ++r) {
                if (ids[r] < 0 || ids[r] >= w->ne[2])
                  continue;
                std::vector<uint8_t> wb(slot_blocks * blk_bytes);
                ggml_backend_tensor_get(w, wb.data(),
                                        (size_t)ids[r] * w->nb[2], wb.size());
                size_t bad_d = 0;
                for (size_t b = 0; b < slot_blocks; ++b) {
                  const ggml_fp16_t d = *reinterpret_cast<const ggml_fp16_t *>(
                      wb.data() + b * blk_bytes);
                  if (!std::isfinite(ggml_fp16_to_fp32(d)))
                    ++bad_d;
                }
                LOG_WARN("[NAN-NODE-W] row=%zu slot=%d blocks=%zu bad_d=%zu "
                         "nb2=%zu buft=%s\n",
                         r, ids[r], slot_blocks, bad_d, (size_t)w->nb[2],
                         w->buffer ? ggml_backend_buffer_name(w->buffer) : "-");
                if (r == 0 && t->src[1] && t->src[1]->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(t->src[1]) &&
                    t->src[1]->ne[0] == w->ne[0]) {
                  const auto *tr = ggml_get_type_traits(w->type);
                  std::vector<float> x((size_t)w->ne[0]);
                  ggml_backend_tensor_get(t->src[1], x.data(), 0,
                                          x.size() * sizeof(float));
                  double xmax = 0;
                  size_t xbad = 0;
                  for (float v : x) {
                    if (!std::isfinite(v)) ++xbad;
                    else xmax = std::max(xmax, (double)std::fabs(v));
                  }
                  std::vector<float> wrow((size_t)w->ne[0]);
                  const size_t row_bytes = ggml_row_size(w->type, w->ne[0]);
                  int shown = 0, ref_finite_of_bad = 0, bad_rows = 0;
                  for (size_t j = 0; j < (size_t)t->ne[0]; ++j) {
                    const float got =
                        reinterpret_cast<const float *>(raw.data())[j];
                    if (std::isfinite(got)) continue;
                    ++bad_rows;
                    tr->to_float(wb.data() + j * row_bytes, wrow.data(),
                                 w->ne[0]);
                    double ref = 0;
                    for (size_t i = 0; i < x.size(); ++i)
                      ref += (double)wrow[i] * x[i];
                    if (std::isfinite(ref)) ++ref_finite_of_bad;
                    if (shown < 3) {
                      ++shown;
                      LOG_WARN("[NAN-NODE-REF] out_idx=%zu got=%g cpu_ref=%g\n",
                               j, got, ref);
                    }
                  }
                  LOG_WARN("[NAN-NODE-REF] src1 nonfinite=%zu absmax=%g "
                           "bad_rows=%d cpu_ref_finite=%d\n",
                           xbad, xmax, bad_rows, ref_finite_of_bad);
                }
              }
            }
            LOG_WARN("[NAN-NODE] #%d op=%s name=%s bad=%zu/%zu ne=%lld,%lld,%lld,%lld "
                     "src0=%s src1=%s\n",
                     idx, ggml_op_name(t->op), t->name, bad, n,
                     (long long)t->ne[0], (long long)t->ne[1],
                     (long long)t->ne[2], (long long)t->ne[3],
                     t->src[0] ? t->src[0]->name : "-",
                     t->src[1] ? t->src[1]->name : "-");
          }
          return true;
        },
        &probe);
  }
  if (g.dynamic_expert_cache && !publish_dynamic_uploads(g)) {
    if (!sync_glm5next_decode_state_to_host(g))
      LOG_ERROR("[GGML-GLM5NEXT-FULL-DECODE] cache-map handoff failed\n");
    g_glm5next_decode_disabled_engine = &engine;
    return false;
  }
  if (engine.hc_stream_buf.size() < (size_t)g.D * g.HC ||
      !g.input || !g.attn_mask || !g.decode_indices)
    return false;

  std::memcpy(g.input->data, engine.hc_stream_buf.data(),
              (size_t)g.D * g.HC * sizeof(float));
  *static_cast<int32_t *>(g.decode_indices->data) = pos_base;
  auto *mask = static_cast<ggml_fp16_t *>(g.attn_mask->data);
  for (int k = 0; k < g.kv_capacity; ++k)
    mask[k] = ggml_fp32_to_fp16(k <= pos_base ? 0.0f : -INFINITY);

  if (g.sparse_decode) {
    const int kpool = g.indexer_kpool;
    const int pool_count = g.kv_capacity / kpool;
    const int complete_pools = (pos_base + 1) / kpool;
    const int tail_count = (pos_base + 1) % kpool;
    const int pool_start = (pos_base / kpool) * kpool;
    auto *pool_mask = static_cast<float *>(g.indexer_pool_mask->data);
    for (int pool = 0; pool < pool_count; ++pool)
      pool_mask[pool] = pool < complete_pools ? 0.0f : -INFINITY;
    auto *update_rows =
        static_cast<int32_t *>(g.indexer_pool_update_rows->data);
    for (int i = 0; i < kpool; ++i)
      update_rows[i] = std::min(pool_start + i, pos_base);
    *static_cast<int32_t *>(g.indexer_pool_update_index->data) =
        pos_base / kpool;
    if (g.indexer_select_tail) {
      auto *tail_indices =
          static_cast<int32_t *>(g.indexer_tail_indices->data);
      auto *tail_mask = static_cast<float *>(g.indexer_tail_mask->data);
      const int tail_start = complete_pools * kpool;
      for (int i = 0; i < kpool - 1; ++i) {
        const bool valid = i < tail_count;
        tail_indices[i] = valid ? tail_start + i : 0;
        tail_mask[i] = valid ? 0.0f : -INFINITY;
      }
    }
  }

  const long avail_mb = MemoryGuard::read_available_mb();
  constexpr long kMinComputeHeadroomMb = 10240;
  if (avail_mb >= 0 && avail_mb < kMinComputeHeadroomMb) {
    LOG_ERROR("[GGML-GLM5NEXT-FULL-DECODE] refusing graph compute: "
              "MemAvailable=%ld MiB < %ld MiB; falling back to legacy decode\n",
              avail_mb, kMinComputeHeadroomMb);
    if (!sync_glm5next_decode_state_to_host(g))
      LOG_ERROR("[GGML-GLM5NEXT-FULL-DECODE] host state handoff failed\n");
    g_glm5next_decode_disabled_engine = &engine;
    return false;
  }

  if (!hold_graph_vram_experts(g)) {
    g_glm5next_decode_disabled_engine = &engine;
    return false;
  }
  struct LayerTimer {
    const FullGraph *g = nullptr;
    std::chrono::steady_clock::time_point last;
    std::vector<double> attn_ms, ffn_ms;
    int tokens = 0;
  };
  static LayerTimer lt;
  const bool layer_timing_on = layer_timing_env() && !nan_probe_on;
  if (layer_timing_on) {
    lt.g = &g;
    if (lt.attn_ms.size() != (size_t)g.n_layers) {
      lt.attn_ms.assign((size_t)g.n_layers, 0.0);
      lt.ffn_ms.assign((size_t)g.n_layers, 0.0);
      lt.tokens = 0;
    }
    ggml_backend_sched_set_eval_callback(
        g.sched,
        [](ggml_tensor *t, bool ask, void *ud) -> bool {
          auto *l = static_cast<LayerTimer *>(ud);
          auto it = l->g->layer_marks.find(t);
          if (it == l->g->layer_marks.end())
            return false;
          if (ask)
            return true;
          const auto now = std::chrono::steady_clock::now();
          const double ms =
              std::chrono::duration<double, std::milli>(now - l->last).count();
          l->last = now;
          (it->second.second == 0 ? l->attn_ms : l->ffn_ms)[(size_t)it->second.first] += ms;
          return true;
        },
        &lt);
    lt.last = std::chrono::steady_clock::now();
  }
  struct OpProf {
    std::chrono::steady_clock::time_point last;
    std::map<std::string, std::pair<double, int>> agg;
    int tokens = 0;
    ggml_backend_sched_t sched = nullptr;
  };
  static OpProf op;
  static const bool op_prof_env =
      moestream::graphcfg::env("MOESTREAM_GLM5NEXT_OP_PROFILE") != nullptr;
  const bool op_prof_on = op_prof_env && !layer_timing_on && !nan_probe_on &&
                          op.tokens < 30;
  if (op_prof_on) {
    ggml_backend_sched_set_eval_callback(
        g.sched,
        [](ggml_tensor *t, bool ask, void *ud) -> bool {
          if (ask)
            return true;
          auto *o = static_cast<OpProf *>(ud);
          const auto now = std::chrono::steady_clock::now();
          const double ms =
              std::chrono::duration<double, std::milli>(now - o->last).count();
          o->last = now;
          std::string key = ggml_op_name(t->op);
          if (t->op == GGML_OP_MUL_MAT_ID || t->op == GGML_OP_MUL_MAT)
            key += std::string("(") + ggml_type_name(t->src[0]->type) + ")" +
                   (t->src[0]->buffer
                        ? ggml_backend_buffer_name(t->src[0]->buffer)
                        : "nobuf") +
                   "/" +
                   ggml_backend_name(
                       ggml_backend_sched_get_tensor_backend(o->sched, t));
          auto &e = o->agg[key];
          e.first += ms;
          ++e.second;
          return true;
        },
        &op);
    op.sched = g.sched;
    op.last = std::chrono::steady_clock::now();
  }
  if (FILE *rt = route_trace_file()) {
    std::fprintf(rt, "P %d\n", pos_base);
    std::fflush(rt);
  }
  for (auto &entry : g.early_ready)
    entry.second->completed = false;
  const auto start = std::chrono::steady_clock::now();
  if (ggml_backend_sched_graph_compute_async(g.sched, g.graph) !=
      GGML_STATUS_SUCCESS) {
    ggml_backend_sched_synchronize(g.sched);
    release_graph_vram_experts(g);
    g.ready = false;
    g_glm5next_decode_disabled_engine = &engine;
    return false;
  }
  ggml_backend_sched_synchronize(g.sched);
  release_graph_vram_experts(g);
  if (!g.early_ready.empty()) {
    static int lane_tokens = 0;
    if (++lane_tokens % 16 == 0) {
      uint64_t calls = 0, split = 0, a_runs = 0, a_experts = 0;
      int64_t wait = 0, compute = 0, stall = 0, a_stall = 0;
      for (auto &entry : g.early_ready) {
        calls += entry.second->calls;
        split += entry.second->split_runs;
        wait += entry.second->wait_us;
        compute += entry.second->compute_us;
        stall += entry.second->split_stall_us;
        a_runs += entry.second->assist_runs;
        a_experts += entry.second->assist_experts;
        a_stall += entry.second->assist_stall_us;
      }
      uint64_t misses = 0, loads = 0, load_us = 0;
      if (OffloadState *o = g_offload.get()) {
        const auto &ct = o->store->counters();
        misses = ct.misses.load();
        loads = ct.loads.load();
        load_us = ct.load_us.load();
      }
      if (a_runs)
        LOG_INFO("[GLM5NEXT-LANE] gpu assist: %llu layer-runs, %llu experts, tail wait %.1fms/token\n",
                 (unsigned long long)a_runs, (unsigned long long)a_experts, a_stall / 1000.0 / lane_tokens);
      LOG_INFO("[GLM5NEXT-LANE] tokens=%d lane calls=%llu split=%llu wait=%.1fms/token compute=%.1fms/token "
               "split stall=%.1fms/token | misses=%llu loads=%llu read=%.1fms/token (%.2f ms per load)\n",
               lane_tokens, (unsigned long long)calls, (unsigned long long)split,
               wait / 1000.0 / lane_tokens, compute / 1000.0 / lane_tokens, stall / 1000.0 / lane_tokens,
               (unsigned long long)misses, (unsigned long long)loads, load_us / 1000.0 / lane_tokens,
               loads ? load_us / 1000.0 / loads : 0.0);
    }
  }
  if (op_prof_on) {
    ggml_backend_sched_set_eval_callback(g.sched, nullptr, nullptr);
    if (++op.tokens == 30) {
      std::vector<std::pair<std::string, std::pair<double, int>>> v(
          op.agg.begin(), op.agg.end());
      std::sort(v.begin(), v.end(), [](auto &a, auto &b) {
        return a.second.first > b.second.first;
      });
      for (size_t i = 0; i < v.size() && i < 25; ++i)
        LOG_WARN("[GLM5NEXT-OP-PROFILE] %-28s total/token=%.2fms nodes/token=%.1f avg=%.1fus\n",
                 v[i].first.c_str(), v[i].second.first / 30,
                 v[i].second.second / 30.0,
                 1000.0 * v[i].second.first / v[i].second.second);
    }
  }
  if (nan_probe_on) {
    ggml_backend_sched_set_eval_callback(g.sched, nullptr, nullptr);
    probe.reported = std::max(probe.reported, 1);
  }
  if (layer_timing_on) {
    ggml_backend_sched_set_eval_callback(g.sched, nullptr, nullptr);
    if (++lt.tokens % 100 == 0) {
      double ta = 0, tf = 0;
      std::string line;
      for (int l = 0; l < g.n_layers; ++l) {
        ta += lt.attn_ms[(size_t)l];
        tf += lt.ffn_ms[(size_t)l];
        char b[64];
        std::snprintf(b, sizeof(b), " L%d=%.2f/%.2f", l,
                      lt.attn_ms[(size_t)l] / lt.tokens,
                      lt.ffn_ms[(size_t)l] / lt.tokens);
        line += b;
      }
      LOG_WARN("[GLM5NEXT-LAYER-TIMING] tokens=%d mean per token: attn=%.1fms "
               "ffn=%.1fms (marked-node sync inflates absolute time)\n"
               "[GLM5NEXT-LAYER-TIMING] attn/ffn ms per layer:%s\n",
               lt.tokens, ta / lt.tokens, tf / lt.tokens, line.c_str());
    }
  }
  if (!g.probe_x.empty()) {
    // Training/diagnostic tap: the exact pilot input (pre-FFN hidden, f32 [D]) of every layer for this decode token.
    const char *dir = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PROBE_DUMP");
    std::vector<float> row((size_t)g.D);
    for (const auto &lx : g.probe_x) {
      if (!dir || lx.second->type != GGML_TYPE_F32 || (size_t)lx.second->ne[0] != (size_t)g.D)
        continue;
      ggml_backend_tensor_get(lx.second, row.data(), 0, (size_t)g.D * sizeof(float));
      std::ofstream xo(std::string(dir) + "/L" + std::to_string(lx.first) + ".x.bin", std::ios::binary | std::ios::app);
      xo.write(reinterpret_cast<const char *>(row.data()), (std::streamsize)((size_t)g.D * sizeof(float)));
    }
  }
  const auto synced = std::chrono::steady_clock::now();
  const double compute_ms = std::chrono::duration<double, std::milli>(
                                synced - start)
                                .count();
  const auto cache_update_start = std::chrono::steady_clock::now();
  if (route_stats_enabled())
    update_partial_routes(g, pos_base, compute_ms);
  const auto cache_update_done = std::chrono::steady_clock::now();
  if (!g.ready) {
    if (!sync_glm5next_decode_state_to_host(g))
      LOG_ERROR("[GGML-GLM5NEXT-FULL-DECODE] cache update state handoff "
                "failed\n");
    g_glm5next_decode_disabled_engine = &engine;
    return false;
  }
  ggml_backend_tensor_get(g.output, hidden_states, 0,
                          (size_t)g.D * sizeof(float));
  if (moestream::graphcfg::env("MOESTREAM_GLM5NEXT_DEBUG_DECODE_OUT")) {
    size_t bad = 0;
    double sum = 0.0, mx = 0.0;
    for (int i = 0; i < g.D; ++i) {
      const float v = hidden_states[i];
      if (!std::isfinite(v)) { ++bad; continue; }
      sum += v; mx = std::max(mx, (double)std::fabs(v));
    }
    LOG_WARN("[GLM5NEXT-DECODE-OUT] pos=%d nonfinite=%zu/%d mean=%g absmax=%g "
             "h[0..3]=%g %g %g %g\n",
             pos_base, bad, g.D, sum / g.D, mx, hidden_states[0],
             hidden_states[1], hidden_states[2], hidden_states[3]);
  }
  g.decode_last_pos = pos_base;
  engine.glm5next_full_graph_output_ready = true;
  if (g.logits && engine.qwen4exp_full_graph_logits_target &&
      engine.qwen4exp_full_graph_logits_target_size == (int)g.logits->ne[0]) {
    ggml_backend_tensor_get(g.logits, engine.qwen4exp_full_graph_logits_target,
                            0, ggml_nbytes(g.logits));
    engine.qwen4exp_full_graph_logits_ready = true;
  }
  const auto done = std::chrono::steady_clock::now();
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  LOG_INFO("[GGML-GLM5NEXT-FULL-DECODE-TIME] pos=%d sync=%.3f ms "
           "cache_update=%.3f ms output_get=%.3f ms total=%.3f ms\n",
           pos_base, ms(start, synced), ms(cache_update_start, cache_update_done),
           ms(cache_update_done, done), ms(start, done));
  return true;
}

#endif // MOESTREAM_USE_GGML_CUDA
