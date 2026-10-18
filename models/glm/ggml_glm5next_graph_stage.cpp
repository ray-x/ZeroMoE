// GLM5Next full graph: Pilot staging: the prefetch/stage worker, stage miss callback, pilot
// stage graph nodes.
//
// Split out of the former single-file ggml_glm5next_graph.cpp; shared
// declarations live in ggml_glm5next_graph_impl.hpp.

#include "ggml_glm5next_graph_impl.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

using namespace glm5next_graph;

namespace glm5next_graph {

void stage_worker_main(StageRuntime *st) {
  for (;;) {
    StageRuntime::Job job;
    {
      std::unique_lock<std::mutex> lock(st->m);
      st->cv.wait(lock, [&] { return st->stop || !st->q.empty(); });
      if (st->stop && st->q.empty())
        return;
      job = st->q.front();
      st->q.pop_front();
    }
    OffloadState *o = g_offload.get();
    StageResult &res = st->result[job.layer];
    int staged = 0;
    int32_t staged_ids[8] = {};
    if (o && pilot_prefetch_on()) {
      // Read predicted experts that are not in DDR from the SSD now, while the GPU runs the previous layer.
      int32_t missing[8];
      int nm = 0;
      {
        auto arena_lock = o->store->reader_lock();
        for (int j = 0; j < job.n; ++j) {
          const int id = job.ids[j];
          if (id >= 0 && id < o->n_experts && !o->cache->resident({job.layer, id}))
            missing[nm++] = id;
        }
      }
      if (nm > 0) {
        const auto t0 = std::chrono::steady_clock::now();
        const moestream::offload::ExpertResidency::Ensure r =
            o->store->ensure(job.layer, missing, nm,
                             moestream::offload::ExpertCache::Kind::Prefetch);
        st->prefetch_loads.fetch_add((uint64_t)r.loads, std::memory_order_relaxed);
        st->prefetch_us.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count(),
                                  std::memory_order_relaxed);
        if (r.errors > 0)
          LOG_WARN("[GGML-GLM5NEXT-STAGE] prefetch of layer %d had %d load error(s)\n", job.layer, r.errors);
      }
    }
    if (!pilot_copy_on())
      continue;
    if (o) {
      auto arena_lock = o->store->reader_lock();
      int chosen[8] = {};
      for (int j = 0; j < job.n && staged < st->K; ++j) {
        const int id = job.ids[j];
        if (id < 0 || id >= o->n_experts || !o->residency->ready({job.layer, id})) {
          st->skipped_not_resident.fetch_add(1, std::memory_order_relaxed);
          continue;
        }
        chosen[staged] = id;
        staged_ids[staged++] = id;
      }
      if (staged > 0) {
        size_t pb[3];
        for (int k = 0; k < 3; ++k)
          pb[k] = (size_t)o->arena->plane_bytes(job.layer, k);
        st->run_pieces(staged * 3, [&](int piece) {
          const int slot = piece / 3, k = piece % 3;
          char *src = static_cast<char *>(o->arena->ptr(job.layer, chosen[slot], k));
          if (o->is_pinned(job.layer, chosen[slot])) {
            // The arena slot is page-locked: the DMA engine reads DDR directly, no CPU copy.
            cudaMemcpyAsync(st->dev_base[job.set][k] + (size_t)slot * pb[k], src, pb[k], cudaMemcpyHostToDevice, st->stream);
            st->direct.fetch_add(1, std::memory_order_relaxed);
            return;
          }
          char *host = static_cast<char *>(st->pinned[job.set][k]) + (size_t)slot * pb[k];
          std::memcpy(host, src, pb[k]);
          cudaMemcpyAsync(st->dev_base[job.set][k] + (size_t)slot * pb[k], host, pb[k], cudaMemcpyHostToDevice, st->stream);
        });
      }
    }
    cudaEventRecord(st->done[job.set], st->stream);
    st->copied.fetch_add((uint64_t)staged, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(st->m);
      res.n = staged;
      std::memcpy(res.ids, staged_ids, sizeof(staged_ids));
      res.state.store(2, std::memory_order_release);
    }
    st->done_cv.notify_all();
  }
}

StageRuntime *get_stage(FullGraph &g) {
  if (g_stage)
    return g_stage.get();
  if (g_stage_failed)
    return nullptr;
  StageRuntime *r = get_stage_impl(g);
  if (!r)
    g_stage_failed = true;
  return r;
}

StageRuntime *get_stage_impl(FullGraph &g) {
  OffloadState *o = g_offload.get();
  const int K = pilot_stage_k();
  if (!o || K <= 0 || !g.cuda)
    return nullptr;
  auto st = std::make_unique<StageRuntime>();
  st->K = K;
  // Per-layer quant types differ in this GGUF (e.g. the down projection), so the ring is sized for the largest plane and each layer
  // addresses it with its own expert stride.
  for (int k = 0; k < 3; ++k) {
    st->plane_bytes[k] = 0;
    for (int l = o->first; l < o->first + o->n_layers; ++l)
      st->plane_bytes[k] = std::max<uint64_t>(st->plane_bytes[k], o->arena->plane_bytes(l, k));
  }
  if (cudaStreamCreateWithFlags(&st->stream, cudaStreamNonBlocking) != cudaSuccess)
    return nullptr;
  for (int s = 0; s < StageRuntime::kSets; ++s) {
    if (cudaEventCreateWithFlags(&st->done[s], cudaEventDisableTiming) != cudaSuccess)
      return nullptr;
    for (int k = 0; k < 3; ++k) {
      const size_t bytes = (size_t)K * st->plane_bytes[k];
      st->dev_buf[s][k] = ggml_backend_alloc_buffer(g.cuda, bytes + 4096);
      if (!st->dev_buf[s][k]) {
        LOG_WARN("[GGML-GLM5NEXT-STAGE] VRAM ring allocation failed (%zu bytes); pilot staging disabled\n", bytes);
        return nullptr;
      }
      ggml_backend_buffer_set_usage(st->dev_buf[s][k], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
      st->dev_base[s][k] = static_cast<char *>(ggml_backend_buffer_get_base(st->dev_buf[s][k]));
      if (cudaMallocHost(&st->pinned[s][k], bytes) != cudaSuccess) {
        LOG_WARN("[GGML-GLM5NEXT-STAGE] pinned bounce allocation failed\n");
        return nullptr;
      }
    }
  }
  st->worker = std::thread(stage_worker_main, st.get());
  for (int h = 0; h < StageRuntime::kCopyHelpers; ++h)
    st->helpers.emplace_back([p = st.get()] { p->helper_main(); });
  LOG_INFO("[GGML-GLM5NEXT-STAGE] ring ready: K=%d, %d sets, expert planes %llu/%llu/%llu bytes\n", K, StageRuntime::kSets,
           (unsigned long long)st->plane_bytes[0], (unsigned long long)st->plane_bytes[1], (unsigned long long)st->plane_bytes[2]);
  g_stage = std::move(st);
  return g_stage.get();
}

void stage_post_job(StageRuntime *st, int layer, int set, const ggml_tensor *pred) {
  StageRuntime::Job job;
  job.layer = layer;
  job.set = set;
  const int K = (int)(pred->ne[0] / 2), nc = std::min(K, 8);
  const float min_score = pilot_min_score();
  StageRuntime::Cand &cd = st->cand[layer];
  cd.n = nc;
  job.n = 0;
  for (int j = 0; j < nc; ++j) {
    const int32_t id = (int32_t)*(const float *)((const char *)pred->data + (size_t)j * pred->nb[0]);
    const float sc = *(const float *)((const char *)pred->data + (size_t)(K + j) * pred->nb[0]);
    cd.ids[j] = id;
    cd.score[j] = sc;
    if (sc >= min_score)
      job.ids[job.n++] = id;
  }
  static std::atomic<int> logged{0};
  if (logged.fetch_add(1) < 4)
    LOG_INFO("[GGML-GLM5NEXT-STAGE] pilot candidates for layer %d: scores %.4f %.4f %.4f %.4f (min_score %g, kept %d/%d)\n", layer,
             nc > 0 ? cd.score[0] : 0.f, nc > 1 ? cd.score[1] : 0.f, nc > 2 ? cd.score[2] : 0.f, nc > 3 ? cd.score[3] : 0.f,
             (double)min_score, job.n, nc);
  {
    std::lock_guard<std::mutex> lock(st->m);
    if (pilot_copy_on())
      st->result[layer].state.store(1, std::memory_order_release);
    st->q.push_back(job);
  }
  st->jobs.fetch_add(1, std::memory_order_relaxed);
  st->cv.notify_one();
}

// One CPU node per layer: (a) wait for the staged copy of this layer, (b) emit, per route, the cold id for the CPU branch (-1 if the route
// is served from the VRAM ring), the ring slot (-1 if not staged) and a staged flag, laid out as [cold ids | slots | flags] (3*topk int32),
// (c) make the remaining cold experts resident, (d) start the copy for the next layer. Everything the CPU branch needs is produced here
// so the cold MUL_MAT_ID stays in the same scheduler split as this node.
// a: static cold ids [topk,1]; b: real routed ids [topk,1]; c: pilot ids of the next layer [K,1].
void stage_miss_cb(ggml_tensor *dst, const ggml_tensor *a, const ggml_tensor *b, const ggml_tensor *c, int ith, int, void *ud) {
  if (ith != 0)
    return;
  StageMissSite *site = static_cast<StageMissSite *>(ud);
  StageRuntime *st = g_stage.get();
  OffloadState &o = *site->off->state;
  const auto cb_t0 = std::chrono::steady_clock::now();
  const int n = (int)a->ne[0];
  int staged = 0;
  int32_t staged_ids[8] = {};
  StageResult &res = st->result[site->layer];
  {
    StageRuntime::Cand &cd = st->cand[site->layer];
    for (int j = 0; j < cd.n; ++j) {
      bool used = false;
      for (int i = 0; i < n; ++i)
        used = used || *(const int32_t *)((const char *)b->data + i * b->nb[0]) == cd.ids[j];
      const int sb = StageRuntime::score_bucket(cd.score[j]);
      st->rank_n[j].fetch_add(1, std::memory_order_relaxed);
      st->sb_n[sb].fetch_add(1, std::memory_order_relaxed);
      if (used) {
        st->rank_hit[j].fetch_add(1, std::memory_order_relaxed);
        st->sb_hit[sb].fetch_add(1, std::memory_order_relaxed);
      }
    }
    cd.n = 0;
  }
  if (res.state.load(std::memory_order_acquire) != 0) {
    const auto t0 = std::chrono::steady_clock::now();
    {
      std::unique_lock<std::mutex> lock(st->m);
      st->done_cv.wait(lock, [&] { return res.state.load(std::memory_order_acquire) != 1; });
      staged = res.n;
      std::memcpy(staged_ids, res.ids, sizeof(staged_ids));
      res.state.store(0, std::memory_order_release);
    }
    if (staged > 0)
      cudaEventSynchronize(st->done[site->set]);
    st->wait_us.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count(),
                          std::memory_order_relaxed);
  }
  std::vector<int32_t> ids((size_t)n);
  int32_t *out = static_cast<int32_t *>(dst->data);
  uint64_t hit = 0;
  for (int i = 0; i < n; ++i) {
    const int32_t cold = *(const int32_t *)((const char *)a->data + i * a->nb[0]);
    const int32_t real = *(const int32_t *)((const char *)b->data + i * b->nb[0]);
    int slot = -1;
    for (int j = 0; j < staged; ++j)
      if (staged_ids[j] == real) {
        slot = j;
        break;
      }
    if (slot >= 0 && cold >= 0) {
      out[i] = -1;
      out[n + i] = slot;
      out[2 * n + i] = 1;
      ids[(size_t)i] = -1;
      ++hit;
    } else {
      out[i] = cold;
      out[n + i] = -1;
      out[2 * n + i] = 0;
      ids[(size_t)i] = cold;
    }
  }
  st->staged_routes.fetch_add(hit, std::memory_order_relaxed);
  st->routes_seen.fetch_add((uint64_t)n, std::memory_order_relaxed);
  const moestream::offload::ExpertResidency::Ensure r =
      o.store->ensure(site->layer, ids.data(), n, moestream::offload::ExpertCache::Kind::Demand);
  o.store->counters().calls.fetch_add(1, std::memory_order_relaxed);
  o.store->counters().routes.fetch_add((uint64_t)n, std::memory_order_relaxed);
  o.store->counters().cold_routes.fetch_add((uint64_t)std::count_if(ids.begin(), ids.end(), [](int32_t v) { return v >= 0; }), std::memory_order_relaxed);
  o.store->counters().misses.fetch_add((uint64_t)r.misses, std::memory_order_relaxed);
  o.store->counters().loads.fetch_add((uint64_t)r.loads, std::memory_order_relaxed);
  o.store->counters().evictions.fetch_add((uint64_t)r.evictions, std::memory_order_relaxed);
  o.store->counters().load_us.fetch_add((uint64_t)(r.load_ms * 1000.0), std::memory_order_relaxed);
  if (r.errors > 0) {
    o.store->counters().errors.fetch_add((uint64_t)r.errors, std::memory_order_relaxed);
    LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] layer %d: %d expert load error(s); aborting\n", site->layer, r.errors);
    o.store->report("fatal");
    std::abort();
  }
  if (site->next_layer >= 0)
    stage_post_job(st, site->next_layer, site->next_set, c);
  o.store->counters().cb_us.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - cb_t0).count(),
                    std::memory_order_relaxed);
  const uint64_t rs = st->resolves.fetch_add(1, std::memory_order_relaxed) + 1;
  if (rs % 4704 == 0)
    LOG_INFO("[GGML-GLM5NEXT-STAGE] progress: jobs=%llu experts_copied=%llu resolves=%llu staged_routes=%llu/%llu not_resident=%llu "
             "wait=%.1f ms direct_pieces=%llu prefetch_loads=%llu prefetch=%.1f ms\n",
             (unsigned long long)st->jobs.load(), (unsigned long long)st->copied.load(), (unsigned long long)rs,
             (unsigned long long)st->staged_routes.load(), (unsigned long long)st->routes_seen.load(),
             (unsigned long long)st->skipped_not_resident.load(), st->wait_us.load() / 1000.0, (unsigned long long)st->direct.load(),
             (unsigned long long)st->prefetch_loads.load(), st->prefetch_us.load() / 1000.0);
  if (rs % 4704 == 0) {
    std::string pr = "pilot precision by rank:", ps = "by score bucket (<.01 <.02 <.05 <.1 <.2 <.4 <.8 <1.6 >=1.6):";
    char buf[64];
    for (int j = 0; j < 8; ++j)
      if (st->rank_n[j].load() > 0) {
        std::snprintf(buf, sizeof buf, " r%d %.0f%%(%llu)", j + 1, 100.0 * (double)st->rank_hit[j].load() / (double)st->rank_n[j].load(),
                      (unsigned long long)st->rank_n[j].load());
        pr += buf;
      }
    for (int j = 0; j < StageRuntime::kScoreBuckets; ++j) {
      const uint64_t nn = st->sb_n[j].load();
      std::snprintf(buf, sizeof buf, " %.0f%%(%llu)", nn ? 100.0 * (double)st->sb_hit[j].load() / (double)nn : 0.0, (unsigned long long)nn);
      ps += buf;
    }
    LOG_INFO("[GGML-GLM5NEXT-STAGE] %s\n[GGML-GLM5NEXT-STAGE] %s\n", pr.c_str(), ps.c_str());
  }
  if (o.store->counters().calls.load() - o.store->counters().last_report_calls.load() >= 544) {
    o.store->counters().last_report_calls.store(o.store->counters().calls.load());
    o.store->report("progress");
  }
}

bool stage_layer_active(const FullGraph &g, int layer) {
  const auto &c = g.engine->model_file.architecture_config;
  return g.decode_mode && g.S == 1 && pilot_stage_k() > 0 && offload_layer(layer) && layer < c.n_layers &&
         layer - 1 >= c.first_dense;
}

// 0 for cold experts, -1e9 for experts that live in VRAM (hot, GPU1 hot): the pilot only ranks what the CPU would otherwise compute.
ggml_tensor *stage_hot_mask(FullGraph &g, ggml_context *ctx, int layer) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_tensor *t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.n_experts);
  if (!t)
    return nullptr;
  std::vector<float> m((size_t)c.n_experts, 0.0f);
  if (PartialExpertLayer *p = partial_layer(g, layer)) {
    for (int e : p->hot)
      if (e >= 0 && e < c.n_experts)
        m[(size_t)e] = -1e9f;
    if (g.gpu1)
      for (int e : p->hot1)
        if (e >= 0 && e < c.n_experts)
          m[(size_t)e] = -1e9f;
  }
  if (!arena_alloc(g.dense.buffer, g.dense.base, g.dense.capacity, g.dense.used, t, m.data(), ggml_nbytes(t)))
    return nullptr;
  ggml_set_name(t, ("glm5next_stage_hot_mask_" + std::to_string(layer)).c_str());
  return t;
}

bool build_pilot_stage(FullGraph &g, ggml_context *ctx, ggml_tensor *cur_pre, int l) {
  const auto &c = g.engine->model_file.architecture_config;
  const int next = l + pilot_lookahead();
  if (!stage_layer_active(g, next))
    return true;
  StageRuntime *st = get_stage(g);
  if (!st)
    return true;
  const std::string pn = "blk." + std::to_string(next) + ".";
  ggml_tensor *fn = resident_tensor(g, ctx, pn + "ffn_norm.weight");
  ggml_tensor *router = resident_tensor(g, ctx, pn + "ffn_gate_inp.weight");
  ggml_tensor *bias = resident_tensor(g, ctx, pn + "exp_probs_b.bias");
  if (!fn || !router) {
    LOG_WARN("[GGML-GLM5NEXT-STAGE] layer %d: missing next-layer norm/router tensor\n", l);
    return false;
  }
  ggml_tensor *rn = ggml_rms_norm(ctx, cur_pre, c.eps);
  ggml_tensor *z = ggml_mul(ctx, rn, fn);
  ggml_tensor *logits = ggml_mul_mat(ctx, router, z);
  ggml_tensor *probs = nullptr;
  switch (c.gating_func) {
  case GatingFunc::Sigmoid:
    probs = ggml_sigmoid(ctx, logits);
    break;
  case GatingFunc::SqrtSoftplus:
    probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));
    break;
  case GatingFunc::Softmax:
  default:
    probs = ggml_soft_max(ctx, logits);
    break;
  }
  ggml_tensor *score = bias ? ggml_add(ctx, probs, bias) : probs;
  // Learned correction (MOESTREAM_GLM5NEXT_PILOT_RIDGE): score += (rms(x) - mu) R + c, fitted offline on the pilot's residual.
  static const PilotRidgeTable ridge = PilotRidgeTable::load("MOESTREAM_GLM5NEXT_PILOT_RIDGE", "GLM-PILOT-RIDGE");
  if (const PilotRidgeTable::Record *rec = ridge.find(next);
      rec && ridge.d == (int)rn->ne[0] && ridge.e == (int)router->ne[1] && g.cuda) {
    ggml_tensor *mu_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ridge.d, 1);
    ggml_tensor *r_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ridge.d, ridge.e);
    ggml_tensor *c_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ridge.e, 1);
    ggml_tensor *parts[3] = {mu_t, r_t, c_t};
    size_t offsets[3], total = 0;
    for (int i = 0; i < 3; ++i) {
      offsets[i] = total;
      total += (ggml_nbytes(parts[i]) + 255) & ~size_t(255);
    }
    ggml_backend_buffer_t rb = ggml_backend_alloc_buffer(g.cuda, total);
    if (!rb) {
      LOG_WARN("[GGML-GLM5NEXT-STAGE] layer %d: pilot ridge buffer allocation failed; using the plain pilot\n", next);
    } else {
      auto *base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(rb));
      bool placed = true;
      for (int i = 0; i < 3 && placed; ++i)
        placed = ggml_backend_tensor_alloc(rb, parts[i], base + offsets[i]) == GGML_STATUS_SUCCESS;
      if (!placed) {
        ggml_backend_buffer_free(rb);
      } else {
        g.ridge_buffers.push_back(rb);
        ggml_backend_tensor_set(mu_t, rec->mu.data(), 0, ggml_nbytes(mu_t));
        ggml_backend_tensor_set(r_t, rec->r.data(), 0, ggml_nbytes(r_t));
        ggml_backend_tensor_set(c_t, rec->c.data(), 0, ggml_nbytes(c_t));
        ggml_tensor *corr = ggml_mul_mat(ctx, r_t, ggml_sub(ctx, rn, mu_t));
        score = ggml_add(ctx, ggml_add(ctx, score, corr), c_t);
      }
    }
  }
  ggml_tensor *mask = stage_hot_mask(g, ctx, next);
  if (!mask) {
    LOG_WARN("[GGML-GLM5NEXT-STAGE] layer %d: hot-mask allocation failed (dense arena %zu/%zu)\n", next, g.dense.used, g.dense.capacity);
    return false;
  }
  score = ggml_add(ctx, score, mask);
  const int topk = std::max(st->K, pilot_prefetch_k());
  ggml_tensor *pred = ggml_argsort_top_k(ctx, score, topk);
  // Packed [ids as f32 (topk) | router scores of those ids (topk)] so the CPU node can rank, filter and calibrate candidates.
  ggml_tensor *pred_c = ggml_cont(ctx, pred);
  ggml_tensor *ids_f = ggml_cast(ctx, pred_c, GGML_TYPE_F32);
  ggml_tensor *top_scores = ggml_get_rows(ctx, ggml_reshape_2d(ctx, ggml_cont(ctx, score), 1, c.n_experts), pred_c);
  ggml_tensor *packed = ggml_concat(ctx, ggml_reshape_1d(ctx, ids_f, topk), ggml_reshape_1d(ctx, top_scores, topk), 0);
  ggml_set_name(packed, ("glm5next_stage_pred_" + std::to_string(next)).c_str());
  g.stage_pred[next] = packed;
  return true;
}

ggml_tensor *stage_ring_tensor(FullGraph &g, ggml_context *ctx, StageRuntime &st, int layer, int plane) {
  static const char *stems[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
  const GGUFTensor *m = model_tensor(*g.engine, "blk." + std::to_string(layer) + "." + stems[plane]);
  if (!m || !type_ok(m->type) || m->shape.size() != 3)
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : m->shape)
    shape.push_back((int64_t)d);
  shape[2] = st.K;
  ggml_tensor *t = new_tensor(ctx, (ggml_type)m->type, shape);
  const int set = layer % StageRuntime::kSets;
  OffloadState *off = g_offload.get();
  if (!t || !off || (uint64_t)t->nb[2] != off->arena->plane_bytes(layer, plane) || ggml_nbytes(t) > (size_t)st.K * st.plane_bytes[plane]) {
    LOG_ERROR("[GGML-GLM5NEXT-STAGE] ring shape mismatch layer %d plane %d\n", layer, plane);
    return nullptr;
  }
  if (ggml_backend_tensor_alloc(st.dev_buf[set][plane], t, st.dev_base[set][plane]) != GGML_STATUS_SUCCESS) {
    LOG_WARN("[GGML-GLM5NEXT-STAGE] ring tensor alloc failed layer %d plane %d\n", layer, plane);
    return nullptr;
  }
  ggml_set_name(t, ("glm5next_stage_ring_" + std::to_string(layer) + "_" + std::to_string(plane)).c_str());
  return t;
}

ggml_tensor *stage_miss_node(FullGraph &g, ggml_context *ctx, StageRuntime &st, int layer, ggml_tensor *cold_ids, ggml_tensor *route_ids) {
  OffloadState *o = get_offload(*g.engine);
  if (!o || !cold_ids || !route_ids)
    return nullptr;
  std::lock_guard<std::mutex> lock(g_offload_mutex);
  o->sites.push_back({o, layer, false});
  StageMissSite ms;
  ms.off = &o->sites.back();
  ms.layer = layer;
  ms.set = layer % StageRuntime::kSets;
  ggml_tensor *pred_next = nullptr;
  const int target = layer + pilot_lookahead();
  if (stage_layer_active(g, target)) {
    auto it = g.stage_pred.find(target);
    if (it != g.stage_pred.end()) {
      pred_next = it->second;
      ms.next_layer = target;
      ms.next_set = target % StageRuntime::kSets;
    }
  }
  st.miss_sites.push_back(ms);
  ggml_tensor *out = ggml_map_custom3(ctx, cold_ids, route_ids, pred_next ? pred_next : route_ids, stage_miss_cb, 1, &st.miss_sites.back());
  if (out) {
    // The callback writes [cold ids | slots | flags]; the result was shaped like cold_ids ([topk,1] int32), so widen it before allocation.
    const int64_t n = out->ne[0];
    if (out->ne[1] != 1 || out->ne[2] != 1 || out->ne[3] != 1 || out->type != GGML_TYPE_I32)
      return nullptr;
    out->ne[0] = 3 * n;
    out->nb[1] = out->nb[2] = out->nb[3] = (size_t)(3 * n) * sizeof(int32_t);
    ggml_set_name(out, ("glm5next_stage_miss_" + std::to_string(layer)).c_str());
  }
  return out;
}

} // namespace glm5next_graph

#endif // MOESTREAM_USE_GGML_CUDA
