// GLM5Next full graph: SSD-offload routed experts: the ExpertStore-backed miss node and
// offload weight tensors.
//
// Split out of the former single-file ggml_glm5next_graph.cpp; shared
// declarations live in ggml_glm5next_graph_impl.hpp.

#include <future>
#include <map>
#include "ggml_glm5next_graph_impl.hpp"

#include <filesystem>

#if defined(MOESTREAM_USE_GGML_CUDA)

using namespace glm5next_graph;

namespace glm5next_graph {

// Created on first use and shared by the prompt and decode graphs.
OffloadState *get_offload(const StreamingEngine &engine,
                          const std::vector<PartialExpertLayer> *partial)  {
  std::lock_guard<std::mutex> lock(g_offload_mutex);
  if (g_offload)
    return g_offload.get();
  const auto &c = engine.model_file.architecture_config;
  auto o = std::make_unique<OffloadState>();
  o->first = offload_first_layer();
  o->n_layers = c.n_layers - o->first;
  o->n_experts = c.n_experts;
  if (o->first < c.first_dense || o->n_layers <= 0 || o->n_experts <= 0) {
    LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] bad first layer %d\n", o->first);
    return nullptr;
  }
  moestream::offload::ExpertStoreConfig sc;
  sc.first_layer = o->first;
  sc.n_layers = o->n_layers;
  sc.n_experts = o->n_experts;
  sc.log_tag = "[GGML-GLM5NEXT-OFFLOAD]";
  for (int l = o->first; l < c.n_layers; ++l) {
    std::array<uint64_t, 3> pb{};
    int k = 0;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(l) + "." + stem);
      if (!m || m->size_bytes % (uint64_t)o->n_experts != 0) {
        LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] layer %d %s missing/uneven\n", l,
                  stem);
        return nullptr;
      }
      pb[(size_t)k] = m->size_bytes / (uint64_t)o->n_experts;
      if (pb[(size_t)k] % 4096 != 0) {
        LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] layer %d %s per-expert %llu is not "
                  "a multiple of 4096\n",
                  l, stem, (unsigned long long)pb[(size_t)k]);
        return nullptr;
      }
      ++k;
    }
    sc.plane_bytes.push_back(pb);
  }
  sc.huge_pages = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_THP", "0") == "1";
  sc.store_path = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_STORE");
  {
    const std::string first_shard = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_MODEL");
    if (first_shard.empty()) {
      LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] MOESTREAM_GLM5NEXT_OFFLOAD_MODEL "
                "(first shard) is required\n");
      return nullptr;
    }
    const size_t pos = first_shard.rfind("-00001-of-");
    std::error_code dir_ec;
    if (std::filesystem::is_directory(first_shard, dir_ec)) {
      // A directory of shards (what `gguf_tool pack` writes: repacked-<name>-partNNNNN-of-NNNNN.gguf): every .gguf, sorted.
      for (const auto &entry : std::filesystem::directory_iterator(first_shard))
        if (entry.is_regular_file() && entry.path().extension() == ".gguf")
          sc.model_shards.push_back(entry.path().string());
      std::sort(sc.model_shards.begin(), sc.model_shards.end());
    } else if (pos == std::string::npos) {
      sc.model_shards.push_back(first_shard);
    } else {
      const std::string head = first_shard.substr(0, pos);
      const int total = std::atoi(first_shard.c_str() + pos + 10);
      char buf[64];
      for (int i = 1; i <= std::max(1, total); ++i) {
        std::snprintf(buf, sizeof buf, "-%05d-of-%05d.gguf", i, total);
        sc.model_shards.push_back(head + buf);
      }
    }
  }
  {
    // MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB=<n> fixes the pool; unset (or "auto") sizes it from the memory this
    // process may actually use: available RAM (or the cgroup limit if lower) minus 36 GiB for the rest of the
    // engine and the OS, at least 16 GiB (60 GiB before the Q2_K/Q3_K GSQ-RCO file, whose ~74 GiB cold set needs more than 58). The store caps it at the total expert bytes. Measured on a 125 GB
    // box: decode 5.3 / 7.3 / 8.2 tok/s at 32 / 48 / 58 GiB, flat beyond (everything resident).
    const std::string v = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB", "auto");
    if (v == "auto" || v.empty()) {
      double avail_gib = 0.0;
      if (FILE *f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f)) {
          unsigned long long kb = 0;
          if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
            avail_gib = (double)kb / (1024.0 * 1024.0);
            break;
          }
        }
        std::fclose(f);
      }
      if (FILE *f = std::fopen("/sys/fs/cgroup/memory.max", "r")) {
        unsigned long long bytes = 0;
        if (std::fscanf(f, "%llu", &bytes) == 1)
          avail_gib = std::min(avail_gib, (double)bytes / (1024.0 * 1024.0 * 1024.0));
        std::fclose(f);
      }
      sc.ddr_gib = avail_gib > 0.0 ? std::max(16.0, avail_gib - 36.0) : 40.0;
      LOG_INFO("[GGML-GLM5NEXT-OFFLOAD] auto DDR pool %.1f GiB (available %.1f GiB)\n", sc.ddr_gib, avail_gib);
    } else {
      sc.ddr_gib = std::atof(v.c_str());
    }
  }
  sc.policy = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_POLICY", "lru") == "seg"
                  ? moestream::offload::ExpertCache::Policy::SegmentedLru
                  : moestream::offload::ExpertCache::Policy::Lru;
  sc.pinned_fraction =
      std::atof(offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_PINNED", "0.5").c_str());
  sc.probation_fraction =
      std::atof(offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_PROBATION", "0.25").c_str());
  sc.byte_capacity = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_BYTE_CAP", "0") == "1";
  sc.async_loads = offload_async();
  // Page-lock the arena slots (DMA straight from DDR) for staging and for the lane's GPU assist.
  if ((pilot_stage_k() > 0 || gpu_assist_slots() > 0) &&
      offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_PIN", gpu_assist_slots() > 0 ? "1" : "0") == "1") {
    sc.pin_fn = [](void *ptr, size_t bytes) {
      if (cudaHostRegister(ptr, bytes, cudaHostRegisterPortable) == cudaSuccess)
        return true;
      cudaGetLastError();
      return false;
    };
    sc.unpin_fn = [](void *ptr) { cudaHostUnregister(ptr); };
  }

  std::string err;
  o->store = moestream::offload::ExpertStore::create(sc, &err);
  if (!o->store) {
    LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] %s\n", err.c_str());
    return nullptr;
  }
  o->cache = &o->store->cache();
  o->arena = &o->store->arena();
  o->residency = &o->store->residency();

  const std::string usage = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_USAGE");
  if (!usage.empty()) {
    const auto t0 = std::chrono::steady_clock::now();
    auto ranked = moestream::offload::rank_hottest_first(
        usage, o->first, o->n_layers, o->n_experts);
    if (partial) {
      // Experts that live in VRAM are served by the GPU branch; do not spend DDR on them.
      // GPU1-hot experts are served by the GPU1 branch in decode, but a batched prefill graph has no such
      // branch and runs them as cold experts. Seed them last so they only take DDR that the rest leave free
      // (otherwise prefill reads them from the SSD, ~0.7 s inline plus ~6 s of background reads at 805 tokens).
      const bool seed_hot1 = offload_env("MOESTREAM_GLM5NEXT_PREFILL_OP_OFFLOAD", "0") == "1" &&
                             offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_SEED_GPU1_HOT", "1") == "1";
      std::vector<std::vector<uint8_t>> hot((size_t)c.n_layers), hot1((size_t)c.n_layers);
      size_t n_hot = 0;
      for (size_t l = 0; l < partial->size() && l < hot.size(); ++l) {
        hot[l].assign((size_t)o->n_experts, 0);
        hot1[l].assign((size_t)o->n_experts, 0);
        for (int e : (*partial)[l].hot)
          if (e >= 0 && e < o->n_experts) {
            hot[l][(size_t)e] = 1;
            ++n_hot;
          }
        for (int e : (*partial)[l].hot1)
          if (e >= 0 && e < o->n_experts) {
            (seed_hot1 ? hot1 : hot)[l][(size_t)e] = 1;
            if (!seed_hot1) ++n_hot;
          }
      }
      const auto in_set = [](const std::vector<std::vector<uint8_t>> &m, const moestream::offload::ExpertKey &k) {
        return (size_t)k.layer < m.size() && !m[(size_t)k.layer].empty() && m[(size_t)k.layer][(size_t)k.expert];
      };
      std::vector<moestream::offload::ExpertKey> late;
      ranked.erase(std::remove_if(ranked.begin(), ranked.end(),
                                  [&](const moestream::offload::ExpertKey &k) {
                                    if (in_set(hot1, k)) {
                                      late.push_back(k);
                                      return true;
                                    }
                                    return in_set(hot, k);
                                  }),
                   ranked.end());
      ranked.insert(ranked.end(), late.begin(), late.end());
      LOG_INFO("[GGML-GLM5NEXT-OFFLOAD] excluded %zu VRAM-hot experts from the "
               "DDR seed\n", n_hot);
    }
    const size_t failed = o->store->seed(ranked);
    if (o->store->pinning())
      LOG_INFO("[GGML-GLM5NEXT-OFFLOAD] pinned %llu experts (%llu failed) for direct DMA\n",
               (unsigned long long)o->store->pin_calls(), (unsigned long long)o->store->pin_failures());
    LOG_INFO("[GGML-GLM5NEXT-OFFLOAD] seeded %zu/%zu experts (%zu failed) in "
             "%.1f s\n",
             o->cache->size(), o->store->capacity(), failed,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                 .count());
  }
  LOG_INFO("[GGML-GLM5NEXT-OFFLOAD] layers %d..%d, capacity %zu experts "
           "(%.1f GiB), pinned %.2f, store=%s\n",
           o->first, c.n_layers - 1, o->store->capacity(), sc.ddr_gib,
           sc.pinned_fraction,
           sc.store_path.empty() ? "<none>" : sc.store_path.c_str());
  g_offload = std::move(o);
  return g_offload.get();
}

void offload_miss_cb(ggml_tensor *dst, const ggml_tensor *a, int ith, int,
                     void *ud) {
  if (ith != 0)
    return;
  OffloadSite *site = static_cast<OffloadSite *>(ud);
  OffloadState &o = *site->state;
  const auto cb_t0 = std::chrono::steady_clock::now();
  std::vector<int32_t> ids;
  ids.reserve((size_t)ggml_nelements(a));
  for (int64_t i1 = 0; i1 < a->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < a->ne[0]; ++i0)
      ids.push_back(*(const int32_t *)((const char *)a->data + i1 * a->nb[1] +
                                       i0 * a->nb[0]));
  static const bool dbg = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_OFFLOAD_DEBUG") != nullptr;
  static std::atomic<int> dbg_lines{0};
  if (dbg && !site->prefill)
    for (size_t k = 0; k < ids.size(); ++k)
      if (ids[k] >= 0 && !o.cache->resident({site->layer, ids[k]}) && dbg_lines.fetch_add(1) < 40)
        LOG_INFO("[OFFLOAD-DEBUG] miss layer=%d expert=%d S=%lld k=%zu\n", site->layer, ids[k],
                 (long long)a->ne[1], k);
  moestream::offload::ExpertResidency::Ensure r;
  if (site->prefill) {
    // Experts are read ahead (MOESTREAM_GLM5NEXT_PREFILL_AHEAD layers, default 2) while the GPU computes the layers
    // before this one; MOESTREAM_GLM5NEXT_PREFILL_AHEAD=0 loads each layer on the miss node, as before.
    static const int ahead = [] {
      const char *e = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PREFILL_AHEAD");
      return e ? std::max(0, std::atoi(e)) : 2;
    }();
    static std::map<int, std::future<int>> pending;
    static std::mutex pending_mu;
    std::lock_guard<std::mutex> pl(pending_mu);
    if (site->layer <= o.first) {
      for (auto &kv : pending) kv.second.wait(); // a new prompt: drop read-ahead left from an aborted one
      pending.clear();
    }
    if (auto it = pending.find(site->layer); it != pending.end()) {
      it->second.get();
      pending.erase(it);
    }
    r = o.store->ensure_transient(site->layer, ids.data(), (int)ids.size());
    const int last = o.first + o.n_layers - 1;
    // Read ahead only for prompts that touch most experts of a layer anyway (a small prompt would read ~2x what it uses).
    static const int min_tokens = [] {
      const char *e = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PREFILL_AHEAD_MIN_TOKENS");
      return e ? std::atoi(e) : 64;
    }();
    for (int k = 1; k <= ahead && site->n_tokens >= min_tokens; ++k) {
      const int next = site->layer + k;
      if (next > last || pending.count(next)) continue;
      moestream::offload::ExpertStore *store = o.store.get();
      const OffloadSite *ns = o.site_for_layer(next);
      const std::vector<uint8_t> *skip = ns ? &ns->not_cold : nullptr;
      pending.emplace(next, std::async(std::launch::async, [store, next, skip] { return store->prefetch_transient(next, skip); }));
    }
  } else {
    r = o.store->ensure(site->layer, ids.data(), (int)ids.size(),
                        moestream::offload::ExpertCache::Kind::Demand);
  }
  o.store->counters().calls.fetch_add(1, std::memory_order_relaxed);
  o.store->counters().routes.fetch_add(ids.size(), std::memory_order_relaxed);
  o.store->counters().cold_routes.fetch_add((uint64_t)std::count_if(ids.begin(), ids.end(), [](int32_t v) { return v >= 0; }),
                          std::memory_order_relaxed);
  o.store->counters().misses.fetch_add((uint64_t)r.misses, std::memory_order_relaxed);
  o.store->counters().loads.fetch_add((uint64_t)(site->prefill ? 0 : r.loads),
                    std::memory_order_relaxed);
  o.store->counters().transient_loads.fetch_add((uint64_t)(site->prefill ? r.loads : 0),
                              std::memory_order_relaxed);
  o.store->counters().evictions.fetch_add((uint64_t)r.evictions, std::memory_order_relaxed);
  o.store->counters().load_us.fetch_add((uint64_t)(r.load_ms * 1000.0), std::memory_order_relaxed);
  if (r.errors > 0) {
    o.store->counters().errors.fetch_add((uint64_t)r.errors, std::memory_order_relaxed);
    LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] layer %d: %d expert load error(s); "
              "aborting rather than computing with missing experts\n",
              site->layer, r.errors);
    o.store->report("fatal");
    std::abort();
  }
  // dst has the shape of `a` but is contiguous.
  std::memcpy(dst->data, ids.data(), ids.size() * sizeof(int32_t));
  o.store->counters().cb_us.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - cb_t0).count(),
                    std::memory_order_relaxed);
  if (!site->prefill && o.store->counters().calls.load() - o.store->counters().last_report_calls.load() >= 544) {
    o.store->counters().last_report_calls.store(o.store->counters().calls.load());
    o.store->report("progress");
  }
}

ggml_tensor *offload_weight(FullGraph &g, ggml_context *ctx, int layer,
                            const char *stem) {
  OffloadState *o = get_offload(*g.engine);
  if (!o)
    return nullptr;
  const int plane = std::strcmp(stem, "ffn_gate_exps.weight") == 0   ? 0
                    : std::strcmp(stem, "ffn_up_exps.weight") == 0   ? 1
                    : std::strcmp(stem, "ffn_down_exps.weight") == 0 ? 2
                                                                     : -1;
  const std::string name = "blk." + std::to_string(layer) + "." + stem;
  const GGUFTensor *meta = model_tensor(*g.engine, name);
  if (plane < 0 || !meta || !type_ok(meta->type) || meta->shape.size() != 3)
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : meta->shape)
    shape.push_back((int64_t)d);
  ggml_tensor *t = new_tensor(ctx, (ggml_type)meta->type, shape);
  if (!t || t->nb[2] != o->arena->stride(layer, plane) ||
      (uint64_t)t->ne[2] != (uint64_t)o->n_experts) {
    LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] %s: tensor stride %zu != arena stride\n",
              name.c_str(), t ? t->nb[2] : (size_t)0);
    return nullptr;
  }
  const size_t len = (size_t)t->nb[2] * (size_t)t->ne[2];
  ggml_backend_buffer_t buffer =
      ggml_backend_cpu_buffer_from_ptr(o->arena->ptr(layer, 0, plane), len);
  if (!buffer)
    return nullptr;
  ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  t->data = o->arena->ptr(layer, 0, plane);
  t->buffer = buffer;
  if (ggml_backend_buffer_init_tensor(buffer, t) != GGML_STATUS_SUCCESS) {
    ggml_backend_buffer_free(buffer);
    return nullptr;
  }
  g.host_buffers.push_back(buffer);
  ggml_set_name(t, (name + ".offload").c_str());
  return t;
}

ggml_tensor *offload_miss_node(FullGraph &g, ggml_context *ctx, int layer,
                               ggml_tensor *ids) {
  OffloadState *o = get_offload(*g.engine);
  if (!o || !ids)
    return nullptr;
  std::lock_guard<std::mutex> lock(g_offload_mutex);
  o->sites.push_back({o, layer, g.S > 1, g.S, {}});
  if (const PartialExpertLayer *pl = partial_layer(g, layer)) {
    o->sites.back().not_cold.assign((size_t)o->n_experts, 0);
    // hot1 (GPU1) experts are served by GPU1 only in S=1 decode; a batched graph routes them through the cold branch.
    for (const std::vector<int> *set : {&pl->hot, &pl->hot1})
      if (set == &pl->hot || (g.decode_mode && g.gpu1))
        for (int e : *set)
          if (e >= 0 && e < o->n_experts)
            o->sites.back().not_cold[(size_t)e] = 1;
  }
  ggml_tensor *out =
      ggml_map_custom1(ctx, ids, offload_miss_cb, 1, &o->sites.back());
  if (out)
    ggml_set_name(out, ("glm5next_offload_ids_" + std::to_string(layer)).c_str());
  return out;
}

ggml_tensor *expert_weight_impl(FullGraph &g, ggml_context *ctx, int layer,
                               const char *stem) {
  if (offload_layer(layer))
    return offload_weight(g, ctx, layer, stem);
  if (partial_layer(g, layer) &&
      !(g.vram_reference_experts && partial_layer_ram_ready(g, layer)))
    return load_cold_weight(g, ctx, layer, stem);
  if (partial_layer_ram_ready(g, layer)) {
    if (ggml_tensor *t = graph_ram_weight(g, ctx, layer, stem))
      return t;
  }
  const bool on_device =
      layer >= g.first_dense && layer < g.first_dense + g.device_expert_layers;
  return load_full_weight(g, ctx, "blk." + std::to_string(layer) + "." + stem,
                          on_device, true);
}

ggml_tensor *expert_gate_up_weight_impl(FullGraph &g, ggml_context *ctx,
                                       int layer) {
  if (offload_layer(layer))
    return nullptr; // separate projection planes inside the offload arena
  if (g.vram_reference_experts && partial_layer_ram_ready(g, layer))
    return nullptr;
  if (partial_layer(g, layer))
    return load_gate_up_weight(g, ctx, layer, false);
  // The shared RAM arena stores gate and up as separate planes inside each
  // expert slice. Force the separate projection views so expert e's gate and
  // up bytes are addressed with the same fixed stride.
  if (partial_layer_ram_ready(g, layer))
    return nullptr;
  const bool on_device =
      layer >= g.first_dense && layer < g.first_dense + g.device_expert_layers;
  return load_gate_up_weight(g, ctx, layer, on_device);
}

// Optional second-GPU expert residency (env MOESTREAM_GLM5NEXT_GPU1_LAYERS,
// "first-last"): decode-graph expert tensors of those layers are copied to a
// CUDA buffer on device 1 so the scheduler runs their MUL_MAT_ID there.
bool gpu1_layer(const FullGraph &g, int layer) {
  return g.decode_mode && g.gpu1 && layer >= g.gpu1_first &&
         layer <= g.gpu1_last;
}

ggml_tensor *move_expert_to_gpu1(FullGraph &g, ggml_context *ctx, int layer,
                                 ggml_tensor *host) {
  if (!host || !gpu1_layer(g, layer) || host->ne[3] != 1)
    return host;
  ggml_tensor *t = ggml_new_tensor_3d(ctx, host->type, host->ne[0], host->ne[1],
                                      host->ne[2]);
  if (!t)
    return host;
  const auto buft = ggml_backend_get_default_buffer_type(g.gpu1);
  ggml_backend_buffer_t buf =
      ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(t) + 4096);
  if (!buf) {
    LOG_WARN("[GGML-GLM5NEXT-FULL] GPU1 expert buffer declined layer=%d "
             "(%.2f GiB); keeping host tensor\n",
             layer, (double)ggml_nbytes(t) / (1024.0 * 1024.0 * 1024.0));
    return host;
  }
  ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  if (ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf)) !=
      GGML_STATUS_SUCCESS) {
    ggml_backend_buffer_free(buf);
    return host;
  }
  const size_t per_expert = ggml_row_size(host->type, host->ne[0]) *
                            (size_t)host->ne[1];
  for (int64_t e = 0; e < host->ne[2]; ++e)
    ggml_backend_tensor_set(t, (const char *)host->data + (size_t)e * host->nb[2],
                            (size_t)e * t->nb[2], per_expert);
  ggml_set_name(t, host->name);
  g.gpu1_buffers.push_back(buf);
  g.gpu1_bytes += ggml_nbytes(t);
  return t;
}

} // namespace glm5next_graph

#endif // MOESTREAM_USE_GGML_CUDA
