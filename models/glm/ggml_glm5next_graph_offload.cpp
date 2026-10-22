#include <chrono>
#include <deque>
// GLM5Next full graph: SSD-offload routed experts: the ExpertStore-backed miss node and
// offload weight tensors.
//
// Split out of the former single-file ggml_glm5next_graph.cpp; shared
// declarations live in ggml_glm5next_graph_impl.hpp.

#include <future>
#include <thread>
#include <map>
#include "ggml_glm5next_graph_impl.hpp"
#include "offload_expert_provider.hpp"

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
  // The family only translates its environment into a plain profile; geometry,
  // shard discovery, pool sizing and the store configuration come from the
  // shared provider.
  moestream::graph::OffloadProfile profile;
  profile.first_layer = o->first;
  profile.log_tag = "[GGML-GLM5NEXT-OFFLOAD]";
  profile.huge_pages = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_THP", "0") == "1";
  profile.mirror_dir = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_MIRROR");
  profile.split_reads = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_SPLIT", "0") == "1";
  profile.store_path = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_STORE");
  profile.model_path = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_MODEL");
  {
    // MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB=<n> fixes the pool; unset (or "auto") sizes it from the memory this
    // process may actually use: available RAM (or the cgroup limit if lower) minus 36 GiB for the rest of the
    // engine and the OS, at least 16 GiB (60 GiB before the Q2_K/Q3_K GSQ-RCO file, whose ~74 GiB cold set needs more than 58). The store caps it at the total expert bytes. Measured on a 125 GB
    // box: decode 5.3 / 7.3 / 8.2 tok/s at 32 / 48 / 58 GiB, flat beyond (everything resident).
    const std::string v = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB", "auto");
    if (v == "auto" || v.empty()) {
      profile.ddr_gib = moestream::graph::auto_ddr_gib();
      LOG_INFO("[GGML-GLM5NEXT-OFFLOAD] auto DDR pool %.1f GiB\n", profile.ddr_gib);
    } else {
      profile.ddr_gib = std::atof(v.c_str());
    }
  }
  profile.lru_segmented = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_POLICY", "lru") == "seg";
  profile.pinned_fraction =
      std::atof(offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_PINNED", "0.5").c_str());
  profile.probation_fraction =
      std::atof(offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_PROBATION", "0.25").c_str());
  profile.byte_capacity = offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_BYTE_CAP", "0") == "1";
  profile.async_loads = offload_async();
  // Page-lock the arena slots (DMA straight from DDR) for staging and for the lane's GPU assist.
  if ((pilot_stage_k() > 0 || gpu_assist_slots() > 0) &&
      offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_PIN", gpu_assist_slots() > 0 ? "1" : "0") == "1") {
    profile.pin_for_dma = true;
    profile.pin_fn = [](void *ptr, size_t bytes) {
      if (cudaHostRegister(ptr, bytes, cudaHostRegisterPortable) == cudaSuccess)
        return true;
      cudaGetLastError();
      return false;
    };
    profile.unpin_fn = [](void *ptr) { cudaHostUnregister(ptr); };
  }
  moestream::offload::ExpertStoreConfig sc;
  std::string cfg_err;
  if (!moestream::graph::make_store_config(engine, profile, &sc, &cfg_err)) {
    LOG_ERROR("[GGML-GLM5NEXT-OFFLOAD] %s\n", cfg_err.c_str());
    return nullptr;
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
    moestream::graph::SeedExclusions excl;
    const moestream::graph::SeedExclusions *excl_ptr = nullptr;
    if (partial) {
      // Experts that live in VRAM are served by the GPU branch; do not spend DDR on them.
      // GPU1-hot experts are served by the GPU1 branch in decode, but a batched prefill graph has no such
      // branch and runs them as cold experts. Seed them last so they only take DDR that the rest leave free
      // (otherwise prefill reads them from the SSD, ~0.7 s inline plus ~6 s of background reads at 805 tokens).
      const bool seed_hot1 = offload_env("MOESTREAM_GLM5NEXT_PREFILL_OP_OFFLOAD", "0") == "1" &&
                             offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_SEED_GPU1_HOT", "1") == "1";
      excl.resident.resize((size_t)c.n_layers);
      excl.deferred.resize((size_t)c.n_layers);
      for (size_t l = 0; l < partial->size() && l < excl.resident.size(); ++l) {
        excl.resident[l].assign((size_t)o->n_experts, 0);
        excl.deferred[l].assign((size_t)o->n_experts, 0);
        for (int e : (*partial)[l].hot)
          if (e >= 0 && e < o->n_experts) {
            excl.resident[l][(size_t)e] = 1;
            ++excl.excluded;
          }
        for (int e : (*partial)[l].hot1)
          if (e >= 0 && e < o->n_experts) {
            (seed_hot1 ? excl.deferred : excl.resident)[l][(size_t)e] = 1;
            if (!seed_hot1) ++excl.excluded;
          }
      }
      excl_ptr = &excl;
    }
    moestream::graph::seed_store(*o->store, usage, o->first, o->n_layers,
                                 o->n_experts, excl_ptr, "[GGML-GLM5NEXT-OFFLOAD]");
  }
  LOG_INFO("[GGML-GLM5NEXT-OFFLOAD] layers %d..%d, capacity %zu experts "
           "(%.1f GiB), pinned %.2f, store=%s\n",
           o->first, c.n_layers - 1, o->store->capacity(), sc.ddr_gib,
           sc.pinned_fraction,
           sc.store_path.empty() ? "<none>" : sc.store_path.c_str());
  g_offload = std::move(o);
  return g_offload.get();
}

extern "C" cudaStream_t moestream_ggml_backend_cuda_get_stream(ggml_backend_t backend);

namespace {
bool g_prefetch_built = false, g_prefetch_suppress = false;
int g_ring_fail_s = 0x7fffffff; // smallest prompt batch for which a ring was found not to fit next to the graph
double g_prefetch_main_gib = 0.0;
} // namespace

namespace {

// ---- speculative expert prefetch for batched prefill ------------------------------------------------------------
// A 805-token prompt routes to nearly every cold expert of a layer, so the layer's whole cold set can be copied to the GPU
// before its router has run. The copies of layer L+1 are issued by the miss node of layer L on a dedicated stream (from a
// worker thread, so pageable host memory cannot stall the graph thread), into one of `sets` device buffer sets, and the
// miss node of L+1 waits for them. Experts that were not resident yet are copied by the miss node itself once the used
// ones are known. The cold weights of a layer are a compact [ne0, ne1, n_cold] tensor over the buffers, indexed by the
// rank of an expert among the layer's cold experts; the miss node rewrites the route ids accordingly.
struct PrefetchLayer {
  int layer = 0, set = 0;
  int split = 0;                // compact slots [0, split) live on part 0, the rest on part 1
  std::vector<int> experts;     // compact slot -> expert id
  std::vector<int32_t> rank;    // expert id -> compact slot, -1 if not part of the cold set
  std::vector<uint8_t> copied;  // per slot: the planes are on the device
  size_t bytes[3] = {0, 0, 0};
  bool issued = false;
  std::future<void> worker;
};

// One device's share of the buffers: its own copy stream and one event per set.
struct PrefetchPart {
  int device = 0;
  cudaStream_t stream = nullptr;
  std::vector<cudaEvent_t> events;
  char *base[4][3] = {};
  ggml_backend_buffer_t buf[4][3] = {};
};

struct PrefetchRing {
  int sets = 2, n_parts = 1;
  PrefetchPart parts[2];
  std::map<int, PrefetchLayer> layers;
};

std::unique_ptr<PrefetchRing> g_ring;

void prefetch_copy_slot(PrefetchRing &ring, PrefetchLayer &lp, OffloadState &o, size_t k) {
  const int e = lp.experts[k];
  const int part = (int)k >= lp.split ? 1 : 0;
  const size_t local = k - (part ? (size_t)lp.split : 0);
  PrefetchPart &pp = ring.parts[part];
  cudaSetDevice(pp.device);
  for (int p = 0; p < 3; ++p)
    cudaMemcpyAsync(pp.base[lp.set][p] + local * lp.bytes[p], o.arena->ptr(lp.layer, e, p), lp.bytes[p],
                    cudaMemcpyHostToDevice, pp.stream);
  lp.copied[k] = 1;
}

void prefetch_record(PrefetchRing &ring, const PrefetchLayer &lp) {
  for (int part = 0; part < ring.n_parts; ++part) {
    cudaSetDevice(ring.parts[part].device);
    cudaEventRecord(ring.parts[part].events[(size_t)lp.set], ring.parts[part].stream);
  }
}

// Copies every expert of the layer that is already populated, then records the set's events. Runs on a worker thread.
// The copies are queued a few experts at a time (two chunks in flight per device): a GPU has a single host-to-device copy
// engine, and the activations that cross between the two GPUs are staged through the host, so they wait behind everything
// already queued there. Queueing a layer's whole cold set at once delayed them by the full copy time.
void prefetch_issue(PrefetchRing *ring, PrefetchLayer *lp, OffloadState *o) {
  static const int chunk = std::max(1, moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_PREFILL_PREFETCH_CHUNK", 2));
  // A copy another stream submits while a long run of weight copies is in flight is not served until the run ends (the copy
  // engine does not interleave streams), so the small inputs of the layer that the miss node has just released (the
  // rewritten ids, ...) would wait for the whole run. The run therefore starts after those uploads have been issued.
  static const int delay_us = moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_PREFILL_PREFETCH_DELAY_US", 0);
  if (delay_us > 0)
    std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
  static const bool drain = moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_PREFILL_PREFETCH_DRAIN", 1) != 0;
  const auto issue_t0 = std::chrono::steady_clock::now();
  struct Chunker {
    cudaEvent_t ev[2] = {nullptr, nullptr};
    int n = 0, in_chunk = 0;
  } ch[2];
  for (int part = 0; part < ring->n_parts; ++part) {
    cudaSetDevice(ring->parts[part].device);
    cudaEventCreateWithFlags(&ch[part].ev[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&ch[part].ev[1], cudaEventDisableTiming);
  }
  // One thread per part: a part waits for each of its chunks (see `drain`), and a single thread doing that for both GPUs
  // serialized the two links (120 ms per layer instead of ~90).
  const size_t n_slots = lp->experts.size();
  auto issue_part = [&](int part) {
    const size_t lo = part == 0 ? 0 : (size_t)lp->split, hi = part == 0 ? (size_t)lp->split : n_slots;
    Chunker &c = ch[part];
    cudaSetDevice(ring->parts[part].device);
    for (size_t k = lo; k < hi; ++k) {
      if (lp->copied[k] || !o->residency->data_ready({lp->layer, lp->experts[k]}))
        continue;
      if (c.in_chunk == 0 && c.n >= 2)
        cudaEventSynchronize(c.ev[c.n % 2]);
      prefetch_copy_slot(*ring, *lp, *o, k);
      if (++c.in_chunk == chunk) {
        cudaEventRecord(c.ev[c.n % 2], ring->parts[part].stream);
        // Waiting for the chunk just queued empties the device's copy queue at the chunk boundary, which is the only point
        // where the small uploads of the compute graph (ids, KDA history, activations crossing between the GPUs) get
        // served; while the queue never empties they wait for the whole layer's run (67 ms measured).
        if (drain)
          cudaEventSynchronize(c.ev[c.n % 2]);
        ++c.n;
        c.in_chunk = 0;
      }
    }
  };
  std::thread second;
  if (ring->n_parts > 1 && lp->split < (int)n_slots)
    second = std::thread(issue_part, 1);
  issue_part(0);
  if (second.joinable())
    second.join();
  prefetch_record(*ring, *lp);
  if (moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PREFETCH_DEBUG"))
    LOG_INFO("[GGML-GLM5NEXT-PREFETCH] issue layer %d chunk=%d chunks=%d/%d issued in %.1f ms\n", lp->layer, chunk, ch[0].n,
             ch[1].n, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - issue_t0).count());
  for (int part = 0; part < ring->n_parts; ++part) {
    cudaSetDevice(ring->parts[part].device);
    cudaEventSynchronize(ch[part].ev[0] ? ch[part].ev[0] : nullptr); // recorded or never used: both complete
    cudaEventDestroy(ch[part].ev[0]);
    cudaEventDestroy(ch[part].ev[1]);
  }
}

// Called by the miss node of `layer` with its route ids (expert ids, negative = inactive): returns once every used expert
// is on the device, rewrites `ids` to compact slots and launches the copies of the layers after it.
void prefetch_service(OffloadState &o, int layer, std::vector<int32_t> &ids) {
  PrefetchRing &ring = *g_ring;
  PrefetchLayer &lp = ring.layers.at(layer);
  static const bool dbg = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PREFETCH_DEBUG") != nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  if (lp.worker.valid())
    lp.worker.get();
  const auto t1 = std::chrono::steady_clock::now();
  std::vector<uint8_t> seen(lp.experts.size(), 0);
  bool fixed = false;
  size_t n_fixed = 0, n_copied_before = 0;
  for (uint8_t c : lp.copied) n_copied_before += c;
  for (int32_t id : ids) {
    if (id < 0)
      continue;
    const int32_t k = (size_t)id < lp.rank.size() ? lp.rank[(size_t)id] : -1;
    if (k < 0) {
      LOG_ERROR("[GGML-GLM5NEXT-PREFETCH] layer %d routed to expert %d outside its cold set\n", layer, id);
      std::abort();
    }
    if (!lp.copied[(size_t)k] && !seen[(size_t)k]) {
      seen[(size_t)k] = 1;
      prefetch_copy_slot(ring, lp, o, (size_t)k);
      fixed = true;
      ++n_fixed;
    }
  }
  const auto t2 = std::chrono::steady_clock::now();
  if (lp.issued)
    for (int part = 0; part < ring.n_parts; ++part)
      cudaEventSynchronize(ring.parts[part].events[(size_t)lp.set]);
  if (fixed)
    for (int part = 0; part < ring.n_parts; ++part)
      cudaStreamSynchronize(ring.parts[part].stream);
  if (dbg) {
    const auto t3 = std::chrono::steady_clock::now();
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    LOG_INFO("[GGML-GLM5NEXT-PREFETCH] layer %d slots=%zu issued=%d copied_before=%zu fixup=%zu worker_wait=%.1f scan=%.1f "
             "event_wait=%.1f ms\n", layer, lp.experts.size(), (int)lp.issued, n_copied_before, n_fixed, ms(t0, t1),
             ms(t1, t2), ms(t2, t3));
  }
  for (int32_t &id : ids)
    if (id >= 0)
      id = lp.rank[(size_t)id];
  for (int d = 1; d < ring.sets; ++d) {
    auto it = ring.layers.find(layer + d);
    if (it == ring.layers.end() || it->second.issued)
      continue;
    it->second.issued = true;
    it->second.worker = std::async(std::launch::async, prefetch_issue, &ring, &it->second, &o);
  }
}

// Builds the ring and the per-layer plans for the prompt graph being built (called when its first offload layer is built).
bool prefetch_prepare(FullGraph &g, OffloadState &o) {
  // Any previous ring must be drained and freed before its buffers can be
  // replaced; g_ring owns them now (they are no longer graph-owned).
  if (g_ring)
    prefetch_ring_join();
  g_prefetch_built = false;
  if (g_prefetch_suppress || g.S >= g_ring_fail_s)
    return false;
  // On by default when the second GPU takes part (the ring is split between the two links); off for a single GPU, where the
  // ring would not fit next to the 3090's expert pool.
  const char *want = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PREFILL_PREFETCH");
  if (!(want ? std::atoi(want) != 0 : g.gpu1 != nullptr) || g.decode_mode || g.S < 64 || !g.cuda)
    return false;
  const auto &c = g.engine->model_file.architecture_config;
  auto ring = std::make_unique<PrefetchRing>();
  ring->sets = std::max(2, std::min(4, moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_PREFILL_PREFETCH_SETS", 2)));
  // The cold experts of a layer are split between the two GPUs in the ratio of their PCIe links (the main GPU's share is
  // MOESTREAM_GLM5NEXT_PREFILL_SPLIT, default 0.63), so both links copy at once and each GPU needs less buffer.
  const bool two = g.gpu1 != nullptr;
  ring->n_parts = two ? 2 : 1;
  double ratio = 0.63;
  if (const char *e = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PREFILL_SPLIT"))
    ratio = std::atof(e);
  if (ratio <= 0.0 || ratio >= 1.0)
    ratio = 1.0;
  if (!two)
    ratio = 1.0;
  ring->parts[0].device = g.device;
  if (two) {
    int cur = 0;
    cudaGetDevice(&cur);
    ring->parts[1].device = second_gpu_device(g.device);
    (void)cur;
  }
  size_t cap[2][4][3] = {};
  int ordinal = 0;
  for (int l = o.first; l < c.n_layers; ++l) {
    const PartialExpertLayer *pl = partial_layer(g, l);
    if (!pl || pl->cold.empty() || !offload_layer(l))
      continue;
    std::vector<uint8_t> skip((size_t)o.n_experts, 0);
    if (g.gpu1)
      for (int e : pl->hot1)
        if (e >= 0 && e < o.n_experts)
          skip[(size_t)e] = 1;
    PrefetchLayer lp;
    lp.layer = l;
    lp.set = ordinal % ring->sets;
    lp.rank.assign((size_t)o.n_experts, -1);
    std::vector<int> cold(pl->cold.begin(), pl->cold.end());
    std::sort(cold.begin(), cold.end());
    for (int e : cold)
      if (e >= 0 && e < o.n_experts && !skip[(size_t)e]) {
        lp.rank[(size_t)e] = (int32_t)lp.experts.size();
        lp.experts.push_back(e);
      }
    if (lp.experts.empty())
      continue;
    const int n = (int)lp.experts.size();
    lp.split = ratio >= 1.0 ? n : std::max(1, std::min(n - 1, (int)(ratio * n + 0.5)));
    lp.copied.assign(lp.experts.size(), 0);
    for (int p = 0; p < 3; ++p) {
      lp.bytes[p] = (size_t)o.arena->plane_bytes(l, p);
      cap[0][lp.set][p] = std::max(cap[0][lp.set][p], lp.bytes[p] * (size_t)lp.split);
      cap[1][lp.set][p] = std::max(cap[1][lp.set][p], lp.bytes[p] * (size_t)(n - lp.split));
    }
    ++ordinal;
    ring->layers.emplace(l, std::move(lp));
  }
  if (ring->layers.empty())
    return false;
  size_t total[2] = {0, 0};
  bool ok = true;
  {
    size_t want[2] = {0, 0};
    for (int part = 0; part < 2; ++part)
      for (int s = 0; s < ring->sets; ++s)
        for (int p = 0; p < 3; ++p)
          want[part] += cap[part][s][p];
    for (int part = 0; part < ring->n_parts; ++part) {
      size_t freeb = 0, totb = 0;
      cudaSetDevice(ring->parts[part].device);
      cudaMemGetInfo(&freeb, &totb);
      LOG_INFO("[GGML-GLM5NEXT-PREFETCH] device %d: free %.3f GiB before the ring, ring %.3f GiB\n", part,
               (double)freeb / (1024.0 * 1024.0 * 1024.0), (double)want[part] / (1024.0 * 1024.0 * 1024.0));
      // The graph's own workspace is allocated after the ring and grows ~0.8 MiB per token on this device (free memory
      // after the build: 0.20 GiB at 814 tokens, 0.004 GiB at 1065, where the first compute then failed to allocate its
      // runtime resources). The ring is only worth it when it leaves that and a small reserve free.
      // The graph's scheduler allocation runs after this ring and refuses to
      // proceed below its own free-VRAM floor.  Reserve that floor on the main
      // GPU as well as the ring and its workspace; otherwise a ring can be
      // built only to have the graph declined, forcing the retry path.
      size_t floor_bytes = 2ull * 1024ull * 1024ull * 1024ull;
      if (const char *e = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_ALLOC_FLOOR_MIB"))
        floor_bytes = (size_t)std::max(0, std::atoi(e)) << 20;
      const double need = (double)want[part] + 0.85 * 1024.0 * 1024.0 * g.S +
                          0.15 * 1024.0 * 1024.0 * 1024.0 +
                          (part == 0 ? (double)floor_bytes : 0.0);
      if (need > (double)freeb) {
        LOG_INFO("[GGML-GLM5NEXT-PREFETCH] device %d: ring + workspace for %d tokens need %.2f GiB, %.2f GiB free; "
                 "prefetch off for this batch\n", part, g.S, need / (1024.0 * 1024.0 * 1024.0),
                 (double)freeb / (1024.0 * 1024.0 * 1024.0));
        return false;
      }
    }
  }
  for (int part = 0; part < ring->n_parts && ok; ++part) {
    ggml_backend_t be = part == 0 ? g.cuda : g.gpu1;
    for (int s = 0; s < ring->sets && ok; ++s)
      for (int p = 0; p < 3; ++p) {
        if (cap[part][s][p] == 0)
          continue;
        ggml_backend_buffer_t buf =
            ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(be), cap[part][s][p] + 4096);
        if (!buf) {
          LOG_WARN("[GGML-GLM5NEXT-PREFETCH] device %d: %.2f GiB of buffers do not fit; prefetch off for this prompt\n",
                   part, (double)(total[part] + cap[part][s][p]) / (1024.0 * 1024.0 * 1024.0));
          ok = false;
          break;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ring->parts[part].buf[s][p] = buf;
        ring->parts[part].base[s][p] = (char *)ggml_backend_buffer_get_base(buf);
        total[part] += cap[part][s][p];
      }
  }
  if (!ok) {
    for (int part = 0; part < 2; ++part)
      for (int s = 0; s < 4; ++s)
        for (int p = 0; p < 3; ++p)
          if (ring->parts[part].buf[s][p])
            ggml_backend_buffer_free(ring->parts[part].buf[s][p]);
    return false;
  }
  static cudaStream_t streams[2] = {nullptr, nullptr};
  static std::vector<cudaEvent_t> events[2];
  for (int part = 0; part < ring->n_parts; ++part) {
    cudaSetDevice(ring->parts[part].device);
    if (!streams[part])
      cudaStreamCreateWithFlags(&streams[part], cudaStreamNonBlocking);
    while ((int)events[part].size() < 4) {
      cudaEvent_t ev = nullptr;
      cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
      events[part].push_back(ev);
    }
    ring->parts[part].stream = streams[part];
    ring->parts[part].events = events[part];
  }
  cudaSetDevice(ring->parts[0].device);
  if (moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PREFETCH_DEBUG")) {
    unsigned long long id[4] = {0, 0, 0, 0};
    cudaStreamGetId(ring->parts[0].stream, &id[0]);
    if (ring->n_parts > 1)
      cudaStreamGetId(ring->parts[1].stream, &id[1]);
    cudaStreamGetId(moestream_ggml_backend_cuda_get_stream(g.cuda), &id[2]);
    if (g.gpu1)
      cudaStreamGetId(moestream_ggml_backend_cuda_get_stream(g.gpu1), &id[3]);
    LOG_INFO("[GGML-GLM5NEXT-PREFETCH] stream ids: copy %llu/%llu backend %llu/%llu\n", id[0], id[1], id[2], id[3]);
  }
  LOG_INFO("[GGML-GLM5NEXT-PREFETCH] %zu layers, %d sets, buffers %.2f GiB on the main GPU + %.2f GiB on the second\n",
           ring->layers.size(), ring->sets, (double)total[0] / (1024.0 * 1024.0 * 1024.0),
           (double)total[1] / (1024.0 * 1024.0 * 1024.0));
  g_prefetch_built = true;
  g_prefetch_main_gib = (double)total[0] / (1024.0 * 1024.0 * 1024.0);
  g_ring = std::move(ring);
  // The first layers' copies start now, while the dense layers in front of them compute (otherwise the first offload layer
  // has nothing prefetched and the miss node copies its whole cold set itself).
  if (moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_PREFILL_PREFETCH_EARLY", 1) != 0 && !g_ring->layers.empty()) {
    PrefetchLayer &first = g_ring->layers.begin()->second;
    first.issued = true;
    first.worker = std::async(std::launch::async, prefetch_issue, g_ring.get(), &first, &o);
  }
  return true;
}

} // namespace

bool prefetch_ring_was_built() {
  const bool b = g_prefetch_built;
  g_prefetch_built = false;
  return b;
}

bool prefetch_ring_candidate(int S, bool decode_mode, int device) {
  if (decode_mode || S < 64 || g_prefetch_suppress || S >= g_ring_fail_s)
    return false;
  const char *want = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PREFILL_PREFETCH");
  if (want)
    return std::atoi(want) != 0;
  return second_gpu_device(device) >= 0;
}
double prefetch_ring_main_gib() { return g_prefetch_main_gib; }
void prefetch_suppress_prompt(bool on) { g_prefetch_suppress = on; }
void prefetch_ring_join() {
  if (!g_ring)
    return;
  for (auto &kv : g_ring->layers) {
    if (kv.second.worker.valid())
      kv.second.worker.wait();
  }
  // A worker's final event record does not wait for the queued copies.  Drain
  // both copy streams before the owning graph frees the destination buffers.
  for (int part = 0; part < g_ring->n_parts; ++part) {
    if (g_ring->parts[part].stream) {
      cudaSetDevice(g_ring->parts[part].device);
      cudaStreamSynchronize(g_ring->parts[part].stream);
    }
  }
  // The ring owns its buffers now.  Free them as soon as the copy workers
  // are drained; this is also what releases the main GPU's VRAM early
  // enough for the decode graph to reclaim the borrowed expert cache.
  for (int part = 0; part < g_ring->n_parts; ++part) {
    for (int s = 0; s < g_ring->sets; ++s)
      for (int p = 0; p < 3; ++p) {
        if (g_ring->parts[part].buf[s][p])
          ggml_backend_buffer_free(g_ring->parts[part].buf[s][p]);
        g_ring->parts[part].buf[s][p] = nullptr;
        g_ring->parts[part].base[s][p] = nullptr;
      }
  }
  g_ring.reset();
  g_prefetch_built = false;
}
void prefetch_note_ring_failure(int S) { g_ring_fail_s = std::min(g_ring_fail_s, S); }

bool prefetch_layer_active(int layer) { return g_ring && g_ring->layers.count(layer) != 0; }

int prefetch_layer_slots(int layer, int *split) {
  if (!prefetch_layer_active(layer))
    return 0;
  const PrefetchLayer &lp = g_ring->layers.at(layer);
  if (split)
    *split = lp.split;
  return (int)lp.experts.size();
}

ggml_tensor *prefetch_weight(FullGraph &g, ggml_context *ctx, int layer, const char *stem, int part) {
  (void)g;
  if (!prefetch_layer_active(layer) || part < 0 || part >= g_ring->n_parts)
    return nullptr;
  const int plane = std::strcmp(stem, "ffn_gate_exps.weight") == 0   ? 0
                    : std::strcmp(stem, "ffn_up_exps.weight") == 0   ? 1
                    : std::strcmp(stem, "ffn_down_exps.weight") == 0 ? 2
                                                                     : -1;
  const GGUFTensor *meta = model_tensor(*g.engine, "blk." + std::to_string(layer) + "." + stem);
  if (plane < 0 || !meta || !type_ok(meta->type) || meta->shape.size() != 3)
    return nullptr;
  PrefetchLayer &lp = g_ring->layers.at(layer);
  const int64_t ne0 = (int64_t)meta->shape[0], ne1 = (int64_t)meta->shape[1];
  const size_t per_expert = ggml_row_size((ggml_type)meta->type, ne0) * (size_t)ne1;
  const int count = part == 0 ? lp.split : (int)lp.experts.size() - lp.split;
  PrefetchPart &pp = g_ring->parts[part];
  if (per_expert != lp.bytes[plane] || count <= 0 || !pp.buf[lp.set][plane])
    return nullptr;
  ggml_tensor *t = new_tensor(ctx, (ggml_type)meta->type, {ne0, ne1, (int64_t)count});
  if (!t || ggml_backend_tensor_alloc(pp.buf[lp.set][plane], t, pp.base[lp.set][plane]) != GGML_STATUS_SUCCESS)
    return nullptr;
  ggml_set_name(t, ("blk." + std::to_string(layer) + "." + stem + ".prefetch" + std::to_string(part)).c_str());
  return t;
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
  const auto cb_t_ensured = std::chrono::steady_clock::now();
  if (site->prefetch && g_ring)
    prefetch_service(o, site->layer, ids);
  static const bool cb_dbg = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PREFETCH_DEBUG") != nullptr;
  if (cb_dbg && site->prefill) {
    static auto last_cb = std::chrono::steady_clock::time_point{};
    const auto now = std::chrono::steady_clock::now();
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    LOG_INFO("[GGML-GLM5NEXT-MISSCB] layer %d period=%.1f ensure=%.1f service=%.1f ms\n", site->layer,
             last_cb.time_since_epoch().count() ? ms(last_cb, cb_t0) : 0.0, ms(cb_t0, cb_t_ensured),
             ms(cb_t_ensured, now));
    last_cb = now;
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
  return moestream::graph::make_plane_tensor(
      *g.engine, *o->arena, o->n_experts, ctx, layer, stem,
      [](GGUFType t) { return type_ok(t); }, &g.host_buffers,
      "[GGML-GLM5NEXT-OFFLOAD]");
}

ggml_tensor *offload_miss_node(FullGraph &g, ggml_context *ctx, int layer,
                               ggml_tensor *ids) {
  OffloadState *o = get_offload(*g.engine);
  if (!o || !ids)
    return nullptr;
  std::lock_guard<std::mutex> lock(g_offload_mutex);
  if (g.S >= 64 && !g.decode_mode && layer == o->first)
    prefetch_prepare(g, *o);
  o->sites.push_back({o, layer, g.S > 1, g.S, {}});
  o->sites.back().prefetch = prefetch_layer_active(layer);
  if (const PartialExpertLayer *pl = partial_layer(g, layer)) {
    o->sites.back().not_cold.assign((size_t)o->n_experts, 0);
    // hot1 (GPU1) experts are served by GPU1 whenever the graph has the second backend (decode, and batched prefill when GPU1 is enabled there).
    for (const std::vector<int> *set : {&pl->hot, &pl->hot1})
      if (set == &pl->hot || g.gpu1)
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

namespace {
struct ColdSplitSite {
  int split_at;
  int mode;
};
std::deque<ColdSplitSite> g_cold_split_sites;
std::mutex g_cold_split_mutex;

void cold_split_cb(ggml_tensor *dst, const ggml_tensor *a, int ith, int, void *ud) {
  if (ith != 0)
    return;
  const ColdSplitSite *site = static_cast<const ColdSplitSite *>(ud);
  const int64_t n = ggml_nelements(a);
  for (int64_t i = 0; i < n; ++i) {
    const int32_t id = *(const int32_t *)((const char *)a->data + i * (int64_t)sizeof(int32_t));
    const bool in_lo = id >= 0 && id < site->split_at;
    const bool in_hi = id >= site->split_at;
    int32_t v = 0;
    switch (site->mode) {
    case 0: v = in_lo ? id : -1; break;
    case 1: v = in_hi ? id - site->split_at : -1; break;
    case 2: v = in_lo ? 1 : 0; break;
    default: v = in_hi ? 1 : 0; break;
    }
    *(int32_t *)((char *)dst->data + i * (int64_t)sizeof(int32_t)) = v;
  }
}
} // namespace

ggml_tensor *cold_split_node(ggml_context *ctx, ggml_tensor *ids, int split_at, int mode) {
  if (!ids)
    return nullptr;
  ColdSplitSite *site = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_cold_split_mutex);
    g_cold_split_sites.push_back({split_at, mode});
    site = &g_cold_split_sites.back();
  }
  return ggml_map_custom1(ctx, ids, cold_split_cb, 1, site);
}

ggml_tensor *expert_weight_impl(FullGraph &g, ggml_context *ctx, int layer,
                               const char *stem) {
  if (offload_layer(layer))
    return offload_weight(g, ctx, layer, stem);
  if (partial_layer(g, layer) &&
      !(g.vram_reference_experts && partial_layer_ram_ready(g, layer))) {
    if (g.vram_reference_experts && g.engine &&
        g.engine->family_ops.family == ModelFamily::BailingMoE3) {
      LOG_WARN("[GGML-LING-CACHE] no DDR-backed expert arena for layer=%d; "
               "declining rather than copying cold MoE weights into the graph\n",
               layer);
      return nullptr;
    }
    return load_cold_weight(g, ctx, layer, stem);
  }
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
  if (partial_layer(g, layer)) {
    if (g.vram_reference_experts && g.engine &&
        g.engine->family_ops.family == ModelFamily::BailingMoE3) {
      LOG_WARN("[GGML-LING-CACHE] no DDR-backed expert arena for layer=%d; "
               "declining rather than copying cold gate/up weights into the graph\n",
               layer);
      return nullptr;
    }
    return load_gate_up_weight(g, ctx, layer, false);
  }
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
