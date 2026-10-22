#pragma once

// Cold-side expert provider (R2.1 B): everything a mixed VRAM/DDR model needs for
// the experts that are NOT resident in VRAM, with no dependency on a family graph
// or on FullGraph and no environment lookups.
//
//   OffloadProfile      plain description of the DDR tier (sizes, policy, files)
//   make_store_config   profile + engine geometry -> ExpertStoreConfig
//   seed_store          usage-ranked DDR admission that skips VRAM-hot experts
//   make_plane_tensor   zero-copy ggml tensor over one arena plane
//   make_assist_ring    GPU-assist staging ring shared by every lane
//
// The owner of the ExpertStore is the caller (the family runtime keeps one per
// process); this header only builds and fills it.

#include "early_ready_lane.hpp"
#include "engine.hpp"
#include "expert_cache.hpp"
#include "expert_store.hpp"
#include "graph_core.hpp"
#include "logger.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace moestream::graph {

struct OffloadProfile {
  int first_layer = 0;
  // Model shards that back the store (required). A directory means "every .gguf
  // in it"; a "-00001-of-0000N" first shard expands to the whole set.
  std::string model_path;
  // Explicit shard list; overrides model_path when set (engines already know it).
  std::vector<std::string> model_shards;
  std::string store_path;  // optional expert-contiguous store
  std::string mirror_dir;  // optional second-drive copy
  bool split_reads = false;
  bool huge_pages = false;
  // <= 0 sizes the pool from available memory (see auto_ddr_gib).
  double ddr_gib = 0.0;
  bool lru_segmented = false;
  double pinned_fraction = 0.5;
  double probation_fraction = 0.25;
  bool byte_capacity = false;
  bool async_loads = false;
  // Page-lock arena slots for direct DMA (staging and GPU assist).
  bool pin_for_dma = false;
  std::function<bool(void *, size_t)> pin_fn;
  std::function<void(void *)> unpin_fn;
  const char *log_tag = "[OFFLOAD]";
};

// Pool size from the memory this process may use: MemAvailable (or the cgroup
// limit if lower) minus `reserve_gib` for the rest of the engine, at least
// `floor_gib`; `fallback_gib` when neither can be read.
inline double auto_ddr_gib(double reserve_gib = 36.0, double floor_gib = 16.0,
                           double fallback_gib = 40.0) {
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
  return avail_gib > 0.0 ? std::max(floor_gib, avail_gib - reserve_gib)
                         : fallback_gib;
}

inline std::vector<std::string> expand_model_shards(const std::string &path) {
  std::vector<std::string> shards;
  const size_t pos = path.rfind("-00001-of-");
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec)) {
    for (const auto &entry : std::filesystem::directory_iterator(path))
      if (entry.is_regular_file() && entry.path().extension() == ".gguf")
        shards.push_back(entry.path().string());
    std::sort(shards.begin(), shards.end());
  } else if (pos == std::string::npos) {
    shards.push_back(path);
  } else {
    const std::string head = path.substr(0, pos);
    const int total = std::atoi(path.c_str() + pos + 10);
    char buf[64];
    for (int i = 1; i <= std::max(1, total); ++i) {
      std::snprintf(buf, sizeof buf, "-%05d-of-%05d.gguf", i, total);
      shards.push_back(head + buf);
    }
  }
  return shards;
}

// Builds the store configuration from the engine's expert geometry. Returns false
// (with `err`) when a layer's planes are missing, uneven, or not 4096-aligned.
inline bool make_store_config(const StreamingEngine &engine,
                              const OffloadProfile &profile,
                              moestream::offload::ExpertStoreConfig *out,
                              std::string *err) {
  const auto &c = engine.model_file.architecture_config;
  const int n_layers = c.n_layers - profile.first_layer;
  const auto fail = [&](const std::string &why) {
    if (err) *err = why;
    return false;
  };
  if (profile.first_layer < c.first_dense || n_layers <= 0 || c.n_experts <= 0)
    return fail("bad first offload layer " + std::to_string(profile.first_layer));
  if (profile.model_path.empty() && profile.model_shards.empty())
    return fail("offload model path (first shard) is required");

  moestream::offload::ExpertStoreConfig sc;
  sc.first_layer = profile.first_layer;
  sc.n_layers = n_layers;
  sc.n_experts = c.n_experts;
  sc.log_tag = profile.log_tag;
  for (int l = profile.first_layer; l < c.n_layers; ++l) {
    std::array<uint64_t, 3> pb{};
    int k = 0;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(l) + "." + stem);
      if (!m || m->size_bytes % (uint64_t)c.n_experts != 0)
        return fail("layer " + std::to_string(l) + " " + stem + " missing/uneven");
      pb[(size_t)k] = m->size_bytes / (uint64_t)c.n_experts;
      if (pb[(size_t)k] % 4096 != 0)
        return fail("layer " + std::to_string(l) + " " + stem +
                    " per-expert size is not a multiple of 4096");
      ++k;
    }
    sc.plane_bytes.push_back(pb);
  }
  sc.huge_pages = profile.huge_pages;
  sc.mirror_dir = profile.mirror_dir;
  sc.split_reads = profile.split_reads;
  sc.store_path = profile.store_path;
  sc.model_shards = profile.model_shards.empty()
                        ? expand_model_shards(profile.model_path)
                        : profile.model_shards;
  sc.ddr_gib = profile.ddr_gib > 0.0 ? profile.ddr_gib : auto_ddr_gib();
  sc.policy = profile.lru_segmented
                  ? moestream::offload::ExpertCache::Policy::SegmentedLru
                  : moestream::offload::ExpertCache::Policy::Lru;
  sc.pinned_fraction = profile.pinned_fraction;
  sc.probation_fraction = profile.probation_fraction;
  sc.byte_capacity = profile.byte_capacity;
  sc.async_loads = profile.async_loads;
  if (profile.pin_for_dma && profile.pin_fn && profile.unpin_fn) {
    sc.pin_fn = profile.pin_fn;
    sc.unpin_fn = profile.unpin_fn;
  }
  *out = std::move(sc);
  return true;
}

// Experts served from VRAM; they must not consume DDR admission. `deferred` ones
// (a second GPU's hot set that a batched prefill folds back into the cold
// branch) are seeded last, so they only take DDR the rest leave free.
struct SeedExclusions {
  std::vector<std::vector<uint8_t>> resident; // [layer][expert]
  std::vector<std::vector<uint8_t>> deferred; // [layer][expert]
  size_t excluded = 0;
};

// Seeds the store hottest-first from a usage file. Returns experts that failed.
inline size_t seed_store(moestream::offload::ExpertStore &store,
                         const std::string &usage_file, int first_layer,
                         int n_layers, int n_experts,
                         const SeedExclusions *exclusions, const char *tag) {
  auto ranked = moestream::offload::rank_hottest_first(usage_file, first_layer,
                                                       n_layers, n_experts);
  if (exclusions) {
    const auto in_set = [](const std::vector<std::vector<uint8_t>> &m,
                           const moestream::offload::ExpertKey &k) {
      return (size_t)k.layer < m.size() && !m[(size_t)k.layer].empty() &&
             m[(size_t)k.layer][(size_t)k.expert];
    };
    std::vector<moestream::offload::ExpertKey> late;
    ranked.erase(std::remove_if(ranked.begin(), ranked.end(),
                                [&](const moestream::offload::ExpertKey &k) {
                                  if (in_set(exclusions->deferred, k)) {
                                    late.push_back(k);
                                    return true;
                                  }
                                  return in_set(exclusions->resident, k);
                                }),
                 ranked.end());
    ranked.insert(ranked.end(), late.begin(), late.end());
    LOG_INFO("%s excluded %zu VRAM-hot experts from the DDR seed\n", tag,
             exclusions->excluded);
  }
  const auto t0 = std::chrono::steady_clock::now();
  const size_t failed = store.seed(ranked);
  if (store.pinning())
    LOG_INFO("%s pinned %llu experts (%llu failed) for direct DMA\n", tag,
             (unsigned long long)store.pin_calls(),
             (unsigned long long)store.pin_failures());
  LOG_INFO("%s seeded %zu/%zu experts (%zu failed) in %.1f s\n", tag,
           store.cache().size(), store.capacity(), failed,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
               .count());
  return failed;
}

// Zero-copy tensor over one projection plane of a layer's arena. `plane` is
// 0 gate / 1 up / 2 down. The backing CPU buffer handle is appended to
// `buffers_out` so the caller controls its lifetime.
inline ggml_tensor *make_plane_tensor(
    const StreamingEngine &engine, moestream::offload::VirtualArena &arena,
    int n_experts, ggml_context *ctx, int layer, const char *stem,
    const std::function<bool(GGUFType)> &type_supported,
    std::vector<ggml_backend_buffer_t> *buffers_out, const char *tag) {
  const int plane = std::strcmp(stem, "ffn_gate_exps.weight") == 0   ? 0
                    : std::strcmp(stem, "ffn_up_exps.weight") == 0   ? 1
                    : std::strcmp(stem, "ffn_down_exps.weight") == 0 ? 2
                                                                     : -1;
  const std::string name = "blk." + std::to_string(layer) + "." + stem;
  const GGUFTensor *meta = model_tensor(engine, name);
  if (plane < 0 || !meta || !type_supported(meta->type) ||
      meta->shape.size() != 3)
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : meta->shape)
    shape.push_back((int64_t)d);
  ggml_tensor *t = new_tensor(ctx, (ggml_type)meta->type, shape);
  if (!t || t->nb[2] != arena.stride(layer, plane) ||
      (uint64_t)t->ne[2] != (uint64_t)n_experts) {
    LOG_ERROR("%s %s: tensor stride %zu != arena stride\n", tag, name.c_str(),
              t ? t->nb[2] : (size_t)0);
    return nullptr;
  }
  const size_t len = (size_t)t->nb[2] * (size_t)t->ne[2];
  ggml_backend_buffer_t buffer =
      ggml_backend_cpu_buffer_from_ptr(arena.ptr(layer, 0, plane), len);
  if (!buffer)
    return nullptr;
  ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  t->data = arena.ptr(layer, 0, plane);
  t->buffer = buffer;
  if (ggml_backend_buffer_init_tensor(buffer, t) != GGML_STATUS_SUCCESS) {
    ggml_backend_buffer_free(buffer);
    return nullptr;
  }
  if (buffers_out)
    buffers_out->push_back(buffer);
  ggml_set_name(t, (name + ".offload").c_str());
  return t;
}

// GPU-assist staging ring: `slots` expert planes per projection on `device`. One
// ring is shared by every lane and freed with the last one. `layer_ok` selects
// which layers have to fit (the ring is sized to the largest).
inline std::shared_ptr<GpuAssistShared> make_assist_ring(
    moestream::offload::VirtualArena &arena, int first_layer, int n_layers,
    int device, int slots, const std::function<bool(int)> &layer_ok) {
  auto s = std::make_shared<GpuAssistShared>();
  s->be = ggml_backend_cuda_init(device);
  s->slots = slots;
  bool ok = s->be != nullptr;
  for (int p = 0; ok && p < 3; ++p) {
    for (int l = first_layer; l < first_layer + n_layers; ++l)
      if (layer_ok(l))
        s->plane_cap[p] =
            std::max<size_t>(s->plane_cap[p], (size_t)arena.plane_bytes(l, p));
    s->ring[p] =
        ggml_backend_alloc_buffer(s->be, s->plane_cap[p] * (size_t)s->slots + 4096);
    ok = s->ring[p] != nullptr;
  }
  return ok ? s : nullptr;
}

} // namespace moestream::graph
