// cpu_engine.hpp — Host-only inference engine for moestream_cpu.
//
// No CUDA headers, no VRAM, no GPU dispatch.  Manages:
//   * Host-resident dense weights (loaded once at startup)
//   * Host-resident expert LRU cache with pin/preload
//   * Host KV cache
//   * Tokenizer and generation state
#pragma once

#include "cpu_config.hpp"
#include "gguf_loader.hpp"
#include "gguf_tokenizer.hpp"
#include "kernels_cpu.hpp"
#include "logger.hpp"
#include "route_usage.hpp"

// Forward declaration to break the include cycle with cpu_model_backend.hpp.
class CPUModelBackend;

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <list>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// One dense or expert weight tensor resident in host memory.
struct CPUQuantizedTensor {
  GGUFTensor meta;
  uint8_t *host_ptr = nullptr; // posix_memalign'd, owned
  size_t host_size = 0;

  CPUQuantizedTensor() = default;
  CPUQuantizedTensor(const CPUQuantizedTensor &) = delete;
  CPUQuantizedTensor &operator=(const CPUQuantizedTensor &) = delete;
  CPUQuantizedTensor(CPUQuantizedTensor &&o) noexcept
      : meta(std::move(o.meta)), host_ptr(o.host_ptr), host_size(o.host_size) {
    o.host_ptr = nullptr;
    o.host_size = 0;
  }
  CPUQuantizedTensor &operator=(CPUQuantizedTensor &&o) noexcept {
    if (this != &o) {
      free(host_ptr);
      meta = std::move(o.meta);
      host_ptr = o.host_ptr;
      host_size = o.host_size;
      o.host_ptr = nullptr;
      o.host_size = 0;
    }
    return *this;
  }
  ~CPUQuantizedTensor() { free(host_ptr); }
};

// One routed expert resident in the host LRU cache.
struct CPUExpertSlot {
  int layer = -1;
  int expert = -1;
  uint8_t *slab = nullptr; // gate + up + down, contiguous
  size_t slab_size = 0;
  size_t gate_size = 0;
  size_t up_size = 0;
  size_t down_size = 0;
  GGUFType gate_type = GGUFType::F32;
  GGUFType up_type = GGUFType::F32;
  GGUFType down_type = GGUFType::F32;
  uint64_t last_used = 0;
  uint32_t hit_count = 0;
  bool pinned = false;

  CPUExpertSlot() = default;
  CPUExpertSlot(const CPUExpertSlot &) = delete;
  CPUExpertSlot &operator=(const CPUExpertSlot &) = delete;
  CPUExpertSlot(CPUExpertSlot &&o) noexcept
      : layer(o.layer), expert(o.expert), slab(o.slab), slab_size(o.slab_size),
        gate_size(o.gate_size), up_size(o.up_size), down_size(o.down_size),
        gate_type(o.gate_type), up_type(o.up_type), down_type(o.down_type),
        last_used(o.last_used), hit_count(o.hit_count), pinned(o.pinned) {
    o.slab = nullptr;
    o.slab_size = 0;
  }
  CPUExpertSlot &operator=(CPUExpertSlot &&o) noexcept {
    if (this != &o) {
      free(slab);
      layer = o.layer;
      expert = o.expert;
      slab = o.slab;
      slab_size = o.slab_size;
      gate_size = o.gate_size;
      up_size = o.up_size;
      down_size = o.down_size;
      gate_type = o.gate_type;
      up_type = o.up_type;
      down_type = o.down_type;
      last_used = o.last_used;
      hit_count = o.hit_count;
      pinned = o.pinned;
      o.slab = nullptr;
      o.slab_size = 0;
    }
    return *this;
  }
  ~CPUExpertSlot() { free(slab); }
};

// Host-only LRU expert cache with optional pin/preload from a usage file.
class CPUExpertCache {
public:
  struct Key {
    int layer;
    int expert;
  };

  bool init(const GGUFFile &model, double ram_gb,
            const std::unordered_set<uint64_t> &pin_set);

  // Returns a pointer to a resident expert, loading/evicting as needed.
  // The pointer is valid until the next get() call on this cache.
  CPUExpertSlot *get(int layer, int expert);

  size_t capacity() const { return slots_.size(); }
  size_t resident_count() const { return index_.size(); }
  size_t bytes_in_cache() const;
  uint64_t misses() const { return misses_; }
  uint64_t hits() const { return hits_; }

private:
  const GGUFFile *model_ = nullptr;
  std::vector<CPUExpertSlot> slots_;
  // layer<<32 | expert  -> slot index
  std::unordered_map<uint64_t, size_t> index_;
  // LRU order: front = most recently used, back = least recently used
  std::list<size_t> lru_;
  std::unordered_set<uint64_t> pin_set_;
  uint64_t clock_ = 0;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;

  static uint64_t key(int layer, int expert) {
    return (static_cast<uint64_t>(layer) << 32) | static_cast<uint32_t>(expert);
  }
  bool load_expert(int layer, int expert, CPUExpertSlot &slot);
  void touch(size_t idx);
  size_t pick_victim();
};

// Host-only KV cache for dense QKV attention.
struct CPUKVCache {
  int max_t = 0;
  int n_layers = 0;
  int n_kv_heads = 0;
  int head_dim = 0;
  // Flat layout: [layer, pos, kv_head, head_dim]
  std::vector<float> k;
  std::vector<float> v;

  bool init(int max_t_, int n_layers_, int n_kv_heads_, int head_dim_);

  float *k_ptr(int layer, int pos) {
    return k.data() + ((int64_t)layer * max_t * n_kv_heads * head_dim +
                       (int64_t)pos * n_kv_heads * head_dim);
  }
  float *v_ptr(int layer, int pos) {
    return v.data() + ((int64_t)layer * max_t * n_kv_heads * head_dim +
                       (int64_t)pos * n_kv_heads * head_dim);
  }
  const float *k_ptr(int layer, int pos) const {
    return const_cast<CPUKVCache *>(this)->k_ptr(layer, pos);
  }
  const float *v_ptr(int layer, int pos) const {
    return const_cast<CPUKVCache *>(this)->v_ptr(layer, pos);
  }
};

// Host-only inference engine.
struct CPUEngine {
  CPUConfig config;
  GGUFFile model_file;
  GGUFTokenizer tokenizer;
  RouteUsage route_usage;
  CPUExpertCache expert_cache;
  CPUKVCache kv_cache;
  std::unique_ptr<CPUModelBackend> backend;

  // All dense weights loaded into host memory at startup.
  std::unordered_map<std::string, CPUQuantizedTensor> resident_layers;

  // Architecture shortcuts (mirrors StreamingEngine)
  int max_t = 0;
  int64_t t_gen = 0;

  // Build the model graph: load architecture config, dense weights, KV cache,
  // and model backend.
  bool build(int max_ctx);

  // Alternate build path for .safetensors exports. Parses the file, creates the
  // right backend, and loads raw tensor bytes into resident host memory using
  // canonical GGUF-style tensor names.
  bool build_from_safetensors(const std::string &path, int max_ctx);

  // Helpers
  CPUQuantizedTensor *find_layer(const std::string &name);
  const CPUQuantizedTensor *find_layer(const std::string &name) const;
};

// Generation helpers
struct CPUSampler {
  std::mt19937 rng;
  CPUSampler() { rng.seed(std::random_device{}()); }
  int sample(const std::vector<float> &logits, float temperature, float top_p);
};

// Forward pass entry points
bool cpu_forward_layer(CPUEngine &engine, int layer_idx, float *hidden_states,
                       int S, int pos_base);
bool cpu_lm_head(CPUEngine &engine, const float *hidden, float *logits, int S);
bool cpu_embed_token(CPUEngine &engine, int tok, float *out, int D);
