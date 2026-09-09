// src/cpu_engine.cpp — Host-only engine: dense-weight loading, expert LRU,
// KV cache, and tokenizer setup for moestream_cpu.
#include "cpu_engine.hpp"
#include "cpu_model_backend.hpp"

#include <cmath>
#include <cstring>
#include <numeric>

#ifdef MOESTREAM_USE_GGML
extern "C" int ggml_dequant_row(uint32_t type, const void *src, float *out,
                                int n);
#endif

// Convert a quantized or F16/BF16 dense tensor to F32 in-place so the
// CPU forward pass can call matmul() with plain float pointers.
static bool densify_tensor(CPUQuantizedTensor &qt) {
  if (qt.meta.type == GGUFType::F32)
    return true;

  int64_t ne = 1;
  for (uint64_t s : qt.meta.shape)
    ne *= static_cast<int64_t>(s);
  if (ne <= 0)
    return false;

  float *f32_buf = nullptr;
  if (posix_memalign((void **)&f32_buf, 4096, (size_t)ne * sizeof(float)) !=
      0) {
    std::cerr << "[CPU-ENGINE] Failed to allocate F32 buffer for tensor\n";
    return false;
  }

  if (qt.meta.type == GGUFType::F16 || qt.meta.type == GGUFType::BF16) {
    const uint16_t *src = reinterpret_cast<const uint16_t *>(qt.host_ptr);
    for (int64_t i = 0; i < ne; ++i) {
      if (qt.meta.type == GGUFType::BF16) {
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&f32_buf[i], &bits, sizeof(float));
      } else {
        f32_buf[i] = fp16_to_fp32(src[i]);
      }
    }
  } else {
#ifdef MOESTREAM_USE_GGML
    // Quantized row-major weight [I, O]; dequantize one output row at a time.
    int I = static_cast<int>(qt.meta.shape[0]);
    int O = static_cast<int>(ne / I);
    if (O <= 0) {
      free(f32_buf);
      return false;
    }
    size_t row_bytes = qt.host_size / O;
    for (int o = 0; o < O; ++o) {
      if (!ggml_dequant_row((uint32_t)qt.meta.type,
                            qt.host_ptr + (size_t)o * row_bytes,
                            f32_buf + (size_t)o * I, I)) {
        std::cerr << "[CPU-ENGINE] Failed to dequantize row " << o
                  << " (type=" << (int)qt.meta.type << ")\n";
        free(f32_buf);
        return false;
      }
    }
#else
    std::cerr << "[CPU-ENGINE] Quantized dense tensor requires ggml (set "
                 "MOESTREAM_USE_GGML=ON) or a F32/F16/BF16 model.\n";
    free(f32_buf);
    return false;
#endif
  }

  free(qt.host_ptr);
  qt.host_ptr = reinterpret_cast<uint8_t *>(f32_buf);
  qt.host_size = (size_t)ne * sizeof(float);
  qt.meta.type = GGUFType::F32;
  return true;
}

// ----------------------------------------------------------------------------
// CPUExpertCache
// ----------------------------------------------------------------------------

bool CPUExpertCache::init(const GGUFFile &model, double ram_gb,
                          const std::unordered_set<uint64_t> &pin_set) {
  model_ = &model;
  pin_set_ = pin_set;

  int n_layers = model.architecture_config.n_layers;
  int n_experts = model.architecture_config.n_experts;
  if (n_layers <= 0 || n_experts <= 0) {
    std::cerr << "[CPU-CACHE] No MoE layers/experts in model.\n";
    return false;
  }

  // Compute the maximum per-expert slab size across all layers.
  // Models such as Qwen3.6-35B-A3B use mixed quantization (e.g. down_exps is
  // IQ4_XS for some layers and IQ3_XXS for others), so every slot must be
  // large enough for the biggest layer and sizes/types are refreshed on load.
  size_t max_gb = 0, max_ub = 0, max_db = 0;
  const GGUFTensor *gM = nullptr;
  const GGUFTensor *uM = nullptr;
  const GGUFTensor *dM = nullptr;
  bool has_packed = false;
  for (int l = 0; l < n_layers; ++l) {
    std::string base = "blk." + std::to_string(l) + ".";
    auto gE = model.tensors.find(base + "ffn_gate_exps.weight");
    auto uE = model.tensors.find(base + "ffn_up_exps.weight");
    auto dE = model.tensors.find(base + "ffn_down_exps.weight");
    if (gE == model.tensors.end() || uE == model.tensors.end() ||
        dE == model.tensors.end())
      continue;
    size_t gb = gE->second.size_bytes / n_experts;
    size_t ub = uE->second.size_bytes / n_experts;
    size_t db = dE->second.size_bytes / n_experts;
    if (!has_packed) {
      gM = &gE->second;
      uM = &uE->second;
      dM = &dE->second;
    }
    has_packed = true;
    if (gb > max_gb)
      max_gb = gb;
    if (ub > max_ub)
      max_ub = ub;
    if (db > max_db)
      max_db = db;
  }
  if (!has_packed) {
    std::cerr << "[CPU-CACHE] Packed *_exps.weight tensors not found; "
                 "per-expert format not yet supported by moestream_cpu.\n";
    return false;
  }
  size_t slab_size = max_gb + max_ub + max_db;

  size_t capacity_slots =
      static_cast<size_t>((ram_gb * 1024.0 * 1024.0 * 1024.0) / slab_size);
  if (capacity_slots == 0)
    capacity_slots = 1;

  slots_.resize(capacity_slots);
  for (auto &s : slots_) {
    s.slab_size = slab_size;
    s.gate_size = max_gb;
    s.up_size = max_ub;
    s.down_size = max_db;
    s.gate_type = gM->type;
    s.up_type = uM->type;
    s.down_type = dM->type;
    if (posix_memalign((void **)&s.slab, 4096, slab_size) != 0) {
      std::cerr << "[CPU-CACHE] Failed to allocate expert slab.\n";
      return false;
    }
  }

  // Preload pinned experts into fixed slots at the end of the array.
  size_t pin_start =
      slots_.size() > pin_set_.size() ? slots_.size() - pin_set_.size() : 0;
  size_t idx = pin_start;
  for (uint64_t k : pin_set_) {
    if (idx >= slots_.size())
      break;
    int layer = static_cast<int>(k >> 32);
    int expert = static_cast<int>(k & 0xFFFFFFFFu);
    if (load_expert(layer, expert, slots_[idx])) {
      slots_[idx].layer = layer;
      slots_[idx].expert = expert;
      slots_[idx].pinned = true;
      index_[k] = idx;
      lru_.push_front(idx);
      ++idx;
    }
  }

  LOG_STREAM_INFO() << "[CPU-CACHE] Expert cache: " << capacity_slots
                    << " slots, "
                    << (capacity_slots * slab_size / (1024.0 * 1024.0))
                    << " MB\n";
  LOG_STREAM_INFO() << "[CPU-CACHE] Pin set: " << pin_set_.size()
                    << " experts\n";
  return true;
}

bool CPUExpertCache::load_expert(int layer, int expert, CPUExpertSlot &slot) {
  if (!model_) {
    LOG_ERROR("[CPU-CACHE] load_expert called with no model\n");
    return false;
  }
  std::string base = "blk." + std::to_string(layer) + ".";
  int n_exp = model_->architecture_config.n_experts;

  auto gE = model_->tensors.find(base + "ffn_gate_exps.weight");
  auto uE = model_->tensors.find(base + "ffn_up_exps.weight");
  auto dE = model_->tensors.find(base + "ffn_down_exps.weight");
  if (gE == model_->tensors.end() || uE == model_->tensors.end() ||
      dE == model_->tensors.end()) {
    LOG_ERROR("[CPU-CACHE] Missing packed expert tensors for layer %d\n",
              layer);
    return false;
  }

  const GGUFTensor &gT = gE->second;
  const GGUFTensor &uT = uE->second;
  const GGUFTensor &dT = dE->second;

  size_t gb = gT.size_bytes / n_exp;
  size_t ub = uT.size_bytes / n_exp;
  size_t db = dT.size_bytes / n_exp;

  size_t goff = gT.offset + expert * gb;
  size_t uoff = uT.offset + expert * ub;
  size_t doff = dT.offset + expert * db;

  int fd = model_->fd_for_tensor(gT);
  ssize_t rg = pread(fd, slot.slab, gb, goff);
  ssize_t ru = pread(fd, slot.slab + gb, ub, uoff);
  ssize_t rd = pread(fd, slot.slab + gb + ub, db, doff);
  if (rg != (ssize_t)gb || ru != (ssize_t)ub || rd != (ssize_t)db) {
    LOG_ERROR("[CPU-CACHE] Failed to read expert (%d,%d): gate=%zd/%zu "
              "up=%zd/%zu down=%zd/%zu\n",
              layer, expert, rg, gb, ru, ub, rd, db);
    return false;
  }
  slot.layer = layer;
  slot.expert = expert;
  slot.gate_size = gb;
  slot.up_size = ub;
  slot.down_size = db;
  slot.gate_type = gT.type;
  slot.up_type = uT.type;
  slot.down_type = dT.type;
  return true;
}

CPUExpertSlot *CPUExpertCache::get(int layer, int expert) {
  uint64_t k = key(layer, expert);
  auto it = index_.find(k);
  if (it != index_.end()) {
    size_t idx = it->second;
    touch(idx);
    ++hits_;
    return &slots_[idx];
  }

  ++misses_;
  size_t victim = pick_victim();
  if (victim >= slots_.size())
    return nullptr;

  CPUExpertSlot &slot = slots_[victim];
  if (slot.layer >= 0 && slot.expert >= 0) {
    index_.erase(key(slot.layer, slot.expert));
  }

  if (!load_expert(layer, expert, slot))
    return nullptr;

  slot.layer = layer;
  slot.expert = expert;
  slot.hit_count = 0;
  slot.last_used = ++clock_;
  index_[k] = victim;
  touch(victim);
  return &slot;
}

void CPUExpertCache::touch(size_t idx) {
  // Move idx to front of LRU list.
  auto it = std::find(lru_.begin(), lru_.end(), idx);
  if (it != lru_.end())
    lru_.erase(it);
  lru_.push_front(idx);
  slots_[idx].last_used = ++clock_;
  ++slots_[idx].hit_count;
}

size_t CPUExpertCache::pick_victim() {
  // Prefer unpinned, unused slots first.
  for (size_t i = 0; i < slots_.size(); ++i) {
    if (slots_[i].layer < 0 && !slots_[i].pinned)
      return i;
  }
  // Then LRU among unpinned residents.
  for (auto it = lru_.rbegin(); it != lru_.rend(); ++it) {
    if (!slots_[*it].pinned)
      return *it;
  }
  // All slots pinned — shouldn't happen if pin count <= capacity.
  return slots_.size();
}

size_t CPUExpertCache::bytes_in_cache() const {
  size_t n = 0;
  for (const auto &s : slots_) {
    if (s.layer >= 0)
      n += s.slab_size;
  }
  return n;
}

// ----------------------------------------------------------------------------
// CPUKVCache
// ----------------------------------------------------------------------------

bool CPUKVCache::init(int max_t_, int n_layers_, int n_kv_heads_,
                      int head_dim_) {
  max_t = max_t_;
  n_layers = n_layers_;
  n_kv_heads = n_kv_heads_;
  head_dim = head_dim_;
  int64_t total = (int64_t)n_layers * max_t * n_kv_heads * head_dim;
  try {
    k.assign(total, 0.0f);
    v.assign(total, 0.0f);
  } catch (...) {
    std::cerr << "[CPU-KV] Failed to allocate KV cache ("
              << total * 2 * sizeof(float) / 1e9 << " GB requested)\n";
    return false;
  }
  return true;
}

// ----------------------------------------------------------------------------
// CPUEngine
// ----------------------------------------------------------------------------

static bool is_routed_expert_tensor(const std::string &name) {
  return name.find("ffn_gate_exps.weight") != std::string::npos ||
         name.find("ffn_up_exps.weight") != std::string::npos ||
         name.find("ffn_down_exps.weight") != std::string::npos;
}

static bool is_qwen4exp_deferred_tensor(const GGUFFile &model,
                                        const std::string &name) {
  if (detect_model_family(model) != ModelFamily::Qwen4Exp)
    return false;
  return name == "per_layer_token_embd.weight" ||
         name.find(".indexer_") != std::string::npos ||
         name.find(".indexer.") != std::string::npos;
}

bool CPUEngine::build(int max_ctx) {
  max_t = max_ctx;

  // 1. Fill architecture_config from KV metadata (mirror of graph_initializer).
  ArchitectureConfig &c = model_file.architecture_config;
  c.hidden = model_file.get_config_int("embedding_length", c.hidden);
  c.n_layers = model_file.get_config_int("block_count", c.n_layers);
  c.n_heads = model_file.get_config_int("attention.head_count", c.n_heads);
  c.n_experts = model_file.get_config_int("expert_count", c.n_experts);
  c.topk = model_file.get_config_int("expert_used_count", c.topk);
  c.moe_inter =
      model_file.get_config_int("expert_feed_forward_length", c.moe_inter);
  c.vocab = model_file.get_config_int("vocab_size", c.vocab);
  if (c.vocab <= 0) {
    // Some models (e.g. Qwen3.6-35B-A3B-UD) have no numeric vocab_size KV at
    // all -- fall back to token_embd.weight/output.weight's own shape. GGUF
    // stores ne[] un-reversed, so shape[0] is the innermost/fastest-varying
    // dimension: for these 2D [D, V] tensors that's the embedding width D
    // (matches c.hidden), NOT the vocab size -- shape[1] is the real row
    // count / vocab size. Reading shape[0] here used to silently truncate
    // c.vocab to D (e.g. 2048 instead of the real 248320 for Qwen3.6-35B-A3B),
    // which caps the LM head/sampler to the first D token ids of a much
    // larger vocabulary -- a mechanically clean explanation for "prefill
    // logits look plausible but decode text is incoherent garbage".
    auto it = model_file.tensors.find("token_embd.weight");
    if (it == model_file.tensors.end())
      it = model_file.tensors.find("output.weight");
    if (it != model_file.tensors.end() && it->second.shape.size() >= 2) {
      c.vocab = static_cast<int>(it->second.shape[1]);
    } else if (it != model_file.tensors.end() && !it->second.shape.empty()) {
      c.vocab = static_cast<int>(it->second.shape[0]);
    }
  }
  c.n_head_kv = model_file.get_config_int("attention.head_count_kv", c.n_heads);
  c.head_dim = model_file.get_config_int("attention.key_length", c.head_dim);
  c.dense_inter =
      model_file.get_config_int("feed_forward_length", c.dense_inter);
  if (detect_model_family(model_file) == ModelFamily::Qwen4Exp) {
    c.qwen4exp_recurrent_layers =
        model_file.get_config_int_array("attention.recurrent_layers");
    c.qwen4exp_full_attention_interval =
        model_file.get_config_int("full_attention_interval", 4);
    c.ssm_conv_kernel = model_file.get_config_int("ssm.conv_kernel", 0);
    c.ssm_inner_size = model_file.get_config_int("ssm.inner_size", 0);
    c.ssm_state_size = model_file.get_config_int("ssm.state_size", 0);
    c.ssm_time_step_rank =
        model_file.get_config_int("ssm.time_step_rank", 0);
    c.ssm_group_count = model_file.get_config_int("ssm.group_count", 0);
    c.hc_mult = model_file.get_config_int("hyper_connection.count", 1);
    c.hc_low_rank = model_file.get_config_int("hyper_connection.low_rank", 0);
    c.qwen4exp_compress_ratios =
        model_file.get_config_int_array("attention.compress_ratios");
    c.qwen4exp_indexer_head_count =
        model_file.get_config_int("attention.indexer.head_count", 0);
    c.qwen4exp_indexer_key_length =
        model_file.get_config_int("attention.indexer.key_length", 0);
    c.qwen4exp_indexer_top_k =
        model_file.get_config_int("attention.indexer.top_k", 0);
    c.rope_sections =
        model_file.get_config_int_array("rope.dimension_sections");
    c.rope_n_rot = model_file.get_config_int("rope.dimension_count", 0);
    c.ple_layers = model_file.get_config_int_array("ple.layers");
    c.ple_ngram_size = model_file.get_config_int("ple.ngram_size", 0);
    c.ple_heads_per_ngram =
        model_file.get_config_int("ple.heads_per_ngram", 0);
    c.ple_conv_kernel = model_file.get_config_int("ple.conv_kernel", 0);
    c.ple_eos_token_id = model_file.get_config_int("ple.eos_token_id", -1);
    c.ple_n_embd_per_layer =
        model_file.get_config_int("embedding_length_per_layer_input", 0);
    c.ple_layer_multipliers =
        model_file.get_config_int_array("ple.layer_multipliers");
    c.ple_head_offsets = model_file.get_config_int_array("ple.head_offsets");
    c.ple_head_vocab_sizes =
        model_file.get_config_int_array("ple.head_vocab_sizes");
    c.ple_n_heads = (c.ple_ngram_size - 1) * c.ple_heads_per_ngram;
  }
  c.expert_weights_scale = model_file.get_config_float("expert_weights_scale",
                                                       c.expert_weights_scale);
  c.eps =
      model_file.get_config_float("attention.layer_norm_rms_epsilon", c.eps);
  if (c.eps <= 0.0f)
    c.eps = 1e-5f;

  if (c.hidden <= 0 || c.n_layers <= 0 || c.vocab <= 0) {
    std::cerr << "[CPU-ENGINE] Invalid architecture metadata (hidden="
              << c.hidden << ", layers=" << c.n_layers << ", vocab=" << c.vocab
              << ")\n";
    return false;
  }
  if (c.head_dim <= 0)
    c.head_dim = c.hidden / c.n_heads;
  if (c.n_head_kv <= 0)
    c.n_head_kv = c.n_heads;

  LOG_STREAM_INFO() << "[CPU-ENGINE] Architecture: hidden=" << c.hidden
                    << " layers=" << c.n_layers << " heads=" << c.n_heads
                    << " kv_heads=" << c.n_head_kv << " head_dim=" << c.head_dim
                    << " experts=" << c.n_experts << " topk=" << c.topk
                    << " moe_inter=" << c.moe_inter << "\n";

  // 2. Load tokenizer.
  if (!tokenizer.load(model_file)) {
    std::cerr << "[CPU-ENGINE] Tokenizer load failed.\n";
    return false;
  }

  // 3. Load dense weights (everything except routed expert slabs).
  LOG_STREAM_INFO() << "[CPU-ENGINE] Loading dense weights into host RAM...\n";
  size_t total_loaded = 0;
  size_t dense_count = 0;
  for (auto &[name, t_meta] : model_file.tensors) {
    if (is_routed_expert_tensor(name) ||
        is_qwen4exp_deferred_tensor(model_file, name))
      continue;

    CPUQuantizedTensor qt;
    qt.meta = t_meta;
    if (posix_memalign((void **)&qt.host_ptr, 4096, t_meta.size_bytes) != 0) {
      LOG_ERROR(
          "[CPU-ENGINE] Failed to allocate host buffer for %s (%zu bytes)\n",
          name.c_str(), t_meta.size_bytes);
      return false;
    }
    qt.host_size = t_meta.size_bytes;

    int fd = model_file.fd_for_tensor(t_meta);
    if (fd < 0) {
      LOG_ERROR("[CPU-ENGINE] No file descriptor for tensor %s\n",
                name.c_str());
      free(qt.host_ptr);
      return false;
    }
    ssize_t r = pread(fd, qt.host_ptr, t_meta.size_bytes, t_meta.offset);
    if (r != (ssize_t)t_meta.size_bytes) {
      LOG_ERROR("[CPU-ENGINE] Failed to read tensor %s: expected %zu bytes, "
                "got %zd (errno=%d)\n",
                name.c_str(), t_meta.size_bytes, r, errno);
      free(qt.host_ptr);
      return false;
    }

    const bool keep_qwen4exp_quantized =
        detect_model_family(model_file) == ModelFamily::Qwen4Exp;
    if (!keep_qwen4exp_quantized && !densify_tensor(qt)) {
      LOG_ERROR("[CPU-ENGINE] Failed to densify tensor %s (type=%u shape=[",
                name.c_str(), static_cast<uint32_t>(t_meta.type));
      for (size_t i = 0; i < t_meta.shape.size(); ++i) {
        if (i)
          std::fprintf(stderr, ",");
        std::fprintf(stderr, "%llu",
                     static_cast<unsigned long long>(t_meta.shape[i]));
      }
      std::fprintf(stderr, "])\n");
      free(qt.host_ptr);
      return false;
    }

    // Paranoid sanity check after densification: first/last few values should
    // not all be NaN/Inf.
    if (qt.meta.type == GGUFType::F32 && qt.host_ptr) {
      const float *fp = reinterpret_cast<const float *>(qt.host_ptr);
      int64_t nelem = static_cast<int64_t>(qt.host_size / sizeof(float));
      int bad = 0;
      for (int64_t i = 0; i < std::min<int64_t>(nelem, 16); ++i)
        if (std::isnan(fp[i]) || std::isinf(fp[i]))
          ++bad;
      if (bad == std::min<int64_t>(nelem, 16)) {
        LOG_ERROR("[CPU-ENGINE] Tensor %s is all NaN/Inf after densification\n",
                  name.c_str());
        free(qt.host_ptr);
        return false;
      }
    }

    total_loaded += qt.host_size;
    ++dense_count;
    resident_layers.emplace(name, std::move(qt));
  }
  LOG_STREAM_INFO() << "[CPU-ENGINE] Loaded " << dense_count
                    << " dense tensors, " << (total_loaded / 1e9) << " GB\n";

  // 4. Load existing usage histogram and initialize expert LRU cache.
  if (!config.pin_file.empty()) {
    if (route_usage.load(config.pin_file)) {
      LOG_STREAM_INFO() << "[CPU-ENGINE] Loaded usage histogram: "
                        << route_usage.size() << " entries\n";
    }
  }
  auto pin_set = route_usage.pin_set(
      config.pin_topk_per_layer, static_cast<uint64_t>(config.pin_threshold));
  if (c.n_experts > 0 &&
      !expert_cache.init(model_file, config.host_ram_gb, pin_set)) {
    return false;
  }

  // 5. Initialize KV cache.
  if (!kv_cache.init(max_t, c.n_layers, c.n_head_kv, c.head_dim)) {
    return false;
  }

  // 6. Create model backend.
  backend = CPUModelBackend::create(model_file);
  if (!backend) {
    std::cerr << "[CPU-ENGINE] No CPU backend available for this model.\n";
    return false;
  }

  return true;
}

static GGUFType gguf_type_from_safetensors_dtype(const std::string &dtype) {
  if (dtype == "F32")
    return GGUFType::F32;
  if (dtype == "F16")
    return GGUFType::F16;
  if (dtype == "BF16")
    return GGUFType::BF16;
  if (dtype == "I32")
    return GGUFType::I32;
  if (dtype == "I64")
    return GGUFType::I64;
  return GGUFType::F32;
}

bool CPUEngine::build_from_safetensors(const std::string &path, int max_ctx) {
  max_t = max_ctx;

  SafetensorsModelInfo model;
  if (!SafetensorsAdapter::parse_file(path, model)) {
    std::cerr << "[CPU-ENGINE] Failed to parse safetensors: " << path << "\n";
    return false;
  }

  // Look for an optional HF-style config.json next to the model file.
  std::string config_json_path;
  {
    size_t slash = path.find_last_of("/\\");
    std::string dir = (slash == std::string::npos) ? "" : path.substr(0, slash + 1);
    config_json_path = dir + "config.json";
  }
  DenseCpuConfigJson cfg = SafetensorsAdapter::parse_dense_config_json(config_json_path);

  std::string format = SafetensorsAdapter::detect_format(model);
  auto backend_or = CPUModelBackend::create_from_safetensors_path(
      path, &model_file.architecture_config, config_json_path);
  if (!backend_or) {
    std::cerr << "[CPU-ENGINE] No backend for safetensors file: " << path
              << "\n";
    return false;
  }
  backend = std::move(backend_or);

  ArchitectureConfig &c = model_file.architecture_config;
  if (c.eps <= 0.0f)
    c.eps = 1e-6f;
  if (c.head_dim <= 0 && c.n_heads > 0)
    c.head_dim = c.hidden / c.n_heads;
  if (c.n_head_kv <= 0)
    c.n_head_kv = c.n_heads;

  LOG_STREAM_INFO() << "[CPU-ENGINE] Safetensors architecture: hidden="
                    << c.hidden << " layers=" << c.n_layers
                    << " heads=" << c.n_heads << "/" << c.n_head_kv
                    << " head_dim=" << c.head_dim << " vocab=" << c.vocab
                    << "\n";

  // Load tensor bytes from the safetensors file directly into resident_layers,
  // using the canonical GGUF-style names produced by the adapter.
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "[CPU-ENGINE] Cannot open safetensors file: " << path << "\n";
    return false;
  }

  std::string header_len_bytes(8, '\0');
  file.read(header_len_bytes.data(), 8);
  if (!file)
    return false;
  uint64_t header_len = 0;
  std::memcpy(&header_len, header_len_bytes.data(), sizeof(header_len));
  const uint64_t data_base = 8 + header_len;

  DenseCpuContract contract =
      SafetensorsAdapter::map_dense_cpu_contract(model, format);
  size_t total_loaded = 0;
  for (const auto &kv : contract.resolved_tensors) {
    const std::string &canonical_name = kv.first;
    const std::string &raw_name = kv.second;
    auto it = model.tensors.find(raw_name);
    if (it == model.tensors.end())
      continue;
    const SafetensorsTensorInfo &info = it->second;

    CPUQuantizedTensor qt;
    qt.meta.name = canonical_name;
    qt.meta.type = gguf_type_from_safetensors_dtype(info.dtype);
    qt.meta.shape.assign(info.shape.begin(), info.shape.end());
    qt.meta.size_bytes = info.length;
    qt.meta.offset = data_base + info.data_offset;

    if (posix_memalign((void **)&qt.host_ptr, 4096, info.length) != 0) {
      std::cerr << "[CPU-ENGINE] Failed to allocate tensor " << canonical_name
                << "\n";
      return false;
    }
    qt.host_size = info.length;

    file.seekg(static_cast<std::streamoff>(qt.meta.offset));
    file.read(reinterpret_cast<char *>(qt.host_ptr),
              static_cast<std::streamsize>(info.length));
    if (!file) {
      std::cerr << "[CPU-ENGINE] Failed to read tensor " << canonical_name
                << "\n";
      free(qt.host_ptr);
      return false;
    }

    if (!densify_tensor(qt)) {
      std::cerr << "[CPU-ENGINE] Failed to densify tensor " << canonical_name
                << "\n";
      free(qt.host_ptr);
      return false;
    }

    total_loaded += qt.host_size;
    resident_layers.emplace(canonical_name, std::move(qt));
  }

  LOG_STREAM_INFO() << "[CPU-ENGINE] Loaded " << resident_layers.size()
                    << " safetensors tensors, " << (total_loaded / 1e9)
                    << " GB\n";

  if (!kv_cache.init(max_t, c.n_layers, c.n_head_kv, c.head_dim)) {
    std::cerr << "[CPU-ENGINE] KV cache init failed.\n";
    return false;
  }

  return true;
}

CPUQuantizedTensor *CPUEngine::find_layer(const std::string &name) {
  auto it = resident_layers.find(name);
  return (it != resident_layers.end() && it->second.host_ptr) ? &it->second
                                                              : nullptr;
}

const CPUQuantizedTensor *CPUEngine::find_layer(const std::string &name) const {
  auto it = resident_layers.find(name);
  return (it != resident_layers.end() && it->second.host_ptr) ? &it->second
                                                              : nullptr;
}

// ----------------------------------------------------------------------------
// Sampler
// ----------------------------------------------------------------------------

int CPUSampler::sample(const std::vector<float> &logits, float temperature,
                       float top_p) {
  int V = (int)logits.size();
  if (temperature <= 0.0f) {
    return (int)(std::max_element(logits.begin(), logits.end()) -
                 logits.begin());
  }

  float max_logit = *std::max_element(logits.begin(), logits.end());
  std::vector<float> probs(V);
  double sum = 0.0;
  float inv_temp = 1.0f / std::max(temperature, 1e-4f);
  for (int i = 0; i < V; ++i) {
    probs[i] = std::exp((logits[i] - max_logit) * inv_temp);
    sum += probs[i];
  }
  for (int i = 0; i < V; ++i)
    probs[i] = (float)(probs[i] / sum);

  if (top_p > 0.0f && top_p < 1.0f) {
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return probs[a] > probs[b]; });
    float cum = 0.0f;
    int keep = V;
    for (int i = 0; i < V; ++i) {
      cum += probs[idx[i]];
      if (cum >= top_p) {
        keep = i + 1;
        break;
      }
    }
    std::vector<float> mask(V, 0.0f);
    for (int i = 0; i < keep; ++i)
      mask[idx[i]] = probs[idx[i]];
    float msum = 0.0f;
    for (float x : mask)
      msum += x;
    if (msum > 0.0f) {
      for (int i = 0; i < V; ++i)
        probs[i] = mask[i] / msum;
    }
  }

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r = dist(rng);
  float cum = 0.0f;
  for (int i = 0; i < V; ++i) {
    cum += probs[i];
    if (r <= cum)
      return i;
  }
  return V - 1;
}
