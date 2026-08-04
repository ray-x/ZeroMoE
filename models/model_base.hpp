// models/model_base.hpp — Abstract base interface for model-specific GPU
// forward passes. Each model family (GLM-DSA, HY-V3, DeepSeek-V3, etc.)
// implements this interface.
#pragma once
#include "engine.hpp"
#include "model_family.hpp"
#include "ram_manager.hpp"
#include <memory>
#include <string>

// Abstract base for model-specific computation
class ModelBackend {
public:
  virtual ~ModelBackend() = default;

  // Model identification
  virtual ModelFamily family() const = 0;
  virtual AttentionType attention_type() const = 0;
  virtual const char *name() const = 0;

  // === Full forward pass for one layer ===
  // Computes: input → attn_norm → attention → residual → ffn_norm → FFN →
  // residual → output
  virtual void forward_layer(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                             int layer_idx,
                             float *hidden_states, // [S, D] in/out
                             int S,                // sequence length
                             int pos_base // position offset for KV cache
                             ) = 0;

  // === Attention-only forward (for layer-split pipelines) ===
  virtual void forward_attention(StreamingEngine &engine, int layer_idx,
                                 const float *input, // [S, D] normed input
                                 float *output,      // [S, D] attention output
                                 int S, int pos_base) = 0;

  // === Populate KV cache for current token(s) ===
  virtual void
  update_kv_cache(StreamingEngine &engine, int layer_idx,
                  const float *input, // [S, D] raw hidden states (pre-norm)
                  int S, int pos_base) = 0;

  // Factory: detect model family from GGUF metadata and create backend
  static std::unique_ptr<ModelBackend> create(const GGUFFile &model_file);
};
