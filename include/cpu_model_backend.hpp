// cpu_model_backend.hpp — Abstract model backend for the host-only engine.
//
// This mirrors models/model_base.hpp but targets CPUEngine.  Each model family
// that supports CPU inference implements CPUModelBackend.  New families can be
// added without touching the CUDA-side ModelBackend hierarchy.
#pragma once

#include "cpu_engine.hpp"
#include "model_family.hpp"
#include <memory>
#include <string>

// Per-layer type for hybrid models (e.g. Qwen3.6 has both SSM and DenseQKV).
enum class CPULayerType {
  DenseQKV,        // standard attention + FFN/MoE
  MLA,             // multi-latent attention + FFN/MoE
  LinearAttention, // gated DeltaNet / linear attention + FFN/MoE
  DenseFFN,        // dense feed-forward only (leading dense layer)
};

class CPUModelBackend {
public:
  virtual ~CPUModelBackend() = default;

  virtual ModelFamily family() const = 0;
  virtual const char *name() const = 0;

  // Return the layer type for layer_idx.  Used by the generic generation loop
  // to dispatch to the right forward implementation.
  virtual CPULayerType layer_type(int layer_idx) const = 0;

  // Full forward pass for one layer.
  // hidden_states is [S, D] in-place.
  virtual void forward_layer(CPUEngine &engine, int layer_idx,
                             float *hidden_states, int S, int pos_base) = 0;

  // Optional: final layer-norm + LM head.  Returns false if engine should use
  // a generic fallback.
  virtual bool lm_head(CPUEngine &engine, const float *hidden, float *logits,
                       int S) {
    (void)engine;
    (void)hidden;
    (void)logits;
    (void)S;
    return false;
  }

  // Factory: create the right backend for the loaded GGUF file.
  static std::unique_ptr<CPUModelBackend> create(const GGUFFile &model_file);
};
