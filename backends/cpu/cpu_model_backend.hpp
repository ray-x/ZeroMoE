// cpu_model_backend.hpp — Abstract model backend for the host-only engine.
//
// This mirrors models/model_base.hpp but targets CPUEngine.  Each model family
// that supports CPU inference implements CPUModelBackend.  New families can be
// added without touching the CUDA-side ModelBackend hierarchy.
#pragma once

#include "cpu_engine.hpp"
#include "gguf_loader.hpp"
#include "model_family.hpp"
#include "safetensors_adapter.hpp"
#include <memory>
#include <string>

// Per-layer type for hybrid models (e.g. Qwen3.6 has both SSM and DenseQKV).
enum class CPULayerType {
  DenseQKV,        // standard attention + FFN/MoE
  MLA,             // multi-latent attention + FFN/MoE
  LinearAttention, // gated DeltaNet / linear attention + FFN/MoE
  DenseFFN,        // dense feed-forward only (leading dense layer)
};

struct CPUBackendState {
  virtual ~CPUBackendState() = default;
};

class CPUModelBackend {
public:
  virtual ~CPUModelBackend() = default;

  virtual ModelFamily family() const = 0;
  virtual const char *name() const = 0;
  virtual bool supports_sequence_state() const { return false; }
  virtual void reset_sequence() {}
  virtual size_t sequence_state_bytes() const { return 0; }
  virtual std::unique_ptr<CPUBackendState> save_sequence_state() const { return nullptr; }
  virtual bool restore_sequence_state(const CPUBackendState &) { return false; }

  // Return the layer type for layer_idx.  Used by the generic generation loop
  // to dispatch to the right forward implementation.
  virtual CPULayerType layer_type(int layer_idx) const = 0;

  // Full forward pass for one layer.
  // hidden_states is [S, D] in-place.
  virtual void forward_layer(CPUEngine &engine, int layer_idx,
                             float *hidden_states, int S, int pos_base) = 0;
  virtual bool last_forward_succeeded() const { return true; }

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

  // Factory: normalize a safetensors export into the logical CPU contract
  // before the backend is constructed.  This is intentionally kept as a
  // separate integration seam so the same adapter can later be reused by the
  // future GPU path without scattering format-specific knowledge through the
  // runtime.  If out_cfg is non-null, it receives the architecture values
  // inferred from the tensor shapes.
  static std::unique_ptr<CPUModelBackend> create_from_safetensors_path(
      const std::string &path, ArchitectureConfig *out_cfg = nullptr,
      const std::string &config_json_path = "");
};
