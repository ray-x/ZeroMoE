// models/cpu_dense.hpp — Dense HF-style transformer CPU backend.
//
// This backend implements standard transformer layers (RMSNorm, GQA, RoPE,
// causal attention, SwiGLU MLP, residuals) using canonical GGUF-style tensor
// names. It is the CPU target for safetensors exports that are not Qwen3.6
// hybrid MoE models, and is kept separate from the Qwen hybrid backend so both
// can coexist without interference.
#pragma once

#include "cpu_model_backend.hpp"
#include "kernels_cpu.hpp"
#include <cmath>
#include <cstring>
#include <vector>

struct DenseModelTraits {
  int n_layers = 0;
  int n_heads = 0;
  int n_kv_heads = 0;
  int head_dim = 0;
  int hidden = 0;
  int intermediate = 0;
  int vocab = 0;
  float eps = 1e-6f;
  float rope_theta = 10000.0f;
  float partial_rotary_factor = 1.0f;
};

class CPUDenseBackend : public CPUModelBackend {
 public:
  DenseModelTraits traits;

  ModelFamily family() const override { return ModelFamily::Qwen; }
  const char *name() const override {
    return "Dense GQA transformer (CPU, safetensors)";
  }

  CPULayerType layer_type(int /*layer_idx*/) const override {
    return CPULayerType::DenseQKV;
  }

  void forward_layer(CPUEngine &engine, int layer_idx, float *hidden_states,
                     int S, int pos_base) override;

  bool lm_head(CPUEngine &engine, const float *hidden, float *logits,
               int S) override;

 private:
  void rmsnorm(float *out, const float *x, const float *w, int S, int D);
  void apply_rope(float *Q, float *K, int S, int pos_base);
  void dense_attention(CPUEngine &engine, int layer_idx, const float *Q,
                       const float *K, const float *V, int S, int pos_base,
                       float *out);
};
