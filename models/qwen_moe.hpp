// models/qwen_moe.hpp — GPU ModelBackend for Qwen MoE family.
//
// This mirrors the CPU backend in models/cpu_qwen_moe.hpp and shares the
// QwenModelTraits layer-type detection.  It is the CUDA path for Qwen3.6-
// 35B-A3B and future Qwen MoE models.
#pragma once
#include "model_base.hpp"

class QwenMoEBackend : public ModelBackend {
public:
  QwenMoEBackend() = default;
  ~QwenMoEBackend() override = default;

  ModelFamily family() const override { return ModelFamily::Qwen; }
  AttentionType attention_type() const override {
    return AttentionType::DenseQKV;
  }
  const char *name() const override { return "Qwen-MoE"; }

  void forward_layer(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                     int layer_idx, float *hidden_states, int S,
                     int pos_base) override;

  void forward_attention(StreamingEngine &engine, int layer_idx,
                         const float *input, float *output, int S,
                         int pos_base) override;

  void update_kv_cache(StreamingEngine &engine, int layer_idx,
                       const float *input, int S, int pos_base) override;
};
