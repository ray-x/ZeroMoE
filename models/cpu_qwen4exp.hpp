// models/cpu_qwen4exp.hpp — CPU backend for Qwen3.8-Flash-Next.
#pragma once

#include "cpu_model_backend.hpp"
#include "cpu_qwen_moe.hpp"

#include <cstdint>
#include <string>
#include <vector>

class CPUQwen4ExpBackend final : public CPUModelBackend {
public:
  QwenModelTraits traits;
  std::vector<CPULayerType> layer_types;
  ~CPUQwen4ExpBackend() override;

  ModelFamily family() const override { return ModelFamily::Qwen4Exp; }
  const char *name() const override {
    return "Qwen3.8-Flash-Next (CPU, GDN+full-attention)";
  }
  CPULayerType layer_type(int layer_idx) const override;

  void forward_layer(CPUEngine &engine, int layer_idx, float *hidden_states,
                     int S, int pos_base) override;
  bool lm_head(CPUEngine &engine, const float *hidden, float *logits,
               int S) override;

  bool compatible(const GGUFFile &model) const;

private:
  static const float *tensor(const CPUEngine &engine, const std::string &name);
  static bool dense_matmul(const CPUEngine &engine, const std::string &name,
                           const float *x, float *y, int S, int I, int O);
  static bool dequant_row(const CPUQuantizedTensor &tensor, int row,
                          float *out, int n);
  static bool dequant_tensor(const CPUQuantizedTensor &tensor, float *out,
                             size_t n);
  static void rmsnorm(float *out, const float *x, const float *w, int n,
                      float eps);
  static void l2norm(float *x, int n, float eps);
  static void silu(float *x, int n);
  static void sigmoid(float *x, int n);
  static void rope_interleaved(float *x, int n_rot, int pos, float theta);
  static void softmax(float *x, int n);

  void hc_mix(CPUEngine &engine, int layer_idx, int sub_block,
              float *hidden_states, int S);
  void hc_combine(CPUEngine &engine, float *hidden_states,
                  const float *sub_block_out, int S);
  void forward_gdn(CPUEngine &engine, int layer_idx, float *hidden_states,
                   int S);
  void forward_attention(CPUEngine &engine, int layer_idx,
                         float *hidden_states, int S, int pos_base);
  void forward_moe(CPUEngine &engine, int layer_idx, float *hidden_states,
                   int S);
  bool ple_layer_input(CPUEngine &engine, int layer_idx, int S, int pos_base);

  const uint8_t *ple_table_ = nullptr;
  size_t ple_row_bytes_ = 0;
  size_t ple_map_length_ = 0;
  GGUFType ple_type_ = GGUFType::F32;
  int ple_row_dim_ = 0;
  uint64_t ple_rows_ = 0;
  void *ple_mapping_ = nullptr;
};
