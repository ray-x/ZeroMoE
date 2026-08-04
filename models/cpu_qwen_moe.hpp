// models/cpu_qwen_moe.hpp — Qwen3.6-35B-A3B CPU backend.
//
// Qwen3.6-35B-A3B is a hybrid:
//   * 60 layers use a Gated DeltaNet linear-attention block
//     (attn_qkv, attn_gate, ssm_alpha/beta/a, ssm_conv1d, ssm_norm)
//   * 10 layers use standard dense Q/K/V attention with a gated output
//   * All transformer layers use a routed MoE FFN plus a shared expert
//
// This file implements the CPU-only forward pass.  A future GPU backend can
// reuse the same layer-type dispatch and tensor naming by deriving from the
// shared QwenModelTraits struct below.
#pragma once

#include "cpu_model_backend.hpp"
#include "kernels_cpu.hpp"
#include "rope.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// Standard RMSNorm. The (1 + w) weight offset is a Gemma-specific quirk;
// Qwen norm weights are used directly, not offset by 1.
inline void qwen_rmsnorm_row(float *out, const float *x, const float *w, int D,
                             float eps) {
  double ss = 0.0;
  for (int i = 0; i < D; ++i)
    ss += static_cast<double>(x[i]) * x[i];
  float r = 1.0f / std::sqrt(static_cast<float>(ss / D) + eps);
  for (int i = 0; i < D; ++i)
    out[i] = x[i] * r * w[i];
}

// Shared model traits (usable by both CPU and GPU backends).
struct QwenModelTraits {
  // Detected at load time.
  int n_layers = 0;
  int n_heads = 0;    // full attention heads (e.g. 16)
  int n_kv_heads = 0; // full attention KV heads (e.g. 2)
  int head_dim = 0;   // full attention head dim (e.g. 256)
  int n_experts = 0;
  int topk = 0;
  int moe_inter = 0;
  // Shared expert's own FFN intermediate width. Qwen's shared expert uses a
  // distinct (often much larger) dimension than the routed experts' moe_inter
  // -- must NOT be assumed equal to moe_inter.
  int moe_inter_shared = 0;

  // Gated DeltaNet linear attention dimensions.
  int linear_num_key_heads = 16;
  int linear_key_head_dim = 128;
  int linear_num_value_heads = 32;
  int linear_value_head_dim = 128;
  int linear_conv_kernel = 4;

  // RoPE used by full attention layers.
  float partial_rotary_factor = 0.25f;

  // Layer-type map.  Filled by inspect_tensors() and used by forward dispatch.
  std::vector<CPULayerType> layer_types;

  // Populate from the actual tensor table.
  void inspect_tensors(const GGUFFile &model);

  // Convenience predicate.
  bool is_linear_attention_layer(int layer_idx) const {
    if (layer_idx < 0 || layer_idx >= (int)layer_types.size())
      return false;
    return layer_types[layer_idx] == CPULayerType::LinearAttention;
  }
};

inline void QwenModelTraits::inspect_tensors(const GGUFFile &model) {
  n_layers = model.get_config_int("block_count", n_layers);
  n_heads = model.get_config_int("attention.head_count", n_heads);
  n_kv_heads = model.get_config_int("attention.head_count_kv", n_heads);
  head_dim = model.get_config_int("attention.key_length", head_dim);
  n_experts = model.get_config_int("expert_count", n_experts);
  topk = model.get_config_int("expert_used_count", topk);
  moe_inter = model.get_config_int("expert_feed_forward_length", moe_inter);
  moe_inter_shared = model.get_config_int("expert_shared_feed_forward_length",
                                          moe_inter_shared);
  if (moe_inter_shared <= 0) {
    // Some GGUF exports omit this key; the shared expert's own ffn_up_shexp
    // tensor shape is ground truth for its (often much larger) width.
    for (int l = 0; l < n_layers && moe_inter_shared <= 0; ++l) {
      auto it = model.tensors.find("blk." + std::to_string(l) +
                                   ".ffn_up_shexp.weight");
      if (it != model.tensors.end() && it->second.shape.size() >= 2)
        moe_inter_shared = (int)it->second.shape[1];
    }
  }

  // Try config keys first, then infer from tensor shapes.
  linear_num_key_heads =
      model.get_config_int("linear_num_key_heads", linear_num_key_heads);
  linear_key_head_dim =
      model.get_config_int("linear_key_head_dim", linear_key_head_dim);
  linear_num_value_heads =
      model.get_config_int("linear_num_value_heads", linear_num_value_heads);
  linear_value_head_dim =
      model.get_config_int("linear_value_head_dim", linear_value_head_dim);
  linear_conv_kernel =
      model.get_config_int("linear_conv_kernel_dim", linear_conv_kernel);
  partial_rotary_factor =
      model.get_config_float("partial_rotary_factor", partial_rotary_factor);

  layer_types.resize(n_layers, CPULayerType::DenseQKV);
  for (int l = 0; l < n_layers; ++l) {
    std::string prefix = "blk." + std::to_string(l) + ".";
    bool has_qkv = (model.tensors.count(prefix + "attn_qkv.weight") > 0);
    bool has_gate = (model.tensors.count(prefix + "attn_gate.weight") > 0);
    bool has_ssm_a = (model.tensors.count(prefix + "ssm_a") > 0);
    bool has_ssm_conv = (model.tensors.count(prefix + "ssm_conv1d.weight") > 0);
    if (has_qkv && has_gate && has_ssm_a && has_ssm_conv) {
      layer_types[l] = CPULayerType::LinearAttention;
      // Infer linear attention dims from tensor shapes if not already known.
      if (linear_num_value_heads <= 0 || linear_value_head_dim <= 0) {
        auto it = model.tensors.find(prefix + "attn_gate.weight");
        if (it != model.tensors.end() && it->second.shape.size() >= 2) {
          int value_dim = (int)it->second.shape[1];
          linear_value_head_dim =
              linear_value_head_dim > 0 ? linear_value_head_dim : 128;
          linear_num_value_heads = value_dim / linear_value_head_dim;
        }
      }
      if (linear_num_key_heads <= 0 || linear_key_head_dim <= 0) {
        auto qkv = model.tensors.find(prefix + "attn_qkv.weight");
        auto gate = model.tensors.find(prefix + "attn_gate.weight");
        if (qkv != model.tensors.end() && gate != model.tensors.end() &&
            qkv->second.shape.size() >= 2 && gate->second.shape.size() >= 2) {
          int qkv_out = (int)qkv->second.shape[1];
          int value_dim = (int)gate->second.shape[1];
          int key_total = qkv_out - value_dim;
          linear_num_key_heads =
              linear_num_key_heads > 0 ? linear_num_key_heads : 16;
          linear_key_head_dim = (key_total / 2) / linear_num_key_heads;
        }
      }
    } else {
      layer_types[l] = CPULayerType::DenseQKV;
    }
  }
}

// CPU-only Qwen MoE backend.
class CPUQwenMoEBackend : public CPUModelBackend {
public:
  QwenModelTraits traits;

  ModelFamily family() const override { return ModelFamily::Qwen; }
  const char *name() const override {
    return "Qwen3.6 MoE (CPU, hybrid LinearAttention+DenseQKV)";
  }

  CPULayerType layer_type(int layer_idx) const override {
    return traits.layer_types[layer_idx];
  }

  void forward_layer(CPUEngine &engine, int layer_idx, float *hidden_states,
                     int S, int pos_base) override;

  bool lm_head(CPUEngine &engine, const float *hidden, float *logits,
               int S) override;

private:
  // Standard dense QKV attention block (for the 10 transformer layers).
  void forward_dense_attention(CPUEngine &engine, int layer_idx,
                               float *hidden_states, int S, int pos_base);

  // Gated DeltaNet linear attention block (for the 60 linear_attention layers).
  void forward_linear_attention(CPUEngine &engine, int layer_idx,
                                float *hidden_states, int S, int pos_base);

  // Routed MoE FFN shared by both layer types.
  void forward_moe_ffn(CPUEngine &engine, int layer_idx, float *hidden_states,
                       int S);

  // Helpers
  void project_qkv_dense(const CPUEngine &engine, int layer_idx,
                         const float *normed, int S, float *Q, float *gate,
                         float *K, float *V);
  void dense_attention(CPUEngine &engine, int layer_idx, const float *Q,
                       const float *K, const float *V, int S, int pos_base,
                       const float *gate, float *out);
  void apply_rope_neox(float *Q, float *K, int S, int pos_base, int H,
                       int n_kv_heads, int head_dim, int rotary_dim,
                       float theta_base);
  void head_rmsnorm(float *out, const float *x, const float *w, int dim,
                    float eps);
  void causal_softmax(float *scores, int S, int T);

  // Gated DeltaNet helpers
  void gated_delta_rule(const float *Q, const float *K, const float *V,
                        const float *beta, const float *g_log_decay,
                        float *state, float *out, int S, int n_v_heads,
                        int k_dim, int v_dim);
  void causal_conv1d(const float *x, const float *w, float *out, int S,
                     int channels, int kernel);

  // Per-layer recurrent state for linear attention, persisted across tokens.
  std::vector<std::vector<float>> linear_attn_states;

  // Quantized matmul helper for MoE experts.
  void dequant_matmul(GGUFType type, const uint8_t *w, size_t w_bytes,
                      const float *x, float *y, int I, int O, int S);
};

// ============================================================
// Shared MoE FFN implementation
// ============================================================

inline void CPUQwenMoEBackend::forward_moe_ffn(CPUEngine &engine, int layer_idx,
                                               float *hidden_states, int S) {
  const ArchitectureConfig &c = engine.model_file.architecture_config;
  int D = c.hidden;
  int topk = c.topk;
  int moe_inter = c.moe_inter;

  std::string prefix = "blk." + std::to_string(layer_idx) + ".";

  // Apply post-attention RMSNorm before the FFN.
  // Save the pre-norm hidden state as the FFN residual.
  std::vector<float> ffn_residual((size_t)S * D);
  CPUQuantizedTensor *post_norm =
      engine.find_layer(prefix + "post_attention_norm.weight");
  if (post_norm && post_norm->host_ptr) {
    std::memcpy(ffn_residual.data(), hidden_states,
                (size_t)S * D * sizeof(float));
    for (int s = 0; s < S; ++s)
      qwen_rmsnorm_row(hidden_states + s * D, hidden_states + s * D,
                       reinterpret_cast<const float *>(post_norm->host_ptr), D,
                       c.eps);
  } else {
    std::memcpy(ffn_residual.data(), hidden_states,
                (size_t)S * D * sizeof(float));
  }

  // Router: [S, D] @ [D, n_experts] -> gate logits [S, n_experts]
  std::string router_key = prefix + "ffn_gate_inp.weight";
  const CPUQuantizedTensor *router = engine.find_layer(router_key);
  if (!router || !router->host_ptr)
    return;

  std::vector<float> gate_logits((size_t)S * c.n_experts, 0.0f);
  matmul(gate_logits.data(), hidden_states,
         reinterpret_cast<const float *>(router->host_ptr), S, D, c.n_experts);

  // Per-token: top-k gating with softmax over selected experts.
  std::vector<int> topk_idx((size_t)S * topk);
  std::vector<float> topk_weight((size_t)S * topk);
  for (int s = 0; s < S; ++s) {
    float *logits = gate_logits.data() + s * c.n_experts;
    // Sanitize non-finite logits before softmax/top-k.  NaN breaks the strict
    // weak ordering required by std::partial_sort (notably on libc++), which
    // can hang or crash when NaN propagates from upstream quant/dequant bugs.
    for (int e = 0; e < c.n_experts; ++e) {
      if (std::isnan(logits[e]) || std::isinf(logits[e]))
        logits[e] = -1.0e30f;
    }
    // softmax
    float mx = *std::max_element(logits, logits + c.n_experts);
    double sum = 0.0;
    for (int e = 0; e < c.n_experts; ++e) {
      logits[e] = std::exp(logits[e] - mx);
      sum += logits[e];
    }
    for (int e = 0; e < c.n_experts; ++e)
      logits[e] = (float)(logits[e] / sum);

    // greedy top-k (small n_experts, simple argmax loop)
    std::vector<int> idx(c.n_experts);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });

    float wsum = 0.0f;
    for (int k = 0; k < topk; ++k) {
      topk_idx[s * topk + k] = idx[k];
      topk_weight[s * topk + k] = logits[idx[k]];
      wsum += logits[idx[k]];
    }
    if (wsum > 0.0f) {
      for (int k = 0; k < topk; ++k)
        topk_weight[s * topk + k] /= wsum;
    }

    // Apply the model-level expert weight scale if configured.
    if (c.expert_weights_scale != 0.0f) {
      for (int k = 0; k < topk; ++k)
        topk_weight[s * topk + k] *= c.expert_weights_scale;
    }
  }

  // Accumulate expert outputs.
  std::vector<float> out((size_t)S * D, 0.0f);
  for (int s = 0; s < S; ++s) {
    const float *x = hidden_states + s * D;
    for (int k = 0; k < topk; ++k) {
      int e = topk_idx[s * topk + k];
      engine.route_usage.bump(layer_idx, e);
      CPUExpertSlot *slot = engine.expert_cache.get(layer_idx, e);
      if (!slot)
        continue;

      // gate(x) @ W_gate -> [moe_inter]
      std::vector<float> gate_proj(moe_inter);
      std::vector<float> up_proj(moe_inter);
      std::vector<float> down_proj(D);

      // Dequantize and matmul.  For now use scalar fallback via
      // ggml_dequant_row + dot_f32_avx512.
      dequant_matmul(slot->gate_type, slot->slab, slot->gate_size, x,
                     gate_proj.data(), D, moe_inter, 1);
      dequant_matmul(slot->up_type, slot->slab + slot->gate_size, slot->up_size,
                     x, up_proj.data(), D, moe_inter, 1);

      silu_inplace(gate_proj.data(), moe_inter);
      elem_mul(gate_proj.data(), gate_proj.data(), up_proj.data(), moe_inter);

      dequant_matmul(
          slot->down_type, slot->slab + slot->gate_size + slot->up_size,
          slot->down_size, gate_proj.data(), down_proj.data(), moe_inter, D, 1);

      float w = topk_weight[s * topk + k];
      for (int i = 0; i < D; ++i)
        out[s * D + i] += w * down_proj[i];
    }
  }

  // Shared expert: always-on MoE FFN branch, gated by ffn_gate_inp_shexp.
  // These tensors are densified to F32 by CPUEngine::build, so we use the
  // plain f32 matmul path instead of dequant_matmul.
  {
    const CPUQuantizedTensor *shared_gate =
        engine.find_layer(prefix + "ffn_gate_shexp.weight");
    const CPUQuantizedTensor *shared_up =
        engine.find_layer(prefix + "ffn_up_shexp.weight");
    const CPUQuantizedTensor *shared_down =
        engine.find_layer(prefix + "ffn_down_shexp.weight");
    const CPUQuantizedTensor *shared_router =
        engine.find_layer(prefix + "ffn_gate_inp_shexp.weight");
    if (shared_gate && shared_up && shared_down && shared_router &&
        shared_gate->host_ptr && shared_up->host_ptr && shared_down->host_ptr &&
        shared_router->host_ptr) {
      // Shared expert has its own intermediate width, distinct from the
      // routed experts' moe_inter -- reusing moe_inter here would give
      // down_proj's matmul the wrong row stride and read across row
      // boundaries into adjacent memory.
      int shared_inter =
          traits.moe_inter_shared > 0 ? traits.moe_inter_shared : moe_inter;
      std::vector<float> gate_proj((size_t)S * shared_inter);
      std::vector<float> up_proj((size_t)S * shared_inter);
      std::vector<float> down_proj((size_t)S * D);
      matmul(gate_proj.data(), hidden_states,
             reinterpret_cast<const float *>(shared_gate->host_ptr), S, D,
             shared_inter);
      matmul(up_proj.data(), hidden_states,
             reinterpret_cast<const float *>(shared_up->host_ptr), S, D,
             shared_inter);
      silu_inplace(gate_proj.data(), S * shared_inter);
      elem_mul(gate_proj.data(), gate_proj.data(), up_proj.data(),
               S * shared_inter);
      matmul(down_proj.data(), gate_proj.data(),
             reinterpret_cast<const float *>(shared_down->host_ptr), S,
             shared_inter, D);

      std::vector<float> gate_scalar(S);
      const float *router_w =
          reinterpret_cast<const float *>(shared_router->host_ptr);
      for (int s = 0; s < S; ++s) {
        float acc = 0.0f;
        const float *x = hidden_states + s * D;
        for (int i = 0; i < D; ++i)
          acc += x[i] * router_w[i];
        gate_scalar[s] = 1.0f / (1.0f + std::exp(-acc));
      }
      for (int s = 0; s < S; ++s) {
        float g = gate_scalar[s];
        for (int i = 0; i < D; ++i)
          out[s * D + i] += g * down_proj[s * D + i];
      }
    }
  }

  // Residual add: FFN output is added to the pre-norm residual.
  for (size_t i = 0; i < out.size(); ++i)
    hidden_states[i] = out[i] + ffn_residual[i];

  for (int s = 0; s < S; ++s) {
    int bad = 0;
    float first_bad = 0.0f;
    for (int i = 0; i < D; ++i) {
      float v = hidden_states[s * D + i];
      if (std::isnan(v) || std::isinf(v)) {
        if (bad == 0)
          first_bad = v;
        ++bad;
      }
    }
    if (bad > 0) {
      LOG_ERROR("[CPU-QWEN] forward_moe_ffn layer=%d seq=%d has %d/%d NaN/Inf "
                "(first=%g)\n",
                layer_idx, s, bad, D, first_bad);
    }
  }
}

// Dequantize one row of a quantized weight matrix and multiply by x.
// This is a minimal helper used only by the CPU MoE path; full-engine builds
// use kquant_dequant.hpp which pulls engine.hpp (CUDA).
#ifdef MOESTREAM_USE_GGML
extern "C" int ggml_dequant_row(uint32_t type, const void *src, float *out,
                                int n);
#endif

// Convert a raw bfloat16 bit pattern to float.  BF16 is the upper 16 bits of
// an IEEE-754 float32 (8-bit exponent, 7-bit mantissa), NOT the same layout
// as FP16 (5-bit exponent, 10-bit mantissa).  Feeding BF16 bits to an FP16
// decoder produces garbage exponents and NaN/Inf floods.
static inline float bf16_to_fp32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline int cpu_dequant_row_fallback(uint32_t type, const void *src, float *out,
                                    int n) {
  // Minimal fallback for F32/F16/BF16 when ggml is not linked.  Quantized
  // types need ggml (or a native dequant implementation) and will return 0.
  if (type == 0) { // F32
    std::memcpy(out, src, (size_t)n * sizeof(float));
    return 1;
  }
  if (type == 1) { // F16
    const uint16_t *s = reinterpret_cast<const uint16_t *>(src);
    for (int i = 0; i < n; ++i)
      out[i] = fp16_to_fp32(s[i]);
    return 1;
  }
  if (type == 25) { // BF16
    const uint16_t *s = reinterpret_cast<const uint16_t *>(src);
    for (int i = 0; i < n; ++i)
      out[i] = bf16_to_fp32(s[i]);
    return 1;
  }
  // Keep this function quiet: the caller (dequant_matmul) logs once per matmul
  // if any row fails, avoiding millions of synchronous stderr prints.
  return 0;
}

inline void CPUQwenMoEBackend::dequant_matmul(GGUFType type, const uint8_t *w,
                                              size_t w_bytes, const float *x,
                                              float *y, int I, int O, int S) {
  (void)w_bytes;
  // Dequantize whole weight to f32 then use the optimized matmul kernel.
  std::vector<float> wf((size_t)I * O, 0.0f);
  int row_bytes = (int)w_bytes / O;
  if (row_bytes * O != (int)w_bytes)
    row_bytes = (int)w_bytes;
  int ok_rows = 0;
  for (int o = 0; o < O; ++o) {
    int ok = 0;
#ifdef MOESTREAM_USE_GGML
    ok = ggml_dequant_row((uint32_t)type, w + (size_t)o * row_bytes,
                          wf.data() + o * I, I);
#else
    ok = cpu_dequant_row_fallback((uint32_t)type, w + (size_t)o * row_bytes,
                                  wf.data() + o * I, I);
#endif
    if (ok)
      ++ok_rows;
  }
  if (ok_rows != O) {
    LOG_ERROR("[CPU-QWEN] dequant_matmul: only %d/%d rows dequantized (type=%u "
              "I=%d O=%d S=%d)\n",
              ok_rows, O, static_cast<uint32_t>(type), I, O, S);
  }
  matmul(y, x, wf.data(), S, I, O);
}
