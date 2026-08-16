// src/cpu_dense.cpp — Dense HF-style transformer CPU backend.
//
// Implements standard transformer layers for safetensors exports that use
// HuggingFace-style tensor naming (q_proj/k_proj/v_proj/o_proj, gate/up/down,
// input/post_attention_layernorm). The tensor names are mapped to canonical
// GGUF-style names by SafetensorsAdapter so this file stays free of format
// specifics.
#include "cpu_dense.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

static inline const float *tensor_f32(const CPUQuantizedTensor *qt) {
  return reinterpret_cast<const float *>(qt->host_ptr);
}

static inline float *tensor_f32(CPUQuantizedTensor *qt) {
  return reinterpret_cast<float *>(qt->host_ptr);
}

void CPUDenseBackend::rmsnorm(float *out, const float *x, const float *w, int S,
                              int D) {
  for (int s = 0; s < S; ++s) {
    double ss = 0.0;
    const float *row = x + (int64_t)s * D;
    float *dst = out + (int64_t)s * D;
    for (int i = 0; i < D; ++i)
      ss += static_cast<double>(row[i]) * row[i];
    float r = 1.0f / std::sqrt(static_cast<float>(ss / D) + traits.eps);
    for (int i = 0; i < D; ++i)
      dst[i] = row[i] * r * w[i];
  }
}

void CPUDenseBackend::apply_rope(float *Q, float *K, int S, int pos_base) {
  const int H = traits.n_heads;
  const int n_kv_heads = traits.n_kv_heads;
  const int d_head = traits.head_dim;
  const int rotary_dim =
      static_cast<int>(std::floor(d_head * traits.partial_rotary_factor + 0.5f));
  if (rotary_dim < 2)
    return;
  const int half = rotary_dim / 2;

  for (int s = 0; s < S; ++s) {
    const int pos = pos_base + s;
    for (int h = 0; h < H; ++h) {
      float *x = Q + ((int64_t)s * H + h) * d_head;
      for (int i = 0; i < half; ++i) {
        const int j = i + half;
        const float inv_freq =
            1.0f / std::pow(traits.rope_theta, 2.0f * i / rotary_dim);
        const float angle = static_cast<float>(pos) * inv_freq;
        const float c = std::cos(angle), si = std::sin(angle);
        const float x0 = x[i], x1 = x[j];
        x[i] = x0 * c - x1 * si;
        x[j] = x0 * si + x1 * c;
      }
    }
    for (int h = 0; h < n_kv_heads; ++h) {
      float *x = K + ((int64_t)s * n_kv_heads + h) * d_head;
      for (int i = 0; i < half; ++i) {
        const int j = i + half;
        const float inv_freq =
            1.0f / std::pow(traits.rope_theta, 2.0f * i / rotary_dim);
        const float angle = static_cast<float>(pos) * inv_freq;
        const float c = std::cos(angle), si = std::sin(angle);
        const float x0 = x[i], x1 = x[j];
        x[i] = x0 * c - x1 * si;
        x[j] = x0 * si + x1 * c;
      }
    }
  }
}

void CPUDenseBackend::dense_attention(CPUEngine &engine, int layer_idx,
                                      const float *Q, const float *K,
                                      const float *V, int S, int pos_base,
                                      float *out) {
  const int H = traits.n_heads;
  const int n_kv_heads = traits.n_kv_heads;
  const int d_head = traits.head_dim;
  const int n_groups = H / n_kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));

  const int T = std::min(pos_base + S, engine.max_t);

  // Write current K/V into the KV cache.
  for (int s = 0; s < S; ++s) {
    const int pos = pos_base + s;
    if (pos >= engine.max_t)
      continue;
    std::memcpy(engine.kv_cache.k_ptr(layer_idx, pos),
                K + (int64_t)s * n_kv_heads * d_head,
                (size_t)n_kv_heads * d_head * sizeof(float));
    std::memcpy(engine.kv_cache.v_ptr(layer_idx, pos),
                V + (int64_t)s * n_kv_heads * d_head,
                (size_t)n_kv_heads * d_head * sizeof(float));
  }

  std::vector<float> head_out((size_t)S * H * d_head, 0.0f);
  for (int s = 0; s < S; ++s) {
    const int context_len = std::min(pos_base + s + 1, T);
    for (int h = 0; h < H; ++h) {
      const int kv_h = h / n_groups;
      const float *q_h = Q + ((int64_t)s * H + h) * d_head;
      std::vector<float> scores(context_len, 0.0f);
      for (int t = 0; t < context_len; ++t) {
        const float *k_t =
            engine.kv_cache.k_ptr(layer_idx, t) + kv_h * d_head;
        float acc = 0.0f;
        for (int i = 0; i < d_head; ++i)
          acc += q_h[i] * k_t[i];
        scores[t] = acc * scale;
      }
      softmax_row(scores.data(), context_len);
      float *o_h = head_out.data() + ((int64_t)s * H + h) * d_head;
      for (int t = 0; t < context_len; ++t) {
        const float *v_t =
            engine.kv_cache.v_ptr(layer_idx, t) + kv_h * d_head;
        const float a = scores[t];
        for (int i = 0; i < d_head; ++i)
          o_h[i] += a * v_t[i];
      }
    }
  }

  const std::string prefix = "blk." + std::to_string(layer_idx) + ".";
  const CPUQuantizedTensor *wo = engine.find_layer(prefix + "attn_output.weight");
  if (wo && wo->host_ptr) {
    matmul(out, head_out.data(), tensor_f32(wo), S, H * d_head, traits.hidden);
  } else {
    std::memset(out, 0, (size_t)S * traits.hidden * sizeof(float));
  }
}

void CPUDenseBackend::forward_layer(CPUEngine &engine, int layer_idx,
                                    float *hidden_states, int S, int pos_base) {
  const std::string prefix = "blk." + std::to_string(layer_idx) + ".";
  const int D = traits.hidden;

  // 1. Pre-attention RMSNorm.
  std::vector<float> normed((size_t)S * D);
  const CPUQuantizedTensor *attn_norm = engine.find_layer(prefix + "attn_norm.weight");
  if (attn_norm && attn_norm->host_ptr) {
    rmsnorm(normed.data(), hidden_states, tensor_f32(attn_norm), S, D);
  } else {
    std::memcpy(normed.data(), hidden_states, (size_t)S * D * sizeof(float));
  }

  // 2. QKV projections.
  const int H = traits.n_heads;
  const int n_kv_heads = traits.n_kv_heads;
  const int d_head = traits.head_dim;
  std::vector<float> Q((size_t)S * H * d_head);
  std::vector<float> K((size_t)S * n_kv_heads * d_head);
  std::vector<float> V((size_t)S * n_kv_heads * d_head);

  const CPUQuantizedTensor *wq = engine.find_layer(prefix + "attn_q.weight");
  const CPUQuantizedTensor *wk = engine.find_layer(prefix + "attn_k.weight");
  const CPUQuantizedTensor *wv = engine.find_layer(prefix + "attn_v.weight");
  if (wq && wq->host_ptr)
    matmul(Q.data(), normed.data(), tensor_f32(wq), S, D, H * d_head);
  if (wk && wk->host_ptr)
    matmul(K.data(), normed.data(), tensor_f32(wk), S, D, n_kv_heads * d_head);
  if (wv && wv->host_ptr)
    matmul(V.data(), normed.data(), tensor_f32(wv), S, D, n_kv_heads * d_head);

  // Optional Q/K RMSNorm (e.g. Qwen3-style).
  const CPUQuantizedTensor *qnorm = engine.find_layer(prefix + "attn_q_norm.weight");
  const CPUQuantizedTensor *knorm = engine.find_layer(prefix + "attn_k_norm.weight");
  if (qnorm && qnorm->host_ptr) {
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < H; ++h)
        rmsnorm(Q.data() + ((int64_t)s * H + h) * d_head,
                Q.data() + ((int64_t)s * H + h) * d_head,
                tensor_f32(qnorm), 1, d_head);
  }
  if (knorm && knorm->host_ptr) {
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < n_kv_heads; ++h)
        rmsnorm(K.data() + ((int64_t)s * n_kv_heads + h) * d_head,
                K.data() + ((int64_t)s * n_kv_heads + h) * d_head,
                tensor_f32(knorm), 1, d_head);
  }

  apply_rope(Q.data(), K.data(), S, pos_base);

  // 3. Causal attention + residual.
  std::vector<float> attn_out((size_t)S * D, 0.0f);
  dense_attention(engine, layer_idx, Q.data(), K.data(), V.data(), S, pos_base,
                  attn_out.data());
  for (size_t i = 0; i < attn_out.size(); ++i)
    hidden_states[i] += attn_out[i];

  // 4. Post-attention RMSNorm.
  std::vector<float> ffn_in((size_t)S * D);
  const CPUQuantizedTensor *ffn_norm = engine.find_layer(prefix + "ffn_norm.weight");
  if (ffn_norm && ffn_norm->host_ptr) {
    rmsnorm(ffn_in.data(), hidden_states, tensor_f32(ffn_norm), S, D);
  } else {
    std::memcpy(ffn_in.data(), hidden_states, (size_t)S * D * sizeof(float));
  }

  // 5. SwiGLU MLP.
  const int I = traits.intermediate;
  const CPUQuantizedTensor *gate = engine.find_layer(prefix + "ffn_gate.weight");
  const CPUQuantizedTensor *up = engine.find_layer(prefix + "ffn_up.weight");
  const CPUQuantizedTensor *down = engine.find_layer(prefix + "ffn_down.weight");
  if (gate && up && down && gate->host_ptr && up->host_ptr && down->host_ptr) {
    std::vector<float> gate_buf((size_t)S * I);
    std::vector<float> up_buf((size_t)S * I);
    matmul(gate_buf.data(), ffn_in.data(), tensor_f32(gate), S, D, I);
    matmul(up_buf.data(), ffn_in.data(), tensor_f32(up), S, D, I);
    silu_inplace(gate_buf.data(), (int)gate_buf.size());
    elem_mul(up_buf.data(), gate_buf.data(), up_buf.data(), (int)up_buf.size());
    std::vector<float> mlp_out((size_t)S * D, 0.0f);
    matmul(mlp_out.data(), up_buf.data(), tensor_f32(down), S, I, D);
    for (size_t i = 0; i < mlp_out.size(); ++i)
      hidden_states[i] += mlp_out[i];
  }
}

bool CPUDenseBackend::lm_head(CPUEngine &engine, const float *hidden,
                              float *logits, int S) {
  const int D = traits.hidden;
  const int V = traits.vocab;

  // Final RMSNorm.
  std::vector<float> normed((size_t)S * D);
  const CPUQuantizedTensor *norm_w = engine.find_layer("output_norm.weight");
  if (norm_w && norm_w->host_ptr) {
    rmsnorm(normed.data(), hidden, tensor_f32(norm_w), S, D);
  } else {
    std::memcpy(normed.data(), hidden, (size_t)S * D * sizeof(float));
  }

  const CPUQuantizedTensor *lm_head = engine.find_layer("output.weight");
  if (!lm_head || !lm_head->host_ptr)
    return false;
  matmul(logits, normed.data(), tensor_f32(lm_head), S, D, V);
  return true;
}
