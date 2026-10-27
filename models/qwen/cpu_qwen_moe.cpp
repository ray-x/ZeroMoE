// src/cpu_qwen_moe.cpp — CPU-only forward pass for Qwen3.6 MoE.
#include "cpu_qwen_moe.hpp"
#include "logger.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Per-layer hidden-state traces are available at the standard VERBOSE
// logging level. Reports the RMS and first
// few values of the LAST sequence position only -- during prefill that's
// the position the LM head actually reads, and during decode S is always 1
// anyway, so this is the one row worth watching every call.
static bool debug_layers_enabled() {
  return Logger::instance().would_log(LogLevel::VERBOSE);
}

// `abs_pos` is the token's ABSOLUTE position in the sequence (pos_base +
// local pos), not the local index into this call's own S-row buffer. Using
// the absolute position is what makes the tag unambiguous across calls: a
// real multi-token prefill's local pos 0 and a later single-token decode
// step's (always-local-pos-0) row used to both print "@0", making it
// impossible to tell from the tag alone which call a line came from -- this
// bit in practice once BOS-handling was fixed and prefill itself dropped to
// S=1, since then EVERY call (prefill and every decode step) had S=1 and
// printed local pos 0. Threading pos_base through fixes that at the source.
static void debug_print_norm_at(const char *tag, int layer_idx,
                                const float *x, int D, int local_pos,
                                int abs_pos) {
  if (!debug_layers_enabled())
    return;
  const float *row = x + (size_t)local_pos * D;
  double ss = 0.0;
  for (int i = 0; i < D; ++i)
    ss += static_cast<double>(row[i]) * row[i];
  double rms = std::sqrt(ss / D);
  LOG_STREAM_VERBOSE() << "[CPU-QWEN-DEBUG] " << tag << "@" << abs_pos
                       << " layer=" << layer_idx << " rms=" << rms
                       << " first3=[" << row[0] << "," << row[1] << ","
                       << row[2] << "]\n";
}

// Prints the FIRST token (local pos 0) in addition to the LAST (local pos
// S-1, the position every other checkpoint in this file was already
// printing) -- each tagged with its ABSOLUTE sequence position
// (pos_base + local pos) so prefill rows and decode-step rows are never
// confusable, even when both happen to have S=1 (see debug_print_norm_at's
// comment for why that specifically matters here).
static void debug_print_norm(const char *tag, int layer_idx, const float *x,
                             int S, int D, int pos_base) {
  if (!debug_layers_enabled())
    return;
  if (S > 1)
    debug_print_norm_at(tag, layer_idx, x, D, 0, pos_base);
  debug_print_norm_at(tag, layer_idx, x, D, S - 1, pos_base + S - 1);
}

static inline float *tensor_f32(CPUQuantizedTensor *qt) {
  return (qt && qt->host_ptr) ? reinterpret_cast<float *>(qt->host_ptr)
                              : nullptr;
}

static inline const float *tensor_f32(const CPUQuantizedTensor *qt) {
  return (qt && qt->host_ptr) ? reinterpret_cast<const float *>(qt->host_ptr)
                              : nullptr;
}

static bool has_tensor(const CPUEngine &engine, const std::string &name) {
  return engine.find_layer(name) != nullptr;
}

static inline float silu_f(float x) { return x / (1.0f + std::exp(-x)); }

// ---------------------------------------------------------------------------
// Top-level layer dispatch
// ---------------------------------------------------------------------------

static bool all_finite(const float *x, int n, const char *what, int layer_idx) {
  int bad = 0;
  float first_bad = 0.0f;
  for (int i = 0; i < n; ++i) {
    if (std::isnan(x[i]) || std::isinf(x[i])) {
      if (bad == 0)
        first_bad = x[i];
      ++bad;
    }
  }
  if (bad > 0) {
    LOG_ERROR(
        "[CPU-QWEN] %s layer=%d contains %d/%d NaN/Inf values (first=%g)\n",
        what, layer_idx, bad, n, first_bad);
    return false;
  }
  return true;
}

void CPUQwenMoEBackend::forward_layer(CPUEngine &engine, int layer_idx,
                                      float *hidden_states, int S,
                                      int pos_base) {
  forward_ok_ = true;
  CPULayerType lt = layer_type(layer_idx);
  int D = engine.model_file.architecture_config.hidden;
  const char *kind = lt == CPULayerType::LinearAttention ? "linear" : "dense";

  debug_print_norm((std::string("pre-attn(") + kind + ")").c_str(), layer_idx,
                   hidden_states, S, D, pos_base);

  if (lt == CPULayerType::LinearAttention) {
    forward_linear_attention(engine, layer_idx, hidden_states, S, pos_base);
  } else {
    forward_dense_attention(engine, layer_idx, hidden_states, S, pos_base);
  }
  if (!forward_ok_) return;

  debug_print_norm("post-attn", layer_idx, hidden_states, S, D, pos_base);

  // Both layer types are followed by the same routed MoE FFN + residual.
  forward_moe_ffn(engine, layer_idx, hidden_states, S);

  debug_print_norm("post-ffn", layer_idx, hidden_states, S, D, pos_base);
}

bool CPUQwenMoEBackend::lm_head(CPUEngine &engine, const float *hidden,
                                float *logits, int S) {
  const ArchitectureConfig &c = engine.model_file.architecture_config;
  int D = c.hidden;
  int V = c.vocab;

  // Final layer norm.
  std::vector<float> normed((size_t)S * D);
  CPUQuantizedTensor *norm_w = engine.find_layer("output_norm.weight");
  if (norm_w && norm_w->host_ptr) {
    for (int s = 0; s < S; ++s)
      qwen_rmsnorm_row(normed.data() + s * D, hidden + s * D,
                       tensor_f32(norm_w), D, c.eps);
  } else {
    std::memcpy(normed.data(), hidden, (size_t)S * D * sizeof(float));
  }

  // LM head: [S, D] @ [V, D]^T -> [S, V]
  CPUQuantizedTensor *lm_head = engine.find_layer("output.weight");
  if (!lm_head || !lm_head->host_ptr)
    return false;

  matmul(logits, normed.data(), tensor_f32(lm_head), S, D, V);
  return true;
}

// ---------------------------------------------------------------------------
// Dense QKV attention block (standard transformer layers)
// ---------------------------------------------------------------------------

void CPUQwenMoEBackend::head_rmsnorm(float *out, const float *x, const float *w,
                                     int dim, float eps) {
  double ss = 0.0;
  for (int i = 0; i < dim; ++i)
    ss += static_cast<double>(x[i]) * x[i];
  float r = 1.0f / std::sqrt(static_cast<float>(ss / dim) + eps);
  for (int i = 0; i < dim; ++i)
    out[i] = x[i] * r * w[i];
}

void CPUQwenMoEBackend::forward_dense_attention(CPUEngine &engine,
                                                int layer_idx,
                                                float *hidden_states, int S,
                                                int pos_base) {
  const ArchitectureConfig &c = engine.model_file.architecture_config;
  std::string prefix = "blk." + std::to_string(layer_idx) + ".";
  int D = c.hidden;
  // Use the validated ArchitectureConfig values for dense attention.  The
  // KV cache is sized from these, so using traits.* here can create a
  // mismatch (e.g. traits.head_dim from linear-attention metadata).
  int H = c.n_heads > 0 ? c.n_heads : traits.n_heads;
  int n_kv_heads = c.n_head_kv > 0 ? c.n_head_kv : traits.n_kv_heads;
  int d_head = c.head_dim > 0 ? c.head_dim : traits.head_dim;
  if (d_head <= 0 && H > 0)
    d_head = D / H;
  if (H <= 0 || n_kv_heads <= 0 || d_head <= 0 || H % n_kv_heads != 0) {
    LOG_ERROR("[CPU-QWEN] Dense attention dimension mismatch in %s: "
              "D=%d H=%d n_kv_heads=%d d_head=%d\n",
              prefix.c_str(), D, H, n_kv_heads, d_head);
    forward_ok_ = false;
    return;
  }
  int rotary_dim = static_cast<int>(
      std::floor(d_head * traits.partial_rotary_factor + 0.5f));
  if (rotary_dim < 2)
    rotary_dim = d_head;
  // Read the model's own RoPE base instead of assuming Qwen's common default
  // -- a checkpoint fine-tuned for longer context can carry a different value.
  float theta_base =
      engine.model_file.get_config_float("rope.freq_base", 1000000.0f);

  // 1. Attention RMSNorm.
  std::vector<float> normed((size_t)S * D);
  CPUQuantizedTensor *attn_norm =
      engine.find_layer(prefix + "attn_norm.weight");
  if (attn_norm && attn_norm->host_ptr) {
    for (int s = 0; s < S; ++s)
      qwen_rmsnorm_row(normed.data() + s * D, hidden_states + s * D,
                       tensor_f32(attn_norm), D, c.eps);
  } else {
    std::memcpy(normed.data(), hidden_states, (size_t)S * D * sizeof(float));
  }

  // 2. QKV projection.  Qwen stores Q+gate together in attn_q.weight.
  std::vector<float> Q((size_t)S * H * d_head);
  std::vector<float> gate((size_t)S * H * d_head);
  std::vector<float> K((size_t)S * n_kv_heads * d_head);
  std::vector<float> V((size_t)S * n_kv_heads * d_head);
  project_qkv_dense(engine, layer_idx, normed.data(), S, Q.data(), gate.data(),
                    K.data(), V.data());
  if (!forward_ok_) return;

  // 3. Per-head Q/K RMSNorm, then partial RoPE.
  const CPUQuantizedTensor *qnorm =
      engine.find_layer(prefix + "attn_q_norm.weight");
  const CPUQuantizedTensor *knorm =
      engine.find_layer(prefix + "attn_k_norm.weight");
  if (qnorm && qnorm->host_ptr) {
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < H; ++h)
        head_rmsnorm(Q.data() + (s * H + h) * d_head,
                     Q.data() + (s * H + h) * d_head, tensor_f32(qnorm), d_head,
                     c.eps);
  }
  if (knorm && knorm->host_ptr) {
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < n_kv_heads; ++h)
        head_rmsnorm(K.data() + (s * n_kv_heads + h) * d_head,
                     K.data() + (s * n_kv_heads + h) * d_head,
                     tensor_f32(knorm), d_head, c.eps);
  }

  apply_rope_neox(Q.data(), K.data(), S, pos_base, H, n_kv_heads, d_head,
                  rotary_dim, theta_base);

  // 4. Causal attention; gate is applied to the per-head attention output
  // before the output projection maps H*d_head -> D.
  std::vector<float> attn_out((size_t)S * D, 0.0f);
  dense_attention(engine, layer_idx, Q.data(), K.data(), V.data(), S, pos_base,
                  gate.data(), attn_out.data());
  if (!forward_ok_) return;

  // 5. Residual add.
  for (size_t i = 0; i < attn_out.size(); ++i)
    hidden_states[i] += attn_out[i];

  all_finite(hidden_states, S * D, "forward_dense_attention post-residual",
             layer_idx);
}

void CPUQwenMoEBackend::project_qkv_dense(const CPUEngine &engine,
                                          int layer_idx, const float *normed,
                                          int S, float *Q, float *gate,
                                          float *K, float *V) {
  const ArchitectureConfig &c = engine.model_file.architecture_config;
  int D = c.hidden;
  int H = c.n_heads > 0 ? c.n_heads : traits.n_heads;
  int n_kv_heads = c.n_head_kv > 0 ? c.n_head_kv : traits.n_kv_heads;
  int d_head = c.head_dim > 0 ? c.head_dim : traits.head_dim;
  if (d_head <= 0 && H > 0)
    d_head = D / H;
  std::string prefix = "blk." + std::to_string(layer_idx) + ".";

  const CPUQuantizedTensor *q = engine.find_layer(prefix + "attn_q.weight");
  const CPUQuantizedTensor *k = engine.find_layer(prefix + "attn_k.weight");
  const CPUQuantizedTensor *v = engine.find_layer(prefix + "attn_v.weight");
  if (!q || !k || !v || !q->host_ptr || !k->host_ptr || !v->host_ptr) {
    LOG_ERROR("[CPU-QWEN] Missing dense-attention projections in %s\n", prefix.c_str());
    forward_ok_ = false;
    return;
  }

  if (q && q->host_ptr) {
    // Qwen attn_q projects hidden -> interleaved-per-head [Q, gate] pairs:
    // row = [Q_h0(d_head), gate_h0(d_head), Q_h1(d_head), gate_h1(d_head), ...],
    // NOT a block split of [all-heads Q, all-heads gate]. Confirmed against
    // llama.cpp's qwen35moe.cpp build_layer_attn(), whose Qcur/gate views both
    // use stride 2*d_head per head with gate at a d_head offset within that
    // same per-head span.
    std::vector<float> qg((size_t)S * H * d_head * 2);
    matmul(qg.data(), normed, tensor_f32(q), S, D, H * d_head * 2);
    for (int s = 0; s < S; ++s) {
      const float *src = qg.data() + s * H * d_head * 2;
      for (int h = 0; h < H; ++h) {
        const float *head_src = src + h * d_head * 2;
        std::memcpy(Q + (s * H + h) * d_head, head_src,
                    (size_t)d_head * sizeof(float));
        std::memcpy(gate + (s * H + h) * d_head, head_src + d_head,
                    (size_t)d_head * sizeof(float));
      }
    }
  }
  if (k && k->host_ptr)
    matmul(K, normed, tensor_f32(k), S, D, n_kv_heads * d_head);
  if (v && v->host_ptr)
    matmul(V, normed, tensor_f32(v), S, D, n_kv_heads * d_head);
}

void CPUQwenMoEBackend::apply_rope_neox(float *Q, float *K, int S, int pos_base,
                                        int H, int n_kv_heads, int head_dim,
                                        int rotary_dim, float theta_base) {
  int half = rotary_dim / 2;
  for (int s = 0; s < S; ++s) {
    int pos = pos_base + s;
    for (int h = 0; h < H; ++h) {
      float *x = Q + (s * H + h) * head_dim;
      for (int i = 0; i < half; ++i) {
        int j = i + half;
        float inv_freq = 1.0f / std::pow(theta_base, 2.0f * i / rotary_dim);
        float angle = pos * inv_freq;
        float c = std::cos(angle), si = std::sin(angle);
        float x0 = x[i], x1 = x[j];
        x[i] = x0 * c - x1 * si;
        x[j] = x0 * si + x1 * c;
      }
    }
    for (int h = 0; h < n_kv_heads; ++h) {
      float *x = K + (s * n_kv_heads + h) * head_dim;
      for (int i = 0; i < half; ++i) {
        int j = i + half;
        float inv_freq = 1.0f / std::pow(theta_base, 2.0f * i / rotary_dim);
        float angle = pos * inv_freq;
        float c = std::cos(angle), si = std::sin(angle);
        float x0 = x[i], x1 = x[j];
        x[i] = x0 * c - x1 * si;
        x[j] = x0 * si + x1 * c;
      }
    }
  }
}

void CPUQwenMoEBackend::dense_attention(CPUEngine &engine, int layer_idx,
                                        const float *Q, const float *K,
                                        const float *V, int S, int pos_base,
                                        const float *gate, float *out) {
  const ArchitectureConfig &c = engine.model_file.architecture_config;
  int D = c.hidden;
  int H = c.n_heads > 0 ? c.n_heads : traits.n_heads;
  int n_kv_heads = c.n_head_kv > 0 ? c.n_head_kv : traits.n_kv_heads;
  int d_head = c.head_dim > 0 ? c.head_dim : traits.head_dim;
  if (d_head <= 0 && H > 0)
    d_head = D / H;
  int n_groups = H / n_kv_heads;
  float scale = 1.0f / std::sqrt((float)d_head);

  // Update KV cache with current K/V.
  for (int s = 0; s < S; ++s) {
    int pos = pos_base + s;
    if (pos >= engine.max_t)
      continue;
    float *k_dst = engine.kv_cache.k_ptr(layer_idx, pos);
    float *v_dst = engine.kv_cache.v_ptr(layer_idx, pos);
    std::memcpy(k_dst, K + s * n_kv_heads * d_head,
                (size_t)n_kv_heads * d_head * sizeof(float));
    std::memcpy(v_dst, V + s * n_kv_heads * d_head,
                (size_t)n_kv_heads * d_head * sizeof(float));
  }

  int T = pos_base + S; // total cached positions available
  if (T > engine.max_t)
    T = engine.max_t;

  std::vector<float> head_out((size_t)S * H * d_head, 0.0f);

  for (int s = 0; s < S; ++s) {
    int context_len = pos_base + s + 1;
    if (context_len > T)
      context_len = T;

    for (int h = 0; h < H; ++h) {
      int kv_h = h / n_groups;
      const float *q_h = Q + (s * H + h) * d_head;

      // scores[t] = q · k[t]
      std::vector<float> scores(context_len, 0.0f);
      for (int t = 0; t < context_len; ++t) {
        const float *k_t = engine.kv_cache.k_ptr(layer_idx, t) + kv_h * d_head;
        float acc = 0.0f;
        for (int i = 0; i < d_head; ++i)
          acc += q_h[i] * k_t[i];
        scores[t] = acc * scale;
      }

      softmax_row(scores.data(), context_len);

      float *o_h = head_out.data() + (s * H + h) * d_head;
      std::memset(o_h, 0, (size_t)d_head * sizeof(float));
      for (int t = 0; t < context_len; ++t) {
        const float *v_t = engine.kv_cache.v_ptr(layer_idx, t) + kv_h * d_head;
        float a = scores[t];
        for (int i = 0; i < d_head; ++i)
          o_h[i] += a * v_t[i];
      }
    }
  }

  // Gated output (Qwen full-attention layers): gate is per-head and has the
  // same [S, H, d_head] shape as head_out. Apply BEFORE the output projection
  // that maps H*d_head -> D.
  if (gate) {
    for (size_t i = 0; i < (size_t)S * H * d_head; ++i)
      head_out[i] *= 1.0f / (1.0f + std::exp(-gate[i]));
  }

  // Output projection: head_out [S, H*d_head] -> out [S, D].
  std::string prefix = "blk." + std::to_string(layer_idx) + ".";
  const CPUQuantizedTensor *wo =
      engine.find_layer(prefix + "attn_output.weight");
  if (wo && wo->host_ptr) {
    matmul(out, head_out.data(), tensor_f32(wo), S, H * d_head, D);
  } else {
    LOG_ERROR("[CPU-QWEN] Missing attention output projection in %s\n", prefix.c_str());
    forward_ok_ = false;
  }
}

// ---------------------------------------------------------------------------
// Gated DeltaNet linear attention block
// ---------------------------------------------------------------------------

void CPUQwenMoEBackend::causal_conv1d(const float *x, const float *w,
                                      float *out, int S, int channels,
                                      int kernel, float *history) {
  // x/out: [S, channels]
  // w: [channels, kernel] row-major (matches the GGUF layout of
  // ssm_conv1d.weight, which stores one kernel-size vector per channel).
  // history: [channels, kernel-1] row-major, persistent across calls -- the
  // last (kernel-1) samples per channel from the previous call (zero at
  // sequence start). Updated in place to hold the last (kernel-1) samples of
  // this call for the NEXT call.
  //
  // Two things had to match llama.cpp's ggml_ssm_conv exactly (confirmed by
  // reading ggml_compute_forward_ssm_conv_f32, ggml/src/ggml-cpu/ops.cpp, and
  // build_conv_state, src/models/delta-net-base.cpp):
  //  1. Kernel-index direction: w[k] applies to virtual time offset
  //     s-(kernel-1)+k, i.e. k=0 is the OLDEST tap and k=kernel-1 is the
  //     CURRENT sample -- this file previously did the opposite (idx = s-k,
  //     k=0 = current), silently applying every trained conv weight to the
  //     wrong relative time offset (these are real trained weights, not a
  //     symmetric kernel, so this corrupted every call including prefill).
  //  2. Cross-call state: without `history`, every decode call (S=1, see
  //     cpu_main.cpp's decode loop) would only ever see the single new
  //     sample, since idx<0 was simply skipped -- silently truncating the
  //     receptive field to 1/kernel of its intended size for every decode
  //     step (prefill's S>1 batch happened to look fine since same-call
  //     history was still visible within that one call).
  int hist = kernel - 1;
  for (int c = 0; c < channels; ++c) {
    const float *h = history + (size_t)c * hist;
    const float *wc = w + (size_t)c * kernel;
    for (int s = 0; s < S; ++s) {
      float acc = 0.0f;
      for (int k = 0; k < kernel; ++k) {
        int j = s + k; // position in the virtual [history ++ x] window
        float val = (j < hist) ? h[j] : x[(size_t)(j - hist) * channels + c];
        acc += val * wc[k];
      }
      out[(size_t)s * channels + c] = silu_f(acc);
    }
  }
  // Roll the history forward: keep the last `hist` samples of the virtual
  // [history ++ x] window (length hist+S) for the next call.
  if (hist > 0) {
    std::vector<float> new_hist((size_t)channels * hist);
    for (int c = 0; c < channels; ++c) {
      const float *h = history + (size_t)c * hist;
      for (int j = 0; j < hist; ++j) {
        int pos = S + j;
        new_hist[(size_t)c * hist + j] =
            (pos < hist) ? h[pos] : x[(size_t)(pos - hist) * channels + c];
      }
    }
    std::memcpy(history, new_hist.data(),
                (size_t)channels * hist * sizeof(float));
  }
}

void CPUQwenMoEBackend::gated_delta_rule(const float *Q, const float *K,
                                         const float *V, const float *beta,
                                         const float *g_log_decay, float *state,
                                         float *out, int S, int n_v_heads,
                                         int k_dim, int v_dim) {
  // Q/K/V/beta/g_log_decay: [S, n_v_heads, ...] after repeat_interleave
  // state: [n_v_heads, k_dim, v_dim] row-major
  // out: [S, n_v_heads, v_dim]
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < n_v_heads; ++h) {
      float *state_h = state + (size_t)h * k_dim * v_dim;
      float decay = std::exp(g_log_decay[s * n_v_heads + h]);
      // state *= decay
      for (int i = 0; i < k_dim * v_dim; ++i)
        state_h[i] *= decay;

      const float *k_h = K + ((size_t)s * n_v_heads + h) * k_dim;
      const float *q_h_raw = Q + ((size_t)s * n_v_heads + h) * k_dim;
      const float *v_h = V + ((size_t)s * n_v_heads + h) * v_dim;
      float *out_h = out + ((size_t)s * n_v_heads + h) * v_dim;
      float b = beta[s * n_v_heads + h];

      // kv_mem = state^T @ k  => [v_dim]
      std::vector<float> kv_mem(v_dim, 0.0f);
      for (int k = 0; k < k_dim; ++k) {
        float kk = k_h[k];
        const float *row = state_h + (size_t)k * v_dim;
        for (int v = 0; v < v_dim; ++v)
          kv_mem[v] += row[v] * kk;
      }

      // delta = (v - kv_mem) * beta
      std::vector<float> delta(v_dim);
      for (int v = 0; v < v_dim; ++v)
        delta[v] = (v_h[v] - kv_mem[v]) * b;

      // state += k[:,None] * delta[None,:]
      for (int k = 0; k < k_dim; ++k)
        for (int v = 0; v < v_dim; ++v)
          state_h[(size_t)k * v_dim + v] += k_h[k] * delta[v];

      // out = state^T @ (q * scale), scale = 1/sqrt(k_dim)
      float q_scale = 1.0f / std::sqrt(static_cast<float>(k_dim));
      for (int v = 0; v < v_dim; ++v)
        out_h[v] = 0.0f;
      for (int k = 0; k < k_dim; ++k) {
        float qk = q_h_raw[k] * q_scale;
        const float *row = state_h + (size_t)k * v_dim;
        for (int v = 0; v < v_dim; ++v)
          out_h[v] += row[v] * qk;
      }
    }
  }
}

void CPUQwenMoEBackend::forward_linear_attention(CPUEngine &engine,
                                                 int layer_idx,
                                                 float *hidden_states, int S,
                                                 int pos_base) {
  const ArchitectureConfig &c = engine.model_file.architecture_config;
  int D = c.hidden;
  int n_k_h = traits.linear_num_key_heads;
  int k_dim = traits.linear_key_head_dim;
  int n_v_h = traits.linear_num_value_heads;
  int v_dim = traits.linear_value_head_dim;
  int conv_k = traits.linear_conv_kernel;

  // Defensive: all linear-attention dims must be positive and consistent.
  if (n_k_h <= 0 || k_dim <= 0 || n_v_h <= 0 || v_dim <= 0 || conv_k <= 0) {
    LOG_ERROR("[CPU-QWEN] Invalid linear-attention dims at layer=%d: "
              "n_k_h=%d k_dim=%d n_v_h=%d v_dim=%d conv_k=%d\n",
              layer_idx, n_k_h, k_dim, n_v_h, v_dim, conv_k);
    forward_ok_ = false;
    return;
  }
  if (n_v_h % n_k_h != 0) {
    LOG_ERROR("[CPU-QWEN] n_v_h=%d is not divisible by n_k_h=%d at layer=%d\n",
              n_v_h, n_k_h, layer_idx);
    forward_ok_ = false;
    return;
  }

  int key_total = n_k_h * k_dim;
  int value_total = n_v_h * v_dim;
  int qkv_total = 2 * key_total + value_total;

  std::string prefix = "blk." + std::to_string(layer_idx) + ".";

  // 1. Input norm.
  std::vector<float> normed((size_t)S * D);
  CPUQuantizedTensor *attn_norm =
      engine.find_layer(prefix + "attn_norm.weight");
  if (attn_norm && attn_norm->host_ptr) {
    for (int s = 0; s < S; ++s)
      qwen_rmsnorm_row(normed.data() + s * D, hidden_states + s * D,
                       tensor_f32(attn_norm), D, c.eps);
  } else {
    std::memcpy(normed.data(), hidden_states, (size_t)S * D * sizeof(float));
  }

  // 2. Projections: QKV, z-gate, beta (decay) and a (dt offset).
  CPUQuantizedTensor *qkv = engine.find_layer(prefix + "attn_qkv.weight");
  CPUQuantizedTensor *zgate = engine.find_layer(prefix + "attn_gate.weight");
  CPUQuantizedTensor *alpha_t = engine.find_layer(prefix + "ssm_alpha.weight");
  CPUQuantizedTensor *beta_t = engine.find_layer(prefix + "ssm_beta.weight");
  CPUQuantizedTensor *conv_w = engine.find_layer(prefix + "ssm_conv1d.weight");
  CPUQuantizedTensor *ssm_a = engine.find_layer(prefix + "ssm_a");
  CPUQuantizedTensor *ssm_dt_bias = engine.find_layer(prefix + "ssm_dt.bias");
  CPUQuantizedTensor *norm_weight =
      engine.find_layer(prefix + "ssm_norm.weight");
  CPUQuantizedTensor *out_proj = engine.find_layer(prefix + "ssm_out.weight");
  if (!tensor_f32(qkv) || !tensor_f32(zgate) || !tensor_f32(alpha_t) ||
      !tensor_f32(beta_t) || !tensor_f32(conv_w) || !tensor_f32(ssm_a) ||
      !tensor_f32(ssm_dt_bias) || !tensor_f32(norm_weight) || !tensor_f32(out_proj)) {
    LOG_ERROR("[CPU-QWEN] Missing linear-attention tensors in %s\n",
              prefix.c_str());
    forward_ok_ = false;
    return;
  }
  if (!all_finite(tensor_f32(qkv), qkv->meta.shape[0] * qkv->meta.shape[1],
                  "qkv weight", layer_idx) ||
      !all_finite(tensor_f32(out_proj),
                  out_proj->meta.shape[0] * out_proj->meta.shape[1],
                  "ssm_out weight", layer_idx)) {
    forward_ok_ = false;
    return;
  }

  std::vector<float> mixed_qkv((size_t)S * qkv_total);
  std::vector<float> z((size_t)S * value_total);
  std::vector<float> a((size_t)S * n_v_h);
  std::vector<float> b((size_t)S * n_v_h);
  matmul(mixed_qkv.data(), normed.data(), tensor_f32(qkv), S, D, qkv_total);
  matmul(z.data(), normed.data(), tensor_f32(zgate), S, D, value_total);
  matmul(a.data(), normed.data(), tensor_f32(alpha_t), S, D, n_v_h);
  matmul(b.data(), normed.data(), tensor_f32(beta_t), S, D, n_v_h);

  // Cross-check against llama.cpp's "linear_attn_qkv_mixed" cb() tensor --
  // raw wqkv projection output, BEFORE causal_conv1d. Isolates whether a
  // divergence lives in the projection itself vs. in causal_conv1d.
  debug_print_norm("mixed_qkv_predelta", layer_idx, mixed_qkv.data(), S,
                    qkv_total, pos_base);

  // 3. Causal conv1d over mixed QKV (groups=channels, silu activation).
  // History must persist across calls (decode calls this function once per
  // token, S=1 each time) -- see causal_conv1d's own comment.
  int conv_hist = conv_k - 1;
  if (conv_history.size() <= (size_t)layer_idx)
    conv_history.resize(layer_idx + 1);
  if ((int)conv_history[layer_idx].size() != qkv_total * conv_hist)
    conv_history[layer_idx].assign((size_t)qkv_total * conv_hist, 0.0f);

  std::vector<float> conv_out((size_t)S * qkv_total);
  causal_conv1d(mixed_qkv.data(), tensor_f32(conv_w), conv_out.data(), S,
                qkv_total, conv_k, conv_history[layer_idx].data());

  // Cross-check against llama.cpp's "conv_output_silu" cb() tensor -- post
  // causal_conv1d AND post-silu, still unsplit (Q+K+V all in one buffer).
  debug_print_norm("conv_output_silu", layer_idx, conv_out.data(), S,
                    qkv_total, pos_base);

  // 4. Split into Q, K, V and reshape to [S, heads, head_dim].
  std::vector<float> Q((size_t)S * n_k_h * k_dim);
  std::vector<float> K((size_t)S * n_k_h * k_dim);
  std::vector<float> Vraw((size_t)S * n_v_h * v_dim);
  for (int s = 0; s < S; ++s) {
    const float *src = conv_out.data() + (size_t)s * qkv_total;
    for (int i = 0; i < key_total; ++i)
      Q[(size_t)s * n_k_h * k_dim + i] = src[i];
    for (int i = 0; i < key_total; ++i)
      K[(size_t)s * n_k_h * k_dim + i] = src[key_total + i];
    for (int i = 0; i < value_total; ++i)
      Vraw[(size_t)s * n_v_h * v_dim + i] = src[2 * key_total + i];
  }

  // 5. L2-normalize Q and K per head.
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < n_k_h; ++h) {
      float *q_h = Q.data() + ((size_t)s * n_k_h + h) * k_dim;
      float *k_h = K.data() + ((size_t)s * n_k_h + h) * k_dim;
      float sq = 0.0f, sk = 0.0f;
      for (int i = 0; i < k_dim; ++i) {
        sq += q_h[i] * q_h[i];
        sk += k_h[i] * k_h[i];
      }
      float rq = 1.0f / std::sqrt(sq + 1e-6f);
      float rk = 1.0f / std::sqrt(sk + 1e-6f);
      for (int i = 0; i < k_dim; ++i) {
        q_h[i] *= rq;
        k_h[i] *= rk;
      }
    }
  }

  // 6. Repeat key heads to match value-head count (n_v_h / n_k_h times).
  //
  // MUST match ggml_repeat_4d's own broadcast convention (confirmed by
  // reading ggml_compute_forward_repeat_f32, ggml/src/ggml-cpu/ops.cpp):
  // dst index j along the repeated dim maps to src index (j % src_extent) --
  // i.e. the WHOLE [0..n_k_h) block is tiled `rep` times in sequence
  // (TILE / cyclic: dst = [h0,h1,...,h_{n_k_h-1], h0,h1,...]), NOT each
  // source head replicated `rep` times contiguously (BLOCK / repeat_interleave:
  // dst = [h0,h0,...,h1,h1,...]). This file used to build dst_h = h*rep+r
  // (BLOCK), which for n_k_h=16/n_v_h=32 pairs value-head 1 with key-head 0
  // instead of the real key-head 1 -- every odd-indexed value head across all
  // 30 Gated-DeltaNet layers received a scrambled K/Q source, more than
  // enough on its own to produce fluent-looking-per-token but incoherent
  // decode output despite finite/plausible prefill logits.
  int rep = n_v_h / n_k_h;
  std::vector<float> Qrep((size_t)S * n_v_h * k_dim);
  std::vector<float> Krep((size_t)S * n_v_h * k_dim);
  for (int s = 0; s < S; ++s) {
    for (int dst_h = 0; dst_h < n_v_h; ++dst_h) {
      int h = dst_h % n_k_h; // TILE, matches ggml_repeat_4d
      const float *q_src = Q.data() + ((size_t)s * n_k_h + h) * k_dim;
      const float *k_src = K.data() + ((size_t)s * n_k_h + h) * k_dim;
      std::memcpy(Qrep.data() + ((size_t)s * n_v_h + dst_h) * k_dim, q_src,
                  (size_t)k_dim * sizeof(float));
      std::memcpy(Krep.data() + ((size_t)s * n_v_h + dst_h) * k_dim, k_src,
                  (size_t)k_dim * sizeof(float));
    }
  }
  (void)rep; // retained for the comment's math / potential future asserts

  // Cross-check points against llama.cpp's eval-callback trace (see
  // build_layer_attn_linear's cb() calls for "q_conv_predelta"/
  // "k_conv_predelta"/"v_conv_predelta" -- captured post L2-norm/repeat for
  // Q/K, pre-anything for V, exactly matching Qrep/Krep/Vraw here).
  debug_print_norm("q_conv_predelta", layer_idx, Qrep.data(), S, n_v_h * k_dim,
                    pos_base);
  debug_print_norm("k_conv_predelta", layer_idx, Krep.data(), S, n_v_h * k_dim,
                    pos_base);
  debug_print_norm("v_conv_predelta", layer_idx, Vraw.data(), S, n_v_h * v_dim,
                    pos_base);

  // 7. Compute log decay per head: g = A * softplus(a + dt_bias).
  //    The ssm_a tensor stores A = -exp(A_log) (continuous-time decay),
  //    following llama.cpp's qwen35moe convention, NOT raw A_log.
  //    beta is the output of ssm_beta projected from the input and sigmoided.
  std::vector<float> g_log_decay((size_t)S * n_v_h);
  for (int i = 0; i < (int)b.size(); ++i)
    b[i] = 1.0f / (1.0f + std::exp(-b[i]));

  // Cross-check against llama.cpp's "beta_sigmoid" cb() tensor.
  debug_print_norm("beta_sigmoid", layer_idx, b.data(), S, n_v_h, pos_base);

  const float *A_param = tensor_f32(ssm_a);
  const float *dt_bias = tensor_f32(ssm_dt_bias);
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < n_v_h; ++h) {
      float aa = a[s * n_v_h + h];
      float sp = std::log1p(std::exp(aa + dt_bias[h]));
      g_log_decay[s * n_v_h + h] = A_param[h] * sp;
    }
  }

  // Cross-check against llama.cpp's "gate" cb() tensor (pre-exp log-decay,
  // i.e. alpha_softplus * ssm_a -- NOT yet exponentiated).
  debug_print_norm("gate", layer_idx, g_log_decay.data(), S, n_v_h, pos_base);

  // 8. Ensure persistent recurrent state exists for this layer.
  // Cap at a sane size to catch corrupt dimensions before malloc traps.
  constexpr size_t MAX_STATE_FLOATS = 256 * 1024 * 1024; // 1 GB of floats
  size_t state_size = (size_t)n_v_h * k_dim * v_dim;
  if (state_size > MAX_STATE_FLOATS) {
    LOG_ERROR("[CPU-QWEN] Refusing to allocate absurd linear-attention state "
              "at layer=%d: n_v_h=%d k_dim=%d v_dim=%d -> %zu floats\n",
              layer_idx, n_v_h, k_dim, v_dim, state_size);
    forward_ok_ = false;
    return;
  }
  if (linear_attn_states.size() <= (size_t)layer_idx)
    linear_attn_states.resize(layer_idx + 1);
  // Only initialize once; zeroing every call destroys recurrence and can mask
  // or trigger heap issues if the vector was previously corrupted.
  if (linear_attn_states[layer_idx].size() != state_size)
    linear_attn_states[layer_idx].assign(state_size, 0.0f);

  // 9. Recurrent gated delta rule.
  std::vector<float> core_out((size_t)S * n_v_h * v_dim);
  gated_delta_rule(Qrep.data(), Krep.data(), Vraw.data(), b.data(),
                   g_log_decay.data(), linear_attn_states[layer_idx].data(),
                   core_out.data(), S, n_v_h, k_dim, v_dim);

  // Cross-check against llama.cpp's "attn_output" cb() tensor (delta-net-base.cpp
  // build_recurrent_attn) -- the RAW recurrence output, before z-gating/out_proj.
  // Bisects whether a divergence lives in gated_delta_rule() itself vs. in the
  // gated-RMSNorm/out_proj steps below.
  debug_print_norm("attn_output", layer_idx, core_out.data(), S, n_v_h * v_dim,
                   pos_base);

  // 10. Gated RMSNorm per value head: x * rms(x) * norm_w * silu(z).
  std::vector<float> gated((size_t)S * value_total);
  const float *norm_w = tensor_f32(norm_weight);
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < n_v_h; ++h) {
      const float *x = core_out.data() + ((size_t)s * n_v_h + h) * v_dim;
      const float *z_h = z.data() + ((size_t)s * n_v_h + h) * v_dim;
      float *g_h = gated.data() + ((size_t)s * n_v_h + h) * v_dim;
      // eps must be added AFTER dividing by v_dim (mean(x^2) + eps), matching
      // qwen_rmsnorm_row's convention -- the prior "msq = eps; msq += sum(x^2);
      // msq /= v_dim" order instead computed (eps + sum(x^2)) / v_dim, shrinking
      // eps's contribution by a factor of v_dim. Invisible for normal-magnitude
      // hidden states, but core_out here can have a very small per-head
      // magnitude (mean(x^2) only ~5e-5), right where eps should matter --
      // confirmed via real cross-engine trace: attn_output (pre-gate, straight
      // out of gated_delta_rule) matched llama.cpp almost exactly, but
      // final_output (post this gated-RMSNorm) was ~24% too large on ZeroMoE.
      double msq = 0.0;
      for (int i = 0; i < v_dim; ++i)
        msq += static_cast<double>(x[i]) * x[i];
      float inv = 1.0f / std::sqrt(static_cast<float>(msq / v_dim) + c.eps);
      for (int i = 0; i < v_dim; ++i)
        g_h[i] = x[i] * inv * norm_w[i] * silu_f(z_h[i]);
    }
  }

  // Cross-check against llama.cpp's "final_output" cb() tensor (qwen35moe.cpp
  // build_layer_attn_linear) -- post z-gating, pre out_proj.
  debug_print_norm("final_output", layer_idx, gated.data(), S, value_total,
                   pos_base);

  // 11. Output projection and residual.
  std::vector<float> attn_out((size_t)S * D, 0.0f);
  if (out_proj && out_proj->host_ptr)
    matmul(attn_out.data(), gated.data(), tensor_f32(out_proj), S, value_total,
           D);

  // Cross-check against llama.cpp's "linear_attn_out" cb() tensor -- post
  // out_proj (ssm_out), pre residual add.
  debug_print_norm("linear_attn_out", layer_idx, attn_out.data(), S, D,
                   pos_base);

  for (size_t i = 0; i < attn_out.size(); ++i)
    hidden_states[i] += attn_out[i];

  all_finite(hidden_states, S * D, "forward_linear_attention post-residual",
             layer_idx);
}
