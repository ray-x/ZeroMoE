// models/hy_v3.hpp — Hunyuan MoE family backend (V1/V2/V3)
// All versions use dense QKV attention + MoE FFN.
// Differences between versions are handled by config flags:
//   HunyuanMoE (original): QKV → RoPE → Norm → Attn | softmax router | all layers MoE
//   HY-V3 (latest):        QKV → Norm → RoPE → Attn | sigmoid router + bias | leading dense + MTP
#pragma once
#include "model_base.hpp"
#include "kernels_cpu.hpp"
#include "perf_timer.hpp"
#include <vector>
#include <cmath>
#include <cstring>

class HYV3Backend : public ModelBackend {
public:
    // Configuration: differs between HunyuanMoE and HY-V3
    bool rope_before_norm = false;  // HunyuanMoE: true, HY-V3: false (norm then RoPE)
    bool use_sigmoid_router = true; // HY-V3: true (sigmoid+bias), HunyuanMoE: false (softmax)
    bool has_leading_dense = false; // HY-V3: true, HunyuanMoE: false

    ModelFamily family() const override { return ModelFamily::Hunyuan; }
    AttentionType attention_type() const override { return AttentionType::DenseQKV; }
    const char* name() const override { return "Hunyuan MoE (Dense QKV)"; }

    void forward_layer(
        StreamingEngine& engine,
        RAMSafetyManager& ram_mgr,
        int layer_idx,
        float* hidden_states,
        int S,
        int pos_base
    ) override;

    void forward_attention(
        StreamingEngine& engine,
        int layer_idx,
        const float* input,
        float* output,
        int S,
        int pos_base
    ) override;

    void update_kv_cache(
        StreamingEngine& engine,
        int layer_idx,
        const float* input,
        int S,
        int pos_base
    ) override;

private:
    // Standard dense Q/K/V projection + RoPE
    void project_qkv(
        StreamingEngine& engine, int layer_idx,
        const float* normed_input, int S,
        float* Q_out, float* K_out, float* V_out  // [S, H, d_head]
    );

    // Standard causal attention with dense KV cache
    void dense_attention(
        StreamingEngine& engine, int layer_idx,
        const float* Q, const float* K, const float* V,
        int S, int pos_base, float* output
    );
};

// ============================================================
// Implementation
// ============================================================

inline void HYV3Backend::forward_layer(
    StreamingEngine& engine, RAMSafetyManager& ram_mgr,
    int layer_idx, float* hidden_states, int S, int pos_base
) {
    (void)ram_mgr;
    auto& c = engine.model_file.architecture_config;
    int D = c.hidden;

    std::vector<float> attn_out(S * D);
    forward_attention(engine, layer_idx, hidden_states, attn_out.data(), S, pos_base);

    // Residual add
    for (int i = 0; i < S * D; ++i) hidden_states[i] += attn_out[i];
}

inline void HYV3Backend::forward_attention(
    StreamingEngine& engine, int layer_idx,
    const float* input, float* output, int S, int pos_base
) {
    PERF_SCOPE(t_attn);
    auto& c = engine.model_file.architecture_config;
    int D = c.hidden;
    int H = c.n_heads;
    int d_head = D / H;  // For HY-V3: standard head dim

    // 1. Attention norm
    std::string norm_key = "blk." + std::to_string(layer_idx) + ".attn_norm.weight";
    auto norm_it = engine.resident_layers.find(norm_key);
    std::vector<float> normed(S * D);
    if (norm_it != engine.resident_layers.end() && norm_it->second.host_ptr) {
        for (int s = 0; s < S; ++s)
            rmsnorm_row(normed.data() + s * D, input + s * D,
                       reinterpret_cast<float*>(norm_it->second.host_ptr), D, c.eps);
    } else {
        std::memcpy(normed.data(), input, S * D * sizeof(float));
    }

    // 2. Q/K/V projections
    std::vector<float> Q(S * H * d_head), K(S * H * d_head), V(S * H * d_head);
    project_qkv(engine, layer_idx, normed.data(), S, Q.data(), K.data(), V.data());

    // 3. Dense causal attention
    dense_attention(engine, layer_idx, Q.data(), K.data(), V.data(), S, pos_base, output);
}

inline void HYV3Backend::project_qkv(
    StreamingEngine& engine, int layer_idx,
    const float* normed_input, int S,
    float* Q_out, float* K_out, float* V_out
) {
    auto& c = engine.model_file.architecture_config;
    int D = c.hidden;
    int H = c.n_heads;
    int d_head = D / H;
    std::string prefix = "blk." + std::to_string(layer_idx) + ".";

    // HY-V3 tensor names: attn_q.weight, attn_k.weight, attn_v.weight
    // or packed attn_qkv.weight — try both
    auto q_it = engine.resident_layers.find(prefix + "attn_q.weight");
    auto k_it = engine.resident_layers.find(prefix + "attn_k.weight");
    auto v_it = engine.resident_layers.find(prefix + "attn_v.weight");

    int n_kv_heads = c.n_heads;  // TODO: GQA support (n_head_kv from config)

    if (q_it != engine.resident_layers.end() && q_it->second.host_ptr) {
        matmul(Q_out, normed_input,
               reinterpret_cast<const float*>(q_it->second.host_ptr), S, D, H * d_head);
    }
    if (k_it != engine.resident_layers.end() && k_it->second.host_ptr) {
        matmul(K_out, normed_input,
               reinterpret_cast<const float*>(k_it->second.host_ptr), S, D, n_kv_heads * d_head);
    }
    if (v_it != engine.resident_layers.end() && v_it->second.host_ptr) {
        matmul(V_out, normed_input,
               reinterpret_cast<const float*>(v_it->second.host_ptr), S, D, n_kv_heads * d_head);
    }

    // Norm/RoPE ordering differs by Hunyuan version:
    //   HunyuanMoE (original): QKV → RoPE(Q,K) → K_norm → Q_norm
    //   HY-V3 (latest):        QKV → Q_norm → K_norm → RoPE(Q,K)
    auto qn_it = engine.resident_layers.find(prefix + "attn_q_norm.weight");
    auto kn_it = engine.resident_layers.find(prefix + "attn_k_norm.weight");

    if (rope_before_norm) {
        // HunyuanMoE order: RoPE first, then norms
        // TODO: apply RoPE(Q_out, pos), RoPE(K_out, pos) here

        if (kn_it != engine.resident_layers.end() && kn_it->second.host_ptr) {
            float* kn_w = reinterpret_cast<float*>(kn_it->second.host_ptr);
            for (int s = 0; s < S; ++s)
                for (int h = 0; h < n_kv_heads; ++h)
                    rmsnorm_row(K_out + (s * n_kv_heads + h) * d_head,
                               K_out + (s * n_kv_heads + h) * d_head,
                               kn_w, d_head, c.eps);
        }
        if (qn_it != engine.resident_layers.end() && qn_it->second.host_ptr) {
            float* qn_w = reinterpret_cast<float*>(qn_it->second.host_ptr);
            for (int s = 0; s < S; ++s)
                for (int h = 0; h < H; ++h)
                    rmsnorm_row(Q_out + (s * H + h) * d_head,
                               Q_out + (s * H + h) * d_head,
                               qn_w, d_head, c.eps);
        }
    } else {
        // HY-V3 order: norms first, then RoPE
        if (qn_it != engine.resident_layers.end() && qn_it->second.host_ptr) {
            float* qn_w = reinterpret_cast<float*>(qn_it->second.host_ptr);
            for (int s = 0; s < S; ++s)
                for (int h = 0; h < H; ++h)
                    rmsnorm_row(Q_out + (s * H + h) * d_head,
                               Q_out + (s * H + h) * d_head,
                               qn_w, d_head, c.eps);
        }
        if (kn_it != engine.resident_layers.end() && kn_it->second.host_ptr) {
            float* kn_w = reinterpret_cast<float*>(kn_it->second.host_ptr);
            for (int s = 0; s < S; ++s)
                for (int h = 0; h < n_kv_heads; ++h)
                    rmsnorm_row(K_out + (s * n_kv_heads + h) * d_head,
                               K_out + (s * n_kv_heads + h) * d_head,
                               kn_w, d_head, c.eps);
        }

        // TODO: apply RoPE(Q_out, pos), RoPE(K_out, pos) here
    }
}

inline void HYV3Backend::dense_attention(
    StreamingEngine& engine, int layer_idx,
    const float* Q, const float* K, const float* V,
    int S, int pos_base, float* output
) {
    auto& c = engine.model_file.architecture_config;
    int D = c.hidden;
    int H = c.n_heads;
    int d_head = D / H;
    float scale = 1.0f / std::sqrt((float)d_head);

    // For HY-V3: standard causal attention per-head
    // Q[S, H, d] @ K[T, H, d]^T → scores[S, H, T] → softmax → @ V[T, H, d] → out[S, H, d]
    std::vector<float> head_contexts(S * H * d_head, 0.0f);

    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int s = 0; s < S; ++s) {
        for (int h = 0; h < H; ++h) {
            int context_len = pos_base + s + 1;
            const float* q_h = Q + (s * H + h) * d_head;

            // For now: simplified single-token decode (context from KV cache)
            // Full implementation would read from a dense KV cache
            std::vector<float> scores(context_len, 0.0f);
            // TODO: Read K/V from dense KV cache for past positions
            // For current position: score[pos_base+s] = Q·K_current
            scores[pos_base + s] = 0.0f;
            for (int i = 0; i < d_head; ++i)
                scores[pos_base + s] += q_h[i] * K[(s * H + h) * d_head + i];
            scores[pos_base + s] *= scale;

            softmax_row(scores.data(), context_len);

            // Weighted V sum (simplified: only current token's V for now)
            float* out_h = head_contexts.data() + (s * H + h) * d_head;
            float alpha = scores[pos_base + s];
            const float* v_h = V + (s * H + h) * d_head;
            for (int i = 0; i < d_head; ++i)
                out_h[i] = alpha * v_h[i];
        }
    }

    // Output projection: [S, H*d_head] → [S, D]
    std::string wo_key = "blk." + std::to_string(layer_idx) + ".attn_output.weight";
    auto wo_it = engine.resident_layers.find(wo_key);
    if (wo_it != engine.resident_layers.end() && wo_it->second.host_ptr) {
        matmul(output, head_contexts.data(),
               reinterpret_cast<const float*>(wo_it->second.host_ptr), S, H * d_head, D);
    } else {
        std::memset(output, 0, S * D * sizeof(float));
    }
}

inline void HYV3Backend::update_kv_cache(
    StreamingEngine& engine, int layer_idx,
    const float* input, int S, int pos_base
) {
    // TODO: For HY-V3, populate a dense KV cache with projected K/V
    (void)engine; (void)layer_idx; (void)input; (void)S; (void)pos_base;
}
