// models/model_base.hpp — Abstract base interface for model-specific forward passes
// Each model family (GLM-DSA, HY-V3, DeepSeek-V3, etc.) implements this interface.
#pragma once
#include "engine.hpp"
#include "ram_manager.hpp"
#include <string>
#include <memory>

// Attention backend type (detected from GGUF metadata)
enum class AttentionType {
    MLA,        // Multi-head Latent Attention (GLM-5.2, DeepSeek-V3)
    DenseQKV,   // Standard dense Q/K/V projections (HY-V3, Llama, Qwen)
};

// Model family identifier
enum class ModelFamily {
    Unknown,
    GLM_DSA,       // GLM-5.2 with DSA indexer + MLA + MoE
    GLM4_MOE,      // GLM-4.5 with standard attention + MoE
    Hunyuan,       // HunyuanMoE family (V1/V2/V3) — dense QKV + MoE
    DeepSeek_V3,   // DeepSeek-V3 with MLA + MoE (similar to GLM-DSA)
    Llama,         // Llama-family (dense FFN, standard attention)
};

// Abstract base for model-specific computation
class ModelBackend {
public:
    virtual ~ModelBackend() = default;

    // Model identification
    virtual ModelFamily family() const = 0;
    virtual AttentionType attention_type() const = 0;
    virtual const char* name() const = 0;

    // === Full forward pass for one layer ===
    // Computes: input → attn_norm → attention → residual → ffn_norm → FFN → residual → output
    virtual void forward_layer(
        StreamingEngine& engine,
        RAMSafetyManager& ram_mgr,
        int layer_idx,
        float* hidden_states,   // [S, D] in/out
        int S,                  // sequence length
        int pos_base            // position offset for KV cache
    ) = 0;

    // === Attention-only forward (for layer-split pipelines) ===
    virtual void forward_attention(
        StreamingEngine& engine,
        int layer_idx,
        const float* input,     // [S, D] normed input
        float* output,          // [S, D] attention output
        int S,
        int pos_base
    ) = 0;

    // === Populate KV cache for current token(s) ===
    virtual void update_kv_cache(
        StreamingEngine& engine,
        int layer_idx,
        const float* input,     // [S, D] raw hidden states (pre-norm)
        int S,
        int pos_base
    ) = 0;

    // Factory: detect model family from GGUF metadata and create backend
    static std::unique_ptr<ModelBackend> create(const GGUFFile& model_file);
};

// Detect model family from GGUF architecture string
inline ModelFamily detect_model_family(const GGUFFile& model_file) {
    auto it = model_file.kv_string.find("general.architecture");
    if (it == model_file.kv_string.end()) return ModelFamily::Unknown;

    const std::string& arch = it->second;
    if (arch == "glm-dsa")       return ModelFamily::GLM_DSA;
    if (arch == "glm4moe")       return ModelFamily::GLM4_MOE;
    if (arch == "hy-v3")         return ModelFamily::Hunyuan;
    if (arch == "hunyuan-moe")   return ModelFamily::Hunyuan;
    if (arch == "deepseek2")     return ModelFamily::DeepSeek_V3;
    if (arch == "llama")         return ModelFamily::Llama;

    // Heuristic fallbacks
    if (model_file.kv_uint.count("glm-dsa.embedding_length")) return ModelFamily::GLM_DSA;
    if (model_file.kv_uint.count("hy-v3.embedding_length"))   return ModelFamily::Hunyuan;

    return ModelFamily::Unknown;
}
