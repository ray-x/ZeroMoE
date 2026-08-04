// model_family.hpp — Model family enum and architecture detection.
//
// This header is deliberately free of engine.hpp / CUDA headers so it can be
// included by both the GPU ModelBackend hierarchy and the CPU-only backend
// hierarchy.  Adding a new model family only requires updating this file and
// the corresponding backend factory.
#pragma once
#include "gguf_loader.hpp"
#include <string>

// Attention backend type (detected from GGUF metadata)
enum class AttentionType {
  MLA,      // Multi-head Latent Attention (GLM-5.2, DeepSeek-V3)
  DenseQKV, // Standard dense Q/K/V projections (HY-V3, Llama, Qwen dense
            // layers)
  SSM,      // State-space model / Mamba block (Qwen3.6-35B-A3B hybrid layers)
};

// Model family identifier
enum class ModelFamily {
  Unknown,
  GLM_DSA,     // GLM-5.2 with DSA indexer + MLA + MoE
  GLM4_MOE,    // GLM-4.5 with standard attention + MoE
  Hunyuan,     // HunyuanMoE family (V1/V2/V3) — dense QKV + MoE
  DeepSeek_V3, // DeepSeek-V3 with MLA + MoE (similar to GLM-DSA)
  Llama,       // Llama-family (dense FFN, standard attention)
  Qwen, // Qwen MoE family (Qwen3.6-35B-A3B and later) — hybrid SSM + DenseQKV
        // + MoE
  GemmaMoE, // Google Gemma MoE family (future)
  KimiMoE,  // Moonshot Kimi MoE family (future)
};

// Detect model family from GGUF architecture string.
// This is the single place that maps a GGUF file to a ModelFamily.  New
// families (GemmaMoE, KimiMoE, ...) are added here and in the backend
// factories.
inline ModelFamily detect_model_family(const GGUFFile &model_file) {
  auto it = model_file.kv_string.find("general.architecture");
  const std::string arch = (it != model_file.kv_string.end()) ? it->second : "";

  if (arch == "glm-dsa")
    return ModelFamily::GLM_DSA;
  if (arch == "glm4moe")
    return ModelFamily::GLM4_MOE;
  if (arch == "hy-v3")
    return ModelFamily::Hunyuan;
  if (arch == "hunyuan-moe")
    return ModelFamily::Hunyuan;
  if (arch == "deepseek2")
    return ModelFamily::DeepSeek_V3;
  if (arch == "llama")
    return ModelFamily::Llama;
  if (arch == "qwen" || arch == "qwen2" || arch == "qwen2moe" ||
      arch == "qwen3" || arch == "qwen3moe")
    return ModelFamily::Qwen;
  if (arch == "gemma2" || arch == "gemma3" || arch == "gemma2b")
    return ModelFamily::GemmaMoE;

  // Heuristic fallbacks
  if (model_file.kv_uint.count("glm-dsa.embedding_length"))
    return ModelFamily::GLM_DSA;
  if (model_file.kv_uint.count("hy-v3.embedding_length"))
    return ModelFamily::Hunyuan;
  // Qwen3.6-35B-A3B hybrid SSM+MoE files carry attention.head_count and
  // expert_count but may not set general.architecture=qwen*; look for SSM
  // gate tensors together with an expert_count.
  if (model_file.tensors.count("blk.0.attn_gate.weight") &&
      model_file.get_config_int("expert_count", 0) > 0) {
    return ModelFamily::Qwen;
  }

  return ModelFamily::Unknown;
}
