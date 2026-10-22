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
  DeepSeek_V4, // DEPRECATED (2026-10-18): graph path removed, use DeepSeek_V41.
               // DeepSeek-V4/V4-Flash: no-absorption MLA + attention sinks +
               // grouped output projection + Hyper-Connections + sqrtsoftplus
               // MoE gating. Structurally distinct from DeepSeek_V3 despite
               // the shared name -- see models/family_ops.cpp's resolve_
               // family_ops table and CLAUDE.md's DeepSeek-V4 plan writeup.
               // Phase 1 only: correct within each layer's local SWA window;
               // CSA/HCA/Lightning-Indexer long-context tiers are future work.
  DeepSeek_V41, // DeepSeek-V4.1-Flash: shared ratio-1/2 compressed MLA,
                // Hyper-Connections, and conditional-memory Engram layers.
  GLM5NEXT,    // GLM 5.3 Flash (GGUF arch "glm5next"): HYBRID per-layer
               // architecture -- recurrent KDA layers (head_count_kv[l]==0)
               // interleaved with nope-only absorbed-MLA DSA layers, plus
               // mHC hyper-connections (same per-layer form as DeepSeek_V4,
               // but the model-level collapse is an unweighted hc_mean, not
               // DSV4's learned output_hc_* head). NOT a variant of GLM_DSA
               // (GLM 5.3) -- a different architecture family entirely.
               // Reference: llama.cpp branch glm5next (src/models/glm5next.cpp).
               // v1: DSA layers run dense (Lightning indexer deferred).
  Qwen4Exp,    // Qwen3.8-Flash-Next (GGUF arch "qwen4exp"): HYBRID per-layer
               // architecture -- Gated DeltaNet recurrent layers interleaved
               // with full-attention layers every full_attention_interval,
               // low-rank hyper-connections in place of every layer norm
               // (INCLUDING the model-level output mixer: no output_norm
               // tensor), QSA sparse attention on the full-attention layers
               // (v1: runs dense, indexer deferred -- same staging as
               // GLM5NEXT's own v1), PLE n-gram hash embedding on select
               // layers, and a 512-expert MoE with a sigmoid-gated shared
               // expert. NOT a variant of Qwen -- a different architecture
               // family entirely. Reference: llama.cpp branch
               // qwen4exp/qwen3.8-flash-next (src/models/qwen4exp.cpp,
               // conversion/qwen4exp.py).
  BailingMoE3, // Ling 3.1 Flash (GGUF arch "bailingmoe3"): KDA + gated MLA.
  MiMo2,      // MiMo V2 family: mixed full/SWA GQA, sigmoid-routed MoE.
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
  if (arch == "deepseek4")
    return ModelFamily::DeepSeek_V4;
  if (arch == "deepseek41")
    return ModelFamily::DeepSeek_V41;
  if (arch == "glm5next" || arch == "glm5-next") // "glm5-next": GSQ-RCO converter spelling
    return ModelFamily::GLM5NEXT;
  if (arch == "qwen4exp")
    return ModelFamily::Qwen4Exp;
  if (arch == "bailingmoe3")
    return ModelFamily::BailingMoE3;
  if (arch == "mimo2")
    return ModelFamily::MiMo2;
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
  // Qwen3.8-Flash-Next ALSO carries blk.0.attn_gate.weight (its GDN layers'
  // z-gate projection) + expert_count, so it must be distinguished from the
  // generic-Qwen heuristic below FIRST: its low-rank hyper-connection
  // tensors (hc_*/output_hc_*) exist in no other family.
  if (model_file.tensors.count("blk.0.hc_attn_inject.weight") ||
      model_file.tensors.count("output_hc_norm.weight")) {
    return ModelFamily::Qwen4Exp;
  }
  // Qwen3.6-35B-A3B hybrid SSM+MoE files carry attention.head_count and
  // expert_count but may not set general.architecture=qwen*; look for SSM
  // gate tensors together with an expert_count.
  if (model_file.tensors.count("blk.0.attn_gate.weight") &&
      model_file.get_config_int("expert_count", 0) > 0) {
    return ModelFamily::Qwen;
  }

  return ModelFamily::Unknown;
}
