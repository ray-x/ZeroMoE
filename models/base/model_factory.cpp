// models/model_factory.cpp — Factory for model backend creation
#include "glm_dsa.hpp"
#include "hy_v3.hpp"
#include "model_base.hpp"
#include "qwen_moe.hpp"
#include <iostream>

std::unique_ptr<ModelBackend> ModelBackend::create(const GGUFFile &model_file) {
  ModelFamily fam = detect_model_family(model_file);

  switch (fam) {
  case ModelFamily::MiMo2:
    std::cerr << "[MODEL] Detected: MiMo V2 -- use the registered full-graph engine path.\n";
    return nullptr;
  case ModelFamily::GLM_DSA:
  case ModelFamily::DeepSeek_V3: // DeepSeek-V3 shares MLA architecture
    std::cerr << "[MODEL] Detected: GLM-DSA / DeepSeek-V3 (MLA attention)\n";
    return std::make_unique<GLMDSABackend>();

  case ModelFamily::Hunyuan: {
    // Detect version: HY-V3 (48L, sigmoid router, MTP) vs original (32L,
    // softmax)
    auto backend = std::make_unique<HYV3Backend>();
    int n_layers = model_file.get_config_int("block_count", 0);
    bool is_v3 = (n_layers >= 48);
    if (is_v3) {
      // HY-V3: norm → RoPE, sigmoid router, leading dense, MTP
      backend->rope_before_norm = false;
      backend->use_sigmoid_router = true;
      backend->has_leading_dense = true;
      std::cerr << "[MODEL] Detected: Hunyuan-V3 (norm→RoPE, sigmoid, MTP)\n";
    } else {
      // Original HunyuanMoE: RoPE → norm, softmax router, all MoE
      backend->rope_before_norm = true;
      backend->use_sigmoid_router = false;
      backend->has_leading_dense = false;
      std::cerr << "[MODEL] Detected: HunyuanMoE (RoPE→norm, softmax)\n";
    }
    return backend;
  }

  case ModelFamily::GLM4_MOE:
    // GLM-4.5 uses standard attention (like HY-V3), not MLA
    std::cerr << "[MODEL] Detected: GLM-4.5 MoE (Dense QKV attention)\n";
    return std::make_unique<HYV3Backend>();

  case ModelFamily::Llama:
    std::cerr << "[MODEL] Detected: Llama-family (Dense QKV attention)\n";
    return std::make_unique<HYV3Backend>();

  case ModelFamily::Qwen:
    std::cerr << "[MODEL] Detected: Qwen MoE (hybrid SSM + DenseQKV + MoE)\n";
    return std::make_unique<QwenMoEBackend>();

  case ModelFamily::GLM5NEXT:
    // GLM 5.3 Flash is served by the LIVE engine path (FamilyOps /
    // glm5next_attention_gguf), not this legacy backend hierarchy -- no
    // ModelBackend subclass implements the hybrid KDA+DSA forward.
    std::cerr << "[MODEL] Detected: GLM 5.3 Flash (glm5next, hybrid KDA+DSA) -- "
                 "not supported by the legacy ModelBackend hierarchy; use the "
                 "main zeromoe engine path.\n";
    return nullptr;

  case ModelFamily::Qwen4Exp:
    // Qwen3.8-Flash-Next is served by the LIVE engine path (FamilyOps /
    // qwen4exp_attention_gguf), not this legacy backend hierarchy -- no
    // ModelBackend subclass implements the hybrid GDN+full-attention forward.
    std::cerr << "[MODEL] Detected: Qwen3.8-Flash-Next (qwen4exp, hybrid "
                 "GDN+full-attention + low-rank HC + PLE) -- not supported by "
                 "the legacy ModelBackend hierarchy; use the main zeromoe "
                 "engine path.\n";
    return nullptr;

  case ModelFamily::BailingMoE3:
    std::cerr << "[MODEL] Detected: Ling 3.1 Flash (bailingmoe3) -- "
                 "use the main zeromoe engine path.\n";
    return nullptr;

  default:
    // Default to GLM-DSA for unknown architectures with kv_lora_rank > 0
    if (model_file.get_config_int("attention.kv_lora_rank") > 0) {
      std::cerr << "[MODEL] Unknown architecture with MLA indicators, "
                   "defaulting to GLM-DSA backend\n";
      return std::make_unique<GLMDSABackend>();
    }
    std::cerr
        << "[MODEL] Unknown architecture, defaulting to Dense QKV backend\n";
    return std::make_unique<HYV3Backend>();
  }
}
