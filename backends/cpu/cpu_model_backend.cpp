// src/cpu_model_backend.cpp — Factory for CPU-only model backends.
#include "cpu_model_backend.hpp"
#include "cpu_dense.hpp"
#include "cpu_qwen_moe.hpp"
#include "cpu_qwen4exp.hpp"

#include <iostream>

std::unique_ptr<CPUModelBackend>
CPUModelBackend::create(const GGUFFile &model_file) {
  ModelFamily fam = detect_model_family(model_file);

  switch (fam) {
  case ModelFamily::Qwen: {
    auto backend = std::make_unique<CPUQwenMoEBackend>();
    backend->traits.inspect_tensors(model_file);
    int lin = 0, dense = 0;
    for (auto lt : backend->traits.layer_types) {
      if (lt == CPULayerType::LinearAttention)
        ++lin;
      else
        ++dense;
    }
    std::cerr << "[CPU-MODEL] Detected: Qwen MoE (" << lin
              << " linear-attention layers, " << dense << " dense layers)\n";
    return backend;
  }

  case ModelFamily::GLM_DSA:
  case ModelFamily::DeepSeek_V3:
    std::cerr << "[CPU-MODEL] MLA backend (GLM-DSA/DeepSeek-V3) not yet "
                 "implemented for zeromoe_cpu.\n";
    return nullptr;

  case ModelFamily::GLM5NEXT:
    std::cerr << "[CPU-MODEL] GLM 5.3 Flash (glm5next, hybrid KDA+DSA) not yet "
                 "implemented for zeromoe_cpu -- use the main zeromoe "
                 "engine path.\n";
    return nullptr;

  case ModelFamily::Qwen4Exp:
    {
      auto backend = std::make_unique<CPUQwen4ExpBackend>();
      backend->traits.inspect_tensors(model_file);
      backend->layer_types.assign(
          static_cast<size_t>(model_file.architecture_config.n_layers),
          CPULayerType::DenseQKV);
      for (int l = 0; l < model_file.architecture_config.n_layers; ++l)
        if (qwen4exp_is_gdn_layer(model_file.architecture_config, l))
          backend->layer_types[static_cast<size_t>(l)] =
              CPULayerType::LinearAttention;
      if (!backend->compatible(model_file)) {
        std::cerr << "[CPU-MODEL] Qwen3.8-Flash-Next metadata is incomplete "
                    "for the CPU backend.\n";
        return nullptr;
      }
      std::cerr << "[CPU-MODEL] Detected: Qwen3.8-Flash-Next (qwen4exp, "
                  "hybrid GDN+full-attention)\n";
      return backend;
    }

  case ModelFamily::Hunyuan:
  case ModelFamily::GLM4_MOE:
  case ModelFamily::Llama:
    std::cerr << "[CPU-MODEL] DenseQKV backend (Hunyuan/Llama) not yet "
                 "implemented for zeromoe_cpu.\n";
    return nullptr;

  default:
    std::cerr << "[CPU-MODEL] Unknown architecture for CPU inference; cannot "
                 "proceed.\n";
    return nullptr;
  }
}

std::unique_ptr<CPUModelBackend>
CPUModelBackend::create_from_safetensors_path(const std::string &path,
                                              ArchitectureConfig *out_cfg,
                                              const std::string &config_json_path) {
  SafetensorsModelInfo model;
  if (!SafetensorsAdapter::parse_file(path, model)) {
    std::cerr << "[CPU-SAFETENSORS] Failed to parse safetensors file: " << path
              << "\n";
    return nullptr;
  }

  std::string format = SafetensorsAdapter::detect_format(model);
  SafetensorsFamily family = SafetensorsAdapter::detect_family(model);

  DenseCpuConfigJson cfg;
  if (!config_json_path.empty())
    cfg = SafetensorsAdapter::parse_dense_config_json(config_json_path);

  if (family == SafetensorsFamily::QwenHybridMoE) {
    QwenCpuContract contract = SafetensorsAdapter::map_qwen_cpu_contract(model, format);

    auto backend = std::make_unique<CPUQwenMoEBackend>();
    backend->traits.n_layers = contract.n_layers;
    backend->traits.n_heads = contract.n_heads;
    backend->traits.n_kv_heads = contract.n_kv_heads;
    backend->traits.head_dim = contract.head_dim;
    backend->traits.n_experts = contract.n_experts;
    backend->traits.topk = contract.topk;
    backend->traits.moe_inter = contract.moe_inter;
    backend->traits.moe_inter_shared = contract.moe_inter_shared;
    backend->traits.layer_types.assign(contract.n_layers, CPULayerType::DenseQKV);

    if (out_cfg) {
      out_cfg->n_layers = contract.n_layers;
      out_cfg->n_heads = contract.n_heads;
      out_cfg->n_head_kv = contract.n_kv_heads;
      out_cfg->head_dim = contract.head_dim;
      out_cfg->n_experts = contract.n_experts;
      out_cfg->topk = contract.topk;
      out_cfg->moe_inter = contract.moe_inter;
    }

    std::cerr << "[CPU-SAFETENSORS] Detected " << format
              << " Qwen-hybrid export with " << contract.n_layers
              << " layers, " << contract.n_heads << " heads, "
              << contract.n_experts << " experts\n";
    return backend;
  }

  if (family == SafetensorsFamily::DenseGQA ||
      family == SafetensorsFamily::Unknown) {
    DenseCpuContract contract = SafetensorsAdapter::map_dense_cpu_contract(model, format);
    contract = SafetensorsAdapter::apply_dense_config_json(contract, cfg);

    auto backend = std::make_unique<CPUDenseBackend>();
    backend->traits = DenseModelTraits{
        .n_layers = contract.n_layers,
        .n_heads = contract.n_heads,
        .n_kv_heads = contract.n_kv_heads,
        .head_dim = contract.head_dim,
        .hidden = contract.hidden,
        .intermediate = contract.intermediate,
        .vocab = contract.vocab,
        .eps = contract.eps,
        .rope_theta = contract.rope_theta,
        .partial_rotary_factor = contract.partial_rotary_factor,
    };

    if (out_cfg) {
      out_cfg->n_layers = contract.n_layers;
      out_cfg->n_heads = contract.n_heads;
      out_cfg->n_head_kv = contract.n_kv_heads;
      out_cfg->head_dim = contract.head_dim;
      out_cfg->hidden = contract.hidden;
      out_cfg->dense_inter = contract.intermediate;
      out_cfg->vocab = contract.vocab;
      out_cfg->eps = contract.eps;
      out_cfg->sliding_window = contract.sliding_window;
    }

    std::cerr << "[CPU-SAFETENSORS] Detected " << format
              << " dense-GQA export with " << contract.n_layers
              << " layers, hidden=" << contract.hidden << ", heads="
              << contract.n_heads << "/" << contract.n_kv_heads << "\n";
    return backend;
  }

  std::cerr << "[CPU-SAFETENSORS] Unsupported safetensors family for: " << path
            << "\n";
  return nullptr;
}
