// src/cpu_model_backend.cpp — Factory for CPU-only model backends.
#include "cpu_model_backend.hpp"
#include "cpu_qwen_moe.hpp"

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
                 "implemented for moestream_cpu.\n";
    return nullptr;

  case ModelFamily::Hunyuan:
  case ModelFamily::GLM4_MOE:
  case ModelFamily::Llama:
    std::cerr << "[CPU-MODEL] DenseQKV backend (Hunyuan/Llama) not yet "
                 "implemented for moestream_cpu.\n";
    return nullptr;

  default:
    std::cerr << "[CPU-MODEL] Unknown architecture for CPU inference; cannot "
                 "proceed.\n";
    return nullptr;
  }
}
