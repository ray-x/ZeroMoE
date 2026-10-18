// src/qwen_moe.cpp — GPU Qwen MoE backend scaffold.
//
// This file is compiled into the main moestream binary (CUDA builds only).
// It currently wires Qwen detection into the GPU backend hierarchy.  The
// dense QKV + MoE FFN paths reuse the existing HY-V3 / GLM-DSA kernels;
// the hybrid SSM layers are a planned extension that will call dedicated
// CUDA kernels once the exact Qwen tensor layout is verified.
#include "qwen_moe.hpp"
#include <iostream>

void QwenMoEBackend::forward_layer(StreamingEngine &engine,
                                   RAMSafetyManager &ram_mgr, int layer_idx,
                                   float *hidden_states, int S, int pos_base) {
  (void)engine;
  (void)ram_mgr;
  (void)layer_idx;
  (void)hidden_states;
  (void)S;
  (void)pos_base;
  std::cerr << "[QWEN-GPU] forward_layer not yet implemented for layer "
            << layer_idx << "\n";
}

void QwenMoEBackend::forward_attention(StreamingEngine &engine, int layer_idx,
                                       const float *input, float *output, int S,
                                       int pos_base) {
  (void)engine;
  (void)layer_idx;
  (void)input;
  (void)output;
  (void)S;
  (void)pos_base;
  std::cerr << "[QWEN-GPU] forward_attention not yet implemented for layer "
            << layer_idx << "\n";
}

void QwenMoEBackend::update_kv_cache(StreamingEngine &engine, int layer_idx,
                                     const float *input, int S, int pos_base) {
  (void)engine;
  (void)layer_idx;
  (void)input;
  (void)S;
  (void)pos_base;
  std::cerr << "[QWEN-GPU] update_kv_cache not yet implemented for layer "
            << layer_idx << "\n";
}
