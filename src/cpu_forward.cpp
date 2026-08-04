// src/cpu_forward.cpp — Generic CPU forward helpers used by moestream_cpu.
#include "cpu_engine.hpp"
#include "cpu_model_backend.hpp"

#include <cstring>

bool cpu_embed_token(CPUEngine &engine, int tok, float *out, int D) {
  CPUQuantizedTensor *emb = engine.find_layer("token_embd.weight");
  if (!emb || !emb->host_ptr)
    return false;

  // token_embd.weight is [V, D]; copy row tok.
  const float *embd = reinterpret_cast<const float *>(emb->host_ptr);
  std::memcpy(out, embd + (int64_t)tok * D, (size_t)D * sizeof(float));
  return true;
}

bool cpu_forward_layer(CPUEngine &engine, int layer_idx, float *hidden_states,
                       int S, int pos_base) {
  if (!engine.backend)
    return false;
  engine.backend->forward_layer(engine, layer_idx, hidden_states, S, pos_base);
  return true;
}

bool cpu_lm_head(CPUEngine &engine, const float *hidden, float *logits, int S) {
  if (!engine.backend)
    return false;
  return engine.backend->lm_head(engine, hidden, logits, S);
}
