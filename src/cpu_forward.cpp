// src/cpu_forward.cpp — Generic CPU forward helpers used by moestream_cpu.
#include "cpu_engine.hpp"
#include "cpu_model_backend.hpp"

#include <cstring>

bool cpu_embed_token(CPUEngine &engine, int tok, float *out, int D) {
  CPUQuantizedTensor *emb = engine.find_layer("token_embd.weight");
  if (emb && emb->host_ptr) {
    // token_embd.weight is [V, D]; copy row tok.
    const float *embd = reinterpret_cast<const float *>(emb->host_ptr);
    std::memcpy(out, embd + (int64_t)tok * D, (size_t)D * sizeof(float));
    return true;
  }

  // Fallback: tied embedding/LM head where only output.weight [D, V] exists.
  // In that layout, column `tok` is the embedding vector for that token.
  CPUQuantizedTensor *head = engine.find_layer("output.weight");
  if (head && head->host_ptr && head->meta.shape.size() >= 2 &&
      static_cast<int>(head->meta.shape[0]) == D) {
    int V = static_cast<int>(head->meta.shape[1]);
    if (tok >= 0 && tok < V) {
      const float *w = reinterpret_cast<const float *>(head->host_ptr);
      for (int i = 0; i < D; ++i)
        out[i] = w[(int64_t)i * V + tok];
      return true;
    }
  }
  return false;
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
