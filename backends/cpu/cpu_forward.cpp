// src/cpu_forward.cpp — Generic CPU forward helpers used by zeromoe_cpu.
#include "cpu_engine.hpp"
#include "cpu_model_backend.hpp"

#include <cstring>

extern "C" int ggml_dequant_row(uint32_t type, const void *src, float *out,
                                int n);

bool cpu_embed_token(CPUEngine &engine, int tok, float *out, int D) {
  CPUQuantizedTensor *emb = engine.find_layer("token_embd.weight");
  if (emb && emb->host_ptr) {
    const int rows = emb->meta.shape.size() >= 2
                         ? static_cast<int>(emb->meta.shape[1])
                         : static_cast<int>(emb->meta.shape[0]);
    if (tok >= 0 && tok < rows) {
      const size_t row_bytes = emb->host_size / static_cast<size_t>(rows);
      const uint8_t *src = emb->host_ptr + static_cast<size_t>(tok) * row_bytes;
      if (emb->meta.type == GGUFType::F32) {
        std::memcpy(out, src, (size_t)D * sizeof(float));
        return true;
      }
      if (emb->meta.type == GGUFType::F16) {
        const auto *p = reinterpret_cast<const uint16_t *>(src);
        for (int i = 0; i < D; ++i)
          out[i] = fp16_to_fp32(p[i]);
        return true;
      }
      if (emb->meta.type == GGUFType::BF16) {
        const auto *p = reinterpret_cast<const uint16_t *>(src);
        for (int i = 0; i < D; ++i) {
          uint32_t bits = static_cast<uint32_t>(p[i]) << 16;
          std::memcpy(out + i, &bits, sizeof(float));
        }
        return true;
      }
      return ggml_dequant_row(static_cast<uint32_t>(emb->meta.type), src,
                              out, D) != 0;
    }
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
  return engine.backend->last_forward_succeeded();
}

bool cpu_lm_head(CPUEngine &engine, const float *hidden, float *logits, int S) {
  if (!engine.backend)
    return false;
  return engine.backend->lm_head(engine, hidden, logits, S);
}
