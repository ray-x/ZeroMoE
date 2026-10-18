#pragma once

#include <vector>

struct StreamingEngine;

// Gather and dequantize the token-selected Engram rows for one Engram layer.
// The returned row matrix is [n_hash_cols * head_dim, S] in GGML order.
bool deepseek41_engram_embedding_rows(StreamingEngine &engine, int layer_idx,
                                      int S, int pos_base,
                                      const int *token_ids,
                                      std::vector<float> &embedding);
