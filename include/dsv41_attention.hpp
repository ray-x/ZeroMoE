#pragma once

struct StreamingEngine;

// Decode bridge: projections are already normalized and computed by GGML.
// Updates the engine-owned caches and returns the grouped output projection.
void deepseek41_attention_projected(StreamingEngine &engine, int layer,
                                    int pos, const float *normed,
                                    const float *q_lora, const float *query,
                                    const float *raw_kv, float *output);
// Context-only variant for graph backends that provide their own grouped
// output projection. The result is [n_heads * (qk_nope + qk_rope)].
void deepseek41_attention_projected_context(
    StreamingEngine &engine, int layer, int pos, const float *normed,
    const float *q_lora, const float *query, const float *raw_kv,
    float *contexts);
