#pragma once

// Whole-prefill graph for DeepSeek-V4.1-Flash. The graph covers Engram,
// shared ratio-compressed attention, hyper-connections, and routed FFNs.

struct StreamingEngine;
class RAMSafetyManager;
class LockFreePrefetcher;
struct LayerScratch;

// Non-owning access to the runtime already bound by layers_forward_gguf.
bool deepseek_v41_graph_runtime(StreamingEngine &, RAMSafetyManager *&,
                                LockFreePrefetcher *&, LayerScratch *&);
bool deepseek_v41_full_graph_enabled();
bool deepseek_v41_full_graph_decode_enabled();
void deepseek_v41_bind_graph_runtime(StreamingEngine &engine,
                                     RAMSafetyManager &ram,
                                     LockFreePrefetcher &prefetch,
                                     LayerScratch &scratch);
void deepseek_v41_full_graph_moe_forward(StreamingEngine &engine, int layer,
                                         const float *input, int S,
                                         int pos_base,
                                         float *output, const int *token_ids);
void deepseek_v41_full_graph_attention_forward(StreamingEngine &engine,
                                               int layer,
                                               const float *input, int S,
                                               int pos_base, float *output);
bool deepseek_v41_full_graph_decode_prepare(StreamingEngine &engine,
                                            int pos, const int *token_ids);
bool deepseek_v41_full_graph_decode_run(StreamingEngine &engine,
                                        float *hidden, int pos,
                                        const int *token_ids,
                                        float *logits = nullptr, int vocab = 0);
void deepseek_v41_full_graph_decode_shutdown();
int deepseek_v41_full_graph_device_expert_layers();
bool deepseek_v41_full_graph_prepare(StreamingEngine &engine, int S,
                                     int pos_base, const int *token_ids);
bool deepseek_v41_full_graph_run(StreamingEngine &engine, float *hidden_states,
                                 int S, int pos_base, const int *token_ids);
void deepseek_v41_full_graph_shutdown();
