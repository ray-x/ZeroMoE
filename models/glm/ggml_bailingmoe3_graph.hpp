#pragma once

struct StreamingEngine;
class RAMSafetyManager;
class LockFreePrefetcher;
struct LayerScratch;

// Ling's cache-backed full graph. The graph contains one host callback per
// layer; attention and MoE reuse the engine's native cache-aware dispatch.
bool bailingmoe3_native_graph_enabled();
// Hybrid path: GLM5Next's GGML/CUDA attention graph with native cache-backed
// Ling MoE callbacks. Requires both native graph flags plus the hybrid opt-in.
bool bailingmoe3_hybrid_cuda_graph_enabled();
void bailingmoe3_native_graph_bind_runtime(StreamingEngine &engine,
                                           RAMSafetyManager &ram,
                                           LockFreePrefetcher &prefetch,
                                           LayerScratch &scratch);
void bailingmoe3_native_graph_ffn(StreamingEngine &engine, int layer_idx,
                                 const float *hidden_states, int S,
                                 float *output, const int *token_ids);
bool bailingmoe3_native_graph_prepare(StreamingEngine &engine, int S,
                                      int pos_base, const int *token_ids);
bool bailingmoe3_native_graph_run(StreamingEngine &engine, float *hidden, int S,
                                  int pos_base, const int *token_ids);
void bailingmoe3_native_graph_shutdown();
