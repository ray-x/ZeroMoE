#pragma once

struct StreamingEngine;
class RAMSafetyManager;
class LockFreePrefetcher;
struct LayerScratch;

// Ling's cache-backed full graph. The graph contains one host callback per
// layer; attention and MoE reuse the engine's native cache-aware dispatch.
bool bailingmoe3_native_graph_enabled();
void bailingmoe3_native_graph_bind_runtime(StreamingEngine &engine,
                                           RAMSafetyManager &ram,
                                           LockFreePrefetcher &prefetch,
                                           LayerScratch &scratch);
bool bailingmoe3_native_graph_prepare(StreamingEngine &engine, int S,
                                      int pos_base, const int *token_ids);
bool bailingmoe3_native_graph_run(StreamingEngine &engine, float *hidden, int S,
                                  int pos_base, const int *token_ids);
void bailingmoe3_native_graph_shutdown();
