#pragma once

struct StreamingEngine;
class RAMSafetyManager;
class LockFreePrefetcher;
struct LayerScratch;

struct GpuTopology;
bool mimo2_full_graph_enabled();
// True when MiMo's routed MoE runs in the graph (hot experts alias the VRAM cache),
// which needs row-aligned cache slot strides.
bool mimo2_ingraph_moe_enabled();
// The engine's VRAM expert cache, borrowed by the in-graph MoE (set once at startup).
void mimo2_full_graph_set_vram_cache(GpuTopology *topology);
void mimo2_bind_runtime(StreamingEngine &, RAMSafetyManager &, LockFreePrefetcher &, LayerScratch &);
bool mimo2_full_graph_prepare(StreamingEngine &, int tokens, int position, const int *ids);
bool mimo2_full_graph_run(StreamingEngine &, float *hidden, int tokens, int position,
                          const int *ids, float *logits = nullptr, int vocab = 0);
void mimo2_full_graph_shutdown();
