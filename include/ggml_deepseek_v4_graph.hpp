#pragma once

// Opt-in DeepSeek-V4/V4-Flash prefill graph.  The graph covers the complete
// hyper-connected trunk, including raw SWA, CSA/HCA compressed attention,
// attention sinks, grouped output projection, and MoE FFNs.  It is kept
// behind this bridge so a failed or unsupported graph build falls back to the
// established layer-by-layer implementation.

struct StreamingEngine;

// Enabled only when MOESTREAM_DEEPSEEK_V4_FULL_GRAPH=1 is present.
bool deepseek_v4_full_graph_enabled();

// Number of complete early MoE layers whose expert tensors are placed in the
// graph CUDA arena. Zero leaves routed expert projections host-backed so the
// GGML scheduler can queue their selective transfers.
int deepseek_v4_full_graph_device_expert_layers();

// Build or reuse a graph for a position-zero prefill of length S.  Preparation
// is outside the measured prefill window, matching BeeLLama's graph setup.
bool deepseek_v4_full_graph_prepare(StreamingEngine &engine, int S,
                                    int pos_base, const int *token_ids);

// Execute the prepared graph, write the final [S,D] activation to
// hidden_states, and publish raw/compressed attention state for decode.
bool deepseek_v4_full_graph_run(StreamingEngine &engine, float *hidden_states,
                                int S, int pos_base, const int *token_ids);

void deepseek_v4_full_graph_shutdown();
