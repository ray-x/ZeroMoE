#pragma once

// Opt-in Qwen4Exp prefill graph.  The implementation is deliberately kept
// behind this small bridge so the existing per-layer path remains available
// as a correctness fallback while the full graph is validated on hardware.

struct StreamingEngine;

// Enabled only when MOESTREAM_QWEN4EXP_FULL_GRAPH=1 is present.  The mode is
// prefill-only for now: the graph includes the complete 48-layer trunk and
// updates the existing host KV/recurrent state at the request boundary.
bool qwen4exp_full_graph_enabled();

// Number of complete early MoE layers whose expert tensors should be placed
// in the graph CUDA weight arena. Zero keeps all routed experts host-backed.
int qwen4exp_full_graph_device_expert_layers();

// Build/rebuild the graph for a prompt shape.  This is intended to run before
// the measured prefill window, just as Bee builds its graph before execution.
bool qwen4exp_full_graph_prepare(StreamingEngine &engine, int S,
                                 int pos_base, const int *token_ids);

// Execute the prepared graph for a fresh prompt (pos_base == 0), write the
// final output-mixer result into hidden_states [S,D], and return true.  A
// false return means the caller must use the established per-layer path.
bool qwen4exp_full_graph_run(StreamingEngine &engine, float *hidden_states,
                             int S, int pos_base, const int *token_ids);

// Destroy the graph while the shared CUDA backend is still alive.  The
// bridge owns scheduler buffers, so leaving this to static destruction runs
// after CudaManager has torn down the driver and makes cudaFree report
// "driver shutting down".
void qwen4exp_full_graph_shutdown();
