#pragma once

// Opt-in Qwen4Exp full graph.  The implementation is deliberately kept
// behind this small bridge so the existing per-layer path remains available
// as a correctness fallback while the full graph is validated on hardware.

struct StreamingEngine;

// Enabled only when MOESTREAM_QWEN4EXP_FULL_GRAPH=1 is present.  The mode is
// The prefill graph is enabled by MOESTREAM_QWEN4EXP_FULL_GRAPH=1.  Decode
// additionally requires MOESTREAM_QWEN4EXP_FULL_GRAPH_DECODE=1, keeping the
// established layered decode path as the default and as the fallback.
bool qwen4exp_full_graph_enabled();
bool qwen4exp_full_graph_decode_enabled();

// Number of complete early MoE layers whose expert tensors should be placed
// in the graph CUDA weight arena. Zero keeps all routed experts host-backed.
int qwen4exp_full_graph_device_expert_layers();

// Build/rebuild the graph for a prompt shape.  This is intended to run before
// the measured prefill window, just as Bee builds its graph before execution.
bool qwen4exp_full_graph_prepare(StreamingEngine &engine, int S,
                                 int pos_base, const int *token_ids);

// Build/reuse the one-token decode graph.  This is Qwen-only and requires a
// complete preloaded DDR expert arena for every layer.  KV/GDN/PLE recurrent
// state is seeded once from the existing host state and remains on the device
// for subsequent submissions.
bool qwen4exp_full_graph_decode_prepare(StreamingEngine &engine, int pos_base,
                                         const int *token_ids);

// Execute the prepared graph for a fresh prompt (pos_base == 0), write the
// final output-mixer result into hidden_states [S,D], and return true.  A
// false return means the caller must use the established per-layer path.
bool qwen4exp_full_graph_run(StreamingEngine &engine, float *hidden_states,
                             int S, int pos_base, const int *token_ids);

// Execute one decode token with one scheduler submission and one final sync.
// If logits/vocab are supplied, the graph also publishes model logits; the
// optional form lets non-generation callers use only the hidden output.
bool qwen4exp_full_graph_decode_run(StreamingEngine &engine,
                                    float *hidden_states, int pos_base,
                                    const int *token_ids, float *logits,
                                    int vocab);

// Destroy the graph while the shared CUDA backend is still alive.  The
// bridge owns scheduler buffers, so leaving this to static destruction runs
// after CudaManager has torn down the driver and makes cudaFree report
// "driver shutting down".
void qwen4exp_full_graph_shutdown();
