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

// Configure full-graph enablement from the TOML profile. Environment
// variables remain supported and take precedence in the query functions.
void qwen4exp_full_graph_configure(bool enabled, bool decode_enabled,
                                   bool prefill_state_handoff = false);
bool qwen4exp_full_graph_sync_state_to_host(StreamingEngine &engine);

// Startup warmup that loads prompt-independent static weight arenas before the
// first request. Enabled by default for full-graph decode; set
// MOESTREAM_QWEN4EXP_GRAPH_BOOTSTRAP=0 to keep lazy preparation. The prompt
// shape defaults to 16 and can be set with
// MOESTREAM_QWEN4EXP_GRAPH_BOOTSTRAP_TOKENS.
bool qwen4exp_full_graph_bootstrap(StreamingEngine &engine);

// Number of complete early MoE layers whose expert tensors should be placed
// in the graph CUDA weight arena. Zero keeps all routed experts host-backed.
int qwen4exp_full_graph_device_expert_layers();
// Whether the configured full-graph resident block is at the end of the
// model. Used by usage-driven Option B selection so it does not steal
// residency from the baseline's already-complete GPU layers.
bool qwen4exp_full_graph_device_expert_trailing();

// Build/rebuild the graph for a prompt shape.  This is intended to run before
// the measured prefill window, just as llama.cpp builds its graph before execution.
bool qwen4exp_full_graph_prepare(StreamingEngine &engine, int S,
                                 int pos_base, const int *token_ids);

// Build/reuse the one-token decode graph. This is Qwen-only and requires
// complete preloaded DDR expert arenas for host-resident layers; the configured
// device block is loaded directly into VRAM. KV/GDN/PLE recurrent state is
// seeded once from the existing host state and remains on the device for
// subsequent submissions.
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

// Diagnostic-only two-layer decode-shaped graph. It builds Layers 0 and 1
// with S=1, retains their real inter-layer dependency, and publishes the
// selected boundaries as graph outputs. It is active only when
// MOESTREAM_QWEN4EXP_TWO_LAYER_S1_DIAG=1 and never replaces production
// execution.
bool qwen4exp_two_layer_s1_diag_run(StreamingEngine &engine,
                                    int pos_base, const int *token_ids);

// Destroy the graph while the shared CUDA backend is still alive.  The
// bridge owns scheduler buffers, so leaving this to static destruction runs
// after CudaManager has torn down the driver and makes cudaFree report
// "driver shutting down".
void qwen4exp_full_graph_shutdown();
