#pragma once

// Qwen4Exp full graph.  The implementation is deliberately kept behind this
// small bridge so the per-layer path remains available as an explicit
// correctness fallback, but both graph phases are on by default.

#include <cstddef>

struct StreamingEngine;

// Full-graph prefill and decode are enabled by default.  A TOML profile may
// explicitly set qwen4exp_full_graph/qwen4exp_full_graph_decode to false for
// a documented fallback or diagnostic; there are no environment overrides.
bool qwen4exp_full_graph_enabled();
bool qwen4exp_full_graph_decode_enabled();

// Configure full-graph enablement from the TOML profile.  The defaults are
// true/true; the call is kept so config parsing has one explicit handoff.
void qwen4exp_full_graph_configure(bool enabled, bool decode_enabled,
                                   bool prefill_state_handoff = false);
bool qwen4exp_full_graph_sync_state_to_host(StreamingEngine &engine);

// Startup warmup that loads prompt-independent static weight arenas before the
// first request. Eligible full-graph decode always bootstraps reusable arenas;
// the prompt shape is fixed at up to 16 tokens.
bool qwen4exp_full_graph_bootstrap(StreamingEngine &engine);

// Number of complete early MoE layers whose expert tensors should be placed
// in the graph CUDA weight arena. Zero keeps all routed experts host-backed.
int qwen4exp_full_graph_device_expert_layers();

// Batch size (tokens) the last declined prefill build worked out would fit in memory; 0 = none.
int qwen4exp_full_graph_suggested_batch();
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

// Run a window of S consecutive tokens through the decode graph (built with
// engine.qwen4exp_decode_window == S) in one submission. The caller has already
// written the S HC-expanded embeddings to engine.hc_stream_buf (S * D * HC floats)
// and called qwen4exp_full_graph_decode_prepare(engine, pos_base, token_ids).
// logits receives S * vocab floats, wide S * D * HC (pre-head residual per token).
bool qwen4exp_full_graph_window_run(StreamingEngine &engine, int S, int pos_base,
                                    const int *token_ids, float *logits,
                                    float *wide);

// Keep the first m tokens of the window last run (1 <= m <= S): the device
// state becomes the state after exactly those tokens (per-token GDN snapshots,
// shifted conv/PLE histories; KV needs nothing) and host metadata advances by m.
// Must be called before the next window run. m == S is free.
bool qwen4exp_full_graph_window_commit(StreamingEngine &engine, int m);

// Device-side checkpoints of the decode graph's recurrent state (GDN state and
// conv history, PLE history, plus the host PLE window), for graph-native MTP
// verification. Valid only on the live decode graph that took them; a rebuilt
// graph invalidates every checkpoint. KV is not copied: decode writes only row
// `position` and masks reads to `<= position`, so restoring the position
// restores KV. Device blocks are pooled; release returns them to the pool.
bool qwen4exp_full_graph_state_save(StreamingEngine &engine, int &handle);
bool qwen4exp_full_graph_state_restore(StreamingEngine &engine, int handle);
void qwen4exp_full_graph_state_release(int handle);
// Bytes one checkpoint occupies (0 when there is no live decode graph).
size_t qwen4exp_full_graph_state_bytes(StreamingEngine &engine);

// Destroy the graph while the shared CUDA backend is still alive.  The
// bridge owns scheduler buffers, so leaving this to static destruction runs
// after CudaManager has torn down the driver and makes cudaFree report
// "driver shutting down".
void qwen4exp_full_graph_shutdown();
