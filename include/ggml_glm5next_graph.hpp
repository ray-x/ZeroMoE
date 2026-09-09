#pragma once

// Opt-in GLM-5.3-Flash prefill graph.  The bridge is separate from the
// established layer-at-a-time implementation so graph bring-up can fall back
// without changing decode or other model families.

struct StreamingEngine;

// Enabled only when MOESTREAM_GLM5NEXT_FULL_GRAPH=1 is present.  The graph is
// prefill-only while recurrent and absorbed-MLA state writeback is validated.
bool glm5next_full_graph_enabled();

// Load the graph's host expert source arena during engine startup.  This is
// independent of prompt length; the graph topology is still built lazily for
// the first prompt, but the 70--78 GiB routed-weight source is loaded once and
// retained for later prompt shapes.
bool glm5next_full_graph_bootstrap(StreamingEngine &engine);

// Number of complete MoE layers whose three expert projection tensors are
// placed in the graph's CUDA arena.  Zero leaves all expert weights backed by
// pinned/pageable host memory and lets ggml schedule their copies.
int glm5next_full_graph_device_expert_layers();

// Build or reuse a graph for a position-zero prompt of length S.  Preparation
// is deliberately outside the measured prefill window, like Bee's graph
// construction and static weight upload.
bool glm5next_full_graph_prepare(StreamingEngine &engine, int S,
                                 int pos_base, const int *token_ids);

// Execute the prepared graph for a fresh prompt, write the final [S,D]
// activation into hidden_states, and update the engine's existing KDA/MLA
// state buffers.  false means the caller should use the legacy path.
bool glm5next_full_graph_run(StreamingEngine &engine, float *hidden_states,
                             int S, int pos_base, const int *token_ids);

// Release scheduler buffers before the shared CUDA backend is torn down.
void glm5next_full_graph_shutdown();
