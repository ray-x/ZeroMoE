#pragma once

// Opt-in GLM-5.3-Flash full graph. The prompt graph and reusable one-token
// decode graph are separate static topologies and can independently fall
// back to the established layer-at-a-time implementation.

struct StreamingEngine;
class GpuTopology;

// Enabled by default for GLM5NEXT; ZEROMOE_GLM5NEXT_FULL_GRAPH=0 selects the legacy layered path.
bool glm5next_full_graph_enabled();
// Ling uses the shared KDA/MLA graph implementation behind an explicit
// opt-in while its full-graph parity is being established.
bool ling_full_graph_enabled();

// Load the graph's host expert source arena during engine startup.  This is
// independent of prompt length; the graph topology is still built lazily for
// the first prompt, but the 70--78 GiB routed-weight source is loaded once and
// retained for later prompt shapes.
bool glm5next_full_graph_bootstrap(StreamingEngine &engine);

// Supply the engine-owned VRAM expert cache before bootstrap.  GLM graph
// tensors alias its resident slots rather than uploading another expert copy.
void glm5next_full_graph_set_vram_cache(GpuTopology *topology);

// Number of complete MoE layers whose three expert projection tensors are
// placed in the graph's CUDA arena.  Zero leaves all expert weights backed by
// pinned/pageable host memory and lets ggml schedule their copies.
int glm5next_full_graph_device_expert_layers();

// Build or reuse a graph for a position-zero prompt of length S.  Preparation
// is deliberately outside the measured prefill window, like llama.cpp's graph
// construction and static weight upload.
bool glm5next_full_graph_prepare(StreamingEngine &engine, int S,
                                 int pos_base, const int *token_ids);

// Execute the prepared graph for a fresh prompt, write the final [S,D]
// activation into hidden_states, and update the engine's existing KDA/MLA
// state buffers.  false means the caller should use the legacy path.
bool glm5next_full_graph_run(StreamingEngine &engine, float *hidden_states,
                             int S, int pos_base, const int *token_ids);

// The batch size (tokens) the last declined prefill build worked out would fit in memory; 0 = none.
int glm5next_full_graph_suggested_batch();

// Build once at the first decode position (seeding from the host prefill
// state), then reuse the same graph and device KDA/DSA state for consecutive
// decode tokens.
bool glm5next_full_graph_decode_prepare(StreamingEngine &engine, int pos_base,
                                        const int *token_ids);
bool glm5next_full_graph_decode_run(StreamingEngine &engine,
                                    float *hidden_states, int pos_base,
                                    const int *token_ids);

// Release scheduler buffers before the shared CUDA backend is torn down.
void glm5next_full_graph_shutdown();
