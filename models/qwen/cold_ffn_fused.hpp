#pragma once

// Fused CPU evaluation of a layer's cold (host-resident) routed experts.
//
// The stock cold branch is a chain of ggml nodes: get_rows (route remap), get_rows (mask), mul,
// MUL_MAT_ID (up), MUL_MAT_ID (gate), SWIGLU, MUL_MAT_ID (down), mul. Each is its own parallel region,
// and MUL_MAT_ID inserts further barriers (quantize the activations, group rows by expert, compute). A decode
// window routes only 2-3 cold experts per layer, so those barriers cost more than the work.
//
// This op does the whole branch in one node with two internal barriers:
//   phase 1  per (expert, 32 rows): gate/up dots, SwiGLU, quantize the activation row slice
//   barrier
//   phase 2  per (expert, 16 rows): down dots, scaled by the routing weight
// It calls the same ggml dot kernels with the same row grouping and the same quantizers as MUL_MAT_ID, so the
// result is bit-identical (tests/test_cold_ffn_fused.cpp checks this against the stock graph).
//
// Inputs (all ggml tensors):
//   x      [D, 1, S] f32  layer input (token t at nb[2])
//   ids    [topk, S] i32  global expert ids chosen by the router (row t at nb[1])
//   w      [1, topk, S] f32 router weights
//   remap  [1, E, 1]  i32 remap[e] = e when expert e is cold, -1 when it is resident on the GPU
//   mask   [1, E, 1]  f32 1.0 when cold, 0.0 otherwise
//   gate, up [D, FF, E] and down [FF, D, E] expert tensors in host memory (any strides, Q2_0)
// Output: [D, topk, S] f32, slot (k, t) = down(swiglu(gate x, up x)) * (w * mask) for a cold expert, else 0.

#include "ggml.h"

#include <memory>

namespace qwen4exp_cold {

struct Context;  // opaque; owns scratch and the barrier, one per layer

// True when every expert tensor is a type this op reproduces exactly (currently Q2_0 with Q8_0 activations)
// and the shapes satisfy its alignment assumptions.
bool supported(const ggml_tensor *gate, const ggml_tensor *up, const ggml_tensor *down, int topk, int n_tokens);

// Builds the node. `ctx_owner` receives the scratch the node refers to and must outlive the graph.
ggml_tensor *build(ggml_context *ctx, std::shared_ptr<Context> &ctx_owner, ggml_tensor *x, ggml_tensor *ids,
                   ggml_tensor *w, ggml_tensor *remap, ggml_tensor *mask, ggml_tensor *gate, ggml_tensor *up,
                   ggml_tensor *down, int max_threads);

}  // namespace qwen4exp_cold
