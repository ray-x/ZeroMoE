#pragma once
// Fused hyper-connection nodes for the Qwen4Exp graph. The ZeroMoE ggml patch (patches/ggml/0015) executes a GGML_OP_ADD
// node tagged op_params[7] == ZM_HC_MAGIC as one CUDA kernel that reproduces, bit for bit, the node chain it replaces
// (hc_combine: SCALE, SIGMOID, SCALE, REPEAT, MUL, ADD; hc_mix tail: SIGMOID, MUL, CONT, ADD..., SCALE).
// Against a ggml without the patch the weak symbol below is null and the graph builders emit the plain chain.
#include "ggml.h"

#include <cstring>

extern "C" int zeromoe_ggml_cuda_hc_fused_version(void) __attribute__((weak));

namespace zeromoe::graph {

inline bool &zm_hc_scope_flag() {
  static thread_local bool v = false;
  return v;
}

// Active while a graph that runs entirely on the CUDA backend is being built (decode graphs, the MTP draft head).
struct ZmHcScope {
  bool prev;
  explicit ZmHcScope(bool on) : prev(zm_hc_scope_flag()) { zm_hc_scope_flag() = on; }
  ~ZmHcScope() { zm_hc_scope_flag() = prev; }
};

inline bool zm_hc_enabled() { return zm_hc_scope_flag() && zeromoe_ggml_cuda_hc_fused_version != nullptr; }

// kind 0 = combine (output shaped like src a), 1 = mix (output [D, S]).
inline ggml_tensor *zm_hc_node(ggml_context *ctx, int kind, const int64_t ne[GGML_MAX_DIMS], int D, int HC, float k,
                               ggml_tensor *a, ggml_tensor *b, ggml_tensor *c) {
  ggml_tensor *t = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, ne);
  t->op = GGML_OP_ADD;
  t->src[0] = a;
  t->src[1] = b;
  t->src[2] = c;
  t->op_params[0] = kind;
  t->op_params[1] = D;
  t->op_params[2] = HC;
  std::memcpy(&t->op_params[3], &k, sizeof(float));
  t->op_params[7] = 0x5a4d4843;  // "ZMHC"
  return t;
}

}  // namespace zeromoe::graph
