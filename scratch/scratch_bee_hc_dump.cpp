// Diagnostic-only harness for the Bee/llama-vs-MoEStream HC divergence
// investigation. Extends scratch_bee_unbatched_logits.cpp's model-loading
// shape (same tensor_buft_overrides, same unbatched S=1 decode loop) with a
// cb_eval hook that dumps layer-0's HC pre-mix intermediates (hc_norm,
// down-proj+SiLU, hc_gate, hc_mixed, hc_inject) for BOTH sub-blocks (attn
// first, ffn second -- graph build order) at selected token positions, in a
// file-naming scheme that lines up with MoEStream's own
// MOESTREAM_GGML_HC_INTERNAL_DUMP output:
//   <out_base>.l00.s<0|1>.call<token_index>.<label>.f32
// label in {normed, lowrank_silu, gate, mixed, inject} -- same as MoEStream.
//
// usage: scratch_bee_hc_dump MODEL TOKEN_FILE OUT_BASE POS1 [POS2 ...]
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

enum class Kind { None, Norm, Gate, Mixed, Inject, Silu, AttnOut, StateIn, Qkv };

static const char *kind_label(Kind k) {
  switch (k) {
    case Kind::Norm: return "normed";
    case Kind::Gate: return "gate";
    case Kind::Mixed: return "mixed";
    case Kind::Inject: return "inject";
    case Kind::Silu: return "lowrank_silu";
    case Kind::AttnOut: return "attn_out";
    case Kind::StateIn: return "state_in";
    case Kind::Qkv: return "qkv";
    default: return "unknown";
  }
}

struct HcDumpState {
  int token_index = -1;
  bool active = false;
  int norm_count = 0;
  int current_sub = -1;
  bool expecting_silu = false;
  Kind pending_kind = Kind::None;
  int pending_sub = -1;
  std::string out_base;
};

static bool hc_dump_cb(struct ggml_tensor *t, bool ask, void *user_data) {
  auto *st = (HcDumpState *)user_data;
  if (!st->active) return false;

  const char *name = t->name;
  static const bool debug_all = std::getenv("BEE_HC_DEBUG_ALL") != nullptr;
  if (debug_all && ask) {
    std::cerr << "[BEE-HC-NODE] tok=" << st->token_index << " name=" << name
              << " op=" << ggml_op_name(t->op) << "\n";
  }
  // Target layer is configurable (BEE_HC_LAYER, default 0): layer 0 folds
  // hc_gate-0/hc_mixed-0 away entirely under this build (every HC stream
  // starts identical at layer 0, since res_hc is a REPEAT of one embedding --
  // ggml's optimizer collapses the now-redundant per-stream gate/mix compute
  // and never materializes those two named nodes). Layer 1 onward has no
  // such degeneracy and exposes all five intermediates normally.
  static const int target_layer = [] {
    const char *v = std::getenv("BEE_HC_LAYER");
    return v ? std::atoi(v) : 0;
  }();
  static const std::string norm_name = "hc_norm-" + std::to_string(target_layer);
  static const std::string gate_name = "hc_gate-" + std::to_string(target_layer);
  static const std::string mixed_name = "hc_mixed-" + std::to_string(target_layer);
  static const std::string inject_name = "hc_inject-" + std::to_string(target_layer);
  // The GDN/linear-attention output that feeds build_hc_combine's block_out
  // -- same quantity as MoEStream's own "attn_out" boundary dump.
  static const std::string attn_out_name =
      "linear_attn_out-" + std::to_string(target_layer);
  // The GDN recurrent state as it enters THIS step (Bee's "state_predelta"),
  // same quantity as MoEStream's "state_in" GDN-diag dump. Walking this
  // across token positions finds the first step where the two engines'
  // carried recurrent state actually diverges, rather than where a
  // downstream nonlinearity merely makes an existing gap visible.
  static const std::string state_in_name =
      "state_predelta-" + std::to_string(target_layer);
  static const std::string qkv_name =
      "linear_attn_qkv_mixed-" + std::to_string(target_layer);
  Kind kind = Kind::None;
  if (name == norm_name) kind = Kind::Norm;
  else if (name == gate_name) kind = Kind::Gate;
  else if (name == mixed_name) kind = Kind::Mixed;
  else if (name == inject_name) kind = Kind::Inject;
  else if (name == attn_out_name) kind = Kind::AttnOut;
  else if (name == state_in_name) kind = Kind::StateIn;
  else if (name == qkv_name) kind = Kind::Qkv;
  else if (st->expecting_silu && t->op == GGML_OP_UNARY &&
           ggml_get_unary_op(t) == GGML_UNARY_OP_SILU)
    kind = Kind::Silu;

  if (kind == Kind::None) return false;

  if (ask) {
    if (kind == Kind::Norm) {
      st->current_sub = st->norm_count++;
      st->expecting_silu = true;
    } else if (kind == Kind::Silu) {
      st->expecting_silu = false;
    }
    st->pending_kind = kind;
    st->pending_sub = st->current_sub;
    return true;
  }

  // ask == false: the node has just been computed; fetch and dump it.
  if (t->type != GGML_TYPE_F32) {
    std::cerr << "[BEE-HC-DUMP] skip non-F32 tensor " << name << " type="
              << t->type << "\n";
    return true;
  }
  const size_t n = (size_t)ggml_nelements(t);
  std::vector<float> host(n);
  if (ggml_backend_buffer_is_host(t->buffer)) {
    std::memcpy(host.data(), t->data, n * sizeof(float));
  } else {
    ggml_backend_tensor_get(t, host.data(), 0, n * sizeof(float));
  }
  char path[512];
  std::snprintf(path, sizeof(path), "%s.l%02d.s%d.call%d.%s.f32",
                st->out_base.c_str(), target_layer, st->pending_sub,
                st->token_index, kind_label(st->pending_kind));
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (out) {
    out.write(reinterpret_cast<const char *>(host.data()),
              (std::streamsize)(n * sizeof(float)));
  }
  std::cerr << "[BEE-HC-DUMP] " << path << " elements=" << n << "\n";
  return true;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "usage: scratch_bee_hc_dump MODEL TOKEN_FILE OUT_BASE POS1 "
                 "[POS2 ...]\n";
    return 2;
  }

  std::ifstream in(argv[2]);
  if (!in) return 3;
  std::vector<llama_token> tokens;
  int token = 0;
  while (in >> token) tokens.push_back(token);
  if (tokens.empty()) return 4;

  std::set<int> want_positions;
  for (int i = 4; i < argc; ++i) want_positions.insert(std::atoi(argv[i]));

  llama_backend_init();
  ggml_backend_load_all();

  // A mixed CPU/GPU split combined with cb_eval segfaults deep in
  // ggml_graph_compute_thread: ggml_backend_sched slices a CPU split into
  // per-node ggml_graph_view()s whenever callback_eval is set (see
  // ggml-backend.cpp's "fast path without callback_eval" branch), and this
  // build's CUDA-graph-capturing decode path does not tolerate that when a
  // CPU split coexists with a captured CUDA graph. Run fully on CPU instead
  // -- no cross-backend split, no CUDA graph capture, no crash -- since the
  // HC math itself is backend-independent (only float rounding differs, at
  // the ~1e-6 level GDN already showed, far below the ~1e-3 HC deltas this
  // harness exists to check).
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  llama_model *model = llama_model_load_from_file(argv[1], mp);
  if (!model) return 5;

  HcDumpState state;
  state.out_base = argv[3];

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 512;
  cp.n_batch = 1;
  cp.n_ubatch = 1;
  cp.n_outputs_max = 1;
  cp.n_threads = 12;
  cp.n_threads_batch = 12;
  cp.cb_eval = hc_dump_cb;
  cp.cb_eval_user_data = &state;
  llama_context *ctx = llama_init_from_model(model, cp);
  if (!ctx) return 6;

  for (size_t i = 0; i < tokens.size(); ++i) {
    state.token_index = (int)i;
    state.active = want_positions.count((int)i) > 0;
    state.norm_count = 0;
    state.current_sub = -1;
    state.expecting_silu = false;
    state.pending_kind = Kind::None;
    state.pending_sub = -1;

    llama_token one = tokens[i];
    llama_batch batch = llama_batch_get_one(&one, 1);
    if (llama_decode(ctx, batch) != 0) return 7;
  }

  llama_synchronize(ctx);
  std::cerr << "tokens=" << tokens.size() << " done\n";
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
