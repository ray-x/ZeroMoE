// Batched-mode counterpart to scratch_llama_cpp_hc_dump.cpp's AttnOut capture.
// Decodes the WHOLE token file in ONE llama_decode call (matching
// MoEStream's normal, non-unbatched prefill) and dumps GDN's per-token
// "linear_attn_out-<layer>" column for each requested position, so it can be
// diffed against MoEStream's own now-batched-capable attn_out boundary dump
// (<out_base>.l%02d.p%d.attn_out.f32, same as MoEStream's own naming).
//
// usage: scratch_llama_cpp_batched_attn_out MODEL TOKEN_FILE OUT_BASE LAYER POS1 [POS2 ...]
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

struct DumpState {
  std::string out_base;
  int layer = 0;
  std::set<int> positions;
};

static bool attn_out_cb(struct ggml_tensor *t, bool ask, void *user_data) {
  auto *st = (DumpState *)user_data;
  static const std::string target =
      "linear_attn_out-" + std::to_string(st->layer);
  if (t->name != target) return false;
  if (ask) return true;

  if (t->type != GGML_TYPE_F32) {
    std::cerr << "[LLAMA-CPP-BATCHED-ATTN-OUT] unexpected type=" << t->type << "\n";
    return true;
  }
  const int64_t d_model = t->ne[0];
  const int64_t n_tokens = t->ne[1];
  const size_t n = (size_t)ggml_nelements(t);
  std::vector<float> host(n);
  if (ggml_backend_buffer_is_host(t->buffer)) {
    std::memcpy(host.data(), t->data, n * sizeof(float));
  } else {
    ggml_backend_tensor_get(t, host.data(), 0, n * sizeof(float));
  }
  for (int pos : st->positions) {
    if (pos < 0 || pos >= n_tokens) continue;
    char path[512];
    std::snprintf(path, sizeof(path), "%s.l%02d.p%d.attn_out.f32",
                  st->out_base.c_str(), st->layer, pos);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (out)
      out.write(reinterpret_cast<const char *>(host.data() +
                                                (size_t)pos * d_model),
                (std::streamsize)(d_model * sizeof(float)));
    std::cerr << "[LLAMA-CPP-BATCHED-ATTN-OUT] " << path << " elements=" << d_model
              << "\n";
  }
  return true;
}

int main(int argc, char **argv) {
  if (argc < 6) {
    std::cerr << "usage: scratch_llama_cpp_batched_attn_out MODEL TOKEN_FILE "
                 "OUT_BASE LAYER POS1 [POS2 ...]\n";
    return 2;
  }

  std::ifstream in(argv[2]);
  if (!in) return 3;
  std::vector<llama_token> tokens;
  int token = 0;
  while (in >> token) tokens.push_back(token);
  if (tokens.empty()) return 4;

  DumpState state;
  state.out_base = argv[3];
  state.layer = std::atoi(argv[4]);
  for (int i = 5; i < argc; ++i) state.positions.insert(std::atoi(argv[i]));

  llama_backend_init();
  ggml_backend_load_all();

  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  llama_model *model = llama_model_load_from_file(argv[1], mp);
  if (!model) return 5;

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 512;
  cp.n_batch = (uint32_t)tokens.size();
  cp.n_ubatch = (uint32_t)tokens.size();
  cp.n_outputs_max = (uint32_t)tokens.size();
  cp.n_threads = 12;
  cp.n_threads_batch = 12;
  cp.cb_eval = attn_out_cb;
  cp.cb_eval_user_data = &state;
  llama_context *ctx = llama_init_from_model(model, cp);
  if (!ctx) return 6;

  llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
  if (llama_decode(ctx, batch) != 0) return 7;

  llama_synchronize(ctx);
  std::cerr << "tokens=" << tokens.size() << " done\n";
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
