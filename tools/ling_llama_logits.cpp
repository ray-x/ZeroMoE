// Reference prefill/decode logits for MoEStream's binary dump format.
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static bool set_tokens(llama_batch_ext *batch, const llama_token *tokens,
                       int count, llama_pos first_pos) {
  llama_batch_ext_clear(batch);
  for (int i = 0; i < count; ++i) {
    const int idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
    const llama_pos pos = first_pos + i;
    if (idx < 0 || !llama_batch_ext_set_pos(batch, idx, &pos))
      return false;
  }
  return llama_batch_ext_set_output_logits(batch, count - 1, true);
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: ling_llama_logits MODEL PROMPT STEPS OUTPUT\n";
    return 2;
  }
  const std::string prompt = argv[2];
  const int steps = std::stoi(argv[3]);
  if (steps < 1) return 2;
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  mp.use_extra_bufts = false;
  llama_model *model = llama_model_load_from_file(argv[1], mp);
  if (!model) return 1;
  const llama_vocab *vocab = llama_model_get_vocab(model);
  const int n_prompt = -llama_tokenize(vocab, prompt.data(), prompt.size(),
                                       nullptr, 0, true, true);
  if (n_prompt < 1) return 1;
  std::vector<llama_token> tokens(n_prompt);
  if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(),
                     n_prompt, true, true) != n_prompt) return 1;
  std::cerr << "prompt ids:";
  for (llama_token id : tokens) std::cerr << ' ' << id;
  std::cerr << '\n';
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = std::max(64, n_prompt + steps + 1);
  cp.n_batch = cp.n_ctx;
  cp.n_ubatch = cp.n_ctx;
  cp.n_threads = 8;
  cp.n_threads_batch = 8;
  llama_context *ctx = llama_init_from_model(model, cp);
  if (!ctx) return 1;
  llama_batch_ext *batch = llama_batch_ext_init(ctx);
  std::ofstream out(argv[4], std::ios::binary | std::ios::trunc);
  if (!batch || !out) return 1;
  const uint32_t vocab_size = static_cast<uint32_t>(llama_vocab_n_tokens(vocab));
  for (int step = 0; step < steps; ++step) {
    if (!set_tokens(batch, tokens.data(), static_cast<int>(tokens.size()),
                    step == 0 ? 0 : n_prompt + step - 1)) return 1;
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) return 1;
    const float *logits = llama_get_logits_ith(ctx, -1);
    if (!logits) return 1;
    const uint32_t step_id = static_cast<uint32_t>(step);
    out.write(reinterpret_cast<const char *>(&step_id), sizeof(step_id));
    out.write(reinterpret_cast<const char *>(&vocab_size), sizeof(vocab_size));
    out.write(reinterpret_cast<const char *>(logits),
              static_cast<std::streamsize>(vocab_size * sizeof(float)));
    const int next = static_cast<int>(
        std::max_element(logits, logits + vocab_size) - logits);
    std::cerr << "step " << step << " next " << next << '\n';
    tokens.assign(1, next);
  }
  llama_batch_ext_free(batch);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
