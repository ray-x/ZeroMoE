// Sanity-check counterpart to scratch_llama_cpp_unbatched_logits.cpp: decodes the
// WHOLE token file in one llama_decode call (single ubatch) instead of one
// token at a time, and dumps final logits the same way. If unbatched
// (token-by-token) decode is correctly carrying GDN/SSM recurrent state,
// this and the unbatched script's output must match closely for a causal
// model. A mismatch means the unbatched harness itself is mismanaging
// recurrent state -- not a genuine MoEStream-vs-llama.cpp difference.
#include "llama.h"
#include "ggml-backend.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "usage: scratch_llama_cpp_batched_logits MODEL TOKEN_FILE OUTPUT\n";
        return 2;
    }

    std::ifstream in(argv[2]);
    if (!in) return 3;
    std::vector<llama_token> tokens;
    int token = 0;
    while (in >> token) tokens.push_back(token);
    if (tokens.empty()) return 4;

    llama_backend_init();
    ggml_backend_load_all();

    std::vector<std::string> patterns;
    std::vector<llama_model_tensor_buft_override> overrides;
    for (int layer = 0; layer < 26; ++layer) {
        patterns.push_back("blk\\." + std::to_string(layer) +
                           "\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
        overrides.push_back({patterns.back().c_str(), ggml_backend_cpu_buffer_type()});
    }
    overrides.push_back({nullptr, nullptr});

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 5;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = (uint32_t)tokens.size();
    cp.n_ubatch = (uint32_t)tokens.size();
    cp.n_outputs_max = (uint32_t)tokens.size();
    cp.n_threads = 12;
    cp.n_threads_batch = 12;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 6;

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    if (llama_decode(ctx, batch) != 0) return 7;

    llama_synchronize(ctx);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits(ctx);
    if (!logits) return 8;
    std::ofstream out(argv[3], std::ios::binary | std::ios::trunc);
    if (!out) return 9;
    out.write(reinterpret_cast<const char *>(logits),
              static_cast<std::streamsize>(vocab * sizeof(float)));
    std::cerr << "tokens=" << tokens.size() << " vocab=" << vocab << "\n";
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
