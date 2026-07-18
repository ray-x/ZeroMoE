#include "llama.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: scratch_decode_tokens MODEL TOKEN_FILE\n";
        return 2;
    }

    std::ifstream in(argv[2]);
    if (!in) return 3;

    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], params);
    if (!model) return 4;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::ostringstream out;
    int token = 0;
    bool first = true;
    while (in >> token) {
        char buf[4096];
        const int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n < 0) return 5;
        if (!first) out << "";
        out.write(buf, n);
        first = false;
    }
    std::cout << out.str();
    llama_model_free(model);
    return 0;
}
