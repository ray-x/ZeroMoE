#include "exl3_decode.hpp"
#include "cpu_model_backend.hpp"

#include <cmath>
#include <chrono>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

std::expected<void, std::string> run_exl3_decode(const Exl3DecodeOptions &options) {
  try {
#ifdef _OPENMP
    omp_set_num_threads(std::max(1, options.threads));
#endif
    CPUEngine engine;
    engine.config.model_path = options.model_path;
    engine.config.model_format = "exl3";
    engine.config.metadata_model_path = options.metadata_model_path;
    if (!engine.build_from_exl3(options.model_path, options.metadata_model_path,
                                options.context_size))
      return std::unexpected("EXL3 Qwen backend initialization failed");
    std::string prompt = options.prompt;
    if (prompt.empty()) {
      std::cout << "Prompt: " << std::flush;
      if (!std::getline(std::cin, prompt) || prompt.empty())
        return std::unexpected("EXL3 decode needs a non-empty prompt");
    }
    std::vector<int> tokens;
    const auto append_text = [&](const std::string &text) {
      const auto encoded = engine.tokenizer.encode(text);
      tokens.insert(tokens.end(), encoded.begin(), encoded.end());
    };
    if (!options.raw_prompt) {
      if (!engine.tokenizer.has_chatml_template())
        return std::unexpected("EXL3 Qwen tokenizer has no ChatML tokens");
      tokens.push_back(engine.tokenizer.im_start_id);
      append_text("user\n" + prompt);
      tokens.push_back(engine.tokenizer.im_end_id);
      append_text("\n");
      tokens.push_back(engine.tokenizer.im_start_id);
      append_text("assistant\n");
      if (engine.tokenizer.ds_think_start_id < 0 || engine.tokenizer.ds_think_end_id < 0)
        return std::unexpected("EXL3 Qwen tokenizer lacks thinking markers");
      tokens.push_back(engine.tokenizer.ds_think_start_id);
      append_text(options.enable_thinking ? "\n" : "\n\n");
      if (!options.enable_thinking) {
        tokens.push_back(engine.tokenizer.ds_think_end_id);
        append_text("\n\n");
      }
    } else append_text(prompt);
    if (tokens.empty() || options.max_new_tokens < 0 ||
        tokens.size() + static_cast<size_t>(options.max_new_tokens) >
            static_cast<size_t>(options.context_size))
      return std::unexpected("EXL3 prompt and decode exceed context capacity");
    const auto &cfg = engine.model_file.architecture_config;
    std::vector<float> hidden(cfg.hidden), logits(cfg.vocab);
    const auto forward = [&](int token, int position) {
      engine.qwen4exp_ple_token_history.push_back(token);
      if (!cpu_embed_token(engine, token, hidden.data(), cfg.hidden))
        throw std::runtime_error("EXL3 token embedding failed");
      for (int layer = 0; layer < cfg.n_layers; ++layer) {
        if (!cpu_forward_layer(engine, layer, hidden.data(), 1, position))
          throw std::runtime_error("EXL3 forward failed at layer " + std::to_string(layer));
        for (float value : engine.qwen4exp_hc_stream_buf)
          if (!std::isfinite(value))
            throw std::runtime_error("Non-finite EXL3 layer " + std::to_string(layer));
        if (position == 0 && layer % 8 == 0)
          LOG_STREAM_INFO() << "[EXL3] Initial token layer " << layer << " complete\n";
      }
    };
    const auto start = std::chrono::steady_clock::now();
    LOG_STREAM_INFO() << "[EXL3] Prefill " << tokens.size() << " tokens\n";
    for (size_t position = 0; position < tokens.size(); ++position) {
      forward(tokens[position], static_cast<int>(position));
      LOG_STREAM_INFO() << "[EXL3] Prefill position " << position << " complete\n";
    }
    CPUSampler sampler;
    std::cout << "\n";
    for (int step = 0; step < options.max_new_tokens; ++step) {
      if (!cpu_lm_head(engine, hidden.data(), logits.data(), 1))
        throw std::runtime_error("EXL3 output head failed");
      if (!std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("Non-finite EXL3 logits");
      const int token = sampler.sample(logits, options.temperature, options.top_p);
      LOG_STREAM_INFO() << "[EXL3] Decode step=" << step << " token=" << token
                        << " logit=" << logits[token] << "\n";
      if (engine.tokenizer.is_stop_token(token)) break;
      std::cout << engine.tokenizer.decode(token) << std::flush;
      if (step + 1 < options.max_new_tokens)
        forward(token, static_cast<int>(tokens.size()) + step);
    }
    std::cout << "\n";
    LOG_STREAM_INFO() << "[EXL3] Decode completed in "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                      << " s; resident tensors=" << engine.resident_layers.size() << "\n";
    return {};
  } catch (const std::exception &error) {
    return std::unexpected(error.what());
  }
}
