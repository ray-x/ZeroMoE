#pragma once

#include <expected>
#include <string>

struct Exl3DecodeOptions {
  std::string model_path;
  std::string metadata_model_path;
  std::string prompt;
  int context_size = 256;
  int max_new_tokens = 8;
  int threads = 4;
  float temperature = 0.0f;
  float top_p = 0.9f;
  bool raw_prompt = false;
  bool enable_thinking = false;
};

std::expected<void, std::string> run_exl3_decode(const Exl3DecodeOptions &options);
