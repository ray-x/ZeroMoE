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
  bool gpu = false;
  bool gpu_hc = false;
  int gpu_device = 0;
  double gpu_cache_gb = 8.0;
  int gpu_expert_layer_begin = 0;
  int gpu_expert_split_layer = -1;
  int gpu_split_device = 1;
  double gpu_split_cache_gb = 8.0;
  double cpu_expert_cache_gb = 24.0;
  bool cpu_expert_int8_vnni = false;
};

std::expected<void, std::string> run_exl3_decode(const Exl3DecodeOptions &options);
