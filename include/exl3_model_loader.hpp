#pragma once

#include "safetensors_adapter.hpp"

#include <cstdint>
#include <string>
#include <vector>

bool load_exl3_mul1_matrix(const SafetensorsModelInfo &model,
                           const Exl3LinearInfo &linear,
                           std::vector<uint8_t> &packed,
                           std::string &error);

bool read_exl3_tensor(const SafetensorsModelInfo &model, const std::string &name,
                      std::vector<uint8_t> &bytes, std::string &error);
std::string exl3_qwen_tensor_name(const std::string &name);
