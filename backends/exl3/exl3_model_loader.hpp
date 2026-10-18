#pragma once

#include "safetensors_adapter.hpp"
#include "exl3_cpu.hpp"

#include <cstdint>
#include <memory>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct Exl3Mul1MatrixData {
  std::vector<uint8_t> trellis;
  std::vector<uint8_t> suh;
  std::vector<uint8_t> svh;
  std::vector<uint8_t> bias;
  int input_features = 0;
  int output_features = 0;
  int bits = 0;

  size_t byte_size() const {
    return trellis.size() + suh.size() + svh.size() + bias.size();
  }

  Exl3Mul1MatrixView view() const {
    return {reinterpret_cast<const uint16_t *>(trellis.data()),
            reinterpret_cast<const uint16_t *>(suh.data()),
            reinterpret_cast<const uint16_t *>(svh.data()),
            bias.empty() ? nullptr : reinterpret_cast<const uint16_t *>(bias.data()),
            input_features, output_features, bits};
  }
};

bool load_exl3_mul1_matrix(const SafetensorsModelInfo &model,
                           const Exl3LinearInfo &linear,
                           std::vector<uint8_t> &packed,
                           std::string &error);
bool load_exl3_mul1_components(const SafetensorsModelInfo &model,
                               const Exl3LinearInfo &linear,
                               Exl3Mul1MatrixData &matrix,
                               std::string &error);

struct Exl3CpuMatrixCacheStats {
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t evictions = 0;
  size_t resident_bytes = 0;
};

class Exl3CpuMatrixCache {
public:
  explicit Exl3CpuMatrixCache(size_t budget_bytes);
  ~Exl3CpuMatrixCache();
  Exl3CpuMatrixCache(const Exl3CpuMatrixCache &) = delete;
  Exl3CpuMatrixCache &operator=(const Exl3CpuMatrixCache &) = delete;
  std::shared_ptr<const Exl3Mul1MatrixData> get(const std::string &key);
  std::shared_ptr<const Exl3Mul1MatrixData> insert(
      const std::string &key, Exl3Mul1MatrixData matrix);
  Exl3CpuMatrixCacheStats stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

bool read_exl3_tensor(const SafetensorsModelInfo &model, const std::string &name,
                      std::vector<uint8_t> &bytes, std::string &error);
std::string exl3_qwen_tensor_name(const std::string &name);

class Exl3NgramReader {
public:
  explicit Exl3NgramReader(const SafetensorsModelInfo &model);
  ~Exl3NgramReader();
  Exl3NgramReader(const Exl3NgramReader &) = delete;
  Exl3NgramReader &operator=(const Exl3NgramReader &) = delete;
  bool read_row(int head, int64_t row, float *output, int dimensions);
  uint64_t row_reads() const;
  uint64_t cache_hits() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
