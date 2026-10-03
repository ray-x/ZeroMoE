#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct Exl3Mul1MatrixView {
  const uint16_t *trellis = nullptr;
  const uint16_t *suh = nullptr;
  const uint16_t *svh = nullptr;
  const uint16_t *bias = nullptr;
  int input_features = 0;
  int output_features = 0;
  int bits = 0;
};

bool exl3_mul1_reconstruct(const Exl3Mul1MatrixView &matrix, float *weight);

bool exl3_mul1_matmul(const Exl3Mul1MatrixView &matrix, const float *input,
                      float *output, int rows);

std::vector<uint8_t>
exl3_mul1_pack_matrix(const Exl3Mul1MatrixView &matrix);

bool exl3_mul1_view_from_packed(const uint8_t *packed, size_t packed_size,
                                Exl3Mul1MatrixView &matrix);

bool exl3_mul1_matmul_packed(const uint8_t *packed, const float *input,
                             float *output, int rows, int input_features,
                             int output_features);
