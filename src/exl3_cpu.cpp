#include "exl3_cpu.hpp"

#include "kernels_cpu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kMul1Multiplier = 0x83DCD12Du;
constexpr float kHadamardScale = 0.08838834764831845f;
constexpr uint64_t kPackedMagic = 0x314c554d334c5845ull;

struct Exl3PackedMatrixHeader {
  uint64_t magic;
  uint64_t total_size;
  uint64_t trellis_offset;
  uint64_t suh_offset;
  uint64_t svh_offset;
  uint64_t bias_offset;
  uint32_t version;
  uint32_t input_features;
  uint32_t output_features;
  uint32_t bits;
  uint32_t flags;
  uint32_t reserved[3];
};

constexpr uint32_t kPackedHasBias = 1u;

size_t align64(size_t value) { return (value + 63u) & ~size_t(63u); }

constexpr std::array<uint16_t, 256> make_tensor_core_permutation() {
  std::array<uint16_t, 256> permutation{};
  for (int thread = 0; thread < 32; ++thread) {
    const int row0 = (thread % 4) * 2;
    const int row1 = row0 + 1;
    const int row2 = row0 + 8;
    const int row3 = row0 + 9;
    const int column0 = thread / 4;
    const int column1 = column0 + 8;
    permutation[thread * 8 + 0] = row0 * 16 + column0;
    permutation[thread * 8 + 1] = row1 * 16 + column0;
    permutation[thread * 8 + 2] = row2 * 16 + column0;
    permutation[thread * 8 + 3] = row3 * 16 + column0;
    permutation[thread * 8 + 4] = row0 * 16 + column1;
    permutation[thread * 8 + 5] = row1 * 16 + column1;
    permutation[thread * 8 + 6] = row2 * 16 + column1;
    permutation[thread * 8 + 7] = row3 * 16 + column1;
  }
  return permutation;
}

constexpr auto kTensorCorePermutation = make_tensor_core_permutation();

uint32_t load_u32(const uint16_t *source, int index) {
  uint32_t value = 0;
  std::memcpy(&value, source + index * 2, sizeof(value));
  return value;
}

template <int Bits>
uint16_t decode_state(const uint16_t *packed, int tensor_core_offset) {
  constexpr int words32 = Bits * 256 / 32;
  const int bit_begin =
      tensor_core_offset * Bits + Bits - 16 + 256 * Bits;
  const int bit_end = bit_begin + 16;
  const int shift = ((bit_end - 1) / 32 + 1) * 32 - bit_end;
  const uint64_t merged =
      (static_cast<uint64_t>(
           load_u32(packed, (bit_begin / 32) % words32))
       << 32) |
      load_u32(packed, ((bit_end - 1) / 32) % words32);
  return static_cast<uint16_t>(merged >> shift);
}

float decode_mul1(uint16_t state) {
  static const auto codebook = [] {
    std::array<float, 65536> values{};
    const float inverse = fp16_to_fp32(0x1eee);
    const float bias = fp16_to_fp32(0xc931);
    for (uint32_t index = 0; index < values.size(); ++index) {
      const uint32_t product = index * kMul1Multiplier;
      const int sum = (product & 255u) + ((product >> 8) & 255u) +
                      ((product >> 16) & 255u) + (product >> 24);
      values[index] = fp16_to_fp32(fp32_to_fp16(
          std::fma(static_cast<float>(1024 + sum), inverse, bias)));
    }
    return values;
  }();
  return codebook[state];
}

void hadamard_128(float *values) {
  for (int width = 1; width < 128; width *= 2) {
    for (int base = 0; base < 128; base += 2 * width) {
      for (int index = 0; index < width; ++index) {
        const float left = values[base + index];
        const float right = values[base + width + index];
        values[base + index] = left + right;
        values[base + width + index] = left - right;
      }
    }
  }
}

bool valid_matrix(const Exl3Mul1MatrixView &matrix) {
  return matrix.trellis && matrix.suh && matrix.svh && matrix.bits >= 1 &&
         matrix.bits <= 8 && matrix.input_features > 0 &&
         matrix.output_features > 0 && matrix.input_features % 128 == 0 &&
         matrix.output_features % 128 == 0;
}

template <int Bits>
void decode_rotated_weight(const Exl3Mul1MatrixView &matrix, float *weight) {
  const int input_tiles = matrix.input_features / 16;
  const int output_tiles = matrix.output_features / 16;
  constexpr int packed_words = 16 * Bits;
  decode_mul1(0);
  #pragma omp parallel for schedule(static)
  for (int input_tile = 0; input_tile < input_tiles; ++input_tile) {
    for (int output_tile = 0; output_tile < output_tiles; ++output_tile) {
      const uint16_t *packed =
          matrix.trellis +
          (static_cast<size_t>(input_tile) * output_tiles + output_tile) *
              packed_words;
      for (int offset = 0; offset < 256; ++offset) {
        const int row_major = kTensorCorePermutation[offset];
        const int row = input_tile * 16 + row_major / 16;
        const int column = output_tile * 16 + row_major % 16;
        weight[static_cast<size_t>(row) * matrix.output_features + column] =
            decode_mul1(decode_state<Bits>(packed, offset));
      }
    }
  }
}

template <int Bits>
void multiply_rotated(const Exl3Mul1MatrixView &matrix, const float *input,
                      float *output, int rows) {
  const int input_tiles = matrix.input_features / 16;
  const int output_tiles = matrix.output_features / 16;
  constexpr int packed_words = 16 * Bits;
  decode_mul1(0);
  #pragma omp parallel for collapse(2) schedule(static)
  for (int row_index = 0; row_index < rows; ++row_index) {
    for (int output_tile = 0; output_tile < output_tiles; ++output_tile) {
      float accumulator[16]{};
      for (int input_tile = 0; input_tile < input_tiles; ++input_tile) {
        const uint16_t *packed =
            matrix.trellis +
            (static_cast<size_t>(input_tile) * output_tiles + output_tile) *
                packed_words;
        float tile[256];
        for (int offset = 0; offset < 256; ++offset)
          tile[kTensorCorePermutation[offset]] =
              decode_mul1(decode_state<Bits>(packed, offset));
        const float *input_tile_data =
            input + static_cast<size_t>(row_index) * matrix.input_features +
            input_tile * 16;
        for (int input_row = 0; input_row < 16; ++input_row) {
          for (int column = 0; column < 16; ++column) {
            accumulator[column] +=
                input_tile_data[input_row] * tile[input_row * 16 + column];
          }
        }
      }
      std::memcpy(output +
                      static_cast<size_t>(row_index) * matrix.output_features +
                      output_tile * 16,
                  accumulator, sizeof(accumulator));
    }
  }
}

template <typename Function>
bool dispatch_bits(int bits, Function &&function) {
  switch (bits) {
  case 1:
    function.template operator()<1>();
    return true;
  case 2:
    function.template operator()<2>();
    return true;
  case 3:
    function.template operator()<3>();
    return true;
  case 4:
    function.template operator()<4>();
    return true;
  case 5:
    function.template operator()<5>();
    return true;
  case 6:
    function.template operator()<6>();
    return true;
  case 7:
    function.template operator()<7>();
    return true;
  case 8:
    function.template operator()<8>();
    return true;
  default:
    return false;
  }
}

} // namespace

bool exl3_mul1_reconstruct(const Exl3Mul1MatrixView &matrix, float *weight) {
  if (!valid_matrix(matrix) || !weight)
    return false;
  const bool decoded = dispatch_bits(matrix.bits, [&]<int Bits>() {
    decode_rotated_weight<Bits>(matrix, weight);
  });
  if (!decoded)
    return false;

  std::vector<float> transform(128);
  for (int input_block = 0; input_block < matrix.input_features;
       input_block += 128) {
    for (int column = 0; column < matrix.output_features; ++column) {
      for (int row = 0; row < 128; ++row)
        transform[row] =
            weight[static_cast<size_t>(input_block + row) *
                       matrix.output_features +
                   column];
      hadamard_128(transform.data());
      for (int row = 0; row < 128; ++row) {
        weight[static_cast<size_t>(input_block + row) *
                   matrix.output_features +
               column] = transform[row] * kHadamardScale *
                         fp16_to_fp32(matrix.suh[input_block + row]);
      }
    }
  }

  for (int row = 0; row < matrix.input_features; ++row) {
    float *weight_row =
        weight + static_cast<size_t>(row) * matrix.output_features;
    for (int output_block = 0; output_block < matrix.output_features;
         output_block += 128) {
      hadamard_128(weight_row + output_block);
      for (int column = 0; column < 128; ++column) {
        weight_row[output_block + column] *=
            kHadamardScale *
            fp16_to_fp32(matrix.svh[output_block + column]);
      }
    }
  }
  return true;
}

bool exl3_mul1_matmul(const Exl3Mul1MatrixView &matrix, const float *input,
                      float *output, int rows) {
  if (!valid_matrix(matrix) || !input || !output || rows <= 0)
    return false;

  std::vector<float> transformed_input(
      static_cast<size_t>(rows) * matrix.input_features);
  for (int row = 0; row < rows; ++row) {
    float *destination = transformed_input.data() +
                         static_cast<size_t>(row) * matrix.input_features;
    const float *source =
        input + static_cast<size_t>(row) * matrix.input_features;
    for (int input_block = 0; input_block < matrix.input_features;
         input_block += 128) {
      for (int index = 0; index < 128; ++index) {
        destination[input_block + index] =
            source[input_block + index] *
            fp16_to_fp32(matrix.suh[input_block + index]);
      }
      hadamard_128(destination + input_block);
      for (int index = 0; index < 128; ++index)
        destination[input_block + index] *= kHadamardScale;
    }
  }

  const bool multiplied = dispatch_bits(matrix.bits, [&]<int Bits>() {
    multiply_rotated<Bits>(matrix, transformed_input.data(), output, rows);
  });
  if (!multiplied)
    return false;

  for (int row = 0; row < rows; ++row) {
    float *output_row =
        output + static_cast<size_t>(row) * matrix.output_features;
    for (int output_block = 0; output_block < matrix.output_features;
         output_block += 128) {
      hadamard_128(output_row + output_block);
      for (int index = 0; index < 128; ++index) {
        const int column = output_block + index;
        output_row[column] *=
            kHadamardScale * fp16_to_fp32(matrix.svh[column]);
        if (matrix.bias)
          output_row[column] += fp16_to_fp32(matrix.bias[column]);
      }
    }
  }
  return true;
}

std::vector<uint8_t>
exl3_mul1_pack_matrix(const Exl3Mul1MatrixView &matrix) {
  if (!valid_matrix(matrix))
    return {};
  const size_t trellis_size =
      static_cast<size_t>(matrix.input_features / 16) *
      (matrix.output_features / 16) * 16 * matrix.bits * sizeof(uint16_t);
  const size_t suh_size =
      static_cast<size_t>(matrix.input_features) * sizeof(uint16_t);
  const size_t svh_size =
      static_cast<size_t>(matrix.output_features) * sizeof(uint16_t);
  const size_t bias_size = matrix.bias ? svh_size : 0;

  Exl3PackedMatrixHeader header{};
  header.magic = kPackedMagic;
  header.version = 1;
  header.input_features = static_cast<uint32_t>(matrix.input_features);
  header.output_features = static_cast<uint32_t>(matrix.output_features);
  header.bits = static_cast<uint32_t>(matrix.bits);
  header.flags = matrix.bias ? kPackedHasBias : 0u;
  header.trellis_offset = align64(sizeof(header));
  header.suh_offset = align64(header.trellis_offset + trellis_size);
  header.svh_offset = align64(header.suh_offset + suh_size);
  header.bias_offset =
      matrix.bias ? align64(header.svh_offset + svh_size) : 0;
  header.total_size = align64(matrix.bias ? header.bias_offset + bias_size
                                          : header.svh_offset + svh_size);

  std::vector<uint8_t> packed(static_cast<size_t>(header.total_size));
  std::memcpy(packed.data(), &header, sizeof(header));
  std::memcpy(packed.data() + header.trellis_offset, matrix.trellis,
              trellis_size);
  std::memcpy(packed.data() + header.suh_offset, matrix.suh, suh_size);
  std::memcpy(packed.data() + header.svh_offset, matrix.svh, svh_size);
  if (matrix.bias)
    std::memcpy(packed.data() + header.bias_offset, matrix.bias, bias_size);
  return packed;
}

bool exl3_mul1_view_from_packed(const uint8_t *packed, size_t packed_size,
                                Exl3Mul1MatrixView &matrix) {
  if (!packed || packed_size < sizeof(Exl3PackedMatrixHeader))
    return false;
  Exl3PackedMatrixHeader header{};
  std::memcpy(&header, packed, sizeof(header));
  if (header.magic != kPackedMagic || header.version != 1 ||
      header.total_size > packed_size || header.input_features == 0 ||
      header.output_features == 0 || header.bits < 1 || header.bits > 8)
    return false;

  const size_t trellis_size =
      static_cast<size_t>(header.input_features / 16) *
      (header.output_features / 16) * 16 * header.bits * sizeof(uint16_t);
  const size_t suh_size =
      static_cast<size_t>(header.input_features) * sizeof(uint16_t);
  const size_t svh_size =
      static_cast<size_t>(header.output_features) * sizeof(uint16_t);
  const auto in_bounds = [&](uint64_t offset, size_t size) {
    return offset <= header.total_size &&
           size <= static_cast<size_t>(header.total_size - offset);
  };
  if (!in_bounds(header.trellis_offset, trellis_size) ||
      !in_bounds(header.suh_offset, suh_size) ||
      !in_bounds(header.svh_offset, svh_size))
    return false;
  if ((header.flags & kPackedHasBias) &&
      !in_bounds(header.bias_offset, svh_size))
    return false;

  matrix.trellis =
      reinterpret_cast<const uint16_t *>(packed + header.trellis_offset);
  matrix.suh = reinterpret_cast<const uint16_t *>(packed + header.suh_offset);
  matrix.svh = reinterpret_cast<const uint16_t *>(packed + header.svh_offset);
  matrix.bias = (header.flags & kPackedHasBias)
                    ? reinterpret_cast<const uint16_t *>(packed +
                                                         header.bias_offset)
                    : nullptr;
  matrix.input_features = static_cast<int>(header.input_features);
  matrix.output_features = static_cast<int>(header.output_features);
  matrix.bits = static_cast<int>(header.bits);
  return valid_matrix(matrix);
}

bool exl3_mul1_matmul_packed(const uint8_t *packed, const float *input,
                             float *output, int rows, int input_features,
                             int output_features) {
  if (!packed)
    return false;
  Exl3PackedMatrixHeader header{};
  std::memcpy(&header, packed, sizeof(header));
  Exl3Mul1MatrixView matrix;
  if (!exl3_mul1_view_from_packed(packed,
                                  static_cast<size_t>(header.total_size),
                                  matrix) ||
      matrix.input_features != input_features ||
      matrix.output_features != output_features)
    return false;
  return exl3_mul1_matmul(matrix, input, output, rows);
}
