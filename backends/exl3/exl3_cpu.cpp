#include "exl3_cpu.hpp"
#include "exl3_format.hpp"

#include "kernels_cpu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace {

constexpr uint32_t kMul1Multiplier = 0x83DCD12Du;
constexpr float kHadamardScale = 0.08838834764831845f;

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

constexpr std::array<uint16_t, 256> make_inverse_tensor_core_permutation() {
  std::array<uint16_t, 256> inverse{};
  for (int offset = 0; offset < 256; ++offset)
    inverse[kTensorCorePermutation[offset]] = static_cast<uint16_t>(offset);
  return inverse;
}

constexpr auto kInverseTensorCorePermutation =
    make_inverse_tensor_core_permutation();

void hadamard_128(float *values);

#if defined(__x86_64__) && defined(__GNUC__)
template <int Bits, int InputRow>
struct VnniStateIndices {
  static constexpr int kWords = Bits * 8;
  static constexpr std::array<int32_t, 16> first = [] {
    std::array<int32_t, 16> result{};
    for (int column = 0; column < 16; ++column) {
      const int offset = kInverseTensorCorePermutation[InputRow * 16 + column];
      const int bit_begin = (offset + 257) * Bits - 16;
      result[column] = (bit_begin / 32) % kWords;
    }
    return result;
  }();
  static constexpr std::array<int32_t, 16> last = [] {
    std::array<int32_t, 16> result{};
    for (int column = 0; column < 16; ++column) {
      const int offset = kInverseTensorCorePermutation[InputRow * 16 + column];
      const int bit_end = (offset + 257) * Bits;
      result[column] = ((bit_end - 1) / 32) % kWords;
    }
    return result;
  }();
  static constexpr std::array<int32_t, 16> shift = [] {
    std::array<int32_t, 16> result{};
    for (int column = 0; column < 16; ++column) {
      const int offset = kInverseTensorCorePermutation[InputRow * 16 + column];
      const int bit_end = (offset + 257) * Bits;
      result[column] = ((bit_end - 1) / 32 + 1) * 32 - bit_end;
    }
    return result;
  }();
};

struct VnniPackedTile {
  __m512i words[4];
};

template <int Count>
constexpr __mmask16 packed_word_mask() {
  if constexpr (Count >= 16) {
    return 0xffff;
  } else {
    return static_cast<__mmask16>((1u << Count) - 1u);
  }
}

template <int Bits>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c"), always_inline))
inline VnniPackedTile load_vnni_packed_tile(const uint16_t *packed) {
  constexpr int kWordCount = Bits * 8;
  const auto *words = reinterpret_cast<const int *>(packed);
  VnniPackedTile tile;
  tile.words[0] = _mm512_maskz_loadu_epi32(packed_word_mask<kWordCount>(), words);
  if constexpr (kWordCount > 16)
    tile.words[1] = _mm512_maskz_loadu_epi32(packed_word_mask<kWordCount - 16>(), words + 16);
  else
    tile.words[1] = _mm512_setzero_si512();
  if constexpr (kWordCount > 32)
    tile.words[2] = _mm512_maskz_loadu_epi32(packed_word_mask<kWordCount - 32>(), words + 32);
  else
    tile.words[2] = _mm512_setzero_si512();
  if constexpr (kWordCount > 48)
    tile.words[3] = _mm512_maskz_loadu_epi32(packed_word_mask<kWordCount - 48>(), words + 48);
  else
    tile.words[3] = _mm512_setzero_si512();
  return tile;
}

template <int Bits>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c"), always_inline))
inline __m512i select_vnni_words(const VnniPackedTile &tile,
                                  const std::array<int32_t, 16> &table) {
  const auto indices = _mm512_loadu_si512(table.data());
  if constexpr (Bits <= 2)
    return _mm512_permutexvar_epi32(indices, tile.words[0]);
  if constexpr (Bits <= 4)
    return _mm512_permutex2var_epi32(tile.words[0], indices, tile.words[1]);
  const auto within_pair = _mm512_and_si512(indices, _mm512_set1_epi32(31));
  const auto low = _mm512_permutex2var_epi32(tile.words[0], within_pair, tile.words[1]);
  __m512i high;
  if constexpr (Bits <= 6)
    high = _mm512_permutexvar_epi32(within_pair, tile.words[2]);
  else
    high = _mm512_permutex2var_epi32(tile.words[2], within_pair, tile.words[3]);
  const auto high_mask = _mm512_cmpgt_epi32_mask(indices, _mm512_set1_epi32(31));
  return _mm512_mask_blend_epi32(high_mask, low, high);
}

template <int Bits, int InputRow>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c"), always_inline))
inline __m512i mul1_products_vnni(const VnniPackedTile &tile) {
  const auto first = select_vnni_words<Bits>(tile, VnniStateIndices<Bits, InputRow>::first);
  const auto last = select_vnni_words<Bits>(tile, VnniStateIndices<Bits, InputRow>::last);
  const auto shifts = _mm512_loadu_si512(VnniStateIndices<Bits, InputRow>::shift.data());
  const auto merged = _mm512_or_si512(
      _mm512_srlv_epi32(last, shifts),
      _mm512_sllv_epi32(first, _mm512_sub_epi32(_mm512_set1_epi32(32), shifts)));
  const auto states = _mm512_and_si512(merged, _mm512_set1_epi32(0xffff));
  return _mm512_mullo_epi32(states, _mm512_set1_epi32(kMul1Multiplier));
}

template <int Bits, int InputRow>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c"), always_inline))
inline __m512 decode_mul1_vnni(const VnniPackedTile &packed, __m512i ones,
                                __m512 inverse, __m512 codebook_bias) {
  const __m512i products = mul1_products_vnni<Bits, InputRow>(packed);
  const __m512i byte_sums = _mm512_dpbusd_epi32(_mm512_setzero_si512(), products, ones);
  const __m512 sums = _mm512_cvtepi32_ps(
      _mm512_add_epi32(byte_sums, _mm512_set1_epi32(1024)));
  const __m512 code = _mm512_fmadd_ps(sums, inverse, codebook_bias);
  return _mm512_cvtph_ps(_mm512_cvtps_ph(
      code, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

template <int Bits, int InputRow>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c"), always_inline))
inline void accumulate_exact_vnni_rows(const VnniPackedTile &packed,
                                       const float *input, int input_offset,
                                       __m512i ones, __m512 inverse,
                                       __m512 codebook_bias, __m512 &accumulator) {
  if constexpr (InputRow < 16) {
    const __m512 weight = decode_mul1_vnni<Bits, InputRow>(
        packed, ones, inverse, codebook_bias);
    accumulator = _mm512_fmadd_ps(
        weight, _mm512_set1_ps(input[input_offset + InputRow]), accumulator);
    accumulate_exact_vnni_rows<Bits, InputRow + 1>(
        packed, input, input_offset, ones, inverse, codebook_bias, accumulator);
  }
}

template <int Bits, int InputRow>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c"), always_inline))
inline void accumulate_int8_vnni_rows_pair(
    const VnniPackedTile &first, const VnniPackedTile &second,
    const int8_t *input, int input_offset, __m512i &first_accumulator,
    __m512i &second_accumulator) {
  if constexpr (InputRow < 16) {
    const int8_t value = input[input_offset + InputRow];
    const uint32_t replicated = static_cast<uint8_t>(value) * 0x01010101u;
    const auto activation = _mm512_set1_epi32(static_cast<int32_t>(replicated));
    first_accumulator = _mm512_dpbusd_epi32(
        first_accumulator, mul1_products_vnni<Bits, InputRow>(first), activation);
    second_accumulator = _mm512_dpbusd_epi32(
        second_accumulator, mul1_products_vnni<Bits, InputRow>(second), activation);
    accumulate_int8_vnni_rows_pair<Bits, InputRow + 1>(
        first, second, input, input_offset, first_accumulator, second_accumulator);
  }
}

template <int Bits, int InputRow>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c"), always_inline))
inline void accumulate_int8_vnni_rows_quad(
    const VnniPackedTile &first, const VnniPackedTile &second,
    const VnniPackedTile &third, const VnniPackedTile &fourth,
    const int8_t *input, int input_offset, __m512i &first_accumulator,
    __m512i &second_accumulator, __m512i &third_accumulator,
    __m512i &fourth_accumulator) {
  if constexpr (InputRow < 16) {
    const uint32_t replicated =
        static_cast<uint8_t>(input[input_offset + InputRow]) * 0x01010101u;
    const auto activation = _mm512_set1_epi32(static_cast<int32_t>(replicated));
    first_accumulator = _mm512_dpbusd_epi32(
        first_accumulator, mul1_products_vnni<Bits, InputRow>(first), activation);
    second_accumulator = _mm512_dpbusd_epi32(
        second_accumulator, mul1_products_vnni<Bits, InputRow>(second), activation);
    third_accumulator = _mm512_dpbusd_epi32(
        third_accumulator, mul1_products_vnni<Bits, InputRow>(third), activation);
    fourth_accumulator = _mm512_dpbusd_epi32(
        fourth_accumulator, mul1_products_vnni<Bits, InputRow>(fourth), activation);
    accumulate_int8_vnni_rows_quad<Bits, InputRow + 1>(
        first, second, third, fourth, input, input_offset, first_accumulator,
        second_accumulator, third_accumulator, fourth_accumulator);
  }
}

template <int Bits>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")))
void multiply_rotated_vnni(const Exl3Mul1MatrixView &matrix, const float *input,
                           float *output, int rows) {
  const int input_tiles = matrix.input_features / 16;
  const int output_tiles = matrix.output_features / 16;
  constexpr int kOutputTilesPerBand = 8;
  const __m512i ones = _mm512_set1_epi32(0x01010101);
  const __m512 inverse = _mm512_set1_ps(fp16_to_fp32(0x1eee));
  const __m512 codebook_bias = _mm512_set1_ps(fp16_to_fp32(0xc931));
  #pragma omp parallel for collapse(2) schedule(static)
  for (int row = 0; row < rows; ++row) {
    for (int output_tile_begin = 0; output_tile_begin < output_tiles;
         output_tile_begin += kOutputTilesPerBand) {
      __m512 accumulators[kOutputTilesPerBand];
      for (auto &accumulator : accumulators)
        accumulator = _mm512_setzero_ps();
      const float *input_row = input + static_cast<size_t>(row) * matrix.input_features;
      for (int input_tile = 0; input_tile < input_tiles; ++input_tile) {
        for (int band_column = 0; band_column < kOutputTilesPerBand; ++band_column) {
          const size_t tile_index = static_cast<size_t>(input_tile) * output_tiles +
                                    output_tile_begin + band_column;
          const auto tile = load_vnni_packed_tile<Bits>(
              matrix.trellis + tile_index * 16 * Bits);
          accumulate_exact_vnni_rows<Bits, 0>(
              tile, input_row, input_tile * 16, ones, inverse,
              codebook_bias, accumulators[band_column]);
        }
      }
      for (int band_column = 0; band_column < kOutputTilesPerBand; ++band_column)
        _mm512_storeu_ps(output + static_cast<size_t>(row) * matrix.output_features +
                         (output_tile_begin + band_column) * 16,
                         accumulators[band_column]);
    }
  }
}

template <int Bits>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")))
void multiply_rotated_vnni_batch(const Exl3Mul1BatchItem *items, int count,
                                  const size_t *input_offsets,
                                  const size_t *job_offsets,
                                  float *transformed) {
  constexpr int kOutputTilesPerBand = 8;
  #pragma omp parallel for schedule(static)
  for (int job = 0; job < job_offsets[count]; ++job) {
    int item_index = 0;
    while (job >= static_cast<int>(job_offsets[item_index + 1])) ++item_index;
    const auto &item = items[item_index];
    const auto &matrix = item.matrix;
    if (matrix.bits != Bits) continue;
    const int output_band =
        static_cast<int>(job - job_offsets[item_index]) * kOutputTilesPerBand;
    const int input_tiles = matrix.input_features / 16;
    const int output_tiles = matrix.output_features / 16;
    const size_t transformed_offset = input_offsets[item_index];
    const float *rotated_input = transformed + transformed_offset;
    __m512 accumulators[kOutputTilesPerBand];
    for (auto &accumulator : accumulators) accumulator = _mm512_setzero_ps();
    const __m512i ones = _mm512_set1_epi32(0x01010101);
    const __m512 inverse = _mm512_set1_ps(fp16_to_fp32(0x1eee));
    const __m512 codebook_bias = _mm512_set1_ps(fp16_to_fp32(0xc931));
    for (int input_tile = 0; input_tile < input_tiles; ++input_tile) {
      for (int band_column = 0; band_column < kOutputTilesPerBand; ++band_column) {
        const size_t tile_index = static_cast<size_t>(input_tile) * output_tiles +
                                  output_band + band_column;
        const auto tile = load_vnni_packed_tile<Bits>(
            matrix.trellis + tile_index * 16 * Bits);
        accumulate_exact_vnni_rows<Bits, 0>(
            tile, rotated_input, input_tile * 16, ones, inverse,
            codebook_bias, accumulators[band_column]);
      }
    }
    alignas(64) float values[128];
    for (int band_column = 0; band_column < kOutputTilesPerBand; ++band_column)
      _mm512_store_ps(values + band_column * 16, accumulators[band_column]);
    for (int block = 0; block < 128; block += 128) {
      hadamard_128(values + block);
      for (int column = 0; column < 128; ++column) {
        const int output_column = output_band * 16 + block + column;
        float result = values[block + column] * kHadamardScale *
                       fp16_to_fp32(matrix.svh[output_column]);
        if (matrix.bias) result += fp16_to_fp32(matrix.bias[output_column]);
        item.output[output_column] = result;
      }
    }
  }
}

template <int Bits>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")))
void multiply_rotated_int8_vnni_batch(const Exl3Mul1BatchItem *items, int count,
                                       const size_t *input_offsets,
                                       const size_t *job_offsets,
                                       const int8_t *activations,
                                       const float *activation_scales,
                                       const int32_t *activation_sums) {
  constexpr int kOutputTilesPerBand = 8;
  #pragma omp parallel for schedule(static)
  for (int job = 0; job < job_offsets[count]; ++job) {
    int item_index = 0;
    while (job >= static_cast<int>(job_offsets[item_index + 1])) ++item_index;
    const auto &item = items[item_index];
    const auto &matrix = item.matrix;
    if (matrix.bits != Bits) continue;
    const int output_band =
        static_cast<int>(job - job_offsets[item_index]) * kOutputTilesPerBand;
    const int input_tiles = matrix.input_features / 16;
    const int output_tiles = matrix.output_features / 16;
    const int8_t *source = activations + input_offsets[item_index];
    __m512i accumulators[kOutputTilesPerBand];
    for (auto &accumulator : accumulators) accumulator = _mm512_setzero_si512();
    for (int input_tile = 0; input_tile < input_tiles; ++input_tile) {
      constexpr int kTilesPerGroup = Bits <= 4 ? 4 : 2;
      for (int band_column = 0; band_column < kOutputTilesPerBand;
           band_column += kTilesPerGroup) {
        const size_t tile_index = static_cast<size_t>(input_tile) * output_tiles +
                                  output_band + band_column;
        const auto first = load_vnni_packed_tile<Bits>(
            matrix.trellis + tile_index * 16 * Bits);
        const auto second = load_vnni_packed_tile<Bits>(
            matrix.trellis + (tile_index + 1) * 16 * Bits);
        if constexpr (Bits <= 4) {
          const auto third = load_vnni_packed_tile<Bits>(
              matrix.trellis + (tile_index + 2) * 16 * Bits);
          const auto fourth = load_vnni_packed_tile<Bits>(
              matrix.trellis + (tile_index + 3) * 16 * Bits);
          accumulate_int8_vnni_rows_quad<Bits, 0>(
              first, second, third, fourth, source, input_tile * 16,
              accumulators[band_column], accumulators[band_column + 1],
              accumulators[band_column + 2], accumulators[band_column + 3]);
        } else {
          accumulate_int8_vnni_rows_pair<Bits, 0>(
              first, second, source, input_tile * 16,
              accumulators[band_column], accumulators[band_column + 1]);
        }
      }
    }
    const float scale = activation_scales[item_index] * fp16_to_fp32(0x1eee);
    const __m512 scale_vector = _mm512_set1_ps(scale);
    const __m512 correction = _mm512_set1_ps(
        -510.0f * static_cast<float>(activation_sums[item_index]) * scale);
    alignas(64) float values[128];
    for (int band_column = 0; band_column < kOutputTilesPerBand; ++band_column) {
      const __m512 transformed = _mm512_fmadd_ps(
          _mm512_cvtepi32_ps(accumulators[band_column]), scale_vector, correction);
      _mm512_store_ps(values + band_column * 16, transformed);
    }
    hadamard_128(values);
    for (int column = 0; column < 128; ++column) {
      const int output_column = output_band * 16 + column;
      float result = values[column] * kHadamardScale *
                     fp16_to_fp32(matrix.svh[output_column]);
      if (matrix.bias) result += fp16_to_fp32(matrix.bias[output_column]);
      item.output[output_column] = result;
    }
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")))
void multiply_rotated_q8_vnni_batch(const Exl3Mul1Q8BatchItem *items, int count,
                                     const size_t *input_offsets,
                                     const size_t *job_offsets,
                                     const int8_t *activations,
                                     const float *activation_scales,
                                     const int32_t *activation_sums) {
  constexpr int kOutputTilesPerBand = 8;
  #pragma omp parallel for schedule(static)
  for (int job = 0; job < job_offsets[count]; ++job) {
    int item_index = 0;
    while (job >= static_cast<int>(job_offsets[item_index + 1])) ++item_index;
    const auto &item = items[item_index];
    const auto &matrix = item.matrix;
    const int output_band =
        static_cast<int>(job - job_offsets[item_index]) * kOutputTilesPerBand;
    const int input_tiles = matrix.input_features / 16;
    const int output_tiles = matrix.output_features / 16;
    const int8_t *source = activations + input_offsets[item_index];
    __m512i accumulators[kOutputTilesPerBand];
    for (auto &accumulator : accumulators) accumulator = _mm512_setzero_si512();
    for (int input_tile = 0; input_tile < input_tiles; ++input_tile) {
      for (int group = 0; group < 4; ++group) {
        uint32_t activation_bytes = 0;
        std::memcpy(&activation_bytes, source + input_tile * 16 + group * 4,
                    sizeof(activation_bytes));
        const __m512i activation_vector = _mm512_set1_epi32(activation_bytes);
        for (int band_column = 0; band_column < kOutputTilesPerBand; ++band_column) {
          const size_t tile_index = static_cast<size_t>(input_tile) * output_tiles +
                                    output_band + band_column;
          const auto weights = _mm512_loadu_si512(
              matrix.weights + tile_index * 256 + group * 64);
          accumulators[band_column] = _mm512_dpbusd_epi32(
              accumulators[band_column], weights, activation_vector);
        }
      }
    }
    const float scale = activation_scales[item_index] * fp16_to_fp32(0x1eee);
    const __m512 weight_scale = _mm512_set1_ps(scale * 4.0f);
    const __m512 correction = _mm512_set1_ps(
        -510.0f * static_cast<float>(activation_sums[item_index]) * scale);
    alignas(64) float values[128];
    for (int band_column = 0; band_column < kOutputTilesPerBand; ++band_column) {
      const __m512 transformed = _mm512_fmadd_ps(
          _mm512_cvtepi32_ps(accumulators[band_column]), weight_scale, correction);
      _mm512_store_ps(values + band_column * 16, transformed);
    }
    hadamard_128(values);
    for (int column = 0; column < 128; ++column) {
      const int output_column = output_band * 16 + column;
      float result = values[column] * kHadamardScale *
                     fp16_to_fp32(matrix.svh[output_column]);
      if (matrix.bias) result += fp16_to_fp32(matrix.bias[output_column]);
      item.output[output_column] = result;
    }
  }
}

template <int Bits, int InputRow>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c"), always_inline))
inline __m128i predecode_q8_row(const VnniPackedTile &tile, __m512i ones) {
  const __m512i products = mul1_products_vnni<Bits, InputRow>(tile);
  const __m512i byte_sums = _mm512_dpbusd_epi32(
      _mm512_setzero_si512(), products, ones);
  const __m512i rounded = _mm512_srli_epi32(
      _mm512_add_epi32(byte_sums, _mm512_set1_epi32(2)), 2);
  return _mm512_cvtusepi32_epi8(rounded);
}

template <int Bits, int FirstRow>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c"), always_inline))
inline void predecode_q8_group(const VnniPackedTile &tile, __m512i ones,
                               uint8_t *destination) {
  const __m128i row0 = predecode_q8_row<Bits, FirstRow>(tile, ones);
  const __m128i row1 = predecode_q8_row<Bits, FirstRow + 1>(tile, ones);
  const __m128i row2 = predecode_q8_row<Bits, FirstRow + 2>(tile, ones);
  const __m128i row3 = predecode_q8_row<Bits, FirstRow + 3>(tile, ones);
  const __m128i pair01_low = _mm_unpacklo_epi8(row0, row1);
  const __m128i pair23_low = _mm_unpacklo_epi8(row2, row3);
  const __m128i pair01_high = _mm_unpackhi_epi8(row0, row1);
  const __m128i pair23_high = _mm_unpackhi_epi8(row2, row3);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(destination),
                   _mm_unpacklo_epi16(pair01_low, pair23_low));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(destination + 16),
                   _mm_unpackhi_epi16(pair01_low, pair23_low));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(destination + 32),
                   _mm_unpacklo_epi16(pair01_high, pair23_high));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(destination + 48),
                   _mm_unpackhi_epi16(pair01_high, pair23_high));
}

template <int Bits>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")))
void predecode_q8_vnni(const Exl3Mul1MatrixView &matrix, uint8_t *weights) {
  const int tile_count =
      (matrix.input_features / 16) * (matrix.output_features / 16);
  const __m512i ones = _mm512_set1_epi32(0x01010101);
  #pragma omp parallel for schedule(static)
  for (int tile_index = 0; tile_index < tile_count; ++tile_index) {
    const auto tile = load_vnni_packed_tile<Bits>(
        matrix.trellis + static_cast<size_t>(tile_index) * 16 * Bits);
    uint8_t *destination = weights + static_cast<size_t>(tile_index) * 256;
    predecode_q8_group<Bits, 0>(tile, ones, destination);
    predecode_q8_group<Bits, 4>(tile, ones, destination + 64);
    predecode_q8_group<Bits, 8>(tile, ones, destination + 128);
    predecode_q8_group<Bits, 12>(tile, ones, destination + 192);
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")))
int32_t quantize_activation_vnni(const float *input, int8_t *output,
                                  int dimensions, float &scale) {
  const __m512 sign = _mm512_set1_ps(-0.0f);
  __m512 absolute_max = _mm512_setzero_ps();
  for (int index = 0; index < dimensions; index += 16) {
    const __m512 values = _mm512_andnot_ps(sign, _mm512_loadu_ps(input + index));
    absolute_max = _mm512_max_ps(absolute_max, values);
  }
  const float maximum = _mm512_reduce_max_ps(absolute_max);
  const float reciprocal = maximum > 0.0f ? 127.0f / maximum : 1.0f;
  scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
  const __m512 reciprocal_vector = _mm512_set1_ps(reciprocal);
  const __m512i minimum = _mm512_set1_epi32(-127);
  const __m512i maximum_vector = _mm512_set1_epi32(127);
  __m512i sum = _mm512_setzero_si512();
  for (int index = 0; index < dimensions; index += 16) {
    __m512i values = _mm512_cvtps_epi32(
        _mm512_mul_ps(_mm512_loadu_ps(input + index), reciprocal_vector));
    values = _mm512_min_epi32(_mm512_max_epi32(values, minimum), maximum_vector);
    sum = _mm512_add_epi32(sum, values);
    const __m256i values16 = _mm512_cvtsepi32_epi16(values);
    const __m128i values8 = _mm256_cvtsepi16_epi8(values16);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + index), values8);
  }
  return _mm512_reduce_add_epi32(sum);
}
#endif

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
void predecode_q8(const Exl3Mul1MatrixView &matrix, uint8_t *weights) {
#if defined(__x86_64__) && defined(__GNUC__)
  if (exl3_cpu_vnni_available()) {
    predecode_q8_vnni<Bits>(matrix, weights);
    return;
  }
#endif
  const int input_tiles = matrix.input_features / 16;
  const int output_tiles = matrix.output_features / 16;
  const int tile_count = input_tiles * output_tiles;
  #pragma omp parallel for schedule(static)
  for (int tile_index = 0; tile_index < tile_count; ++tile_index) {
    const uint16_t *packed = matrix.trellis +
        static_cast<size_t>(tile_index) * 16 * Bits;
    uint8_t *destination = weights + static_cast<size_t>(tile_index) * 256;
    for (int offset = 0; offset < 256; ++offset) {
      const int row_major = kTensorCorePermutation[offset];
      const int row = row_major / 16;
      const int column = row_major % 16;
      const uint32_t product =
          static_cast<uint32_t>(decode_state<Bits>(packed, offset)) *
          kMul1Multiplier;
      const int byte_sum = (product & 255u) + ((product >> 8) & 255u) +
                           ((product >> 16) & 255u) + (product >> 24);
      destination[(row / 4) * 64 + column * 4 + row % 4] =
          static_cast<uint8_t>((byte_sum + 2) / 4);
    }
  }
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

bool exl3_cpu_vnni_available() {
#if defined(__x86_64__) && defined(__GNUC__)
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni") &&
         __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#else
  return false;
#endif
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
#if defined(__x86_64__) && defined(__GNUC__)
    if (exl3_cpu_vnni_available()) {
      multiply_rotated_vnni<Bits>(matrix, transformed_input.data(), output, rows);
      return;
    }
#endif
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

bool exl3_mul1_matmul_batch(const Exl3Mul1BatchItem *items, int count,
                            bool int8_activation) {
  if (!items || count <= 0) return false;
  for (int index = 0; index < count; ++index)
    if (!valid_matrix(items[index].matrix) || !items[index].input || !items[index].output)
      return false;
#if defined(__x86_64__) && defined(__GNUC__)
  if (exl3_cpu_vnni_available()) {
    std::vector<size_t> input_offsets(static_cast<size_t>(count) + 1, 0);
    std::vector<size_t> job_offsets(static_cast<size_t>(count) + 1, 0);
    for (int index = 0; index < count; ++index) {
      input_offsets[index + 1] = input_offsets[index] + items[index].matrix.input_features;
      job_offsets[index + 1] = job_offsets[index] + items[index].matrix.output_features / 128;
    }
    std::vector<float> transformed_inputs(input_offsets.back());
    std::vector<int8_t> q8_activations;
    std::vector<float> q8_scales;
    std::vector<int32_t> q8_sums;
    if (int8_activation) {
      q8_activations.resize(input_offsets.back());
      q8_scales.resize(static_cast<size_t>(count));
      q8_sums.resize(static_cast<size_t>(count));
    }
    for (int index = 0; index < count; ++index) {
      const auto &matrix = items[index].matrix;
      const float *source = items[index].input;
      float *target = transformed_inputs.data() + input_offsets[index];
      for (int block = 0; block < matrix.input_features; block += 128) {
        for (int column = 0; column < 128; ++column)
          target[block + column] = source[block + column] * fp16_to_fp32(matrix.suh[block + column]);
        hadamard_128(target + block);
        for (int column = 0; column < 128; ++column)
          target[block + column] *= kHadamardScale;
      }
      if (int8_activation) {
#if defined(__x86_64__) && defined(__GNUC__)
        q8_sums[index] = quantize_activation_vnni(
            target, q8_activations.data() + input_offsets[index],
            matrix.input_features, q8_scales[index]);
#else
        return false;
#endif
      }
    }
    bool present[9]{};
    for (int index = 0; index < count; ++index)
      present[items[index].matrix.bits] = true;
#define EXL3_BATCH_VNNI(Bits) \
    if (present[Bits]) { \
      if (int8_activation) \
        multiply_rotated_int8_vnni_batch<Bits>(items, count, input_offsets.data(), \
            job_offsets.data(), q8_activations.data(), q8_scales.data(), q8_sums.data()); \
      else \
        multiply_rotated_vnni_batch<Bits>(items, count, input_offsets.data(), \
            job_offsets.data(), transformed_inputs.data()); \
    }
    EXL3_BATCH_VNNI(1);
    EXL3_BATCH_VNNI(2);
    EXL3_BATCH_VNNI(3);
    EXL3_BATCH_VNNI(4);
    EXL3_BATCH_VNNI(5);
    EXL3_BATCH_VNNI(6);
    EXL3_BATCH_VNNI(7);
    EXL3_BATCH_VNNI(8);
#undef EXL3_BATCH_VNNI
    return true;
  }
#endif
  for (int index = 0; index < count; ++index)
    if (!exl3_mul1_matmul(items[index].matrix, items[index].input,
                           items[index].output, 1)) return false;
  return true;
}

std::vector<uint8_t> exl3_mul1_predecode_q8(
    const Exl3Mul1MatrixView &matrix) {
  if (!valid_matrix(matrix)) return {};
  std::vector<uint8_t> weights(
      static_cast<size_t>(matrix.input_features) * matrix.output_features);
  const bool decoded = dispatch_bits(matrix.bits, [&]<int Bits>() {
    predecode_q8<Bits>(matrix, weights.data());
  });
  return decoded ? weights : std::vector<uint8_t>{};
}

bool exl3_mul1_q8_matmul_batch(const Exl3Mul1Q8BatchItem *items, int count) {
  if (!items || count <= 0) return false;
  for (int index = 0; index < count; ++index) {
    const auto &matrix = items[index].matrix;
    if (!matrix.weights || !matrix.suh || !matrix.svh || !items[index].input ||
        !items[index].output || matrix.input_features <= 0 ||
        matrix.output_features <= 0 || matrix.input_features % 128 ||
        matrix.output_features % 128)
      return false;
  }
#if defined(__x86_64__) && defined(__GNUC__)
  if (!exl3_cpu_vnni_available()) return false;
  std::vector<size_t> input_offsets(static_cast<size_t>(count) + 1, 0);
  std::vector<size_t> job_offsets(static_cast<size_t>(count) + 1, 0);
  for (int index = 0; index < count; ++index) {
    input_offsets[index + 1] = input_offsets[index] + items[index].matrix.input_features;
    job_offsets[index + 1] = job_offsets[index] +
                             items[index].matrix.output_features / 128;
  }
  std::vector<float> transformed_inputs(input_offsets.back());
  std::vector<int8_t> activations(input_offsets.back());
  std::vector<float> activation_scales(static_cast<size_t>(count));
  std::vector<int32_t> activation_sums(static_cast<size_t>(count));
  for (int index = 0; index < count; ++index) {
    const auto &matrix = items[index].matrix;
    const float *source = items[index].input;
    float *target = transformed_inputs.data() + input_offsets[index];
    for (int block = 0; block < matrix.input_features; block += 128) {
      for (int column = 0; column < 128; ++column)
        target[block + column] = source[block + column] *
            fp16_to_fp32(matrix.suh[block + column]);
      hadamard_128(target + block);
      for (int column = 0; column < 128; ++column)
        target[block + column] *= kHadamardScale;
    }
    activation_sums[index] = quantize_activation_vnni(
        target, activations.data() + input_offsets[index],
        matrix.input_features, activation_scales[index]);
  }
  multiply_rotated_q8_vnni_batch(
      items, count, input_offsets.data(), job_offsets.data(), activations.data(),
      activation_scales.data(), activation_sums.data());
  return true;
#else
  return false;
#endif
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
  header.magic = kExl3PackedMagic;
  header.version = 1;
  header.input_features = static_cast<uint32_t>(matrix.input_features);
  header.output_features = static_cast<uint32_t>(matrix.output_features);
  header.bits = static_cast<uint32_t>(matrix.bits);
  header.flags = matrix.bias ? kExl3PackedHasBias : 0u;
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
  if (!exl3_valid_packed_header(header, packed_size))
    return false;

  matrix.trellis =
      reinterpret_cast<const uint16_t *>(packed + header.trellis_offset);
  matrix.suh = reinterpret_cast<const uint16_t *>(packed + header.suh_offset);
  matrix.svh = reinterpret_cast<const uint16_t *>(packed + header.svh_offset);
  matrix.bias = (header.flags & kExl3PackedHasBias)
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
