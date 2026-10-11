#pragma once

#include <cstdint>

constexpr uint64_t kExl3PackedMagic = 0x314c554d334c5845ull;
constexpr uint32_t kExl3PackedHasBias = 1u;

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

static_assert(sizeof(Exl3PackedMatrixHeader) == 80);

inline bool exl3_valid_packed_header(const Exl3PackedMatrixHeader &header,
                                      uint64_t storage_size) {
  if (header.magic != kExl3PackedMagic || header.version != 1 ||
      header.total_size < sizeof(header) || header.total_size > storage_size ||
      header.input_features == 0 || header.output_features == 0 ||
      header.input_features > 0x7fffffffu || header.output_features > 0x7fffffffu ||
      header.input_features % 128 || header.output_features % 128 ||
      header.bits < 1 || header.bits > 8 || (header.flags & ~kExl3PackedHasBias))
    return false;
  const uint64_t trellis_size = static_cast<uint64_t>(header.input_features) *
                                 header.output_features / 8 * header.bits;
  const uint64_t suh_size = static_cast<uint64_t>(header.input_features) * 2;
  const uint64_t svh_size = static_cast<uint64_t>(header.output_features) * 2;
  const auto in_bounds = [&](uint64_t offset, uint64_t bytes, uint64_t alignment) {
    return offset >= sizeof(header) && offset % alignment == 0 &&
           offset <= header.total_size && bytes <= header.total_size - offset;
  };
  return in_bounds(header.trellis_offset, trellis_size, 4) &&
         in_bounds(header.suh_offset, suh_size, 2) &&
         in_bounds(header.svh_offset, svh_size, 2) &&
         (!(header.flags & kExl3PackedHasBias) || in_bounds(header.bias_offset, svh_size, 2));
}
