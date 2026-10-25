#pragma once

// The one table of ggml tensor types: elements per block, bytes per block, and
// whether MUL_MAT_ID takes ggml-cuda's MMQ path for it. Numeric ids are ggml's
// (ggml.h), so this header has no dependency on the GGUF loader or on ggml.
//
// Before this table existed the same facts lived in three places (the GGUF
// loader's size formula, the offload store's row-size table, and two copies of
// the MMQ type list) and drifted: the offload store sized Q2_0 as 10 bytes per
// 32 elements (the real block is 18 bytes per 64) and lacked the legacy
// quants, and GLM's MMQ list lacked MXFP4. tests/test_quant_table.cpp checks
// every row against ggml_row_size.

#include <cstdint>

namespace zeromoe::quant {

struct QuantInfo {
  uint32_t type;
  const char *name;
  uint32_t block;  // elements per block
  uint32_t bytes;  // bytes per block
  // With the patched negative-id skip (patches/ggml/0006), batched MUL_MAT_ID
  // reaches ggml-cuda's MMQ path for this type. Other types (float, IQ1_M, ...)
  // use other batched paths, so their layers keep the valid-dummy-id route.
  bool mmq_ids;
};

inline constexpr QuantInfo kQuantTable[] = {
    {0, "F32", 1, 4, false},       {1, "F16", 1, 2, false},
    {2, "Q4_0", 32, 18, true},     {3, "Q4_1", 32, 20, true},
    {6, "Q5_0", 32, 22, true},     {7, "Q5_1", 32, 24, true},
    {8, "Q8_0", 32, 34, true},     {9, "Q8_1", 32, 36, false},
    {10, "Q2_K", 256, 84, true},   {11, "Q3_K", 256, 110, true},
    {12, "Q4_K", 256, 144, true},  {13, "Q5_K", 256, 176, true},
    {14, "Q6_K", 256, 210, true},  {15, "Q8_K", 256, 292, false},
    {16, "IQ2_XXS", 256, 66, true}, {17, "IQ2_XS", 256, 74, true},
    {18, "IQ3_XXS", 256, 98, true}, {19, "IQ1_S", 256, 50, true},
    {20, "IQ4_NL", 32, 18, true},  {21, "IQ3_S", 256, 110, true},
    {22, "IQ2_S", 256, 82, true},  {23, "IQ4_XS", 256, 136, true},
    {26, "I32", 1, 4, false},      {27, "I64", 1, 8, false},
    {29, "IQ1_M", 256, 56, false}, {30, "BF16", 1, 2, false},
    {34, "TQ1_0", 256, 54, false}, {35, "TQ2_0", 256, 66, false},
    {39, "MXFP4", 32, 17, true},   {40, "NVFP4", 64, 36, false},
    {42, "Q2_0", 64, 18, false},
};

inline const QuantInfo *find(uint32_t type) {
  for (const QuantInfo &q : kQuantTable)
    if (q.type == type) return &q;
  return nullptr;
}

// Bytes of one row of `ne0` elements; 0 for a type not in the table.
inline uint64_t row_bytes(uint32_t type, uint64_t ne0) {
  const QuantInfo *q = find(type);
  return q ? (ne0 / q->block) * q->bytes : 0;
}

// Bytes of `numel` elements. A type not in the table counts one byte per
// element, the GGUF loader's long-standing fallback.
inline uint64_t tensor_bytes(uint32_t type, uint64_t numel) {
  const QuantInfo *q = find(type);
  return q ? (numel / q->block) * q->bytes : numel;
}

inline bool takes_mmq(uint32_t type) {
  const QuantInfo *q = find(type);
  return q && q->mmq_ids;
}

} // namespace zeromoe::quant
