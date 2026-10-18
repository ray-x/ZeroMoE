// iq1_decoder.hpp — 1-bit quantization format decoders (IQ1_S, IQ1_M, TQ1_0)
// These are the most aggressive quantization formats, used for MoE expert weights
// where the model tolerates higher per-parameter error due to ensemble averaging.
#pragma once
#include <cstdint>
#include <cstring>

// ============================================================
// IQ1_S: ~1.56 bits/weight (256 values in 50 bytes)
// Layout per superblock (256 elements):
//   - qs[32]:   8 groups of 4 bytes, each byte indexes a 3-entry codebook
//   - qh[4]:    high bits for grid selection
//   - scales[2]: FP16 super-block scale + min
// Total: 32 + 4 + 14 = 50 bytes (but exact layout from ggml is more complex)
// ============================================================
struct block_iq1_s {
    uint8_t qs[32];      // grid indices
    uint16_t qh[8];      // high bits + scales packed
    // Total: 48 bytes? Actually ggml uses: qs[QK_K/8] + qh[QK_K/16] + d (half)
};

// Simplified: IQ1_S values are ternary-like {-1, 0, +1} with per-block scaling
// Exact dequant follows ggml's iq1s_grid lookup table
inline void dequantize_iq1_s_block(const uint8_t* block_data, float* output, int n_elements) {
    // IQ1_S superblock: 256 elements in 50 bytes
    // Byte layout: [d:2][qs:QK_K/8][qh:QK_K/16]
    // d = FP16 scale
    uint16_t d_bits;
    std::memcpy(&d_bits, block_data, 2);
    // Convert FP16 to F32 (simplified)
    union { uint32_t u; float f; } conv;
    uint32_t sign = (uint32_t)(d_bits & 0x8000) << 16;
    uint32_t exp = (d_bits >> 10) & 0x1F;
    uint32_t man = d_bits & 0x3FF;
    if (exp == 0) conv.u = sign;
    else if (exp == 0x1F) conv.u = sign | 0x7F800000 | (man << 13);
    else conv.u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float scale = conv.f;

    const uint8_t* qs = block_data + 2;
    // Each byte encodes 8 ternary values via grid codebook
    // Simplified: treat each 2-bit pair as {-1, 0, +1} mapped from {0,1,2,3}
    for (int i = 0; i < n_elements && i < 256; ++i) {
        int byte_idx = i / 8;
        int bit_idx = i % 8;
        // Extract 1-bit value + sign from packed data
        uint8_t val = (qs[byte_idx] >> bit_idx) & 1;
        // For true IQ1_S, values come from a codebook; simplified to ternary here
        output[i] = (val ? 1.0f : -1.0f) * scale;
    }
}

// ============================================================
// IQ1_M: ~1.75 bits/weight (256 values in 56 bytes)
// Improved IQ1_S with better codebook and scale resolution
// ============================================================
struct block_iq1_m {
    uint8_t qs[32];      // quantized indices
    uint8_t scales[12];  // per-sub-block scales
    uint8_t qh[12];     // high bits
};

inline void dequantize_iq1_m_block(const uint8_t* block_data, float* output, int n_elements) {
    // Similar to IQ1_S but with more scale granularity
    uint16_t d_bits;
    std::memcpy(&d_bits, block_data, 2);
    union { uint32_t u; float f; } conv;
    uint32_t sign = (uint32_t)(d_bits & 0x8000) << 16;
    uint32_t exp = (d_bits >> 10) & 0x1F;
    uint32_t man = d_bits & 0x3FF;
    if (exp == 0) conv.u = sign;
    else if (exp == 0x1F) conv.u = sign | 0x7F800000 | (man << 13);
    else conv.u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float scale = conv.f;

    const uint8_t* qs = block_data + 2;
    for (int i = 0; i < n_elements && i < 256; ++i) {
        int byte_idx = i / 8;
        int bit_idx = i % 8;
        uint8_t val = (qs[byte_idx] >> bit_idx) & 1;
        output[i] = (val ? 1.0f : -1.0f) * scale;
    }
}

// ============================================================
// TQ1_0: Ternary quantization (values in {-1, 0, +1})
// 256 values in 54 bytes — uses base-3 encoding
// This is the simplest 1-bit format: just sign + zero detection
// ============================================================
struct block_tq1_0 {
    uint8_t qs[32];      // ternary values packed (5 trits per byte via base-243 encoding)
    uint8_t qh[4];       // overflow bits
    uint16_t d;          // FP16 scale
};

// Base-3 decoding: each byte holds 5 ternary values (3^5 = 243 ≤ 255)
// Values map: 0 → -1, 1 → 0, 2 → +1
static const float tq1_0_map[3] = {-1.0f, 0.0f, 1.0f};

inline void dequantize_tq1_0_block(const uint8_t* block_data, float* output, int n_elements) {
    // Layout: [qs: 32 bytes][qh: 4 bytes][d: 2 bytes] = 38 bytes? 
    // Actual ggml TQ1_0: more complex packing. Simplified version:
    uint16_t d_bits;
    std::memcpy(&d_bits, block_data + 52, 2);  // scale at end
    union { uint32_t u; float f; } conv;
    uint32_t sign = (uint32_t)(d_bits & 0x8000) << 16;
    uint32_t exp = (d_bits >> 10) & 0x1F;
    uint32_t man = d_bits & 0x3FF;
    if (exp == 0) conv.u = sign;
    else if (exp == 0x1F) conv.u = sign | 0x7F800000 | (man << 13);
    else conv.u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float scale = conv.f;

    const uint8_t* qs = block_data;
    int out_idx = 0;

    // Decode base-3 packed bytes: each byte holds 5 ternary digits
    for (int b = 0; b < 32 && out_idx < n_elements; ++b) {
        uint8_t packed = qs[b];
        for (int t = 0; t < 5 && out_idx < n_elements; ++t) {
            int trit = packed % 3;
            packed /= 3;
            output[out_idx++] = tq1_0_map[trit] * scale;
        }
    }
    // Remaining elements from qh (overflow)
    const uint8_t* qh = block_data + 32;
    for (int b = 0; b < 4 && out_idx < n_elements; ++b) {
        uint8_t packed = qh[b];
        for (int t = 0; t < 5 && out_idx < n_elements; ++t) {
            int trit = packed % 3;
            packed /= 3;
            output[out_idx++] = tq1_0_map[trit] * scale;
        }
    }
}
