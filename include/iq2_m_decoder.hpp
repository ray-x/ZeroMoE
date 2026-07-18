#pragma once
#include <cstdint>
#include <immintrin.h>

#define QK_K 256

struct block_iq2_m {
    uint8_t scales[16];   
    uint8_t qs[32];       
    uint8_t signs[32];    
    uint8_t grid[4];      
}; // Fixed missing closing bracket

extern const float iq2_m_codebook[256]; 

// Simple extraction mock signature to allow compiling
static inline float extract_iq2_scale(const uint8_t* scales, int i) {
    (void)scales; (void)i; return 1.0f; 
}

void dequantize_iq2_m(const block_iq2_m* __restrict__ x, float* __restrict__ y) {
    float dl[8];
    for (int i = 0; i < 8; ++i) {
        dl[i] = extract_iq2_scale(x->scales, i); 
    }

    int q_idx = 0;
    for (int ib = 0; ib < 8; ++ib) { 
        float scale = dl[ib];
        for (int i = 0; i < 32; ++i) {
            uint8_t q_val = (x->qs[q_idx >> 2] >> ((q_idx & 3) * 2)) & 0x03;
            uint8_t sign_bit = (x->signs[q_idx >> 3] >> (q_idx & 7)) & 0x01;
            float sign = sign_bit ? -1.0f : 1.0f;

            float grid_val = iq2_m_codebook[q_val | (x->grid[0] << 2)]; 
            y[q_idx] = grid_val * sign * scale;
            q_idx++;
        }
    }
} // Fixed trailing loop closures