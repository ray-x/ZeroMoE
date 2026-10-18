#pragma once

#include "gguf_loader.hpp"

#include <cstdint>

void expert_ffn_cpu(const float *input, float *output,
                    const uint8_t *gate_data, const GGUFTensor &gate_meta,
                    const uint8_t *up_data, const GGUFTensor &up_meta,
                    const uint8_t *down_data, const GGUFTensor &down_meta,
                    float *gate_buf, float *up_buf, int S, int D,
                    int moe_inter, float swiglu_clamp, int layer_idx,
                    int expert_id);
