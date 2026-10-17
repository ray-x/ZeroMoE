#pragma once

#include "gguf_loader.hpp"

#include <cstdint>
#include <vector>

struct MoEExpertBlock;

void expert_ffn_cpu(const float *input, float *output,
                    const uint8_t *gate_data, const GGUFTensor &gate_meta,
                    const uint8_t *up_data, const GGUFTensor &up_meta,
                    const uint8_t *down_data, const GGUFTensor &down_meta,
                    float *gate_buf, float *up_buf, int S, int D,
                    int moe_inter, float swiglu_clamp, int layer_idx,
                    int expert_id);

// Group cold routed experts which share one input activation. The routine
// reads each expert directly from its pinned DDR block and does not create a
// packed weight copy. Returns false for mixed/unsupported projection types.
bool expert_ffn_cpu_grouped_ddr(const float *input, float *output,
                                const std::vector<MoEExpertBlock *> &experts,
                                const float *weights, int D, int moe_inter,
                                float swiglu_clamp, int layer_idx);
