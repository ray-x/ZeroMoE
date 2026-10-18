#pragma once

struct MoePipelineContext;
struct QwenRamCpuMoeLayer;

class StreamingEngine;
bool qwen4exp_ram_cpu_moe_layer(const StreamingEngine &engine, int layer_idx,
                                int n_experts,
                                QwenRamCpuMoeLayer &layer);

class TokenDispatchPipeline {
public:
  static void run(MoePipelineContext &context);
};
