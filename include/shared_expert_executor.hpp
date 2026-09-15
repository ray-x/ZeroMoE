#pragma once

class StreamingEngine;
struct ArchitectureConfig;
struct LayerScratch;

bool run_qwen4exp_shared_device(
    StreamingEngine &engine, int layer_idx, int S, int D, int moe_inter,
    const void *device_input, const void *routed_output,
    const ArchitectureConfig &config, const void **combined_output,
    const float *host_routed = nullptr, bool shared_only = false,
    bool shared_add = false);

void run_qwen4exp_shared_host(StreamingEngine &engine, int layer_idx, int S,
                              int D, int moe_inter, const float *normed,
                              float *output, LayerScratch &scratch,
                              const ArchitectureConfig &config);
