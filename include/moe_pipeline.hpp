#pragma once

class StreamingEngine;
class RAMSafetyManager;
class LockFreePrefetcher;
struct LayerScratch;

void moe_forward_gguf(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                      LockFreePrefetcher &async_io_queue, int layer_idx,
                      const float *hidden_states, int S, float *output,
                      LayerScratch &scratch, const int *token_ids);

double diag_get_decode_qwen_cpu_graph_ms();
double diag_get_decode_qwen_shared_submit_ms();

namespace moe_pipeline_detail {
bool qwen4exp_router_pilot_layer_allowed(const StreamingEngine &engine,
                                         int target_layer);
bool qwen4exp_decode_hot_set_stable(int layer_idx);
bool qwen4exp_router_pilot_gpu_async(StreamingEngine &engine, int target_layer,
                                     int D, int n_experts, int topk,
                                     const void **device_ids,
                                     const void **device_weights);
void qwen4exp_process_pilot_predictions(
    StreamingEngine &engine, RAMSafetyManager &ram_mgr,
    LockFreePrefetcher &async_io_queue, LayerScratch &scratch,
    int target_layer, int qwen_k);
bool qwen4exp_decode_ggml_path(const StreamingEngine &engine);
void reset_diagnostics();
} // namespace moe_pipeline_detail
