#pragma once

class StreamingEngine;
class RAMSafetyManager;
class LockFreePrefetcher;
struct LayerScratch;

class MoePipelineOrchestrator {
public:
  static void forward(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                      LockFreePrefetcher &async_io_queue, int layer_idx,
                      const float *hidden_states, int S, float *output,
                      LayerScratch &scratch, const int *token_ids);
};
