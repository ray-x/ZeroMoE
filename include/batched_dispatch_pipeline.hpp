#pragma once

struct MoePipelineContext;

enum class BatchedDispatchResult {
  Continue,
  Handled,
  Abort,
};

class BatchedDispatchPipeline {
public:
  static BatchedDispatchResult run(MoePipelineContext &context);
};
