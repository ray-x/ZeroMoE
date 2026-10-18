#pragma once

struct MoePipelineContext;

class PrefillWarmupManager {
public:
  static void prepare(MoePipelineContext &context);
  static void finish(MoePipelineContext &context);
};
