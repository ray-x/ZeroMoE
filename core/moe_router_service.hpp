#pragma once

struct MoePipelineContext;

class MoeRouterService {
public:
  // Returns true when an integrated scheduler completed the whole layer.
  static bool route(MoePipelineContext &context);
};
