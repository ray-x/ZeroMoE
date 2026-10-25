#pragma once

#if defined(ZEROMOE_USE_GGML_CUDA)

#include "ggml-backend.h"
#include "ggml.h"
#include <cuda_runtime.h>

#include <cstdint>

using Exl3GgmlCustomCompute = void (*)(ggml_backend_t cuda_backend,
                                       ggml_tensor *node, void *user_data);

struct Exl3GgmlCustomNode {
  static constexpr uint64_t kMagic = 0x45584c3347474d4cull;
  uint64_t magic = kMagic;
  Exl3GgmlCustomCompute compute = nullptr;
  void *user_data = nullptr;
  bool graph_capture_safe = false;
};

ggml_backend_t exl3_ggml_cuda_backend_wrap(ggml_backend_t cuda_backend,
                                            bool take_ownership = false);

bool exl3_ggml_cuda_backend_set_graph_capture(ggml_backend_t adapter,
                                              bool enabled);
void exl3_ggml_cuda_backend_reset_graph(ggml_backend_t adapter);

ggml_tensor *exl3_ggml_map_custom2(ggml_context *ctx, ggml_tensor *input,
                                    ggml_tensor *opaque,
                                    Exl3GgmlCustomNode *node);
ggml_tensor *exl3_ggml_map_custom3(ggml_context *ctx, ggml_tensor *input,
                                    ggml_tensor *opaque, ggml_tensor *routing,
                                    Exl3GgmlCustomNode *node);

cudaStream_t exl3_ggml_cuda_stream(ggml_backend_t cuda_backend);

#endif
