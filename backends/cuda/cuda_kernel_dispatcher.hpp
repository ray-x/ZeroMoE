#pragma once

#include "gguf_loader.hpp"

class CudaKernelDispatcher {
public:
  static bool matmul(GGUFType type, float *y, const float *x, const void *w,
                     int S, int I, int O);

private:
  static bool matmul_bf16(GGUFType type, float *y, const float *x,
                          const void *w, int S, int I, int O);
};
