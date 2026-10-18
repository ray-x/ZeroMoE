#pragma once

#include <algorithm>
#include <cmath>

inline void swiglu_clamp_inplace(float *data, int n, float limit) {
  if (limit <= 0.0f)
    return;
  for (int i = 0; i < n; ++i)
    data[i] = std::max(-limit, std::min(limit, data[i]));
}

inline void swiglu_clamp_fused_inplace(float *gate, const float *up, int n,
                                       float limit) {
  for (int i = 0; i < n; ++i) {
    float g = gate[i];
    float u = up[i];
    if (limit > 0.0f) {
      g = std::min(g, limit);
      u = std::max(-limit, std::min(limit, u));
    }
    gate[i] = g / (1.0f + std::exp(-g)) * u;
  }
}
