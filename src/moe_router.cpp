#include "moe_router.hpp"

#include <algorithm>
#include <cmath>

void apply_gating_activation(GatingFunc func, float *row, int n) {
  switch (func) {
  case GatingFunc::Sigmoid:
    for (int e = 0; e < n; ++e)
      row[e] = 1.0f / (1.0f + std::exp(-row[e]));
    break;
  case GatingFunc::SqrtSoftplus:
    for (int e = 0; e < n; ++e) {
      const float x = row[e];
      const float softplus =
          x > 0.0f ? x + std::log1p(std::exp(-x))
                   : std::log1p(std::exp(x));
      row[e] = std::sqrt(softplus);
    }
    break;
  case GatingFunc::Softmax:
  default: {
    float max_l = row[0];
    for (int e = 1; e < n; ++e)
      max_l = std::max(max_l, row[e]);
    float sum = 0.0f;
    for (int e = 0; e < n; ++e) {
      row[e] = std::exp(row[e] - max_l);
      sum += row[e];
    }
    const float inv = 1.0f / sum;
    for (int e = 0; e < n; ++e)
      row[e] *= inv;
    break;
  }
  }
}
