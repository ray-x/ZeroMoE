#pragma once

#include "gguf_loader.hpp"

void apply_gating_activation(GatingFunc func, float *row, int n);
