#pragma once

#include <cstdint>
#include <vector>

struct StreamingEngine;

// Gather/dequantize PLE table rows through the process-level mmap shared by
// the ordinary Qwen layer-input path.  This keeps the full decode graph from
// issuing one synchronous pread_direct_aware() per hashed row on every token.
// The output layout is [rows.size(), ple_n_embd_per_layer].
bool qwen4exp_ple_gather_rows(StreamingEngine &engine,
                              const std::vector<int32_t> &rows,
                              std::vector<float> &out);
