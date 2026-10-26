#pragma once

#include "mtp_graph_runtime.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)
namespace qwen4exp_mtp {

struct Options {
  int device = 0;
  int capacity = 32;
  size_t device_budget = 0;
  bool diagnostics = false;
  // Parity-only. The pinned llama.cpp reference schedules its weightless
  // RMS-norm nodes (enorm, hnorm, attention Q/K norm) on the CPU backend,
  // which accumulates in double; CUDA accumulates in float. Placing the same
  // nodes on the CPU makes a strict bit-exact comparison meaningful. It adds
  // host round trips and is never a production setting.
  bool reference_cpu_norms = false;
};

// The owner must contain the trunk metadata and resident output head, plus
// resident NextN tensors loaded from mtp_model_file. It is retained throughout
// graph lifetime. Calls and resident-weight mutation must be serialized.
std::unique_ptr<zeromoe::graph::MtpGraphRuntime> create(
    std::shared_ptr<StreamingEngine> owner, const Options &options,
    std::string &error);

} // namespace qwen4exp_mtp
#endif
