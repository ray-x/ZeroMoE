#pragma once
// Qwen4Exp recurrent / hyper-connection / PLE state of the CPU engine, kept apart
// from CPUEngine so that engine does not grow a member per supported architecture.
// Allocated lazily by the Qwen4Exp CPU backend; non-Qwen models never touch it.
#include <cstdint>
#include <vector>

struct Qwen4ExpCpuState {
  std::vector<std::vector<float>> conv_history;      // GDN conv history per layer
  std::vector<std::vector<float>> recur_state;       // GDN recurrent state per layer
  std::vector<float> hc_stream_buf;                  // hyper-connection residual streams
  std::vector<float> hc_inject;                      // hyper-connection injection
  std::vector<std::vector<float>> ple_conv_history;  // PLE conv history per layer
  std::vector<int32_t> ple_token_history;            // PLE n-gram token window
  int ple_next_pos = 0;
};
