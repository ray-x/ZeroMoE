// Compile-only contract test for the completion graph boundary. Runtime CUDA
// capture is exercised by the engine A/B harness when a Ready payload is
// available; this CPU-side test verifies the public types and stale-generation
// API remain buildable with the same headers.
#include "qwen4exp_completion_graph.hpp"

#include <type_traits>

static_assert(std::is_move_constructible_v<moestream::qwen4exp::StorageHandle>);
static_assert(std::is_same_v<decltype(&moestream::qwen4exp::CompletionGraph::submit),
                             bool (moestream::qwen4exp::CompletionGraph::*) (
                                 const moestream::qwen4exp::StorageCompletion &,
                                 uint64_t, uint64_t)>);

int main() { return 0; }
