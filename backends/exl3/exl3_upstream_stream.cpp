// Thread-local CUDA stream used by the torch-free stub of at::cuda::getCurrentCUDAStream.
#include "exl3_stub.hpp"

#include <tuple>
#include <vector>
#include "graph.cuh"

// Upstream CUDA-graph parameter recording is not used (graph == nullptr in every
// call MoEStream makes), but the launchers reference the symbol.
void Graph::record_param(void *kernel, int param_id, int param_offset, int size) {
  graph_sites.emplace_back(kernel, param_id, param_offset, size);
}

namespace moestream_exl3 {
namespace {
thread_local cudaStream_t g_stream = nullptr;
}
cudaStream_t current_stream() { return g_stream; }
void set_current_stream(cudaStream_t stream) { g_stream = stream; }
} // namespace moestream_exl3
