// Qwen4Exp full graph: device-side checkpoints of the decode graph's recurrent
// state, used by graph-native MTP verification to undo rejected draft tokens.
//
// What a decode step mutates, and how each part is rolled back:
//  * GDN recurrent state and conv history (per GDN layer) and the PLE
//    convolution history: device buffers rewritten in place every step.
//    These are copied device-to-device into a pooled checkpoint block.
//  * The PLE token window and next position: host metadata, copied by value.
//  * Full-attention KV: each step writes only row `position`, and every read
//    is masked to `<= position`. Restoring the position therefore restores KV
//    exactly; rows past it are dead until overwritten. Not copied.
//  * decode_last_pos: host metadata, copied by value.

#include "ggml_qwen4exp_graph_internal.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)

using namespace qwen4exp_graph;

namespace {

struct Span {
  void *device = nullptr;  // live storage inside the graph
  size_t bytes = 0;
};

struct Checkpoint {
  uint64_t graph_generation = 0;
  const FullGraph *graph = nullptr;
  void *block = nullptr;  // pooled device memory, sum of span bytes
  size_t block_bytes = 0;
  std::vector<int32_t> ple_tok_hist;
  int64_t ple_next_pos = 0;
  int decode_last_pos = 0;
  bool host_state_dirty = false;
};

std::mutex g_mutex;
std::map<int, std::unique_ptr<Checkpoint>> g_checkpoints;
std::vector<std::pair<void *, size_t>> g_free_blocks;  // pooled, never shrunk
int g_next_handle = 1;

// The spans that make up the recurrent state, in a fixed order.
bool collect_spans(FullGraph &g, std::vector<Span> &spans, size_t &total) {
  spans.clear();
  total = 0;
  const auto add = [&](ggml_tensor *t) {
    if (!t || !t->data || !t->buffer || !ggml_is_contiguous(t) ||
        ggml_backend_buffer_is_host(t->buffer))
      return false;
    spans.push_back({t->data, ggml_nbytes(t)});
    total += ggml_nbytes(t);
    return true;
  };
  if (g.gdn_state_storage.size() != g.gdn_layers.size() ||
      g.gdn_hist_storage.size() != g.gdn_layers.size())
    return false;
  for (size_t i = 0; i < g.gdn_layers.size(); ++i)
    if (!add(g.gdn_state_storage[i]) || !add(g.gdn_hist_storage[i]))
      return false;
  if (g.ple_layer >= 0 && !add(g.ple_hist_storage))
    return false;
  return !spans.empty();
}

void *take_block(size_t bytes) {
  for (auto it = g_free_blocks.begin(); it != g_free_blocks.end(); ++it)
    if (it->second == bytes) {
      void *p = it->first;
      g_free_blocks.erase(it);
      return p;
    }
  void *p = nullptr;
  return cudaMalloc(&p, bytes) == cudaSuccess ? p : nullptr;
}

bool copy_spans(const std::vector<Span> &spans, void *block, bool to_block,
                cudaStream_t stream) {
  char *cursor = static_cast<char *>(block);
  for (const Span &s : spans) {
    void *dst = to_block ? static_cast<void *>(cursor) : s.device;
    const void *src = to_block ? s.device : static_cast<const void *>(cursor);
    if (cudaMemcpyAsync(dst, src, s.bytes, cudaMemcpyDeviceToDevice, stream) !=
        cudaSuccess)
      return false;
    cursor += s.bytes;
  }
  return cudaStreamSynchronize(stream) == cudaSuccess;
}

// The graph must be a ready decode graph on this engine with no host-side
// recurrent state pending that would outlive a restore.
FullGraph *live_decode_graph(StreamingEngine &engine) {
  FullGraph *g = g_graph.get();
  if (!g || !g->ready || !g->decode_mode || g->engine != &engine || g->S != 1 ||
      !g->cuda_raw)
    return nullptr;
  return g;
}

}  // namespace

bool qwen4exp_full_graph_state_save(StreamingEngine &engine, int &handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  FullGraph *g = live_decode_graph(engine);
  std::vector<Span> spans;
  size_t total = 0;
  if (!g || !collect_spans(*g, spans, total) || cudaSetDevice(g->device) != cudaSuccess)
    return false;
  void *block = take_block(total);
  if (!block) return false;
  cudaStream_t stream = zeromoe_ggml_backend_cuda_get_stream(g->cuda_raw);
  if (!copy_spans(spans, block, /*to_block=*/true, stream)) {
    g_free_blocks.push_back({block, total});
    return false;
  }
  auto cp = std::make_unique<Checkpoint>();
  cp->graph_generation = g->graph_generation;
  cp->graph = g;
  cp->block = block;
  cp->block_bytes = total;
  cp->ple_tok_hist = engine.qwen4exp_ple_tok_hist;
  cp->ple_next_pos = engine.qwen4exp_ple_next_pos;
  cp->decode_last_pos = g->decode_last_pos;
  cp->host_state_dirty = g->decode_host_state_dirty;
  handle = g_next_handle++;
  g_checkpoints[handle] = std::move(cp);
  return true;
}

bool qwen4exp_full_graph_state_restore(StreamingEngine &engine, int handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  FullGraph *g = live_decode_graph(engine);
  const auto it = g_checkpoints.find(handle);
  std::vector<Span> spans;
  size_t total = 0;
  // A rebuilt graph has different storage; its checkpoints are meaningless.
  if (!g || it == g_checkpoints.end() || it->second->graph != g ||
      it->second->graph_generation != g->graph_generation ||
      !collect_spans(*g, spans, total) || total != it->second->block_bytes ||
      cudaSetDevice(g->device) != cudaSuccess)
    return false;
  cudaStream_t stream = zeromoe_ggml_backend_cuda_get_stream(g->cuda_raw);
  if (!copy_spans(spans, it->second->block, /*to_block=*/false, stream))
    return false;
  engine.qwen4exp_ple_tok_hist = it->second->ple_tok_hist;
  engine.qwen4exp_ple_next_pos = it->second->ple_next_pos;
  g->decode_last_pos = it->second->decode_last_pos;
  g->decode_host_state_dirty = it->second->host_state_dirty;
  return true;
}

void qwen4exp_full_graph_state_release(int handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto it = g_checkpoints.find(handle);
  if (it == g_checkpoints.end()) return;
  g_free_blocks.push_back({it->second->block, it->second->block_bytes});
  g_checkpoints.erase(it);
}

size_t qwen4exp_full_graph_state_bytes(StreamingEngine &engine) {
  std::lock_guard<std::mutex> lock(g_mutex);
  FullGraph *g = live_decode_graph(engine);
  std::vector<Span> spans;
  size_t total = 0;
  return g && collect_spans(*g, spans, total) ? total : 0;
}

void qwen4exp_full_graph_state_shutdown() {
  std::lock_guard<std::mutex> lock(g_mutex);
  for (const auto &entry : g_checkpoints) cudaFree(entry.second->block);
  g_checkpoints.clear();
  for (const auto &block : g_free_blocks) cudaFree(block.first);
  g_free_blocks.clear();
}

#endif  // ZEROMOE_USE_GGML_CUDA
