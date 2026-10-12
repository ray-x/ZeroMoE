#pragma once

// Family-independent state and helpers shared by every full-graph
// implementation (Qwen4Exp, GLM5Next / DeepSeek-V4, ...).
//
// A family's FullGraph derives from GraphCore and adds only what is specific to
// its architecture (recurrent-state tensors, indexer pools, expert-split
// tables, ...).  Everything that used to be copied between the families --
// the backend / context / graph handles, the three device arenas, the owned
// host allocations and the helpers that create host-bound input tensors --
// lives here once.
//
// Requires ggml and the CUDA runtime; include it only from translation units
// compiled with MOESTREAM_USE_GGML_CUDA.

#include <cmath>

#include "engine.hpp"
#include "early_ready_lane.hpp"
#include "logger.hpp"
#include "memory_guard.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace moestream::graph {

// A graph-owned host allocation (posix_memalign), optionally registered for
// DMA, with the MemoryGuard reservation that was charged for it.
struct HostAllocation {
  void *ptr = nullptr;
  size_t bytes = 0;
  bool pinned = false;
  size_t reservation_bytes = 0;
};

// One bump-allocated device buffer (dense weights, routed experts or decode
// state).  `buffer` is owned by whoever created it; a graph that borrows the
// arena of another graph keeps its own view and must not free it.
struct DeviceArena {
  ggml_backend_buffer_t buffer = nullptr;
  void *base = nullptr;
  size_t capacity = 0;
  size_t used = 0;
  int tensors = 0;

  bool valid() const { return buffer != nullptr; }
  // Forget the buffer without freeing it (borrowed or already freed).
  void forget() {
    buffer = nullptr;
    base = nullptr;
    capacity = 0;
    used = 0;
    tensors = 0;
  }
  void free_buffer() {
    if (buffer)
      ggml_backend_buffer_free(buffer);
    forget();
  }
};

struct GraphCore {
  // ---- identity and shape -------------------------------------------------
  StreamingEngine *engine = nullptr;
  int device = -1;
  int S = 0; // prompt length (prefill) or 1 (decode)
  int D = 0; // hidden size
  int HC = 0;
  int n_layers = 0;
  int H = 0;
  int hd = 0;
  int d_conv = 0;
  bool decode_mode = false;
  int kv_capacity = 0;
  int decode_last_pos = -1;
  int device_expert_layers = 0;

  // ---- ggml objects ---------------------------------------------------------
  ggml_backend_t cuda = nullptr;
  ggml_backend_t cpu = nullptr;
  ggml_backend_sched_t sched = nullptr;
  ggml_context *ctx = nullptr;
  ggml_cgraph *graph = nullptr;

  // ---- tensors every family has ----------------------------------------------
  ggml_tensor *input = nullptr;          // [D, HC, S], host input
  ggml_tensor *logits = nullptr;         // decode only: LM-head output
  ggml_tensor *output = nullptr;         // [D, S], after the HC mean
  ggml_tensor *final_streams = nullptr;
  ggml_tensor *attn_mask = nullptr;      // [S, S], causal mask
  ggml_tensor *decode_indices = nullptr; // [1], I32 KV row
  ggml_tensor *positions = nullptr;      // [S], I32 token positions

  // ---- device arenas -----------------------------------------------------------
  DeviceArena dense;  // dense / attention weights
  DeviceArena expert; // device-resident routed experts
  DeviceArena state;  // decode-time recurrent / KV state

  // ---- CPU lanes (per layer: cold experts concurrent with GPU work) ------------
  std::unordered_map<int, std::unique_ptr<EarlyReadyLane>> early_ready;

  // ---- host memory --------------------------------------------------------------
  std::vector<HostAllocation> owned;
  std::vector<ggml_backend_buffer_t> host_buffers;
  std::vector<void *> host_inputs;
  size_t host_weight_bytes = 0;
  size_t host_pinned_bytes = 0;
  size_t host_pin_failures = 0;
  size_t host_reservation_bytes = 0;
  size_t context_reservation_bytes = 0;
  // Bytes already charged to MemoryGuard for allocations this graph is about
  // to make; alloc_host() draws these down before charging more.
  size_t host_precharged_bytes = 0;
  bool ready = false;

  // Prefix of the "[<tag>-MEM]" warning when a host allocation is refused.
  const char *mem_tag = "GGML-FULL-MEM";

  GraphCore() = default;
  GraphCore(const GraphCore &) = delete;
  GraphCore &operator=(const GraphCore &) = delete;

  // Allocates 4096-byte-aligned host memory owned by this graph.  Returns
  // nullptr when the MemoryGuard budget refuses it.
  void *alloc_host(size_t bytes) {
    if (bytes == 0)
      return nullptr;
    size_t reservation = 0;
    if (host_precharged_bytes >= bytes) {
      host_precharged_bytes -= bytes;
    } else {
      reservation = bytes - host_precharged_bytes;
      host_precharged_bytes = 0;
      if (!MemoryGuard::try_reserve_host_bytes(reservation)) {
        LOG_WARN("[%s] host graph allocation refused: %.2f MB "
                 "(reserved=%.2f GB, budget=%.2f GB)\n",
                 mem_tag, bytes / 1e6, MemoryGuard::reserved_host_bytes() / 1e9,
                 MemoryGuard::host_budget_bytes() / 1e9);
        return nullptr;
      }
    }
    void *p = nullptr;
    const int err = posix_memalign(&p, 4096, bytes);
    if (err != 0 || !p) {
      MemoryGuard::release_host_bytes(reservation);
      return nullptr;
    }
    owned.push_back({p, bytes, false, reservation});
    return p;
  }

  // Wraps `ptr` in a CPU backend buffer and binds tensor `t` to it.
  ggml_backend_buffer_t bind_host(ggml_tensor *t, void *ptr, size_t bytes,
                                  ggml_backend_buffer_usage usage) {
    if (!t || !ptr || bytes < ggml_nbytes(t))
      return nullptr;
    ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr(ptr, bytes);
    if (!b)
      return nullptr;
    ggml_backend_buffer_set_usage(b, usage);
    if (ggml_backend_tensor_alloc(b, t, ptr) != GGML_STATUS_SUCCESS) {
      ggml_backend_buffer_free(b);
      return nullptr;
    }
    host_buffers.push_back(b);
    return b;
  }

  // Creates a graph input tensor backed by graph-owned host memory.
  ggml_tensor *host_input(ggml_context *c, ggml_type type,
                          const std::vector<int64_t> &shape, size_t bytes,
                          const char *name);

  // Frees, in this order: host-bound tensor buffers, the ggml context, the
  // owned host allocations (releasing their reservations), the context and
  // source reservations; then clears the tensors and counters common to all
  // families.  A family's destroy() calls this last, after it has released
  // its own backends, arenas and family-specific tensors.
  void release_core() {
    for (ggml_backend_buffer_t b : host_buffers)
      if (b)
        ggml_backend_buffer_free(b);
    host_buffers.clear();
    if (ctx) {
      ggml_free(ctx);
      ctx = nullptr;
    }
    for (HostAllocation &a : owned) {
      if (a.pinned)
        cudaHostUnregister(a.ptr);
      if (a.ptr)
        std::free(a.ptr);
      MemoryGuard::release_host_bytes(a.reservation_bytes);
      a.ptr = nullptr;
      a.pinned = false;
    }
    owned.clear();
    host_inputs.clear();
    cuda = nullptr;
    cpu = nullptr;
    graph = nullptr;
    input = output = logits = final_streams = positions = attn_mask =
        decode_indices = nullptr;
    ready = false;
    host_weight_bytes = 0;
    host_pinned_bytes = 0;
    host_pin_failures = 0;
    MemoryGuard::release_host_bytes(host_reservation_bytes);
    host_reservation_bytes = 0;
    MemoryGuard::release_host_bytes(context_reservation_bytes);
    context_reservation_bytes = 0;
    host_precharged_bytes = 0;
    engine = nullptr;
    S = 0;
    decode_mode = false;
    kv_capacity = 0;
    decode_last_pos = -1;
  }
};

// ---- tensor / model lookup helpers shared by every family -----------------

inline ggml_tensor *new_tensor(ggml_context *ctx, ggml_type type,
                               const std::vector<int64_t> &shape) {
  switch (shape.size()) {
  case 1:
    return ggml_new_tensor_1d(ctx, type, shape[0]);
  case 2:
    return ggml_new_tensor_2d(ctx, type, shape[0], shape[1]);
  case 3:
    return ggml_new_tensor_3d(ctx, type, shape[0], shape[1], shape[2]);
  case 4:
    return ggml_new_tensor_4d(ctx, type, shape[0], shape[1], shape[2],
                              shape[3]);
  default:
    return nullptr;
  }
}

inline const GGUFTensor *model_tensor(const StreamingEngine &engine,
                                      const std::string &name) {
  auto it = engine.model_file.tensors.find(name);
  return it == engine.model_file.tensors.end() ? nullptr : &it->second;
}

inline const QuantizedTensor *resident(const StreamingEngine &engine,
                                       const std::string &name) {
  auto it = engine.resident_layers.find(name);
  if (it == engine.resident_layers.end() || !it->second.host_ptr)
    return nullptr;
  return &it->second;
}

inline ggml_tensor *GraphCore::host_input(ggml_context *c, ggml_type type,
                                          const std::vector<int64_t> &shape,
                                          size_t bytes, const char *name) {
  ggml_tensor *t = new_tensor(c, type, shape);
  if (!t)
    return nullptr;
  const size_t required = std::max(bytes, ggml_nbytes(t));
  void *p = alloc_host(required);
  if (!p || !bind_host(t, p, required, GGML_BACKEND_BUFFER_USAGE_COMPUTE))
    return nullptr;
  ggml_set_input(t);
  ggml_set_name(t, name);
  host_inputs.push_back(p);
  return t;
}

// Call-site-compatible free forms (`bind_host(g, t, ptr, bytes, usage)`).
inline void *alloc_host(GraphCore &g, size_t bytes) { return g.alloc_host(bytes); }
inline ggml_backend_buffer_t bind_host(GraphCore &g, ggml_tensor *t, void *ptr,
                                       size_t bytes,
                                       ggml_backend_buffer_usage usage) {
  return g.bind_host(t, ptr, bytes, usage);
}
inline ggml_tensor *host_input(GraphCore &g, ggml_context *ctx, ggml_type type,
                               const std::vector<int64_t> &shape, size_t bytes,
                               const char *name) {
  return g.host_input(ctx, type, shape, bytes, name);
}

} // namespace moestream::graph
