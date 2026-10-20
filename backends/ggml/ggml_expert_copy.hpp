#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

// New ggml moved selective MoE transfers out of its scheduler and into this
// application callback. Older forks implement patches 0007/0008 internally.
#if defined(__GNUC__) && !defined(_WIN32)
extern "C" void ggml_backend_sched_set_copy_callback(
    ggml_backend_sched_t,
    bool (*)(ggml_backend_t, const ggml_tensor *, ggml_tensor *, ggml_cgraph *, void *),
    void *) __attribute__((weak));
#endif

namespace moestream::ggmlcopy {

struct ExpertCopy {
  const ggml_tensor *last_ids = nullptr;
  std::vector<int32_t> ids;
  std::vector<uint8_t> used;
  uint64_t plane_copies = 0;
  uint64_t bytes = 0;
  bool attached = false;

  void reset() {
    last_ids = nullptr;
    plane_copies = bytes = 0;
  }

  static bool copy(ggml_backend_t backend, const ggml_tensor *src,
                   ggml_tensor *dst, ggml_cgraph *graph, void *opaque) {
    auto &state = *static_cast<ExpertCopy *>(opaque);
    const ggml_tensor *routes = nullptr;
    // A cold branch can start with masking operations, not MUL_MAT_ID.
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
      const ggml_tensor *node = ggml_graph_node(graph, i);
      if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] == dst) {
        routes = node->src[2];
        break;
      }
    }
    if (!routes) {
      // Never fall back to a bulk read of a sparse expert arena.
      if (src->ne[2] > 1 && std::strstr(src->name, "_exps.weight"))
        GGML_ABORT("expert weight copy has no consuming MUL_MAT_ID: %s", src->name);
      return false;
    }
    GGML_ASSERT(src->data && routes->type == GGML_TYPE_I32);
    GGML_ASSERT(routes->ne[2] == 1 && routes->ne[3] == 1);
    GGML_ASSERT(src->ne[2] > 0 && src->ne[2] <= 65536);
    GGML_ASSERT(src->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src->ne[0] == dst->ne[0] && src->ne[1] == dst->ne[1] &&
                src->ne[2] == dst->ne[2] && src->type == dst->type);
    GGML_ASSERT(src->nb[1] == dst->nb[1]);
    const size_t route_bytes = ggml_nbytes(routes);
    // Bounded activation scratch, never an expert-weight mirror.
    GGML_ASSERT(route_bytes <= 1024 * 1024 && route_bytes % sizeof(int32_t) == 0);
    if (state.last_ids != routes || state.used.size() != (size_t)src->ne[2]) {
      state.ids.resize(route_bytes / sizeof(int32_t));
      ggml_backend_tensor_get_async(backend, routes, state.ids.data(), 0, route_bytes);
      ggml_backend_synchronize(backend);
      state.used.assign((size_t)src->ne[2], 0);
      for (int64_t t = 0; t < routes->ne[1]; ++t) {
        for (int64_t k = 0; k < routes->ne[0]; ++k) {
          const int32_t id = state.ids[(t * routes->nb[1] + k * routes->nb[0]) / sizeof(int32_t)];
          if (id < 0) continue; // masked hot routes are inactive in this branch
          GGML_ASSERT(id < src->ne[2]);
          state.used[(size_t)id] = 1;
        }
      }
      state.last_ids = routes;
    }
    const size_t payload = ggml_row_size(src->type, src->ne[0]) +
                           (size_t)(src->ne[1] - 1) * src->nb[1];
    GGML_ASSERT(payload <= src->nb[2] && payload <= dst->nb[2]);
    static const std::array<uint8_t, 512> zeros{};
    for (size_t e = 0; e < state.used.size(); ++e) {
      if (!state.used[e]) continue;
      const size_t from = e * src->nb[2], to = e * dst->nb[2];
      GGML_ASSERT(from <= ggml_nbytes(src) - payload);
      GGML_ASSERT(to <= ggml_nbytes(dst) - payload);
      // Each slot can be registered separately. Do not merge adjacent copies
      // or read padding from the next, possibly unpopulated, source slot.
      ggml_backend_tensor_set_async(backend, dst,
          static_cast<const uint8_t *>(src->data) + from, to, payload);
      const size_t padding = std::min(zeros.size(), ggml_nbytes(dst) - to - payload);
      if (padding)
        ggml_backend_tensor_set_async(backend, dst, zeros.data(), to + payload, padding);
      ++state.plane_copies;
      state.bytes += payload;
    }
    return true;
  }

  bool attach(ggml_backend_sched_t sched) {
#if defined(__GNUC__) && !defined(_WIN32)
    if (ggml_backend_sched_set_copy_callback) {
      ggml_backend_sched_set_copy_callback(sched, copy, this);
      attached = true;
    }
#else
    (void)sched;
#endif
    return attached;
  }
};

} // namespace moestream::ggmlcopy
