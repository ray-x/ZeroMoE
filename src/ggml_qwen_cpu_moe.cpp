// Persistent CPU GGML routed-expert graph for Qwen4Exp.
//
// This file is intentionally independent of the CUDA MoE runner. It is used
// only by execution_layer.cpp's Qwen CPU-affinity path. It supports both the
// original compact fallback (selected experts copied into a temporary expert
// axis) and a BeeLlama-style strided mode that points a persistent
// MUL_MAT_ID graph at complete host-RAM expert arenas.

#include "ggml_moe_backend.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

#if defined(MOESTREAM_USE_GGML)

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

constexpr size_t kWeightContextBytes = 64ull * 1024ull;
constexpr size_t kGraphContextBytes = 2ull * 1024ull * 1024ull;

int default_cpu_threads() {
#if defined(_OPENMP)
  return std::max(1, omp_get_max_threads());
#else
  return 1;
#endif
}

bool type_is_valid(uint32_t type) {
  return type < static_cast<uint32_t>(GGML_TYPE_COUNT);
}

} // namespace

GgmlQwenCpuMoeRunner &GgmlQwenCpuMoeRunner::instance() {
  static GgmlQwenCpuMoeRunner runner;
  return runner;
}

bool GgmlQwenCpuMoeRunner::compiled_in() { return true; }

GgmlQwenCpuMoeRunner::~GgmlQwenCpuMoeRunner() { shutdown(); }

void GgmlQwenCpuMoeRunner::free_graph() {
  // All graph executions are synchronous, so no CPU worker can still be
  // touching these objects when a shape/type change reaches this function.
  if (galloc_) {
    ggml_gallocr_free(static_cast<ggml_gallocr_t>(galloc_));
    galloc_ = nullptr;
  }
  if (graph_ctx_) {
    ggml_free(static_cast<ggml_context *>(graph_ctx_));
    graph_ctx_ = nullptr;
  }
  graph_ = nullptr;
  x_ = ids_ = weights_ = output_ = nullptr;

  if (weight_buf_) {
    ggml_backend_buffer_free(
        static_cast<ggml_backend_buffer_t>(weight_buf_));
    weight_buf_ = nullptr;
  }
  for (void *buf : external_weight_bufs_) {
    if (buf)
      ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(buf));
  }
  external_weight_bufs_.clear();
  if (weight_ctx_) {
    ggml_free(static_cast<ggml_context *>(weight_ctx_));
    weight_ctx_ = nullptr;
  }
  gate_ = up_ = down_ = nullptr;
  if (weight_storage_) std::free(weight_storage_);
  weight_storage_ = nullptr;
  weight_storage_bytes_ = 0;
  n_experts_ = d_model_ = d_ff_ = topk_ = 0;
  gate_type_ = up_type_ = down_type_ = 0;
  strided_mode_ = false;
  expert_stride_ = 0;
  cached_sources_.clear();
}

void GgmlQwenCpuMoeRunner::shutdown() {
  free_graph();
  if (cpu_backend_) {
    ggml_backend_free(static_cast<ggml_backend_t>(cpu_backend_));
    cpu_backend_ = nullptr;
  }
  n_threads_ = 0;
}

bool GgmlQwenCpuMoeRunner::ensure_graph(
    const GgmlQwenCpuExpertRef &ref, int n_experts, int d_model, int d_ff,
    int topk, bool strided, const void *gate_plane, const void *up_plane,
    const void *down_plane, size_t expert_stride) {
  if (!type_is_valid(ref.gate_type) || !type_is_valid(ref.up_type) ||
      !type_is_valid(ref.down_type) || n_experts <= 0 || d_model <= 0 ||
      d_ff <= 0 || topk <= 0 || topk > n_experts ||
      (strided && (!gate_plane || !up_plane || !down_plane ||
                   expert_stride == 0)) ||
      (!strided && topk != n_experts))
    return false;

  const bool shape_matches =
      graph_ && n_experts_ == n_experts && d_model_ == d_model &&
      d_ff_ == d_ff && topk_ == topk && gate_type_ == ref.gate_type &&
      up_type_ == ref.up_type && down_type_ == ref.down_type &&
      strided_mode_ == strided &&
      (!strided || expert_stride_ == expert_stride);
  if (shape_matches) return true;

  free_graph();
  if (!cpu_backend_) {
    cpu_backend_ = ggml_backend_cpu_init();
    if (!cpu_backend_) return false;
    n_threads_ = default_cpu_threads();
    ggml_backend_cpu_set_n_threads(static_cast<ggml_backend_t>(cpu_backend_),
                                   n_threads_);
  }

  ggml_init_params wip = {};
  wip.mem_size = kWeightContextBytes + ggml_tensor_overhead() * 8;
  wip.no_alloc = true;
  ggml_context *wctx = ggml_init(wip);
  if (!wctx) return false;

  const ggml_type gate_type = static_cast<ggml_type>(ref.gate_type);
  const ggml_type up_type = static_cast<ggml_type>(ref.up_type);
  const ggml_type down_type = static_cast<ggml_type>(ref.down_type);
  ggml_tensor *gate = ggml_new_tensor_3d(
      wctx, gate_type, d_model, d_ff, n_experts);
  ggml_tensor *up = ggml_new_tensor_3d(
      wctx, up_type, d_model, d_ff, n_experts);
  ggml_tensor *down = ggml_new_tensor_3d(
      wctx, down_type, d_ff, d_model, n_experts);
  if (!gate || !up || !down) {
    ggml_free(wctx);
    return false;
  }

  const size_t gate_bytes_per_expert =
      ggml_nbytes(gate) / static_cast<size_t>(n_experts);
  const size_t up_bytes_per_expert =
      ggml_nbytes(up) / static_cast<size_t>(n_experts);
  const size_t down_bytes_per_expert =
      ggml_nbytes(down) / static_cast<size_t>(n_experts);
  if (gate_bytes_per_expert == 0 || up_bytes_per_expert == 0 ||
      down_bytes_per_expert == 0 ||
      gate_bytes_per_expert > SIZE_MAX / static_cast<size_t>(n_experts) ||
      up_bytes_per_expert > SIZE_MAX / static_cast<size_t>(n_experts) ||
      down_bytes_per_expert > SIZE_MAX / static_cast<size_t>(n_experts) ||
      (strided && (expert_stride < gate_bytes_per_expert ||
                   expert_stride < up_bytes_per_expert ||
                   expert_stride < down_bytes_per_expert ||
                   (n_experts > 1 &&
                    expert_stride > SIZE_MAX / static_cast<size_t>(n_experts - 1))))) {
    ggml_free(wctx);
    return false;
  }
  const size_t gate_bytes = gate_bytes_per_expert * static_cast<size_t>(n_experts);
  const size_t up_bytes = up_bytes_per_expert * static_cast<size_t>(n_experts);
  const size_t down_bytes = down_bytes_per_expert * static_cast<size_t>(n_experts);
  if (gate_bytes > SIZE_MAX - up_bytes ||
      gate_bytes + up_bytes > SIZE_MAX - down_bytes) {
    ggml_free(wctx);
    return false;
  }

  void *storage = nullptr;
  ggml_backend_buffer_t wbuf = nullptr;
  if (!strided) {
    const size_t storage_bytes = gate_bytes + up_bytes + down_bytes;
    if (posix_memalign(&storage, 64, storage_bytes) != 0 || !storage) {
      ggml_free(wctx);
      return false;
    }
    weight_storage_ = storage;
    weight_storage_bytes_ = storage_bytes;
    wbuf = ggml_backend_cpu_buffer_from_ptr(
        weight_storage_, weight_storage_bytes_);
    if (!wbuf ||
        ggml_backend_tensor_alloc(wbuf, gate, weight_storage_) !=
            GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(wbuf, up,
                                  static_cast<uint8_t *>(weight_storage_) +
                                      gate_bytes) !=
            GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(wbuf, down,
                                  static_cast<uint8_t *>(weight_storage_) +
                                      gate_bytes + up_bytes) !=
            GGML_STATUS_SUCCESS) {
      if (wbuf) ggml_backend_buffer_free(wbuf);
      ggml_free(wctx);
      std::free(weight_storage_);
      weight_storage_ = nullptr;
      weight_storage_bytes_ = 0;
      return false;
    }
  } else {
    // The three pointers refer to the first expert in one host-RAM arena.
    // GGML's expert-axis stride is the arena's fixed slab stride, so the
    // projection views skip each expert's other two projections and padding.
    gate->nb[2] = expert_stride;
    up->nb[2] = expert_stride;
    down->nb[2] = expert_stride;
    gate->nb[3] = expert_stride * static_cast<size_t>(n_experts);
    up->nb[3] = expert_stride * static_cast<size_t>(n_experts);
    down->nb[3] = expert_stride * static_cast<size_t>(n_experts);

    const void *planes[3] = {gate_plane, up_plane, down_plane};
    const size_t plane_bytes[3] = {gate_bytes_per_expert,
                                   up_bytes_per_expert,
                                   down_bytes_per_expert};
    ggml_tensor *tensors[3] = {gate, up, down};
    for (int i = 0; i < 3; ++i) {
      const size_t view_bytes =
          expert_stride * static_cast<size_t>(n_experts - 1) + plane_bytes[i];
      ggml_backend_buffer_t mbuf = ggml_backend_cpu_buffer_from_ptr(
          const_cast<void *>(planes[i]), view_bytes);
      if (!mbuf ||
          ggml_backend_tensor_alloc(mbuf, tensors[i],
                                     const_cast<void *>(planes[i])) !=
              GGML_STATUS_SUCCESS) {
        if (mbuf) ggml_backend_buffer_free(mbuf);
        for (void *old : external_weight_bufs_)
          if (old)
            ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(old));
        external_weight_bufs_.clear();
        ggml_free(wctx);
        return false;
      }
      external_weight_bufs_.push_back(mbuf);
    }
  }

  auto discard_weight_storage = [&]() {
    if (wbuf) {
      ggml_backend_buffer_free(wbuf);
      wbuf = nullptr;
    }
    for (void *buf : external_weight_bufs_) {
      if (buf)
        ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(buf));
    }
    external_weight_bufs_.clear();
    if (weight_storage_) {
      std::free(weight_storage_);
      weight_storage_ = nullptr;
      weight_storage_bytes_ = 0;
    }
  };

  ggml_init_params gip = {};
  gip.mem_size = kGraphContextBytes + ggml_graph_overhead() +
                 ggml_tensor_overhead() * 256;
  gip.no_alloc = true;
  ggml_context *gctx = ggml_init(gip);
  if (!gctx) {
    discard_weight_storage();
    ggml_free(wctx);
    return false;
  }
  ggml_cgraph *graph = ggml_new_graph(gctx);
  ggml_tensor *x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d_model, 1);
  ggml_tensor *ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, topk, 1);
  ggml_tensor *weights =
      ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 1, topk, 1);
  if (!graph || !x || !ids || !weights) {
    ggml_free(gctx);
    discard_weight_storage();
    ggml_free(wctx);
    return false;
  }
  ggml_set_input(x);
  ggml_set_input(ids);
  ggml_set_input(weights);

  ggml_tensor *input3 = ggml_reshape_3d(gctx, x, d_model, 1, 1);
  ggml_tensor *gate_out = ggml_mul_mat_id(gctx, gate, input3, ids);
  ggml_tensor *up_out = ggml_mul_mat_id(gctx, up, input3, ids);
  ggml_tensor *act = ggml_swiglu_split(gctx, gate_out, up_out);
  ggml_tensor *down_out = ggml_mul_mat_id(gctx, down, act, ids);
  ggml_tensor *weighted = ggml_mul(gctx, down_out, weights);
  if (!input3 || !gate_out || !up_out || !act || !down_out || !weighted) {
    ggml_free(gctx);
    discard_weight_storage();
    ggml_free(wctx);
    return false;
  }

  ggml_tensor *sum = nullptr;
  for (int k = 0; k < topk; ++k) {
    ggml_tensor *expert = ggml_view_2d(
        gctx, weighted, d_model, 1, weighted->nb[2],
        static_cast<size_t>(k) * weighted->nb[1]);
    sum = k == 0 ? expert : ggml_add(gctx, sum, expert);
  }
  if (!sum) {
    ggml_free(gctx);
    discard_weight_storage();
    ggml_free(wctx);
    return false;
  }
  if (topk > 1) sum = ggml_cont(gctx, sum);
  ggml_set_output(sum);
  ggml_set_name(sum, "qwen4exp_cpu_moe_out");
  ggml_build_forward_expand(graph, sum);

  ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
  if (!ga || !ggml_gallocr_alloc_graph(ga, graph)) {
    if (ga) ggml_gallocr_free(ga);
    ggml_free(gctx);
    discard_weight_storage();
    ggml_free(wctx);
    return false;
  }

  weight_ctx_ = wctx;
  weight_buf_ = wbuf;
  gate_ = gate;
  up_ = up;
  down_ = down;
  graph_ctx_ = gctx;
  galloc_ = ga;
  graph_ = graph;
  x_ = x;
  ids_ = ids;
  weights_ = weights;
  output_ = sum;
  n_experts_ = n_experts;
  d_model_ = d_model;
  d_ff_ = d_ff;
  topk_ = topk;
  gate_type_ = ref.gate_type;
  up_type_ = ref.up_type;
  down_type_ = ref.down_type;
  strided_mode_ = strided;
  expert_stride_ = strided ? expert_stride : 0;
  return true;
}

bool GgmlQwenCpuMoeRunner::run(
    const float *input, float *output, const GgmlQwenCpuExpertRef *experts,
    int n_experts, const float *router_weights, int d_model, int d_ff,
    int topk) {
  if (!input || !output || !experts || !router_weights || n_experts <= 0 ||
      n_experts != topk || d_model <= 0 || d_ff <= 0)
    return false;

  const GgmlQwenCpuExpertRef &first = experts[0];
  if (!first.gate || !first.up || !first.down || first.gate_bytes == 0 ||
      first.up_bytes == 0 || first.down_bytes == 0 ||
      !type_is_valid(first.gate_type) || !type_is_valid(first.up_type) ||
      !type_is_valid(first.down_type))
    return false;
  for (int i = 1; i < n_experts; ++i) {
    const GgmlQwenCpuExpertRef &e = experts[i];
    if (!e.gate || !e.up || !e.down || e.gate_bytes != first.gate_bytes ||
        e.up_bytes != first.up_bytes || e.down_bytes != first.down_bytes ||
        e.gate_type != first.gate_type || e.up_type != first.up_type ||
        e.down_type != first.down_type)
      return false;
  }
  if (!ensure_graph(first, n_experts, d_model, d_ff, topk, false, nullptr,
                    nullptr, nullptr, 0))
    return false;

  const size_t gate_plane = weight_storage_ == nullptr
                                ? 0
                                : ggml_nbytes(static_cast<ggml_tensor *>(gate_)) /
                                      static_cast<size_t>(n_experts_);
  const size_t up_plane = ggml_nbytes(static_cast<ggml_tensor *>(up_)) /
                          static_cast<size_t>(n_experts_);
  const size_t down_plane = ggml_nbytes(static_cast<ggml_tensor *>(down_)) /
                            static_cast<size_t>(n_experts_);
  if (gate_plane != first.gate_bytes || up_plane != first.up_bytes ||
      down_plane != first.down_bytes)
    return false;

  uint8_t *storage = static_cast<uint8_t *>(weight_storage_);
  const size_t gate_total = gate_plane * static_cast<size_t>(n_experts_);
  const size_t up_total = up_plane * static_cast<size_t>(n_experts_);
  if (cached_sources_.size() != static_cast<size_t>(n_experts_))
    cached_sources_.assign(static_cast<size_t>(n_experts_), {});
  for (int i = 0; i < n_experts_; ++i) {
    const GgmlQwenCpuExpertRef &expert = experts[i];
    auto &cached = cached_sources_[static_cast<size_t>(i)];
    const bool unchanged =
        cached.gate == expert.gate && cached.up == expert.up &&
        cached.down == expert.down && cached.gate_bytes == gate_plane &&
        cached.up_bytes == up_plane && cached.down_bytes == down_plane;
    if (!unchanged) {
      std::memcpy(storage + static_cast<size_t>(i) * gate_plane, expert.gate,
                  gate_plane);
      std::memcpy(storage + gate_total + static_cast<size_t>(i) * up_plane,
                  expert.up, up_plane);
      std::memcpy(storage + gate_total + up_total +
                      static_cast<size_t>(i) * down_plane,
                  expert.down, down_plane);
      cached.gate = expert.gate;
      cached.up = expert.up;
      cached.down = expert.down;
      cached.gate_bytes = gate_plane;
      cached.up_bytes = up_plane;
      cached.down_bytes = down_plane;
    }
  }

  std::memcpy(static_cast<ggml_tensor *>(x_)->data, input,
              static_cast<size_t>(d_model_) * sizeof(float));
  auto *ids = static_cast<int32_t *>(static_cast<ggml_tensor *>(ids_)->data);
  for (int i = 0; i < topk_; ++i) ids[i] = i;
  std::memcpy(static_cast<ggml_tensor *>(weights_)->data, router_weights,
              static_cast<size_t>(topk_) * sizeof(float));

  if (ggml_backend_graph_compute(static_cast<ggml_backend_t>(cpu_backend_),
                                 static_cast<ggml_cgraph *>(graph_)) !=
      GGML_STATUS_SUCCESS)
    return false;
  std::memcpy(output, static_cast<ggml_tensor *>(output_)->data,
              static_cast<size_t>(d_model_) * sizeof(float));
  return true;
}

bool GgmlQwenCpuMoeRunner::run_strided(
    const float *input, float *output, const GgmlQwenCpuExpertRef &layer,
    const void *gate_plane, const void *up_plane, const void *down_plane,
    const int *expert_ids, int n_experts, const float *router_weights,
    int d_model, int d_ff, int topk, size_t expert_stride) {
  if (!input || !output || !gate_plane || !up_plane || !down_plane ||
      !expert_ids || !router_weights || n_experts <= 0 || topk <= 0 ||
      topk > n_experts || d_model <= 0 || d_ff <= 0 ||
      layer.gate_bytes == 0 || layer.up_bytes == 0 || layer.down_bytes == 0 ||
      expert_stride == 0 ||
      !type_is_valid(layer.gate_type) || !type_is_valid(layer.up_type) ||
      !type_is_valid(layer.down_type))
    return false;
  for (int k = 0; k < topk; ++k) {
    if (expert_ids[k] < 0 || expert_ids[k] >= n_experts)
      return false;
  }

  if (!ensure_graph(layer, n_experts, d_model, d_ff, topk, true,
                    gate_plane, up_plane, down_plane, expert_stride))
    return false;

  // The graph and its allocator are persistent, but the source planes can
  // belong to a different layer on the next call. GGML's CPU backend reads
  // tensor->data directly, so retarget the three external views without
  // allocating or copying the model weights.
  static_cast<ggml_tensor *>(gate_)->data = const_cast<void *>(gate_plane);
  static_cast<ggml_tensor *>(up_)->data = const_cast<void *>(up_plane);
  static_cast<ggml_tensor *>(down_)->data = const_cast<void *>(down_plane);

  std::memcpy(static_cast<ggml_tensor *>(x_)->data, input,
              static_cast<size_t>(d_model_) * sizeof(float));
  auto *ids = static_cast<int32_t *>(static_cast<ggml_tensor *>(ids_)->data);
  for (int k = 0; k < topk_; ++k)
    ids[k] = static_cast<int32_t>(expert_ids[k]);
  std::memcpy(static_cast<ggml_tensor *>(weights_)->data, router_weights,
              static_cast<size_t>(topk_) * sizeof(float));

  if (ggml_backend_graph_compute(static_cast<ggml_backend_t>(cpu_backend_),
                                 static_cast<ggml_cgraph *>(graph_)) !=
      GGML_STATUS_SUCCESS)
    return false;
  std::memcpy(output, static_cast<ggml_tensor *>(output_)->data,
              static_cast<size_t>(d_model_) * sizeof(float));
  return true;
}

#else

GgmlQwenCpuMoeRunner &GgmlQwenCpuMoeRunner::instance() {
  static GgmlQwenCpuMoeRunner runner;
  return runner;
}

bool GgmlQwenCpuMoeRunner::compiled_in() { return false; }

GgmlQwenCpuMoeRunner::~GgmlQwenCpuMoeRunner() = default;
void GgmlQwenCpuMoeRunner::shutdown() {}
bool GgmlQwenCpuMoeRunner::run(
    const float *, float *, const GgmlQwenCpuExpertRef *, int, const float *,
    int, int, int) {
  return false;
}
bool GgmlQwenCpuMoeRunner::run_strided(
    const float *, float *, const GgmlQwenCpuExpertRef &, const void *,
    const void *, const void *, const int *, int, const float *, int, int,
    int, size_t) {
  return false;
}

#endif
