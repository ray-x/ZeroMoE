#include "exl3_ggml_backend.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

extern "C" cudaStream_t moestream_ggml_backend_cuda_get_stream(
    ggml_backend_t backend);

namespace {

struct Adapter {
  struct CapturedSegment {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
  };
  struct TensorSignature {
    const ggml_tensor *node = nullptr;
    const void *data = nullptr;
    ggml_op op = GGML_OP_NONE;
    ggml_type type = GGML_TYPE_F32;
    std::array<int64_t, GGML_MAX_DIMS> ne{};
    std::array<size_t, GGML_MAX_DIMS> nb{};
    std::array<const ggml_tensor *, GGML_MAX_SRC> sources{};
    std::array<const void *, GGML_MAX_SRC> source_data{};
  };

  ggml_backend backend{};
  ggml_backend_t cuda_backend = nullptr;
  bool take_ownership = false;
  bool graph_capture_enabled = false;
  bool graph_captured = false;
  std::vector<CapturedSegment> captured_segments;
  std::vector<TensorSignature> captured_signature;
};

const char *adapter_name(ggml_backend_t) { return "EXL3-CUDA-ADAPTER"; }

ggml_backend_t inner_backend(ggml_backend_t backend) {
  auto *adapter = static_cast<Adapter *>(backend->context);
  return adapter->cuda_backend;
}

void adapter_free(ggml_backend_t backend) {
  auto *adapter = static_cast<Adapter *>(backend->context);
  for (const auto &segment : adapter->captured_segments) {
    if (segment.executable)
      cudaGraphExecDestroy(segment.executable);
    if (segment.graph)
      cudaGraphDestroy(segment.graph);
  }
  if (adapter->take_ownership)
    ggml_backend_free(adapter->cuda_backend);
  delete adapter;
}

void adapter_synchronize(ggml_backend_t backend) {
  ggml_backend_synchronize(inner_backend(backend));
}

void adapter_set_tensor_async(ggml_backend_t backend, ggml_tensor *tensor,
                              const void *data, size_t offset, size_t size) {
  ggml_backend_tensor_set_async(inner_backend(backend), tensor, data, offset,
                                size);
}

void adapter_get_tensor_async(ggml_backend_t backend,
                              const ggml_tensor *tensor, void *data,
                              size_t offset, size_t size) {
  ggml_backend_tensor_get_async(inner_backend(backend), tensor, data, offset,
                                size);
}

void adapter_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor *tensor,
                                 const void *data, size_t offset, size_t size,
                                 size_t n_copies, size_t stride_tensor,
                                 size_t stride_data) {
  ggml_backend_tensor_set_2d_async(inner_backend(backend), tensor, data,
                                   offset, size, n_copies, stride_tensor,
                                   stride_data);
}

void adapter_get_tensor_2d_async(ggml_backend_t backend,
                                 const ggml_tensor *tensor, void *data,
                                 size_t offset, size_t size, size_t n_copies,
                                 size_t stride_tensor, size_t stride_data) {
  ggml_backend_tensor_get_2d_async(inner_backend(backend), tensor, data,
                                   offset, size, n_copies, stride_tensor,
                                   stride_data);
}

bool adapter_copy_tensor_async(ggml_backend_t backend_src,
                               ggml_backend_t backend_dst,
                               const ggml_tensor *src, ggml_tensor *dst) {
  const auto unwrap = [](ggml_backend_t backend) {
    if (backend && backend->iface.free == adapter_free)
      return inner_backend(backend);
    return backend;
  };
  ggml_backend_tensor_copy_async(unwrap(backend_src), unwrap(backend_dst), src,
                                 dst);
  return true;
}

ggml_backend_graph_plan_t adapter_graph_plan_create(
    ggml_backend_t backend, const ggml_cgraph *graph) {
  auto *inner = inner_backend(backend);
  return inner->iface.graph_plan_create
             ? inner->iface.graph_plan_create(inner, graph)
             : nullptr;
}

void adapter_graph_plan_free(ggml_backend_t backend,
                             ggml_backend_graph_plan_t plan) {
  ggml_backend_graph_plan_free(inner_backend(backend), plan);
}

void adapter_graph_plan_update(ggml_backend_t backend,
                               ggml_backend_graph_plan_t plan,
                               const ggml_cgraph *graph) {
  auto *inner = inner_backend(backend);
  if (inner->iface.graph_plan_update)
    inner->iface.graph_plan_update(inner, plan, graph);
}

enum ggml_status adapter_graph_plan_compute(ggml_backend_t backend,
                                           ggml_backend_graph_plan_t plan) {
  return ggml_backend_graph_plan_compute(inner_backend(backend), plan);
}

void adapter_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
  ggml_backend_event_record(event, inner_backend(backend));
}

void adapter_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
  ggml_backend_event_wait(inner_backend(backend), event);
}

void adapter_graph_optimize(ggml_backend_t backend, ggml_cgraph *graph,
                            ggml_backend_graph_optimize_params *params) {
  auto *inner = inner_backend(backend);
  if (inner->iface.graph_optimize)
    inner->iface.graph_optimize(inner, graph, params);
}

bool is_exl3_custom(const ggml_tensor *node) {
  if (!node || (node->op != GGML_OP_MAP_CUSTOM2 &&
                node->op != GGML_OP_MAP_CUSTOM3) ||
      !node->extra)
    return false;
  const auto *custom = static_cast<const Exl3GgmlCustomNode *>(node->extra);
  return custom->magic == Exl3GgmlCustomNode::kMagic && custom->compute;
}

enum ggml_status adapter_graph_compute_eager(ggml_backend_t backend,
                                             ggml_cgraph *graph) {
  auto *adapter = static_cast<Adapter *>(backend->context);
  int segment_begin = 0;
  for (int index = 0; index < graph->n_nodes; ++index) {
    if (!is_exl3_custom(graph->nodes[index]))
      continue;
    if (segment_begin < index) {
      ggml_cgraph segment = ggml_graph_view(graph, segment_begin, index);
      if (ggml_backend_graph_compute_async(adapter->cuda_backend, &segment) !=
          GGML_STATUS_SUCCESS)
        return GGML_STATUS_FAILED;
    }
    auto *custom = static_cast<Exl3GgmlCustomNode *>(graph->nodes[index]->extra);
    try {
      custom->compute(adapter->cuda_backend, graph->nodes[index],
                      custom->user_data);
    } catch (const std::exception &error) {
      std::fprintf(stderr, "[EXL3-GGML] custom op failed: %s\n",
                   error.what());
      return GGML_STATUS_FAILED;
    } catch (...) {
      std::fprintf(stderr, "[EXL3-GGML] custom op failed: unknown error\n");
      return GGML_STATUS_FAILED;
    }
    segment_begin = index + 1;
  }
  if (segment_begin < graph->n_nodes) {
    ggml_cgraph segment = ggml_graph_view(graph, segment_begin, graph->n_nodes);
    if (ggml_backend_graph_compute_async(adapter->cuda_backend, &segment) !=
        GGML_STATUS_SUCCESS)
      return GGML_STATUS_FAILED;
  }
  return GGML_STATUS_SUCCESS;
}

enum ggml_status run_exl3_custom(ggml_backend_t cuda_backend,
                                 ggml_tensor *node) {
  auto *custom = static_cast<Exl3GgmlCustomNode *>(node->extra);
  try {
    custom->compute(cuda_backend, node, custom->user_data);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "[EXL3-GGML] custom op failed: %s\n", error.what());
    return GGML_STATUS_FAILED;
  } catch (...) {
    std::fprintf(stderr, "[EXL3-GGML] custom op failed: unknown error\n");
    return GGML_STATUS_FAILED;
  }
  return GGML_STATUS_SUCCESS;
}

bool graph_signature_matches(const Adapter *adapter,
                             const ggml_cgraph *graph) {
  if (!adapter->graph_captured ||
      adapter->captured_signature.size() !=
          static_cast<size_t>(graph->n_nodes))
    return false;
  for (int index = 0; index < graph->n_nodes; ++index) {
    const auto *node = graph->nodes[index];
    const auto &signature =
        adapter->captured_signature[static_cast<size_t>(index)];
    if (signature.node != node || signature.data != node->data ||
        signature.op != node->op || signature.type != node->type)
      return false;
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension)
      if (signature.ne[static_cast<size_t>(dimension)] != node->ne[dimension] ||
          signature.nb[static_cast<size_t>(dimension)] != node->nb[dimension])
        return false;
    for (int source = 0; source < GGML_MAX_SRC; ++source) {
      const ggml_tensor *source_node = node->src[source];
      if (signature.sources[static_cast<size_t>(source)] != source_node ||
          signature.source_data[static_cast<size_t>(source)] !=
              (source_node ? source_node->data : nullptr))
        return false;
    }
  }
  return true;
}

Adapter::TensorSignature tensor_signature(const ggml_tensor *node) {
  Adapter::TensorSignature signature;
  signature.node = node;
  signature.data = node->data;
  signature.op = node->op;
  signature.type = node->type;
  for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
    signature.ne[static_cast<size_t>(dimension)] = node->ne[dimension];
    signature.nb[static_cast<size_t>(dimension)] = node->nb[dimension];
  }
  for (int source = 0; source < GGML_MAX_SRC; ++source) {
    const ggml_tensor *source_node = node->src[source];
    signature.sources[static_cast<size_t>(source)] = source_node;
    signature.source_data[static_cast<size_t>(source)] =
        source_node ? source_node->data : nullptr;
  }
  return signature;
}

enum ggml_status run_captured_segment(ggml_backend_t backend,
                                      ggml_cgraph *graph, int begin, int end,
                                      size_t segment_index, bool capture) {
  auto *adapter = static_cast<Adapter *>(backend->context);
  if (begin == end) {
    if (capture)
      adapter->captured_segments.emplace_back();
    return GGML_STATUS_SUCCESS;
  }
  const cudaStream_t stream =
      moestream_ggml_backend_cuda_get_stream(adapter->cuda_backend);
  if (!stream)
    return GGML_STATUS_FAILED;
  if (!capture) {
    if (segment_index >= adapter->captured_segments.size())
      return GGML_STATUS_FAILED;
    const cudaGraphExec_t executable =
        adapter->captured_segments[segment_index].executable;
    return executable && cudaGraphLaunch(executable, stream) == cudaSuccess
               ? GGML_STATUS_SUCCESS
               : GGML_STATUS_FAILED;
  }

  ggml_cgraph view = ggml_graph_view(graph, begin, end);
  if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) !=
      cudaSuccess)
    return GGML_STATUS_FAILED;
  const enum ggml_status compute_status =
      adapter_graph_compute_eager(backend, &view);
  cudaGraph_t captured_graph = nullptr;
  const cudaError_t end_status = cudaStreamEndCapture(stream, &captured_graph);
  if (compute_status != GGML_STATUS_SUCCESS || end_status != cudaSuccess ||
      !captured_graph) {
    if (captured_graph)
      cudaGraphDestroy(captured_graph);
    return GGML_STATUS_FAILED;
  }
  cudaGraphExec_t executable = nullptr;
  const cudaError_t instantiate_status =
      cudaGraphInstantiate(&executable, captured_graph, nullptr, nullptr, 0);
  if (instantiate_status != cudaSuccess) {
    cudaGraphDestroy(captured_graph);
    return GGML_STATUS_FAILED;
  }
  adapter->captured_segments.push_back({captured_graph, executable});
  return cudaGraphLaunch(executable, stream) == cudaSuccess
             ? GGML_STATUS_SUCCESS
             : GGML_STATUS_FAILED;
}

enum ggml_status adapter_graph_compute(ggml_backend_t backend,
                                       ggml_cgraph *graph) {
  auto *adapter = static_cast<Adapter *>(backend->context);
  if (!adapter->graph_capture_enabled)
    return adapter_graph_compute_eager(backend, graph);

  if (adapter->graph_captured && !graph_signature_matches(adapter, graph))
    exl3_ggml_cuda_backend_reset_graph(backend);

  const bool capture = !adapter->graph_captured;
  const bool signature_matches = graph_signature_matches(adapter, graph);
  if (!capture && !signature_matches)
    return GGML_STATUS_FAILED;

  if (capture)
    adapter->captured_segments.clear();
  const auto fail = [&](enum ggml_status status) {
    if (capture)
      exl3_ggml_cuda_backend_reset_graph(backend);
    return status;
  };

  size_t segment_index = 0;
  int segment_begin = 0;
  for (int index = 0; index < graph->n_nodes; ++index) {
    if (!is_exl3_custom(graph->nodes[index]) ||
        static_cast<const Exl3GgmlCustomNode *>(graph->nodes[index]->extra)
            ->graph_capture_safe)
      continue;
    const enum ggml_status segment_status = run_captured_segment(
        backend, graph, segment_begin, index, segment_index++, capture);
    if (segment_status != GGML_STATUS_SUCCESS)
      return fail(segment_status);
    const enum ggml_status custom_status =
        run_exl3_custom(adapter->cuda_backend, graph->nodes[index]);
    if (custom_status != GGML_STATUS_SUCCESS)
      return fail(custom_status);
    segment_begin = index + 1;
  }
  const enum ggml_status final_status = run_captured_segment(
      backend, graph, segment_begin, graph->n_nodes, segment_index, capture);
  if (final_status != GGML_STATUS_SUCCESS)
    return fail(final_status);

  if (capture) {
    adapter->captured_signature.clear();
    adapter->captured_signature.reserve(static_cast<size_t>(graph->n_nodes));
    for (int index = 0; index < graph->n_nodes; ++index)
      adapter->captured_signature.push_back(
          tensor_signature(graph->nodes[index]));
    adapter->graph_captured = true;
    std::fprintf(stderr,
                 "[EXL3-GGML] captured %zu static graph segments around "
                 "%zu eager custom callbacks (%d graph nodes)\n",
                 adapter->captured_segments.size(), segment_index,
                 graph->n_nodes);
  }
  return GGML_STATUS_SUCCESS;
}

} // namespace

ggml_backend_t exl3_ggml_cuda_backend_wrap(ggml_backend_t cuda_backend,
                                            bool take_ownership) {
  if (!cuda_backend || !ggml_backend_is_cuda(cuda_backend))
    return nullptr;
  auto *adapter = new Adapter;
  adapter->cuda_backend = cuda_backend;
  adapter->take_ownership = take_ownership;
  adapter->backend.guid = ggml_backend_guid(cuda_backend);
  adapter->backend.device = ggml_backend_get_device(cuda_backend);
  adapter->backend.context = adapter;
  adapter->backend.iface = {
      adapter_name,
      adapter_free,
      adapter_set_tensor_async,
      adapter_get_tensor_async,
      adapter_set_tensor_2d_async,
      adapter_get_tensor_2d_async,
      adapter_copy_tensor_async,
      adapter_synchronize,
      adapter_graph_plan_create,
      adapter_graph_plan_free,
      adapter_graph_plan_update,
      adapter_graph_plan_compute,
      adapter_graph_compute,
      adapter_event_record,
      adapter_event_wait,
      adapter_graph_optimize};
  return &adapter->backend;
}

bool exl3_ggml_cuda_backend_set_graph_capture(ggml_backend_t backend,
                                              bool enabled) {
  if (!backend || !backend->context)
    return false;
  auto *adapter = static_cast<Adapter *>(backend->context);
  if (!enabled) {
    adapter->graph_capture_enabled = false;
    exl3_ggml_cuda_backend_reset_graph(backend);
    return true;
  }
  adapter->graph_capture_enabled = true;
  return true;
}

void exl3_ggml_cuda_backend_reset_graph(ggml_backend_t backend) {
  if (!backend || !backend->context)
    return;
  auto *adapter = static_cast<Adapter *>(backend->context);
  ggml_backend_synchronize(adapter->cuda_backend);
  for (const auto &segment : adapter->captured_segments) {
    if (segment.executable)
      cudaGraphExecDestroy(segment.executable);
    if (segment.graph)
      cudaGraphDestroy(segment.graph);
  }
  adapter->captured_segments.clear();
  adapter->captured_signature.clear();
  adapter->graph_captured = false;
}

ggml_tensor *exl3_ggml_map_custom2(ggml_context *ctx, ggml_tensor *input,
                                    ggml_tensor *opaque,
                                    Exl3GgmlCustomNode *node) {
  if (!ctx || !input || !opaque || !node ||
      node->magic != Exl3GgmlCustomNode::kMagic || !node->compute)
    return nullptr;
  auto *result = ggml_map_custom2(ctx, input, opaque, nullptr, 1, nullptr);
  if (result) result->extra = node;
  return result;
}

ggml_tensor *exl3_ggml_map_custom3(ggml_context *ctx, ggml_tensor *input,
                                    ggml_tensor *opaque, ggml_tensor *routing,
                                    Exl3GgmlCustomNode *node) {
  if (!ctx || !input || !opaque || !routing || !node ||
      node->magic != Exl3GgmlCustomNode::kMagic || !node->compute)
    return nullptr;
  auto *result = ggml_map_custom3(ctx, input, opaque, routing, nullptr, 1,
                                  nullptr);
  if (result)
    result->extra = node;
  return result;
}

cudaStream_t exl3_ggml_cuda_stream(ggml_backend_t cuda_backend) {
  return cuda_backend ? moestream_ggml_backend_cuda_get_stream(cuda_backend)
                      : nullptr;
}

#endif
