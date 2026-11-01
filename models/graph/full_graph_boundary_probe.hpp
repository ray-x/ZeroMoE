#pragma once

#include "graph_core.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>

namespace zeromoe::graph {

inline bool full_graph_boundary_probe_enabled_for_layer(int layer) {
  const char *directory = std::getenv("ZEROMOE_FULL_GRAPH_PROBE_DIR");
  if (!directory || !directory[0])
    return false;
  const char *selected = std::getenv("ZEROMOE_FULL_GRAPH_PROBE_LAYER");
  if (!selected || !selected[0])
    return true;
  char *end = nullptr;
  const long parsed = std::strtol(selected, &end, 10);
  if (end == selected || *end != '\0')
    return false;
  return parsed < 0 || parsed == layer;
}

inline std::string full_graph_boundary_safe_name(const std::string &name) {
  std::string result = name;
  for (char &ch : result) {
    const auto c = static_cast<unsigned char>(ch);
    if (!std::isalnum(c) && ch != '_' && ch != '-')
      ch = '_';
  }
  return result;
}

// Register an existing tensor at a model-defined semantic boundary. No GGML
// arithmetic node or copy is introduced by this helper.
inline void capture_full_graph_boundary(GraphCore &graph, int layer,
                                        const char *stage,
                                        ggml_tensor *tensor) {
  if (!tensor || !stage || !graph.graph ||
      !full_graph_boundary_probe_enabled_for_layer(layer))
    return;
  ggml_set_output(tensor);
  ggml_build_forward_expand(graph.graph, tensor);
  graph.boundary_taps.push_back({layer, stage, tensor});
}

// Call after scheduler synchronization. Values are saved as raw GGML bytes;
// metadata includes type, dimensions, strides, op, and backend so the same
// comparator can consume dense or routed tensors from different families.
inline bool dump_full_graph_boundary_taps(GraphCore &graph) {
  const char *directory = std::getenv("ZEROMOE_FULL_GRAPH_PROBE_DIR");
  if (!directory || !directory[0] || graph.boundary_taps.empty())
    return true;

  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) {
    LOG_ERROR("[FULL-GRAPH-PROBE] cannot create %s: %s\n", directory,
              error.message().c_str());
    return false;
  }

  for (const GraphBoundaryTap &tap : graph.boundary_taps) {
    if (!tap.tensor)
      return false;
    const size_t bytes = ggml_nbytes(tap.tensor);
    std::vector<uint8_t> data(bytes);
    if (bytes != 0)
      ggml_backend_tensor_get(tap.tensor, data.data(), 0, bytes);

    const std::string stem = "L" + std::to_string(tap.layer) + "." +
                             full_graph_boundary_safe_name(tap.stage);
    const auto binary_path =
        std::filesystem::path(directory) / (stem + ".bin");
    std::ofstream binary(binary_path, std::ios::binary | std::ios::app);
    if (!binary) {
      LOG_ERROR("[FULL-GRAPH-PROBE] cannot write %s\n",
                binary_path.string().c_str());
      return false;
    }
    if (bytes != 0)
      binary.write(reinterpret_cast<const char *>(data.data()),
                   static_cast<std::streamsize>(bytes));
    if (!binary) {
      LOG_ERROR("[FULL-GRAPH-PROBE] short write to %s\n",
                binary_path.string().c_str());
      return false;
    }

    ggml_backend_t backend =
        graph.sched ? ggml_backend_sched_get_tensor_backend(graph.sched,
                                                            tap.tensor)
                    : nullptr;
    const char *backend_name = backend == graph.cuda ? "CUDA"
                               : backend == graph.cpu ? "CPU"
                                                      : "unknown";
    const auto metadata_path =
        std::filesystem::path(directory) / (stem + ".meta");
    std::ofstream metadata(metadata_path, std::ios::trunc);
    if (!metadata) {
      LOG_ERROR("[FULL-GRAPH-PROBE] cannot write %s\n",
                metadata_path.string().c_str());
      return false;
    }
    metadata << "layer=" << tap.layer << '\n'
             << "stage=" << tap.stage << '\n'
             << "type=" << ggml_type_name(tap.tensor->type) << '\n'
             << "op=" << ggml_op_name(tap.tensor->op) << '\n'
             << "backend=" << backend_name << '\n'
             << "ne=" << tap.tensor->ne[0] << ',' << tap.tensor->ne[1] << ','
             << tap.tensor->ne[2] << ',' << tap.tensor->ne[3] << '\n'
             << "nb=" << tap.tensor->nb[0] << ',' << tap.tensor->nb[1] << ','
             << tap.tensor->nb[2] << ',' << tap.tensor->nb[3] << '\n'
             << "nbytes=" << bytes << '\n';
    if (!metadata)
      return false;
  }
  return true;
}

} // namespace zeromoe::graph
