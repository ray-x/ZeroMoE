// Qwen4Exp full graph: Diagnostics: per-layer timing callback, captured-graph write scan, tensor
// and hidden-state dumps, layout logging, the two-layer S=1 diagnostic.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_qwen4exp_graph_internal.hpp.

#include "ggml_qwen4exp_graph_internal.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)

using namespace qwen4exp_graph;

namespace qwen4exp_graph {

// Runtime guard (2026-09-19): CMakeLists.txt already documents that a plain
// ZEROMOE_USE_GGML_CUDA build can silently compile out ggml-cuda's CUDA
// graph capture/replay path (GGML_CUDA_USE_GRAPHS never defined) -- this
// exact failure mode cost an entire investigation before it was found once
// already (full-graph decode 20 tok/s -> 8 tok/s, mistaken for a
// correctness/scheduling bug). A build-time default is not enough: it can
// still silently drift (wrong CMake cache value, a stale build directory,
// an override left on from debugging). This checks the ACTUAL compiled
// binary's feature list via ggml's own backend-registry API, every single
// run, and logs impossible-to-miss output either way -- no one should have
// to remember to ask "is this still on" again.
void verify_cuda_graphs_compiled_in(ggml_backend_t cuda_backend) {
  static bool checked = false;
  if (checked || !cuda_backend)
    return;
  checked = true;
  ggml_backend_dev_t dev = ggml_backend_get_device(cuda_backend);
  ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
  auto get_features = reg ? reinterpret_cast<ggml_backend_get_features_t>(
                                ggml_backend_reg_get_proc_address(
                                    reg, "ggml_backend_get_features"))
                          : nullptr;
  bool has_graphs = false;
  if (get_features) {
    for (const ggml_backend_feature *f = get_features(reg);
        f && f->name; ++f) {
      if (std::strcmp(f->name, "USE_GRAPHS") == 0) {
        has_graphs = true;
        break;
      }
    }
  }
  if (has_graphs) {
    LOG_INFO("[CUDA-GRAPHS-CHECK] compiled in: yes (GGML_CUDA_USE_GRAPHS "
             "active in this binary)\n");
  } else {
    LOG_WARN("[CUDA-GRAPHS-CHECK] compiled in: NO -- this build silently "
             "lacks CUDA graph capture/replay (GGML_CUDA_USE_GRAPHS was not "
             "defined when ggml-cuda was compiled). Full-graph decode will "
             "run every node as an individual cudaLaunchKernel instead of "
             "one batched cudaGraphLaunch per token -- previously measured "
             "cost: ~20 tok/s -> ~8 tok/s on this same decode path. Rebuild "
             "with -DZEROMOE_ENABLE_CUDA_GRAPHS=ON (the CMakeLists.txt "
             "default) and verify with a fresh 'cmake -B <dir>' configure, "
             "not just an incremental build in a stale build directory.\n");
  }
}

// Parses the "blk.<N>." prefix ggml_qwen4exp_graph.cpp's own tensor names
// always use (see expert_weight/dense_weight/etc.) -- returns -1 if the
// name doesn't match (dense/global tensors with no per-layer index, e.g.
// the embedding or output head). Mirrors load_full_weight's own parsing.
int layer_timing_parse_layer(const char *name) {
  if (!name || std::strncmp(name, "blk.", 4) != 0)
    return -1;
  char *end = nullptr;
  const long n = std::strtol(name + 4, &end, 10);
  if (end == name + 4 || !end || *end != '.')
    return -1;
  return static_cast<int>(n);
}

// ggml_backend_sched_eval_callback: fires with ask=true immediately before
// a node runs, ask=false immediately after. For CPU-backend nodes this
// synchronously brackets real compute time (ggml-cpu has no async
// concept); for CUDA-backend nodes it would only bracket the async launch,
// so those are skipped entirely -- GPU-side timing already exists from the
// nsys work elsewhere in this investigation. See FullGraph::layer_timing_*
// fields and the memory note this feature's plan came from.
bool layer_timing_eval_callback(struct ggml_tensor *t, bool ask,
                                void *user_data) {
  FullGraph *g = static_cast<FullGraph *>(user_data);
  if (!g || !t)
    return true;
  if (g->layer_timing_verbose_enabled && !g->layer_timing_verbose_dumped) {
    // One-shot: log EVERY node (both backends), first decode token only,
    // so the real node/backend sequence bordering a CPU-dispatched MoE
    // layer is visible directly instead of inferred from the undiferen-
    // tiated gap bucket. ask=true is logged with the gap since the last
    // ask=false of ANY node (not just CPU ones, unlike the timing above)
    // so a GPU-backend D2H/H2D copy sitting in that gap shows up as real
    // elapsed time next to its own name/op/backend.
    ggml_backend_t node_be = ggml_backend_sched_get_tensor_backend(g->sched, t);
    const char *be_name = node_be == g->cpu ? "CPU"
                          : node_be == g->cuda ? "CUDA"
                                               : "?";
    const auto vnow = std::chrono::steady_clock::now();
    double gap_ms = 0.0;
    if (g->layer_timing_verbose_have_last)
      gap_ms = std::chrono::duration<double, std::milli>(
                   vnow - g->layer_timing_verbose_last)
                   .count();
    auto vtag_it = g->layer_timing_layer_of.find(t);
    char tag_buf[32] = "-";
    if (vtag_it != g->layer_timing_layer_of.end())
      std::snprintf(tag_buf, sizeof(tag_buf), "L%d/%s",
                    vtag_it->second.first, vtag_it->second.second);
    LOG_INFO("[GGML-QWEN-FULL-LAYER-TIMING-VERBOSE] ask=%d name=%s op=%s "
             "backend=%s tag=%s since_last=%.4fms\n",
             ask ? 1 : 0, ggml_get_name(t), ggml_op_name(t->op), be_name,
             tag_buf, gap_ms);
    g->layer_timing_verbose_last = vnow;
    g->layer_timing_verbose_have_last = true;
    if (!ask && g->n_layers > 0) {
      // End-of-graph marker: the LAST node's ask=false has fired once we
      // see the final layer's down-projection result. Token 1 is reliably
      // JIT/warm-up-contaminated (confirmed: ~4ms MoE ops vs ~0.3-0.4ms
      // steady-state) -- capture several tokens, not just one, so a real
      // steady-state sample is available alongside the warm-up outlier.
      if (t == g->output || t == g->logits) {
        ++g->layer_timing_verbose_tokens_seen;
        if (g->layer_timing_verbose_tokens_seen >= 10)
          g->layer_timing_verbose_dumped = true;
      }
    }
  }
  if (ggml_backend_sched_get_tensor_backend(g->sched, t) != g->cpu)
    return true; // GPU node or unassigned -- not timeable this way, skip
  const auto now = std::chrono::steady_clock::now();
  if (ask) {
    // Gap since the LAST CPU node's ask=false -- dispatch/barrier/
    // scheduling overhead the op-bracket timing below cannot see, since
    // it only covers what happens INSIDE one node's own bracket.
    if (g->layer_timing_have_last_ask_false) {
      g->layer_timing_gap_ms_total +=
          std::chrono::duration<double, std::milli>(
              now - g->layer_timing_last_ask_false)
              .count();
      ++g->layer_timing_gap_count;
    }
    g->layer_timing_t0[t] = now;
    return true;
  }
  g->layer_timing_last_ask_false = now;
  g->layer_timing_have_last_ask_false = true;
  auto it = g->layer_timing_t0.find(t);
  if (it == g->layer_timing_t0.end())
    return true; // no matching ask=true (shouldn't happen, be defensive)
  const double ms =
      std::chrono::duration<double, std::milli>(now - it->second).count();
  g->layer_timing_t0.erase(it);
  // Op result tensors (mul_mat_id/swiglu outputs) get ggml's auto "node_
  // NNNN" name, not a "blk.N.stem" one -- only build_moe()'s own explicit
  // layer_timing_layer_of recording (populated at graph-build time, see
  // its own comment) can attribute these correctly. Fall back to name
  // parsing for anything build_moe() didn't tag (e.g. weight leaves, if
  // this callback ever fires on one) so the op-level map still groups
  // something meaningful rather than silently dropping it.
  auto tag_it = g->layer_timing_layer_of.find(t);
  const bool tagged = tag_it != g->layer_timing_layer_of.end();
  const int layer = tagged ? tag_it->second.first
                           : layer_timing_parse_layer(ggml_get_name(t));
  if (layer >= 0 && layer < static_cast<int>(g->layer_timing_cpu_ms.size()))
    g->layer_timing_cpu_ms[static_cast<size_t>(layer)] += ms;
  std::string stem;
  if (tagged) {
    stem = tag_it->second.second;
  } else {
    const char *name = ggml_get_name(t);
    stem = name;
    if (layer >= 0) {
      // Strip the "blk.<N>." prefix so the op-level map groups by stem
      // across all layers, e.g. "ffn_down_exps.weight" regardless of
      // which layer -- same convention expert_weight() itself uses.
      const char *dot = std::strchr(name + 4, '.');
      if (dot && dot[1] != '\0')
        stem = dot + 1;
    }
  }
  g->layer_timing_op_ms[stem] += ms;
  return true;
}

// Find captured CUDA memcpy/memset nodes that target a protected expert range.
// GGML tensor-node overlap checks cannot see raw transfer nodes introduced by
// the CUDA backend or scheduler, so inspect each split's captured graph too.
void qwen4exp_scan_captured_graph_writes(FullGraph &g,
                                               const void *range_base,
                                               size_t range_bytes) {
  if (!g.graph || !g.sched || !range_base || range_bytes == 0)
    return;
  const uintptr_t protected_begin =
      reinterpret_cast<uintptr_t>(range_base);
  if (range_bytes > UINTPTR_MAX - protected_begin)
    return;
  const uintptr_t protected_end = protected_begin + range_bytes;

  std::unordered_set<cudaGraph_t> captured_graphs;
  for (int index = 0; index < ggml_graph_n_nodes(g.graph); ++index) {
    ggml_tensor *tensor = ggml_graph_node(g.graph, index);
    if (!tensor)
      continue;
    ggml_backend_t backend =
        ggml_backend_sched_get_tensor_backend(g.sched, tensor);
    if (!backend)
      continue;
    cudaGraph_t captured =
        zeromoe_ggml_cuda_get_captured_graph(backend, tensor);
    if (!captured && backend != g.cuda)
      captured = zeromoe_ggml_cuda_get_captured_graph(g.cuda, tensor);
    if (captured)
      captured_graphs.insert(captured);
  }

  size_t graph_nodes = 0;
  size_t memcpy_nodes = 0;
  size_t memset_nodes = 0;
  size_t kernel_nodes = 0;
  size_t kernel_param_checks = 0;
  size_t kernel_param_overlaps = 0;
  size_t kernel_param_scan_failures = 0;
  size_t overlapping_writes = 0;
  size_t overlap_logs = 0;
  auto overlaps = [&](const void *destination, size_t bytes) {
    if (!destination || bytes == 0)
      return false;
    const uintptr_t begin = reinterpret_cast<uintptr_t>(destination);
    if (bytes > UINTPTR_MAX - begin)
      return false;
    const uintptr_t end = begin + bytes;
    return begin < protected_end && protected_begin < end;
  };
  auto add_bytes = [](size_t a, size_t b, size_t *out) {
    if (!out || b > SIZE_MAX - a)
      return false;
    *out = a + b;
    return true;
  };
  auto multiply_bytes = [](size_t a, size_t b, size_t *out) {
    if (!out || (a != 0 && b > SIZE_MAX / a))
      return false;
    *out = a * b;
    return true;
  };

  auto visit_graph = [&](cudaGraph_t graph, int depth,
                         const auto &self) -> void {
    if (!graph || depth > 8)
      return;
    size_t node_count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &node_count) != cudaSuccess)
      return;
    std::vector<cudaGraphNode_t> nodes(node_count);
    if (node_count &&
        cudaGraphGetNodes(graph, nodes.data(), &node_count) != cudaSuccess)
      return;
    graph_nodes += node_count;
    for (size_t index = 0; index < node_count; ++index) {
      const cudaGraphNode_t node = nodes[index];
      cudaGraphNodeType type{};
      if (cudaGraphNodeGetType(node, &type) != cudaSuccess)
        continue;
      if (type == cudaGraphNodeTypeMemcpy) {
        ++memcpy_nodes;
        cudaMemcpy3DParms params{};
        if (cudaGraphMemcpyNodeGetParams(node, &params) != cudaSuccess ||
            !params.dstPtr.ptr)
          continue;
        const size_t width = params.extent.width;
        const size_t height = std::max<size_t>(1, params.extent.height);
        const size_t depth_count = std::max<size_t>(1, params.extent.depth);
        size_t span = width;
        size_t rows = 0;
        if (height > 1 &&
            (!multiply_bytes(height - 1, params.dstPtr.pitch, &rows) ||
             !add_bytes(span, rows, &span)))
          continue;
        if (depth_count > 1) {
          size_t slice_pitch = 0, slices = 0;
          const size_t ysize = params.dstPtr.ysize
                                   ? params.dstPtr.ysize
                                   : height;
          if (!multiply_bytes(params.dstPtr.pitch, ysize, &slice_pitch) ||
              !multiply_bytes(depth_count - 1, slice_pitch, &slices) ||
              !add_bytes(span, slices, &span))
            continue;
        }
        if (overlaps(params.dstPtr.ptr, span)) {
          ++overlapping_writes;
          if (overlap_logs++ < 24)
            LOG_INFO("[QWEN4EXP-P2-UP-CUDA-WRITER] kind=memcpy graph=%p "
                     "node=%p dst=%p span=%zu copy_kind=%d depth=%d\n",
                     static_cast<void *>(graph), static_cast<void *>(node),
                     params.dstPtr.ptr, span, static_cast<int>(params.kind),
                     depth);
        }
      } else if (type == cudaGraphNodeTypeMemset) {
        ++memset_nodes;
        cudaMemsetParams params{};
        if (cudaGraphMemsetNodeGetParams(node, &params) != cudaSuccess ||
            !params.dst)
          continue;
        size_t width = 0, rows = 0, span = 0;
        if (!multiply_bytes(params.width, params.elementSize, &width))
          continue;
        span = width;
        if (params.height > 1 &&
            (!multiply_bytes(params.height - 1, params.pitch, &rows) ||
             !add_bytes(span, rows, &span)))
          continue;
        if (overlaps(params.dst, span)) {
          ++overlapping_writes;
          if (overlap_logs++ < 24)
            LOG_INFO("[QWEN4EXP-P2-UP-CUDA-WRITER] kind=memset graph=%p "
                     "node=%p dst=%p span=%zu depth=%d\n",
                     static_cast<void *>(graph), static_cast<void *>(node),
                     params.dst, span, depth);
        }
      } else if (type == cudaGraphNodeTypeKernel) {
        ++kernel_nodes;
        cudaKernelNodeParams params{};
        if (cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess ||
            !params.func) {
          ++kernel_param_scan_failures;
          continue;
        }
        size_t param_count = 0;
        if (cudaFuncGetParamCount(params.func, &param_count) != cudaSuccess ||
            (param_count != 0 && !params.kernelParams)) {
          ++kernel_param_scan_failures;
          continue;
        }
        const char *function_name = nullptr;
        const bool have_function_name =
            cudaFuncGetName(&function_name, params.func) == cudaSuccess &&
            function_name;
        for (size_t param_index = 0; param_index < param_count;
             ++param_index) {
          size_t param_offset = 0;
          size_t param_size = 0;
          if (cudaFuncGetParamInfo(params.func, param_index, &param_offset,
                                   &param_size) != cudaSuccess) {
            ++kernel_param_scan_failures;
            continue;
          }
          // Kernel pointer parameters are represented as pointer-sized
          // arguments in kernelParams. Scalar arguments are ignored. This
          // finds captured kernels that reference the protected expert arena;
          // it does not assume every reference is a write.
          if (param_size != sizeof(void *) || !params.kernelParams[param_index])
            continue;
          void *argument_value = nullptr;
          std::memcpy(&argument_value, params.kernelParams[param_index],
                      sizeof(argument_value));
          ++kernel_param_checks;
          if (!overlaps(argument_value, 1))
            continue;
          ++kernel_param_overlaps;
          if (overlap_logs++ < 48) {
            const uintptr_t pointer =
                reinterpret_cast<uintptr_t>(argument_value);
            LOG_INFO("[QWEN4EXP-P2-UP-CUDA-WRITER] kind=kernel-param "
                     "graph=%p node=%p fn=%s arg=%zu offset=%zu ptr=%p "
                     "range_offset=%zu depth=%d\n",
                     static_cast<void *>(graph), static_cast<void *>(node),
                     have_function_name ? function_name : "unknown",
                     param_index, param_offset, argument_value,
                     static_cast<size_t>(pointer - protected_begin), depth);
          }
        }
      } else if (type == cudaGraphNodeTypeGraph) {
        cudaGraph_t child = nullptr;
        if (cudaGraphChildGraphNodeGetGraph(node, &child) == cudaSuccess)
          self(child, depth + 1, self);
      }
    }
  };
  for (cudaGraph_t graph : captured_graphs)
    visit_graph(graph, 0, visit_graph);
  LOG_INFO("[QWEN4EXP-P2-UP-CUDA-SCAN] range=[%p,%p) bytes=%zu "
           "captured_graphs=%zu nodes=%zu memcpy=%zu memset=%zu "
           "kernels=%zu kernel_param_checks=%zu kernel_param_hits=%zu "
           "kernel_scan_failures=%zu overlapping_memcpy_memset=%zu\n",
           range_base, reinterpret_cast<void *>(protected_end), range_bytes,
           captured_graphs.size(), graph_nodes, memcpy_nodes, memset_nodes,
           kernel_nodes, kernel_param_checks, kernel_param_overlaps,
           kernel_param_scan_failures, overlapping_writes);
}

void qwen_probe_dump(FullGraph &g) {
  const char *dir = zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_PROBE_DUMP");
  if (!dir || !dir[0] ||
    (g.probe_taps.empty() && g.layer_input_taps.empty() &&
      g.router_logits_taps.empty() && g.gdn_probe_taps.empty()))
    return;
  for (auto &[l, x, p] : g.probe_taps) {
    std::vector<float> a((size_t)ggml_nelements(x)), b((size_t)ggml_nelements(p));
    ggml_backend_tensor_get(x, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_get(p, b.data(), 0, b.size() * sizeof(float));
    const std::string base = std::string(dir) + "/L" + std::to_string(l);
    std::ofstream xo(base + ".x.bin", std::ios::binary | std::ios::app);
    xo.write(reinterpret_cast<const char *>(a.data()), (std::streamsize)(a.size() * sizeof(float)));
    std::ofstream so(base + ".s.bin", std::ios::binary | std::ios::app);
    so.write(reinterpret_cast<const char *>(b.data()), (std::streamsize)(b.size() * sizeof(float)));
  }
  for (auto &[l, logits] : g.router_logits_taps) {
    std::vector<float> values((size_t)ggml_nelements(logits));
    ggml_backend_tensor_get(logits, values.data(), 0,
                            values.size() * sizeof(float));
    const std::string path = std::string(dir) + "/L" +
                             std::to_string(l) + ".router_logits.bin";
    std::ofstream out(path, std::ios::binary | std::ios::app);
    out.write(reinterpret_cast<const char *>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
  }
  for (auto &[l, label, tensor] : g.gdn_probe_taps) {
    if (!tensor || tensor->type != GGML_TYPE_F32)
      continue;
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0,
                            values.size() * sizeof(float));
    const std::string path = std::string(dir) + "/L" + std::to_string(l) +
                             "." + label + ".bin";
    std::ofstream out(path, std::ios::binary | std::ios::app);
    out.write(reinterpret_cast<const char *>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
  }
  for (auto &[l, hidden] : g.layer_input_taps) {
    std::vector<float> values((size_t)ggml_nelements(hidden));
    ggml_backend_tensor_get(hidden, values.data(), 0,
                            values.size() * sizeof(float));
    const std::string path = std::string(dir) + "/L" +
                             std::to_string(l) + ".h.bin";
    std::ofstream out(path, std::ios::binary | std::ios::app);
    out.write(reinterpret_cast<const char *>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
  }
}

void log_qwen_layout_tensor(const FullGraph &g, int layer, bool cpu_only,
                            const char *role, const ggml_tensor *tensor) {
  if (!tensor)
    return;
  ggml_backend_buffer_t buffer = tensor->buffer;
  if (!buffer && tensor->view_src)
    buffer = tensor->view_src->buffer;
  const char *buft = buffer
                         ? ggml_backend_buft_name(
                               ggml_backend_buffer_get_type(buffer))
                         : "unallocated";
  const char *backend_name = "unassigned";
  if (g.sched) {
    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(
        g.sched, const_cast<ggml_tensor *>(tensor));
    if (backend)
      backend_name = ggml_backend_name(backend);
  }
  const char *pinned = "unknown";
  if (buffer && ggml_backend_buffer_is_host(buffer)) {
    void *base = ggml_backend_buffer_get_base(buffer);
    unsigned int flags = 0;
    if (base && cudaHostGetFlags(&flags, base) == cudaSuccess) {
      pinned = "yes";
    } else {
      cudaGetLastError();
      pinned = "no";
    }
  }
  LOG_INFO(
      "[GGML-QWEN-LAYOUT] layer=%d cpu_only=%d role=%s op=%s name=%s "
      "type=%s nbytes=%zu ne=[%lld,%lld,%lld,%lld] "
      "nb=[%zu,%zu,%zu,%zu] buft=%s backend=%s pinned=%s\n",
      layer, cpu_only ? 1 : 0, role, ggml_op_name(tensor->op),
      ggml_get_name(tensor), ggml_type_name(tensor->type),
      ggml_nbytes(tensor), (long long) tensor->ne[0],
      (long long) tensor->ne[1], (long long) tensor->ne[2],
      (long long) tensor->ne[3], tensor->nb[0], tensor->nb[1], tensor->nb[2],
      tensor->nb[3], buft, backend_name, pinned);
}

void dump_qwen_moe_layout(const FullGraph &g) {
  if (!graph_layout_debug() || !g.graph || !g.sched)
    return;
  std::unordered_set<const ggml_tensor *> logged;
  for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
    ggml_tensor *node = ggml_graph_node(g.graph, i);
    if (!node || node->op != GGML_OP_MUL_MAT_ID || !node->src[0])
      continue;
    ggml_backend_t backend =
        ggml_backend_sched_get_tensor_backend(g.sched, node);
    if (backend != g.cpu)
      continue;
    const int layer = layer_timing_parse_layer(ggml_get_name(node->src[0]));
    if (layer < 0)
      continue;
    const bool cpu_only = qwen_moe_layer_cpu_only(g, layer);
    // The existing dump covered only MUL_MAT_ID sources.  Host-node mode
    // changes CUDA ownership at the Layer-13 boundary, so also record the
    // ordinary CPU op's destination: an output buffer changing from the
    // shared CUDA-host arena to a backend-private buffer would explain a
    // split-time fence even when weights and inputs are identical.
    log_qwen_layout_tensor(g, layer, cpu_only, "output", node);
    const char *roles[] = {"weight", "input", "ids"};
    for (int src = 0; src < 3; ++src) {
      ggml_tensor *tensor = node->src[src];
      if (!tensor || !logged.insert(tensor).second)
        continue;
      log_qwen_layout_tensor(g, layer, cpu_only, roles[src], tensor);
    }
  }
}

} // namespace qwen4exp_graph

#endif // ZEROMOE_USE_GGML_CUDA
