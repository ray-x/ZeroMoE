// Qwen4Exp full graph: Diagnostics: per-layer timing callback, captured-graph write scan, tensor
// and hidden-state dumps, layout logging, the two-layer S=1 diagnostic.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_qwen4exp_graph_internal.hpp.

#include "ggml_qwen4exp_graph_internal.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

using namespace qwen4exp_graph;

namespace qwen4exp_graph {

// Runtime guard (2026-09-19): CMakeLists.txt already documents that a plain
// MOESTREAM_USE_GGML_CUDA build can silently compile out ggml-cuda's CUDA
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
             "with -DMOESTREAM_ENABLE_CUDA_GRAPHS=ON (the CMakeLists.txt "
             "default) and verify with a fresh 'cmake -B <dir>' configure, "
             "not just an incremental build in a stale build directory.\n");
  }
}

const char *env_layer0_common_attn_input_path() {
  const char *v =
      moestream::graphcfg::env("MOESTREAM_QWEN4EXP_LAYER0_COMMON_ATTN_INPUT");
  return v && v[0] ? v : nullptr;
}

bool load_exact_f32_file(const char *path, float *dst, size_t elements) {
  if (!path || !dst || elements == 0)
    return false;
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in)
    return false;
  const std::streamoff expected = static_cast<std::streamoff>(
      elements * sizeof(float));
  if (in.tellg() != expected)
    return false;
  in.seekg(0, std::ios::beg);
  in.read(reinterpret_cast<char *>(dst), expected);
  return in.good() || in.eof() && in.gcount() == expected;
}

void log_layer0_diag_tensor(const char *role, const ggml_tensor *t) {
  if (!t) {
    LOG_INFO("[QWEN4EXP-LAYER0-DIAG] %s=<null>\n", role);
    return;
  }
  const char *buft = "unbound";
  const char *dev = "unbound";
  if (t->buffer) {
    const auto type = ggml_backend_buffer_get_type(t->buffer);
    buft = ggml_backend_buft_name(type);
    if (const auto device = ggml_backend_buft_get_device(type))
      dev = ggml_backend_dev_name(device);
  }
  LOG_INFO(
      "[QWEN4EXP-LAYER0-DIAG] %s type=%s backend=%s device=%s "
      "ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] data=%p\n",
      role, ggml_type_name(t->type), buft, dev,
      static_cast<long long>(t->ne[0]), static_cast<long long>(t->ne[1]),
      static_cast<long long>(t->ne[2]), static_cast<long long>(t->ne[3]),
      t->nb[0], t->nb[1], t->nb[2], t->nb[3], t->data);
}

void log_layer0_diag_host_values(const char *role, const float *values,
                                 size_t n) {
  if (!values || n == 0)
    return;
  float min_v = INFINITY;
  float max_v = -INFINITY;
  double sum = 0.0;
  std::ostringstream first;
  for (size_t i = 0; i < n; ++i) {
    min_v = std::min(min_v, values[i]);
    max_v = std::max(max_v, values[i]);
    sum += values[i];
    if (i < 8)
      first << (i ? "," : "") << std::setprecision(9) << values[i];
  }
  LOG_INFO("[QWEN4EXP-LAYER0-DIAG] %s values=%zu first=[%s] min=%.9g "
           "max=%.9g sum=%.9g data=%p\n",
           role, n, first.str().c_str(), min_v, max_v, sum,
           static_cast<const void *>(values));
}

// Every-decode-step, all-layer hidden-state dump for a llama.cpp parity
// trace. One file per decode step: <path>.step<pos>.f32, holding
// n_layers * D * HC * S float32s in layer-major order when every graph layer
// has a snapshot. Partial graphs can leave this vector incomplete; callers
// must validate file length and contents before comparing it.
void dump_all_layers_snapshot(FullGraph &g, int pos) {
  const char *path = env_alllayers_dump_path();
  if (!path || g.all_layers_snapshot.empty())
    return;
  std::ostringstream fname;
  fname << path << ".step" << pos << ".f32";
  std::ofstream out(fname.str(), std::ios::binary | std::ios::trunc);
  if (!out) {
    LOG_WARN("[QWEN4EXP-ALLLAYERS] cannot open %s\n", fname.str().c_str());
    return;
  }
  int missing = 0;
  for (ggml_tensor *t : g.all_layers_snapshot) {
    if (t && t->data) {
      const size_t n = static_cast<size_t>(ggml_nelements(t));
      out.write(static_cast<const char *>(t->data),
                static_cast<std::streamsize>(n * sizeof(float)));
    } else {
      ++missing;
      const size_t n = static_cast<size_t>(g.D) * g.HC * g.S;
      std::vector<float> zeros(n, 0.0f);
      out.write(reinterpret_cast<const char *>(zeros.data()),
                static_cast<std::streamsize>(n * sizeof(float)));
    }
  }
  out.close();
  std::ofstream meta(fname.str() + ".json", std::ios::trunc);
  if (meta)
    meta << "{\"position\":" << pos << ",\"layers\":" << g.n_layers
         << ",\"layer_elements\":"
         << static_cast<size_t>(g.D) * g.HC * g.S
         << ",\"hidden\":" << g.D << ",\"hc\":" << g.HC
         << ",\"sequence\":" << g.S << ",\"missing_layers\":"
         << missing << "}\n";
}

// Join the per-layer snapshots from the Phase-2 Prefix and chosen suffix.
// Prefix stops before the split layer's FFN; the suffix owns that layer's
// post-FFN state and every later layer. Keep the same layer-major .f32 format
// as the ordinary full-graph dump so the saved runs can be compared directly.
void dump_all_layers_snapshot_pair(const FullGraph &prefix,
                                   const FullGraph &suffix, int pos) {
  const char *path = env_alllayers_dump_path();
  if (!path)
    return;
  std::ostringstream fname;
  fname << path << ".step" << pos << ".f32";
  std::ofstream out(fname.str(), std::ios::binary | std::ios::trunc);
  if (!out) {
    LOG_WARN("[QWEN4EXP-ALLLAYERS] cannot open merged snapshot %s\n",
             fname.str().c_str());
    return;
  }
  const size_t layer_elements =
      static_cast<size_t>(prefix.D) * prefix.HC * prefix.S;
  const size_t layer_bytes = layer_elements * sizeof(float);
  std::vector<float> zeros(layer_elements, 0.0f);
  int missing = 0;
  for (int layer = 0; layer < prefix.n_layers; ++layer) {
    ggml_tensor *snapshot = nullptr;
    if (static_cast<size_t>(layer) < suffix.all_layers_snapshot.size())
      snapshot = suffix.all_layers_snapshot[static_cast<size_t>(layer)];
    if (!snapshot &&
        static_cast<size_t>(layer) < prefix.all_layers_snapshot.size())
      snapshot = prefix.all_layers_snapshot[static_cast<size_t>(layer)];
    if (snapshot && snapshot->data && ggml_nelements(snapshot) ==
                                          static_cast<int64_t>(layer_elements)) {
      out.write(static_cast<const char *>(snapshot->data),
                static_cast<std::streamsize>(layer_bytes));
    } else {
      ++missing;
      out.write(reinterpret_cast<const char *>(zeros.data()),
                static_cast<std::streamsize>(layer_bytes));
    }
  }
  LOG_INFO("[QWEN4EXP-ALLLAYERS] merged Prefix/suffix dump pos=%d "
           "layers=%d missing=%d file=%s\n",
           pos, prefix.n_layers, missing, fname.str().c_str());
  out.close();
  std::ofstream meta(fname.str() + ".json", std::ios::trunc);
  if (meta)
    meta << "{\"position\":" << pos << ",\"layers\":"
         << prefix.n_layers << ",\"layer_elements\":" << layer_elements
         << ",\"hidden\":" << prefix.D << ",\"hc\":" << prefix.HC
         << ",\"sequence\":" << prefix.S << ",\"missing_layers\":"
         << missing << "}\n";
}

void dump_layer0_diag_values(FullGraph &g) {
  const char *split_dump_path = env_layer0_split_dump_path();
  if (split_dump_path && g.decode_mode && !g.layer0_split_dumped) {
    g.layer0_split_dumped = true;
    std::unordered_map<std::string, size_t> snapshot_occurrences;
    for (const auto &entry : g.layer0_split_snapshots) {
      const size_t occurrence = snapshot_occurrences[entry.first]++;
      // The side diagnostic graph runs immediately before production decode
      // and uses the same snapshot labels. Keep its files separate so the
      // production replay cannot overwrite controlled side-graph evidence.
      const char *graph_tag = g.two_layer_diag ? ".two_layer." : ".";
      std::string path = std::string(split_dump_path) + graph_tag +
                         entry.first;
      if (occurrence != 0)
        path += "." + std::to_string(occurrence);
      path += ".f32";
      ggml_tensor *t = entry.second;
      if (!t || !t->data)
        continue;
      const size_t n = static_cast<size_t>(ggml_nelements(t));
      std::vector<float> values(n);
      ggml_backend_tensor_get(t, values.data(), 0, n * sizeof(float));
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (!out) {
        LOG_WARN("[QWEN4EXP-LAYER0-SPLIT] cannot open %s\n", path.c_str());
        continue;
      }
      out.write(reinterpret_cast<const char *>(values.data()),
                static_cast<std::streamsize>(n * sizeof(float)));
      float min_v = INFINITY;
      float max_v = -INFINITY;
      double sum = 0.0;
      double sum2 = 0.0;
      for (float v : values) {
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        sum += v;
        sum2 += static_cast<double>(v) * v;
      }
      LOG_INFO("[QWEN4EXP-LAYER0-SPLIT] boundary=%s elements=%zu "
               "fnv64=0x%016llx min=%.9g max=%.9g mean=%.9g rms=%.9g "
               "file=%s\n",
               entry.first.c_str(), n,
               static_cast<unsigned long long>(qwen4exp_fnv64(
                   values.data(), n * sizeof(float))),
               min_v, max_v, n ? sum / static_cast<double>(n) : 0.0,
               n ? std::sqrt(sum2 / static_cast<double>(n)) : 0.0,
               path.c_str());
    }
  }
  const bool hidden_dump = env_hidden01_dump_path() && g.decode_mode &&
                           !g.hidden01_dumped;
  if (!g.layer0_diag_enabled ||
      (g.layer0_diag_dumped && !hidden_dump))
    return;
  if (!g.layer0_diag_dumped) {
    g.layer0_diag_dumped = true;
    if (!g.layer0_diag_gate || !g.layer0_diag_up)
      LOG_INFO("[QWEN4EXP-LAYER0-DIAG] numeric gate/up outputs unavailable "
               "for this graph (host-node or no separate projection result)\n");
    auto dump = [](const char *role, const ggml_tensor *t) {
      if (!t)
        return;
      const size_t n = static_cast<size_t>(ggml_nelements(t));
      std::vector<float> values(n, 0.0f);
      ggml_backend_tensor_get(t, values.data(), 0, n * sizeof(float));
      float min_v = INFINITY;
      float max_v = -INFINITY;
      double sum = 0.0;
      std::ostringstream first;
      for (size_t i = 0; i < n; ++i) {
        min_v = std::min(min_v, values[i]);
        max_v = std::max(max_v, values[i]);
        sum += values[i];
        if (i < 8)
          first << (i ? "," : "") << std::setprecision(9) << values[i];
      }
      LOG_INFO("[QWEN4EXP-LAYER0-DIAG] %s values=%zu first=[%s] min=%.9g "
               "max=%.9g sum=%.9g\n",
               role, n, first.str().c_str(), min_v, max_v, sum);
    };
    if (g.layer0_diag_gate)
      dump("gate_output", g.layer0_diag_gate);
    if (g.layer0_diag_up)
      dump("up_output", g.layer0_diag_up);
    dump("down_input_swiglu", g.layer0_diag_act);
    dump("down_output_raw", g.layer0_diag_down);
    // The following graph-owned tensors may alias later allocations by the
    // time the whole graph has retired. Their exact values are reported from
    // the host snapshots below instead.
    auto dump_snapshot = [](const char *role, const ggml_tensor *t) {
      if (!t || !t->data)
        return;
      log_layer0_diag_host_values(role, static_cast<const float *>(t->data),
                                  static_cast<size_t>(ggml_nelements(t)));
    };
    dump_snapshot("residual_before_moe_snapshot",
                  g.layer0_diag_residual_before_snapshot);
    dump_snapshot("moe_plus_shared_snapshot",
                  g.layer0_diag_moe_plus_shared_snapshot);
    dump_snapshot("residual_after_moe_add_snapshot",
                  g.layer0_diag_residual_after_snapshot);
    dump_snapshot("layer1_first_consumer_snapshot",
                  g.layer0_diag_next_consumer_snapshot);
  }

  const char *dump_path = env_hidden01_dump_path();
  if (!dump_path || !hidden_dump)
    return;
  g.hidden01_dumped = true;
  for (int layer = 0; layer < 2; ++layer) {
    ggml_tensor *t = g.hidden01_snapshot[layer];
    if (!t || !t->data)
      continue;
    const size_t n = static_cast<size_t>(ggml_nelements(t));
    const auto *values = static_cast<const float *>(t->data);
    std::ostringstream suffix;
    suffix << dump_path << ".decode.layer" << layer << ".f32";
    std::ofstream out(suffix.str(), std::ios::binary | std::ios::trunc);
    if (!out) {
      LOG_WARN("[QWEN4EXP-HIDDEN01] cannot open %s\n", suffix.str().c_str());
      continue;
    }
    out.write(reinterpret_cast<const char *>(values),
              static_cast<std::streamsize>(n * sizeof(float)));
    float min_v = INFINITY;
    float max_v = -INFINITY;
    double sum = 0.0;
    double sum2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
      min_v = std::min(min_v, values[i]);
      max_v = std::max(max_v, values[i]);
      sum += values[i];
      sum2 += static_cast<double>(values[i]) * values[i];
    }
    LOG_INFO("[QWEN4EXP-HIDDEN01] layer=%d elements=%zu fnv64=0x%016llx "
             "min=%.9g max=%.9g mean=%.9g rms=%.9g file=%s\n",
             layer, n,
             static_cast<unsigned long long>(qwen4exp_fnv64(
                 values, n * sizeof(float))),
             min_v, max_v, n ? sum / static_cast<double>(n) : 0.0,
             n ? std::sqrt(sum2 / static_cast<double>(n)) : 0.0,
             suffix.str().c_str());
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
        moestream_ggml_cuda_get_captured_graph(backend, tensor);
    if (!captured && backend != g.cuda)
      captured = moestream_ggml_cuda_get_captured_graph(g.cuda, tensor);
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

// Diagnostic-only snapshot for tensors whose GGML allocator storage may be
// reused later in the graph.  Reading such a tensor after the full graph has
// retired can observe a later consumer's data, so materialize the value at
// the exact graph position into a persistent host buffer.
ggml_tensor *layer0_diag_snapshot(FullGraph &g, ggml_context *ctx,
                                   ggml_tensor *src, const char *name) {
  if (!g.layer0_diag_enabled || !src || !g.graph)
    return nullptr;
  if (src->type != GGML_TYPE_F32)
    src = ggml_cast(ctx, src, GGML_TYPE_F32);
  if (!src)
    return nullptr;
  // The snapshot is always materialized as F32.  Using ggml_nbytes(src)
  // under-allocates when the source is BF16/F16 and lets the diagnostic copy
  // overwrite adjacent host inputs (the first split-layer trace exposed this
  // as apparently corrupt layer data).
  const size_t bytes = static_cast<size_t>(ggml_nelements(src)) *
                       sizeof(float);
  const std::vector<int64_t> shape = {src->ne[0], src->ne[1], src->ne[2],
                                      src->ne[3]};
  ggml_tensor *dst = host_input(g, ctx, GGML_TYPE_F32, shape, bytes, name);
  if (!dst)
    return nullptr;
  ggml_tensor *copy = ggml_cpy(ctx, src, dst);
  if (!copy)
    return nullptr;
  ggml_set_name(copy, name);
  ggml_set_output(copy);
  ggml_build_forward_expand(g.graph, copy);
  return dst;
}

ggml_tensor *layer13_moe_diag_snapshot(
    FullGraph &g, ggml_context *ctx, ggml_tensor *src, ggml_type type,
    const std::vector<int64_t> &shape, const char *name) {
  if (!g.layer13_moe_diag_enabled || !src || !g.graph)
    return nullptr;
  if (type == GGML_TYPE_F32 && src->type != GGML_TYPE_F32)
    src = ggml_cast(ctx, src, GGML_TYPE_F32);
  if (!src)
    return nullptr;
  size_t elements = 1;
  for (const int64_t dim : shape) {
    if (dim <= 0 || elements > SIZE_MAX / static_cast<size_t>(dim))
      return nullptr;
    elements *= static_cast<size_t>(dim);
  }
  ggml_tensor *dst = host_input(
      g, ctx, type, shape, elements * ggml_type_size(type), name);
  if (!dst)
    return nullptr;
  ggml_tensor *copy = ggml_cpy(ctx, src, dst);
  if (!copy)
    return nullptr;
  ggml_set_name(copy, name);
  ggml_set_output(copy);
  ggml_build_forward_expand(g.graph, copy);
  return dst;
}

void qwen_probe_dump(FullGraph &g) {
  const char *dir = moestream::graphcfg::env("MOESTREAM_QWEN4EXP_PROBE_DUMP");
  if (!dir || !dir[0] || g.probe_taps.empty())
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

bool qwen4exp_two_layer_s1_diag_run(StreamingEngine &engine, int pos_base,
                                    const int *token_ids) {
  if (!env_two_layer_s1_diag_enabled() ||
      engine.family_ops.family != ModelFamily::Qwen4Exp || pos_base < 0)
    return false;
  if (g_two_layer_diag_done)
    return true;

  const auto &c = engine.model_file.architecture_config;
  if (c.n_layers < 2 || engine.max_t <= 0 || pos_base >= engine.max_t ||
      engine.hc_stream_buf.size() <
          static_cast<size_t>(c.hidden) * c.hc_mult)
    return false;
  const int dev = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  const auto build_start = std::chrono::steady_clock::now();

  auto g = std::make_unique<FullGraph>();
  g->engine = &engine;
  g->device = dev;
  g->S = 1;
  g->D = c.hidden;
  g->HC = c.hc_mult;
  g->n_layers = 2;
  g->H = c.n_heads;
  g->Hkv = std::max(1, c.n_head_kv);
  g->hd = c.head_dim;
  g->Hk_gdn = c.ssm_group_count;
  g->Hv_gdn = c.ssm_time_step_rank;
  g->gdn_hd = c.ssm_state_size;
  g->key_dim = g->Hk_gdn * g->gdn_hd;
  g->value_dim = c.ssm_inner_size;
  g->conv_dim = 2 * g->key_dim + g->value_dim;
  g->d_conv = c.ssm_conv_kernel;
  g->decode_mode = true;
  g->kv_capacity = env_decode_kv_capacity(engine.max_t, pos_base + 1);
  g->decode_last_pos = pos_base - 1;
  g->two_layer_diag = true;
  // Reuse the existing HC boundary instrumentation when the caller also
  // requests a split dump. This keeps the side-graph internals observable
  // without changing the normal two-layer boundary API or graph topology.
  g->layer0_diag_enabled = env_layer0_split_dump_path() != nullptr ||
                           env_alllayers_dump_path() != nullptr;
  g->layer_timing_verbose_enabled = env_layer_timing_verbose();

  g->cuda_raw = static_cast<ggml_backend_t>(ggml_shared_backend(dev));
  g->cuda = full_graph_cuda_scheduler_backend(*g);
  g->cpu = ggml_backend_cpu_init();
  if (!g->cuda_raw || !g->cuda || !g->cpu)
    return false;
  verify_cuda_graphs_compiled_in(g->cuda_raw);
  g->cpu_threads = graph_cpu_threads();
  ggml_backend_cpu_set_n_threads(g->cpu, g->cpu_threads);
  ggml_threadpool_params tp = ggml_threadpool_params_default(g->cpu_threads);
  g->cpu_threadpool = ggml_threadpool_new(&tp);
  if (g->cpu_threadpool)
    ggml_backend_cpu_set_threadpool(g->cpu, g->cpu_threadpool);

  // Only the first two layers can be referenced by this graph. This avoids
  // charging/reserving the remaining 46 layers' expert source bytes when a
  // caller has not enabled the full preloaded DDR arena.
  const size_t expert_bytes = full_expert_source_bytes(engine, g->n_layers);
  if (expert_bytes > 0 &&
      !MemoryGuard::try_reserve_host_bytes(expert_bytes))
    return false;
  g->host_reservation_bytes = expert_bytes;
  g->host_precharged_bytes = expert_bytes;
  if (!init_device_dense_arena(*g) || !init_device_state_arena(*g) ||
      !init_device_expert_arena(*g))
    return false;

  ggml_backend_t sched_cuda = full_graph_cuda_scheduler_backend(*g);
  ggml_backend_t backends[] = {sched_cuda, g->cpu};
  ggml_backend_buffer_type_t sched_bufts[2] = {};
  ggml_backend_buffer_type_t *sched_bufts_ptr = nullptr;
  if (graph_cuda_host_buffers()) {
    sched_bufts[0] = ggml_backend_get_default_buffer_type(g->cuda);
    sched_bufts[1] = ggml_backend_dev_host_buffer_type(
        ggml_backend_get_device(g->cuda));
    if (sched_bufts[1])
      sched_bufts_ptr = sched_bufts;
  }
  // op_offload MUST stay false -- see qwen4exp_full_graph_prepare's own
  // comment for the real S>=32 host-weight-offload OOM/crash this avoids.
  g->sched = ggml_backend_sched_new(backends, sched_bufts_ptr, 2,
                                    kGraphSize, graph_scheduler_parallel(),
                                    false);
  if (!g->sched)
    return false;
  if (g->layer_timing_verbose_enabled) {
    g->layer_timing_cpu_ms.assign(static_cast<size_t>(g->n_layers), 0.0);
    ggml_backend_sched_set_eval_callback(g->sched, layer_timing_eval_callback,
                                         g.get());
  }

  std::vector<float> ple;
  if (!prepare_ple_embedding(*g, ple, 1, pos_base, token_ids) &&
      has_ple_layer(c))
    return false;
  if (!build_graph(*g, has_ple_layer(c) ? &ple : nullptr))
    return false;

  // Seed only this diagnostic graph's private state/cache copies. The live
  // engine state is never touched by the replay or its state-update nodes.
  const size_t state_n = static_cast<size_t>(g->gdn_hd) * g->gdn_hd * g->Hv_gdn;
  const size_t hist_n = static_cast<size_t>(g->d_conv - 1) * g->conv_dim;
  for (size_t i = 0; i < g->gdn_layers.size(); ++i) {
    const int l = g->gdn_layers[i];
    if (l < 0 || l >= static_cast<int>(engine.qwen4exp_recur_state.size()) ||
        l >= static_cast<int>(engine.qwen4exp_conv_history.size()) ||
        engine.qwen4exp_recur_state[l].size() != state_n ||
        engine.qwen4exp_conv_history[l].size() != hist_n)
      return false;
    std::vector<float> native(state_n);
    gdn_state_transpose_host(native.data(), engine.qwen4exp_recur_state[l].data(),
                              g->gdn_hd, g->Hv_gdn);
    ggml_backend_tensor_set(g->gdn_state_storage[i], native.data(), 0,
                            state_n * sizeof(float));
    ggml_backend_tensor_set(g->gdn_hist_storage[i],
                            engine.qwen4exp_conv_history[l].data(), 0,
                            hist_n * sizeof(float));
  }
  const size_t kv_row = static_cast<size_t>(g->Hkv) * g->hd;
  const size_t kv_bytes = static_cast<size_t>(pos_base) * kv_row * sizeof(float);
  for (size_t i = 0; i < g->kv_layers.size(); ++i) {
    const int l = g->kv_layers[i];
    if (l < 0 || l >= static_cast<int>(engine.Kc.size()) ||
        l >= static_cast<int>(engine.Vc.size()) || !engine.Kc[l] ||
        !engine.Vc[l])
      return false;
    if (kv_bytes > 0) {
      ggml_backend_tensor_set(g->kv_state_k[i], engine.Kc[l], 0, kv_bytes);
      ggml_backend_tensor_set(g->kv_state_v[i], engine.Vc[l], 0, kv_bytes);
    }
  }
  // Layer 1 on this model owns the PLE convolution state.  The diagnostic
  // graph has private state tensors, so it must receive the same history as
  // production before replay; otherwise Layer 0 can match while Layer 1's
  // attention input diverges.
  if (g->ple_layer >= 0 && g->ple_hist_storage) {
    const size_t ple_n = static_cast<size_t>(c.ple_conv_kernel - 1) *
                         c.ple_ngram_size * g->HC * g->D;
    if (g->ple_layer >=
            static_cast<int>(engine.qwen4exp_ple_conv_history.size()) ||
        engine.qwen4exp_ple_conv_history[g->ple_layer].size() != ple_n)
      return false;
    ggml_backend_tensor_set(
        g->ple_hist_storage,
        engine.qwen4exp_ple_conv_history[g->ple_layer].data(), 0,
        ple_n * sizeof(float));
    if (env_two_layer_s1_diag_dump_path()) {
      std::vector<float> history(ple_n);
      ggml_backend_tensor_get(g->ple_hist_storage, history.data(), 0,
                              ple_n * sizeof(float));
      const std::string path =
          std::string(env_two_layer_s1_diag_dump_path()) +
          ".ple_history_before.f32";
      qwen4exp_dump_f32(path.c_str(), history.data(), ple_n);
    }
  }
  if (!feed_decode_common_inputs(*g, pos_base))
    return false;
  std::memcpy(g->input->data, engine.hc_stream_buf.data(),
              static_cast<size_t>(g->D) * g->HC * sizeof(float));

  const auto build_done = std::chrono::steady_clock::now();
  const auto replay_start = std::chrono::steady_clock::now();
  if (ggml_backend_sched_graph_compute_async(g->sched, g->graph) !=
      GGML_STATUS_SUCCESS)
    return false;
  ggml_backend_sched_synchronize(g->sched);
  if (g->engine && g->engine->exl3_gpu_executor)
    (void)g->engine->exl3_gpu_executor->stats();
  if (g->layer0_diag_enabled)
    dump_layer0_diag_values(*g);
  if (env_alllayers_dump_path())
    dump_all_layers_snapshot(*g, pos_base);
  const double replay_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - replay_start)
                               .count();

  const char *dump_base = env_two_layer_s1_diag_dump_path();
  auto dump_boundary = [&](const char *label, ggml_tensor *tensor) {
    if (!tensor)
      return;
    const size_t n = static_cast<size_t>(ggml_nelements(tensor));
    std::vector<float> values(n);
    ggml_backend_tensor_get(tensor, values.data(), 0, n * sizeof(float));
    const uint64_t hash = qwen4exp_fnv64(values.data(), n * sizeof(float));
    LOG_INFO("[QWEN4EXP-TWO-LAYER-S1] boundary=%s elements=%zu "
             "fnv64=0x%016llx\n",
             label, n, static_cast<unsigned long long>(hash));
    if (dump_base) {
      std::string path = std::string(dump_base) + "." + label + ".f32";
      const bool ok = qwen4exp_dump_f32(path.c_str(), values.data(), n);
      LOG_INFO("[QWEN4EXP-TWO-LAYER-S1] dump=%s path=%s\n",
               ok ? "ok" : "failed", path.c_str());
    }
  };
  dump_boundary("layer0_post_ffn", g->two_layer_l0_post_ffn);
  dump_boundary("layer1_pre_attn", g->two_layer_l1_pre_attn);
  dump_boundary("layer1_post_attn", g->two_layer_l1_post_attn);
  dump_boundary("layer1_post_ffn", g->two_layer_l1_post_ffn);
  LOG_INFO("[QWEN4EXP-TWO-LAYER-S1] graph built_layers=%d nodes=%d "
           "splits=%d build_to_replay=%.3f ms replay=%.3f ms\n",
           g->n_layers, ggml_graph_n_nodes(g->graph),
           ggml_backend_sched_get_n_splits(g->sched),
           std::chrono::duration<double, std::milli>(build_done - build_start)
               .count(),
           replay_ms);
  g->ready = true;
  g_two_layer_diag_graph = std::move(g);
  g_two_layer_diag_done = true;
  return true;
}

#endif // MOESTREAM_USE_GGML_CUDA
