#include "ggml_qwen_shared_runner.hpp"
#include "ggml_moe_graph_builder.hpp"
#include "ggml_moe_internal.hpp"
#include "ggml_resource_guards.hpp"

#include "logger.hpp"
#include "memory_guard.hpp"
#include "perf_timer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#if defined(MOESTREAM_USE_GGML_CUDA)
#include <cuda_runtime.h>
#include "gpu_topology.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#if defined(_OPENMP)
#include <omp.h>
#endif
using namespace moe::detail;

extern "C" cudaStream_t moestream_ggml_backend_cuda_get_stream(
    ggml_backend_t backend);

// Qwen4Exp shared expert
// ---------------------------------------------------------------------------

GgmlQwenSharedRunner &GgmlQwenSharedRunner::instance() {
  static GgmlQwenSharedRunner r;
  return r;
}

GgmlQwenSharedRunner &GgmlQwenSharedRunner::instance_for_layer(int layer) {
  // The model currently has 48 trunk layers, but keep the pool comfortably
  // above that without exposing a model-specific limit in the public header.
  // Objects are initialized lazily and only the Qwen shared path reaches this
  // selector; the legacy singleton remains available to tests/diagnostics.
  static GgmlQwenSharedRunner runners[256];
  if (layer >= 0 && layer < 256)
    return runners[layer];
  return instance();
}

bool GgmlQwenSharedRunner::compiled_in() { return true; }

GgmlQwenSharedRunner::~GgmlQwenSharedRunner() { shutdown(); }

void GgmlQwenSharedRunner::free_graph() {
  if (graph_ctx_) {
    if (overlap_combine_galloc_) {
      ggml_gallocr_free((ggml_gallocr_t)overlap_combine_galloc_);
      overlap_combine_galloc_ = nullptr;
    }
  }
  if (overlap_shared_galloc_) {
    ggml_gallocr_free((ggml_gallocr_t)overlap_shared_galloc_);
    overlap_shared_galloc_ = nullptr;
  }
  if (galloc_) {
    ggml_gallocr_free((ggml_gallocr_t)galloc_);
    galloc_ = nullptr;
  }
  if (zero_routed_data_) {
    cudaSetDevice(device_);
    cudaFree(zero_routed_data_);
    zero_routed_data_ = nullptr;
  }
  if (graph_ctx_) {
    ggml_free((ggml_context *)graph_ctx_);
    graph_ctx_ = nullptr;
  }
  graph_ = overlap_shared_graph_ = overlap_combine_graph_ = nullptr;
  x_ = routed_ = output_ = nullptr;
  shared_raw_ = shared_ = nullptr;
  overlap_x_ = overlap_routed_ = overlap_shared_input_ = nullptr;
  overlap_shared_output_ = overlap_combined_ = nullptr;
  x_owned_data_ = routed_owned_data_ = nullptr;
  overlap_x_owned_data_ = overlap_routed_owned_data_ = nullptr;
  graph_ready_ = false;
  overlap_graph_ready_ = false;
}

void GgmlQwenSharedRunner::shutdown() {
  if (device_ >= 0) cudaSetDevice(device_);
  if (output_done_event_ &&
      ((ggml_backend_event_t)output_done_event_)->context)
    cudaEventSynchronize(
        (cudaEvent_t)((ggml_backend_event_t)output_done_event_)->context);
  if (backend_) ggml_backend_synchronize((ggml_backend_t)backend_);
  free_graph();
  if (weight_buf_) {
    ggml_backend_buffer_free((ggml_backend_buffer_t)weight_buf_);
    weight_buf_ = nullptr;
  }
  if (weight_ctx_) {
    ggml_free((ggml_context *)weight_ctx_);
    weight_ctx_ = nullptr;
  }
  if (external_buf_) {
    ggml_backend_buffer_free((ggml_backend_buffer_t)external_buf_);
    external_buf_ = nullptr;
  }
  if (output_done_event_) {
    ggml_backend_event_free((ggml_backend_event_t)output_done_event_);
    output_done_event_ = nullptr;
  }
  gate_ = up_ = down_ = gate_inp_ = nullptr;
  owned_gate_data_ = owned_up_data_ = owned_down_data_ = nullptr;
  owned_gate_inp_data_ = nullptr;
  backend_ = nullptr; // borrowed from ggml_shared_backend()
  device_ = -1;
  d_model_ = shared_inter_ = n_tokens_ = 0;
  has_gate_ = false;
  clamp_ = 0.0f;
  gate_type_ = up_type_ = down_type_ = gate_inp_type_ = 0;
}

bool GgmlQwenSharedRunner::ensure(int device, int d_model, int shared_inter,
                                  int n_tokens, bool has_gate, float clamp) {
  if (device < 0 || d_model <= 0 || shared_inter <= 0 || n_tokens <= 0)
    return false;
  if (backend_ && device_ == device && d_model_ == d_model &&
      shared_inter_ == shared_inter && n_tokens_ == n_tokens &&
      has_gate_ == has_gate && clamp_ == clamp)
    return true;
  if (backend_) shutdown();
  cudaSetDevice(device);
  backend_ = ggml_shared_backend(device);
  if (!backend_) return false;
  // Marker buffer for engine-owned CUDA pointers. It does not own the dense
  // pool allocation; it only gives GGML a CUDA buffer/backend association for
  // an externally managed tensor address.
  external_buf_ = ggml_backend_alloc_buffer((ggml_backend_t)backend_, 1);
  if (!external_buf_) {
    backend_ = nullptr;
    return false;
  }
  ggml_backend_buffer_set_usage((ggml_backend_buffer_t)external_buf_,
                                GGML_BACKEND_BUFFER_USAGE_COMPUTE);
  output_done_event_ = ggml_backend_event_new(
      ggml_backend_get_device((ggml_backend_t)backend_));
  if (!output_done_event_) {
    ggml_backend_buffer_free((ggml_backend_buffer_t)external_buf_);
    external_buf_ = nullptr;
    backend_ = nullptr;
    return false;
  }
  device_ = device;
  d_model_ = d_model;
  shared_inter_ = shared_inter;
  n_tokens_ = n_tokens;
  has_gate_ = has_gate;
  clamp_ = clamp;
  LOG_INFO("[GGML-SHARED] enabled on GPU %d (D=%d inter=%d S=%d gate=%s)\n",
           device, d_model, shared_inter, n_tokens, has_gate ? "yes" : "no");
  return true;
}

bool GgmlQwenSharedRunner::record_output_event() {
  if (!output_done_event_ || !backend_)
    return false;
  ggml_backend_event_record((ggml_backend_event_t)output_done_event_,
                            (ggml_backend_t)backend_);
  return true;
}

bool GgmlQwenSharedRunner::build_graph() {
  if (!backend_ || !gate_ || !up_ || !down_ ||
      (has_gate_ && !gate_inp_))
    return false;
  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 160 + ggml_graph_overhead() * 2;
  ip.no_alloc = true;
  moe::GgmlContextGuard ctx(ggml_init(ip));
  if (!ctx) return false;
  ggml_cgraph *gf = ggml_new_graph(ctx.get());
  ggml_tensor *x =
      ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, d_model_, n_tokens_);
  ggml_tensor *routed =
      ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, d_model_, n_tokens_);
  if (!gf || !x || !routed) return false;
  ggml_set_input(x);
  ggml_set_input(routed);
  ggml_set_name(x, "qwen4exp_shared_input");
  ggml_set_name(routed, "qwen4exp_routed_output");

  const moe::SharedGraphNodes shared_nodes = moe::build_shared_topology(
      ctx.get(), (ggml_tensor *)gate_, (ggml_tensor *)up_,
      (ggml_tensor *)down_, has_gate_ ? (ggml_tensor *)gate_inp_ : nullptr,
      x, routed, clamp_);
  ggml_tensor *combined = shared_nodes.combined;
  if (!shared_nodes.gate || !shared_nodes.up || !shared_nodes.activation ||
      !shared_nodes.shared || !combined)
    return false;
  ggml_set_output(combined);
  ggml_set_name(combined, "qwen4exp_routed_plus_shared");
  ggml_build_forward_expand(gf, combined);

  moe::GgmlGraphAllocation allocation =
      moe::allocate_graph(std::move(ctx), gf, (ggml_backend_t)backend_);
  if (!allocation) return false;
  ggml_gallocr_t ga = allocation.release_allocator();
  ggml_context *ctx_raw = allocation.release_context();
  ggml_backend_buffer_type_t bt =
      ggml_backend_get_default_buffer_type((ggml_backend_t)backend_);

  // Build a second, tiny graph for the Qwen CPU-affinity overlap path.  The
  // normal graph above remains unchanged: it computes shared FFN + routed
  // result in one operation.  The overlap graph only adds a host-routed
  // result to a shared result that was produced by the normal graph with a
  // zero routed input.  Keeping this graph separate avoids recomputing the
  // shared expert after the CPU worker finishes.
  ggml_cgraph *overlap_graph = ggml_new_graph(ctx_raw);
  ggml_tensor *overlap_routed =
      ggml_new_tensor_2d(ctx_raw, GGML_TYPE_F32, d_model_, n_tokens_);
  ggml_tensor *overlap_shared_input =
      ggml_new_tensor_2d(ctx_raw, GGML_TYPE_F32, d_model_, n_tokens_);
  ggml_tensor *overlap_combined =
      overlap_routed && overlap_shared_input
          ? ggml_add(ctx_raw, overlap_routed, overlap_shared_input)
          : nullptr;
  ggml_gallocr_t overlap_ga = nullptr;
  if (overlap_graph && overlap_routed && overlap_shared_input &&
      overlap_combined) {
    ggml_set_input(overlap_routed);
    ggml_set_input(overlap_shared_input);
    ggml_set_name(overlap_routed, "qwen4exp_overlap_routed_input");
    ggml_set_name(overlap_shared_input, "qwen4exp_overlap_shared_input");
    ggml_set_name(overlap_combined, "qwen4exp_overlap_routed_plus_shared");
    ggml_set_output(overlap_combined);
    ggml_build_forward_expand(overlap_graph, overlap_combined);
    overlap_ga = ggml_gallocr_new(bt);
    if (!overlap_ga || !ggml_gallocr_alloc_graph(overlap_ga, overlap_graph)) {
      if (overlap_ga)
        ggml_gallocr_free(overlap_ga);
      overlap_ga = nullptr;
    }
  }

  // The overlap graph is an optional Qwen-only optimization.  A failure to
  // allocate its small add workspace must never disable the established
  // shared graph or change any other model family's path.
  if (overlap_ga) {
    const size_t zero_bytes =
        (size_t)d_model_ * (size_t)n_tokens_ * sizeof(float);
    cudaSetDevice(device_);
    if (cudaMalloc(&zero_routed_data_, zero_bytes) == cudaSuccess &&
        cudaMemset(zero_routed_data_, 0, zero_bytes) == cudaSuccess) {
      overlap_shared_galloc_ = nullptr;
      overlap_combine_galloc_ = overlap_ga;
      overlap_combine_graph_ = overlap_graph;
      overlap_routed_ = overlap_routed;
      overlap_shared_input_ = overlap_shared_input;
      overlap_combined_ = overlap_combined;
      overlap_routed_owned_data_ = overlap_routed->data;
      overlap_graph_ready_ = true;
    } else {
      if (zero_routed_data_) {
        cudaFree(zero_routed_data_);
        zero_routed_data_ = nullptr;
      }
      ggml_gallocr_free(overlap_ga);
    }
  }
  LOG_INFO("[GGML-SHARED-MEM] graph workspace: %.2f MiB\n",
           (double)ggml_gallocr_get_buffer_size(ga, 0) /
               (1024.0 * 1024.0));
  graph_ctx_ = ctx_raw;
  galloc_ = ga;
  graph_ = gf;
  x_ = x;
  routed_ = routed;
  output_ = combined;
  shared_raw_ = shared_nodes.shared_raw;
  shared_ = shared_nodes.shared;
  x_owned_data_ = x->data;
  routed_owned_data_ = routed->data;
  graph_ready_ = true;
  return true;
}

bool GgmlQwenSharedRunner::ensure_layer(
    int layer, const GgmlSharedWeightRef &gate,
    const GgmlSharedWeightRef &up, const GgmlSharedWeightRef &down,
    const GgmlSharedWeightRef *gate_inp) {
  const bool debug = [] {
    const char *e = std::getenv("MOESTREAM_GGML_SHARED_DEBUG");
    return e && e[0] == '1';
  }();
  auto fail = [&](const char *reason) {
    if (debug)
      LOG_ERROR("[GGML-SHARED-DEBUG] ensure_layer=%d rejected: %s "
                "backend=%p host=(%p,%p,%p) shape=(%lld,%lld)/(%lld,%lld)/"
                "(%lld,%lld) expected=(%d,%d) types=(%u,%u,%u,%u) bytes=(%zu,%zu,%zu,%zu)\n",
                layer, reason, backend_, gate.host_data, up.host_data,
                down.host_data, (long long)gate.ne0, (long long)gate.ne1,
                (long long)up.ne0, (long long)up.ne1, (long long)down.ne0,
                (long long)down.ne1, d_model_, shared_inter_, gate.type,
                up.type, down.type, gate_inp ? gate_inp->type : 0,
                gate.nbytes, up.nbytes, down.nbytes,
                gate_inp ? gate_inp->nbytes : 0);
    return false;
  };
  if (!backend_ || layer < 0 || !gate.host_data || !up.host_data ||
      !down.host_data || gate.ne0 != d_model_ || gate.ne1 != shared_inter_ ||
      up.ne0 != d_model_ || up.ne1 != shared_inter_ ||
      down.ne0 != shared_inter_ || down.ne1 != d_model_ ||
      (has_gate_ && (!gate_inp || !gate_inp->host_data || gate_inp->ne0 != d_model_ ||
                     gate_inp->ne1 != 1)))
    return fail("basic validation");
  if (gate.type >= (uint32_t)GGML_TYPE_COUNT ||
      up.type >= (uint32_t)GGML_TYPE_COUNT ||
      down.type >= (uint32_t)GGML_TYPE_COUNT ||
      (gate_inp && gate_inp->type >= (uint32_t)GGML_TYPE_COUNT))
    return fail("type range");
  if (!type_supported(gate.type) || !type_supported(up.type) ||
      !type_supported(down.type) ||
      (has_gate_ && !type_supported(gate_inp->type)))
    return fail("type unsupported");

  const uint32_t new_gate_inp_type = has_gate_ ? gate_inp->type : 0;
  const bool type_changed =
      weight_ctx_ &&
      (gate_type_ != gate.type || up_type_ != up.type ||
       down_type_ != down.type || gate_inp_type_ != new_gate_inp_type);
  if (type_changed) {
    ggml_backend_synchronize((ggml_backend_t)backend_);
    free_graph();
    if (weight_buf_) {
      ggml_backend_buffer_free((ggml_backend_buffer_t)weight_buf_);
      weight_buf_ = nullptr;
    }
    if (weight_ctx_) {
      ggml_free((ggml_context *)weight_ctx_);
      weight_ctx_ = nullptr;
    }
    gate_ = up_ = down_ = gate_inp_ = nullptr;
  }

  cudaSetDevice(device_);
  if (!weight_ctx_) {
    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead() * 12;
    ip.no_alloc = true;
    ggml_context *ctx = ggml_init(ip);
    if (!ctx) return fail("weight context allocation");
    ggml_tensor *g = ggml_new_tensor_2d(ctx, (ggml_type)gate.type,
                                        d_model_, shared_inter_);
    ggml_tensor *u = ggml_new_tensor_2d(ctx, (ggml_type)up.type,
                                        d_model_, shared_inter_);
    ggml_tensor *d = ggml_new_tensor_2d(ctx, (ggml_type)down.type,
                                        shared_inter_, d_model_);
    ggml_tensor *gi = has_gate_
                         ? ggml_new_tensor_2d(ctx, (ggml_type)gate_inp->type,
                                              d_model_, 1)
                         : nullptr;
    if (!g || !u || !d || (has_gate_ && !gi)) {
      ggml_free(ctx);
      return fail("weight tensor creation");
    }
    auto bind_external = [&](ggml_tensor *tensor,
                             const GgmlSharedWeightRef &ref) {
      if (!external_buf_ || !ref.device_data ||
          ref.device_bytes < ggml_nbytes(tensor))
        return false;
      tensor->data = const_cast<void *>(ref.device_data);
      tensor->buffer = (ggml_backend_buffer_t)external_buf_;
      ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
      return true;
    };
    const bool ext_gate = bind_external(g, gate);
    const bool ext_up = bind_external(u, up);
    const bool ext_down = bind_external(d, down);
    const bool ext_gate_inp = !has_gate_ || bind_external(gi, *gate_inp);
    const bool all_external = ext_gate && ext_up && ext_down && ext_gate_inp;
    ggml_backend_buffer_t buf = nullptr;
    if (!all_external)
      buf = ggml_backend_alloc_ctx_tensors(ctx, (ggml_backend_t)backend_);
    if (!all_external && !buf) {
      ggml_free(ctx);
      return fail("weight buffer allocation");
    }
    weight_ctx_ = ctx;
    weight_buf_ = buf;
    gate_ = g;
    up_ = u;
    down_ = d;
    gate_inp_ = gi;
    owned_gate_data_ = ext_gate ? nullptr : g->data;
    owned_up_data_ = ext_up ? nullptr : u->data;
    owned_down_data_ = ext_down ? nullptr : d->data;
    owned_gate_inp_data_ = (has_gate_ && !ext_gate_inp) ? gi->data : nullptr;
    gate_type_ = gate.type;
    up_type_ = up.type;
    down_type_ = down.type;
    gate_inp_type_ = new_gate_inp_type;
  }

  const size_t gb = ggml_nbytes((ggml_tensor *)gate_);
  const size_t ub = ggml_nbytes((ggml_tensor *)up_);
  const size_t db = ggml_nbytes((ggml_tensor *)down_);
  if (gate.nbytes < gb || up.nbytes < ub || down.nbytes < db ||
      (has_gate_ && gate_inp->nbytes <
                         ggml_nbytes((ggml_tensor *)gate_inp_)))
    return fail("weight bytes short");
  // Rebind persistent dense-pool pointers before each graph submission. If a
  // weight has no device copy, restore its private allocation and retain the
  // established async host upload fallback. A different-device/partial-pool
  // configuration therefore remains correct without borrowing foreign VRAM.
  auto bind_or_upload = [&](ggml_tensor *tensor, const GgmlSharedWeightRef &ref,
                            void *owned_data, size_t bytes) -> bool {
    const bool external = external_buf_ && ref.device_data &&
                          ref.device_bytes >= ggml_nbytes(tensor);
    if (external) {
      tensor->data = const_cast<void *>(ref.device_data);
      tensor->buffer = (ggml_backend_buffer_t)external_buf_;
      ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
      return true;
    }
    if (!owned_data || !weight_buf_) return false;
    tensor->data = owned_data;
    tensor->buffer = (ggml_backend_buffer_t)weight_buf_;
    ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
    // All weight updates and the following graph run share one GGML CUDA
    // stream. This queues only the weights that are not already in the pool.
    ggml_backend_tensor_set_async((ggml_backend_t)backend_, tensor,
                                  ref.host_data, 0, bytes);
    return true;
  };
  const bool weights_ok =
      bind_or_upload((ggml_tensor *)gate_, gate, owned_gate_data_, gb) &&
      bind_or_upload((ggml_tensor *)up_, up, owned_up_data_, ub) &&
      bind_or_upload((ggml_tensor *)down_, down, owned_down_data_, db) &&
      (!has_gate_ || bind_or_upload((ggml_tensor *)gate_inp_, *gate_inp,
                                    owned_gate_inp_data_,
                                    ggml_nbytes((ggml_tensor *)gate_inp_)));
  if (!weights_ok) return fail("weight bind/upload");
  static std::atomic<bool> logged_external{false};
  if (gate.device_data && up.device_data && down.device_data) {
    bool expected = false;
    if (logged_external.compare_exchange_strong(expected, true,
                                                std::memory_order_relaxed))
      LOG_INFO("[GGML-SHARED] borrowing persistent dense-pool weights; "
               "removing per-layer shared-expert weight H2D copies\n");
  }
  if (!graph_ready_ && !build_graph()) {
    free_graph();
    return fail("graph build");
  }
  if (layer == 0)
    LOG_INFO("[GGML-SHARED] layer weights ready (gate=%u up=%u down=%u%s)\n",
             (unsigned)gate.type, (unsigned)up.type, (unsigned)down.type,
             has_gate_ ? " + scalar gate" : "");
  return true;
}

bool GgmlQwenSharedRunner::run(int layer, const void *device_x,
                               const void *routed_output,
                               const void **combined_output) {
  const bool debug = [] {
    const char *e = std::getenv("MOESTREAM_GGML_SHARED_DEBUG");
    return e && e[0] == '1';
  }();
  auto fail = [&](const char *reason, const ggml_tensor *input,
                  const ggml_tensor *routed_src) {
    if (debug) {
      const auto dump = [](const ggml_tensor *t) {
        if (!t) return std::string("null");
        return "type=" + std::to_string((unsigned)t->type) +
               " ne=" + std::to_string((long long)t->ne[0]) + "," +
               std::to_string((long long)t->ne[1]) + "," +
               std::to_string((long long)t->ne[2]) + "," +
               std::to_string((long long)t->ne[3]) +
               " nb=" + std::to_string(t->nb[0]) + "," +
               std::to_string(t->nb[1]) + " data=" +
               std::to_string((uintptr_t)t->data) + " buffer=" +
               std::to_string((uintptr_t)t->buffer);
      };
      LOG_WARN("[GGML-SHARED-DEBUG] run layer=%d rejected: %s input={%s} "
               "routed={%s} graph_input={%s} graph_routed={%s} ready=%d "
               "direct=%d\n",
               layer, reason, dump(input).c_str(), dump(routed_src).c_str(),
               dump((ggml_tensor *)x_).c_str(),
               dump((ggml_tensor *)routed_).c_str(),
               graph_ready_ ? 1 : 0, use_direct_shared_inputs() ? 1 : 0);
    }
    return false;
  };
  if (!backend_ || !graph_ready_ || layer < 0 || !device_x ||
      !routed_output || !combined_output)
    return fail("missing state/input", (ggml_tensor *)device_x,
                (ggml_tensor *)routed_output);
  ggml_tensor *routed = (ggml_tensor *)routed_output;
  ggml_tensor *x = (ggml_tensor *)x_;
  ggml_tensor *r = (ggml_tensor *)routed_;
  if (r->type != GGML_TYPE_F32 || r->ne[0] != d_model_ ||
      r->ne[1] != n_tokens_ || r->ne[2] != 1 || r->ne[3] != 1 ||
      r->nb[0] != sizeof(float) ||
      r->nb[1] != (size_t)d_model_ * sizeof(float) || !r->data || !r->buffer)
    return fail("persistent routed tensor validation", routed, routed);
  // run_shared_only() temporarily retargets the normal routed input to a
  // device-zero buffer. Restore the graph-owned allocation before a normal
  // routed run uses the same persistent graph again.
  r->data = routed_owned_data_;
  // 2026-09-10: real, confirmed root cause of the shared-expert NaN
  // corruption bug (see Test 35, which reproduces it in isolation with a
  // CPU oracle). Every production caller of run() -- run_qwen4exp_shared_device()
  // in execution_layer.cpp, fed by GgmlHcRunner::pre_mix()'s
  // `*device_output = t_mixed_` -- passes device_x as a `ggml_tensor *`
  // (t_mixed_), exactly like routed_output below. This function instead
  // fed device_x straight into cudaMemcpy() as if it were already a raw
  // CUDA device pointer: copying from a host ggml_tensor struct's address
  // via cudaMemcpyDeviceToDevice is undefined behavior, and in practice
  // silently produced an all-zero (or garbage) x_ input -- silu(0)*0=0
  // down the whole gate/up/act/shared chain, so `combined` collapsed to
  // just `routed` (see the "combined == routed" trace). Dereference it the
  // same way `src` is handled just below, instead of memcpy'ing the wrong
  // bytes.
  ggml_tensor *hc_in = (ggml_tensor *)device_x;
  // GgmlHcRunner keeps one persistent [D,S_cap] state tensor across a
  // request.  Decode reuses that tensor with S=1, so its second dimension can
  // be larger than this shared graph's current token count.  The row stride
  // check below guarantees that the graph consumes only the first S rows.
  if (hc_in->type != GGML_TYPE_F32 || hc_in->ne[0] != d_model_ ||
      hc_in->ne[1] < n_tokens_ || hc_in->ne[2] != 1 || hc_in->ne[3] != 1 ||
      hc_in->nb[0] != sizeof(float) ||
      hc_in->nb[1] != (size_t)d_model_ * sizeof(float) || !hc_in->data ||
      !hc_in->buffer)
    return fail("HC input tensor validation", hc_in, routed);
  ggml_tensor *src = (ggml_tensor *)routed_output;
  if (src->type != GGML_TYPE_F32 || src->ne[0] != d_model_ ||
      src->ne[1] != n_tokens_ || src->ne[2] != 1 || src->ne[3] != 1 ||
      src->nb[0] != sizeof(float) ||
      src->nb[1] != (size_t)d_model_ * sizeof(float) || !src->data ||
      !src->buffer)
    return fail("routed input tensor validation", hc_in, src);

  // Opt-in A/A diagnostic for the first FFN boundary.  The routed tensor is
  // the output of GgmlMoeRunner; the combined tensor is the result of this
  // shared+routed graph.  Keep this synchronizing path strictly diagnostic.
  const char *dump_base = std::getenv("MOESTREAM_GGML_HC_COMPARE_DUMP");
  const bool dump_ffn = dump_base && n_tokens_ == 1 && layer < 3;
  auto dump_device_f32 = [&](const char *label, const ggml_tensor *tensor) {
    if (!dump_ffn || !tensor || !tensor->data) return;
    cudaSetDevice(device_);
    if (cudaDeviceSynchronize() != cudaSuccess) return;
    const size_t n = (size_t)ggml_nelements(tensor);
    std::vector<float> host(n);
    if (cudaMemcpy(host.data(), tensor->data,
                   host.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      return;
    std::string path = std::string(dump_base) + ".l" +
                       (layer < 10 ? "0" : "") + std::to_string(layer) +
                       ".s1." + label + ".f32";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (out)
      out.write(reinterpret_cast<const char *>(host.data()),
                static_cast<std::streamsize>(host.size() * sizeof(float)));
  };
  dump_device_f32("routed_in", src);
  const bool direct_inputs = use_direct_shared_inputs();
  if (direct_inputs) {
    // Match GgmlRouterRunner's already-validated direct-input optimization.
    // Keep each graph input's original buffer association intact; ggml-cuda
    // uses the data pointer for the actual read, while the buffer continues
    // to describe the allocator/backend that owns the graph tensor.
    x->data = hc_in->data;
    r->data = src->data;
    static bool logged_direct = false;
    if (!logged_direct) {
      logged_direct = true;
      LOG_INFO("[GGML-SHARED] direct HC/MoE inputs active: removing two "
               "per-layer D2D copies and host sync\n");
    }
  } else {
    cudaSetDevice(device_);
    const size_t bytes = (size_t)d_model_ * n_tokens_ * sizeof(float);
    if (cudaMemcpy(x->data, hc_in->data, bytes, cudaMemcpyDeviceToDevice) !=
        cudaSuccess)
      return fail("HC device copy", hc_in, src);
    // The routed result is produced by GgmlMoeRunner's graph.  Copy that
    // producer tensor into this graph's independent input; copying the
    // destination tensor to itself would leave stale routed data after the
    // first layer and change all subsequent routing decisions.
    ggml_backend_tensor_copy_async((ggml_backend_t)backend_,
                                   (ggml_backend_t)backend_, src, r);
  }
  if (ggml_backend_graph_compute_async((ggml_backend_t)backend_,
                                       (ggml_cgraph *)graph_) !=
      GGML_STATUS_SUCCESS)
    return fail("graph compute", hc_in, src);
  if (!record_output_event())
    return fail("output event record", hc_in, src);
  // 2026-09-24: real, reproduced GPU correctness bug (found via device-side
  // kernel tracing, not a race). ggml-cuda's mul_mat(down_weight[Q8_0],
  // activation) for this shared-expert down-projection can produce NaN in
  // exactly the LAST output row (d_model-1) for every token, only for
  // batched (n_tokens_>1, i.e. prefill) calls -- confirmed reproducible on
  // qwen3.8-flash-next layer 11 with clean, verified-finite weight bytes
  // (checked against the GGUF file) and a clean, verified-finite activation
  // input; S=1 decode calls with the identical weights never showed it. The
  // corrupted value then silently poisons the persistent HC stream state for
  // the rest of generation (`!!!!!!!!!!` output). Root cause is presumably a
  // narrow ggml-cuda kernel-selection edge case for this exact (K=640,
  // M=2560, N>1, Q8_0) shape; rather than patch vendored ggml-cuda, detect
  // and refuse the corrupted result here so the caller's already-established
  // host-shared-expert fallback (see batched_dispatch_pipeline.cpp) recovers
  // instead of propagating NaN. Scoped to n_tokens_>1 to keep the decode hot
  // path (S=1, called every layer every token) exactly as fast as before --
  // that path was never observed to reproduce this.
  if (n_tokens_ > 1) {
    cudaSetDevice(device_);
    cudaDeviceSynchronize();
    const size_t n = (size_t)d_model_ * n_tokens_;
    static thread_local std::vector<float> finite_check;
    finite_check.resize(n);
    const ggml_tensor *out_t = (const ggml_tensor *)output_;
    const cudaError_t e = cudaMemcpy(finite_check.data(), out_t->data,
                                     n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess)
      return fail("output finiteness readback", hc_in, src);
    for (float v : finite_check) {
      if (!std::isfinite(v)) {
        LOG_ERROR("[GGML-SHARED] layer=%d S=%d: non-finite value in shared+"
                  "routed FFN output -- rejecting device result, falling "
                  "back to host shared-expert path\n",
                  layer, n_tokens_);
        return fail("non-finite shared output", hc_in, src);
      }
    }
  }
  dump_device_f32("shared_raw", (const ggml_tensor *)shared_raw_);
  dump_device_f32("shared_gated", (const ggml_tensor *)shared_);
  if (dump_ffn)
    dump_device_f32("combined_out", (const ggml_tensor *)output_);
  // In the conservative mode, preserve the old request-local completion
  // fence.  In direct-input/async-reuse mode the next operation is HC
  // post-combine on this same backend, so its queued tensor copy is the
  // consumer-side dependency and a host wait would only serialize the graph
  // pipeline.  A later host readback still synchronizes through GGML's normal
  // tensor-get path.
  if (!direct_inputs)
    ggml_backend_synchronize((ggml_backend_t)backend_);
  if (debug) {
    cudaSetDevice(device_);
    cudaDeviceSynchronize();
    const size_t n = (size_t)d_model_ * n_tokens_;
    static thread_local std::vector<float> check;
    check.resize(n);
    const cudaError_t e = cudaMemcpy(
        check.data(), ((ggml_tensor *)output_)->data,
        n * sizeof(float), cudaMemcpyDeviceToHost);
    double l2 = 0.0;
    if (e == cudaSuccess)
      for (float v : check) l2 += (double)v * v;
    LOG_WARN("[GGML-SHARED-DEBUG] output layer=%d S=%d copy=%s l2=%g "
             "first=%g,%g,%g,%g\n",
             layer, n_tokens_, e == cudaSuccess ? "ok" : cudaGetErrorString(e),
             std::sqrt(l2), n > 0 ? check[0] : 0.0f,
             n > 1 ? check[1] : 0.0f, n > 2 ? check[2] : 0.0f,
             n > 3 ? check[3] : 0.0f);
  }
  *combined_output = output_;
  return true;
}

bool GgmlQwenSharedRunner::run_host_routed(int layer, const void *device_x,
                                           const float *host_routed,
                                           const void **combined_output) {
  const bool debug = [] {
    const char *e = std::getenv("MOESTREAM_GGML_SHARED_DEBUG");
    return e && e[0] == '1';
  }();
  if (combined_output)
    *combined_output = nullptr;
  if (!backend_ || !graph_ready_ || layer < 0 || !device_x || !host_routed ||
      !combined_output)
    return false;

  ggml_tensor *hc_in = (ggml_tensor *)device_x;
  ggml_tensor *x = (ggml_tensor *)x_;
  ggml_tensor *r = (ggml_tensor *)routed_;
  const size_t bytes = (size_t)d_model_ * (size_t)n_tokens_ * sizeof(float);
  if (!hc_in || !x || !r || hc_in->type != GGML_TYPE_F32 ||
      hc_in->ne[0] != d_model_ || hc_in->ne[1] < n_tokens_ ||
      hc_in->ne[2] != 1 || hc_in->ne[3] != 1 ||
      hc_in->nb[0] != sizeof(float) ||
      hc_in->nb[1] != (size_t)d_model_ * sizeof(float) || !hc_in->data ||
      !hc_in->buffer || r->type != GGML_TYPE_F32 ||
      r->ne[0] != d_model_ || r->ne[1] != n_tokens_ || r->ne[2] != 1 ||
      r->ne[3] != 1 || r->nb[0] != sizeof(float) ||
      r->nb[1] != (size_t)d_model_ * sizeof(float) || !r->buffer ||
      !routed_owned_data_)
    return false;

  // A previous direct-input run may have pointed the graph input at another
  // runner's output. Restore the persistent allocation before tensor_set_async
  // so the upload cannot overwrite a producer-owned buffer.
  r->data = routed_owned_data_;
  x->data = x_owned_data_;
  if (!r->data || !x->data)
    return false;

  const bool direct_inputs = use_direct_shared_inputs();
  if (direct_inputs) {
    // HC is already on the shared backend. The routed result is the only
    // host-originating input; queue its H2D into the persistent graph input.
    x->data = hc_in->data;
  } else {
    cudaSetDevice(device_);
    if (cudaMemcpy(x->data, hc_in->data, bytes, cudaMemcpyDeviceToDevice) !=
        cudaSuccess)
      return false;
  }
  ggml_backend_tensor_set_async((ggml_backend_t)backend_, r, host_routed, 0,
                                bytes);
  if (ggml_backend_graph_compute_async((ggml_backend_t)backend_,
                                       (ggml_cgraph *)graph_) !=
      GGML_STATUS_SUCCESS) {
    if (debug)
      LOG_WARN(
          "[GGML-SHARED-DEBUG] host-routed graph compute failed layer=%d\n",
          layer);
    return false;
  }
  if (!record_output_event())
    return false;
  // The CPU-affinity Qwen path uses run_host_routed(), not run().  Keep the
  // same opt-in shared-expert snapshots on this production entry point so an
  // A/A comparison cannot accidentally observe only the host fallback.
  if (n_tokens_ == 1 && layer < 3) {
    const char *dump_base = std::getenv("MOESTREAM_GGML_HC_COMPARE_DUMP");
    if (dump_base) {
      cudaSetDevice(device_);
      if (cudaDeviceSynchronize() == cudaSuccess) {
        auto dump = [&](const char *label, const ggml_tensor *tensor) {
          if (!tensor || !tensor->data) return;
          const size_t n = (size_t)ggml_nelements(tensor);
          std::vector<float> host(n);
          if (cudaMemcpy(host.data(), tensor->data,
                         n * sizeof(float), cudaMemcpyDeviceToHost) !=
              cudaSuccess)
            return;
          std::string path = std::string(dump_base) + ".l" +
                             (layer < 10 ? "0" : "") + std::to_string(layer) +
                             ".s1." + label + ".f32";
          std::ofstream out(path, std::ios::binary | std::ios::trunc);
          if (out)
            out.write(reinterpret_cast<const char *>(host.data()),
                  static_cast<std::streamsize>(n * sizeof(float)));
        };
        // Capture the exact shared-expert input for the canonical GGML path;
        // this lets the A/A comparison distinguish a producer mismatch from
        // a Q5_K/Q8_0 projection-kernel mismatch.
        if (hc_in->data) {
          std::vector<float> host((size_t)d_model_ * n_tokens_);
          if (cudaMemcpy(host.data(), hc_in->data,
                         host.size() * sizeof(float),
                         cudaMemcpyDeviceToHost) == cudaSuccess) {
            std::ofstream out(std::string(dump_base) + ".l" +
                                  (layer < 10 ? "0" : "") +
                                  std::to_string(layer) +
                                  ".s1.shared_input.f32",
                              std::ios::binary | std::ios::trunc);
            if (out)
              out.write(reinterpret_cast<const char *>(host.data()),
                        static_cast<std::streamsize>(host.size() *
                                                     sizeof(float)));
          }
        }
        dump("shared_raw", (const ggml_tensor *)shared_raw_);
        dump("shared_gated", (const ggml_tensor *)shared_);
        dump("combined_out", (const ggml_tensor *)output_);
      }
    }
  }
  if (!direct_inputs)
    ggml_backend_synchronize((ggml_backend_t)backend_);
  *combined_output = output_;
  return true;
}

bool GgmlQwenSharedRunner::run_shared_only(int layer, const void *device_x,
                                           const void **shared_output) {
  if (shared_output)
    *shared_output = nullptr;
  if (!backend_ || !graph_ready_ || !overlap_graph_ready_ || layer < 0 ||
      !device_x || !shared_output || !zero_routed_data_)
    return false;

  ggml_tensor *hc_in = (ggml_tensor *)device_x;
  ggml_tensor *x = (ggml_tensor *)x_;
  ggml_tensor *r = (ggml_tensor *)routed_;
  if (!hc_in || !x || !r || hc_in->type != GGML_TYPE_F32 ||
      hc_in->ne[0] != d_model_ || hc_in->ne[1] < n_tokens_ ||
      hc_in->ne[2] != 1 || hc_in->ne[3] != 1 ||
      hc_in->nb[0] != sizeof(float) ||
      hc_in->nb[1] != (size_t)d_model_ * sizeof(float) || !hc_in->data ||
      !hc_in->buffer)
    return false;

  const size_t bytes = (size_t)d_model_ * (size_t)n_tokens_ * sizeof(float);
  const bool direct_inputs = use_direct_shared_inputs();
  if (direct_inputs) {
    // The HC producer and this graph are ordered on the same GGML backend
    // stream.  Retargeting the persistent input removes the blocking D2D
    // copy and lets the CPU worker start while CUDA computes the shared FFN.
    x->data = hc_in->data;
  } else {
    cudaSetDevice(device_);
    if (cudaMemcpy(x->data, hc_in->data, bytes, cudaMemcpyDeviceToDevice) !=
        cudaSuccess)
      return false;
  }
  r->data = zero_routed_data_;
  if (ggml_backend_graph_compute_async(
          (ggml_backend_t)backend_, (ggml_cgraph *)graph_) !=
      GGML_STATUS_SUCCESS)
    return false;
  if (!record_output_event())
    return false;
  if (!direct_inputs)
    ggml_backend_synchronize((ggml_backend_t)backend_);
  *shared_output = output_;
  return true;
}

bool GgmlQwenSharedRunner::run_host_routed_add(
    int layer, const float *host_routed, const void **combined_output) {
  if (combined_output)
    *combined_output = nullptr;
  if (!backend_ || !overlap_graph_ready_ || layer < 0 || !host_routed ||
      !combined_output || !overlap_routed_ || !overlap_shared_input_ ||
      !overlap_combined_ || !overlap_routed_owned_data_ || !output_)
    return false;

  auto *routed = (ggml_tensor *)overlap_routed_;
  auto *shared = (ggml_tensor *)overlap_shared_input_;
  routed->data = overlap_routed_owned_data_;
  // output_ is the shared-only result because run_shared_only() fed the
  // normal graph a zero routed input.  Preserve this graph input's original
  // buffer association while retargeting only its data pointer, exactly like
  // the established direct-input path above.
  shared->data = ((ggml_tensor *)output_)->data;

  const size_t bytes = (size_t)d_model_ * (size_t)n_tokens_ * sizeof(float);
  ggml_backend_tensor_set_async((ggml_backend_t)backend_, routed,
                                host_routed, 0, bytes);
  if (ggml_backend_graph_compute_async(
          (ggml_backend_t)backend_,
          (ggml_cgraph *)overlap_combine_graph_) != GGML_STATUS_SUCCESS)
    return false;
  if (!record_output_event())
    return false;
  if (!use_direct_shared_inputs())
    ggml_backend_synchronize((ggml_backend_t)backend_);
  *combined_output = overlap_combined_;
  return true;
}

bool GgmlQwenSharedRunner::run_device_routed_add(
    int layer, const void *base_device, const void *device_routed,
    const void **combined_output) {
  if (combined_output)
    *combined_output = nullptr;
  if (!backend_ || !overlap_graph_ready_ || layer < 0 || !base_device ||
      !device_routed || !combined_output || !overlap_routed_ ||
      !overlap_shared_input_ || !overlap_combined_)
    return false;

  auto *routed = (ggml_tensor *)overlap_routed_;
  auto *base = (ggml_tensor *)overlap_shared_input_;
  auto *base_output = (ggml_tensor *)const_cast<void *>(base_device);
  if (!routed || !base || routed->type != GGML_TYPE_F32 ||
      base->type != GGML_TYPE_F32 || !base_output ||
      base_output->type != GGML_TYPE_F32 || routed->ne[0] != d_model_ ||
      base_output->ne[0] != d_model_ || routed->ne[1] < n_tokens_ ||
      base_output->ne[1] < n_tokens_ || routed->ne[2] != 1 ||
      base_output->ne[2] != 1 || routed->ne[3] != 1 ||
      base_output->ne[3] != 1 ||
      routed->nb[0] != sizeof(float) || base->nb[0] != sizeof(float) ||
      base_output->nb[0] != sizeof(float) ||
      routed->nb[1] != (size_t)d_model_ * sizeof(float) ||
      base_output->nb[1] != (size_t)d_model_ * sizeof(float) ||
      !routed->buffer || !base->buffer || !base_output->data ||
      !base_output->buffer)
    return false;

  // Opt-in A/A diagnostic: export the two inputs to the final shared+routed
  // add before the combine graph runs.  This is deliberately a synchronizing
  // path and is never enabled during normal inference.
  if (n_tokens_ == 1 && layer < 3) {
    const char *dump_base = std::getenv("MOESTREAM_GGML_HC_COMPARE_DUMP");
    if (dump_base) {
      cudaSetDevice(device_);
      cudaDeviceSynchronize();
      auto dump = [&](const char *label, const ggml_tensor *tensor) {
        if (!tensor || !tensor->data) return;
        std::vector<float> host((size_t)d_model_);
        if (cudaMemcpy(host.data(), tensor->data,
                       host.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
          return;
        std::string path = std::string(dump_base) + ".l" +
                           (layer < 10 ? "0" : "") + std::to_string(layer) +
                           ".s1." + label + ".f32";
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (out)
          out.write(reinterpret_cast<const char *>(host.data()),
                    static_cast<std::streamsize>(host.size() * sizeof(float)));
      };
      dump("shared_out", base_output);
      dump("routed_out", (const ggml_tensor *)device_routed);
    }
  }

  // The graph owns the tensor metadata and buffer association.  Retargeting
  // only `data` is the same external-pointer protocol used by the shared-only
  // path; both pointers are device-resident and the caller has already
  // established their stream dependency.
  routed->data = const_cast<void *>(device_routed);
  base->data = base_output->data;
  if (ggml_backend_graph_compute_async(
          (ggml_backend_t)backend_,
          (ggml_cgraph *)overlap_combine_graph_) != GGML_STATUS_SUCCESS)
    return false;
  if (!record_output_event())
    return false;
  *combined_output = overlap_combined_;
  return true;
}

void *GgmlQwenSharedRunner::backend_stream() const {
  if (!backend_)
    return nullptr;
  return reinterpret_cast<void *>(
      moestream_ggml_backend_cuda_get_stream((ggml_backend_t)backend_));
}

bool GgmlQwenSharedRunner::copy_output_to_host(const void *combined_output,
                                               float *host_out) {
  if (!backend_ || !graph_ready_ || !combined_output || !host_out ||
      (combined_output != output_ && combined_output != overlap_combined_)) {
    const char *diag = std::getenv("MOESTREAM_PREFILL_BOUNDARY_DUMP");
    if (diag) {
      LOG_ERROR("[GGML-SHARED-DIAG] output copy rejected backend=%d ready=%d "
                "arg=%p output=%p overlap=%p host=%p\n",
                backend_ ? 1 : 0, graph_ready_ ? 1 : 0, combined_output,
                output_, overlap_combined_, (void *)host_out);
    }
    return false;
  }
  cudaSetDevice(device_);
  const ggml_tensor *tensor =
      (const ggml_tensor *)combined_output;
  if (!tensor->data || tensor->type != GGML_TYPE_F32 ||
      tensor->ne[0] != d_model_ || tensor->ne[1] < n_tokens_ ||
      tensor->nb[0] != sizeof(float) ||
      tensor->nb[1] != (size_t)d_model_ * sizeof(float)) {
    const char *diag = std::getenv("MOESTREAM_PREFILL_BOUNDARY_DUMP");
    if (diag) {
      LOG_ERROR("[GGML-SHARED-DIAG] output tensor rejected data=%p type=%u "
                "ne=%lld,%lld nb=%zu,%zu expected=%d,%d,%zu\n",
                tensor->data, (unsigned)tensor->type,
                (long long)tensor->ne[0], (long long)tensor->ne[1],
                tensor->nb[0], tensor->nb[1], d_model_, n_tokens_,
                (size_t)d_model_ * sizeof(float));
    }
    return false;
  }
  const size_t bytes = (size_t)d_model_ * n_tokens_ * sizeof(float);
  ggml_backend_tensor_get((ggml_tensor *)combined_output, host_out, 0, bytes);
  return true;
}

// ---------------------------------------------------------------------------
// Qwen CPU-affinity post-down GPU combine
// ---------------------------------------------------------------------------

GgmlQwenDeviceCombineRunner &GgmlQwenDeviceCombineRunner::instance() {
  static GgmlQwenDeviceCombineRunner runner;
  return runner;
}

bool GgmlQwenDeviceCombineRunner::compiled_in() { return true; }

GgmlQwenDeviceCombineRunner::~GgmlQwenDeviceCombineRunner() { shutdown(); }

void GgmlQwenDeviceCombineRunner::shutdown() {
  if (backend_)
    ggml_backend_synchronize((ggml_backend_t)backend_);

  if (galloc_) {
    ggml_gallocr_free((ggml_gallocr_t)galloc_);
    galloc_ = nullptr;
  }
  if (graph_ctx_) {
    ggml_free((ggml_context *)graph_ctx_);
    graph_ctx_ = nullptr;
  }
  graph_ = expert_output_ = weights_ = output_ = nullptr;
  graph_ready_ = false;

  for (void *event : stage_events_) {
    if (event)
      ggml_backend_event_free((ggml_backend_event_t)event);
  }
  stage_events_.clear();
  stage_used_.clear();
  for (void *ptr : host_stage_) {
    if (ptr)
      cudaFreeHost(ptr);
  }
  for (void *ptr : weight_stage_) {
    if (ptr)
      cudaFreeHost(ptr);
  }
  host_stage_.clear();
  weight_stage_.clear();
  backend_ = nullptr;
  device_ = -1;
  d_model_ = topk_ = stage_next_ = 0;
}

bool GgmlQwenDeviceCombineRunner::ensure(int device, int d_model, int topk) {
  if (device < 0 || d_model <= 0 || topk <= 0)
    return false;
  if (graph_ready_ && device_ == device && d_model_ == d_model &&
      topk_ == topk)
    return true;

  if (backend_)
    shutdown();
  backend_ = ggml_shared_backend(device);
  if (!backend_)
    return false;
  cudaSetDevice(device);

  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 128 + ggml_graph_overhead();
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx)
    return false;
  ggml_cgraph *graph = ggml_new_graph(ctx);
  ggml_tensor *expert = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, topk);
  ggml_tensor *weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, topk);
  if (!graph || !expert || !weights) {
    ggml_free(ctx);
    return false;
  }
  ggml_set_input(expert);
  ggml_set_input(weights);
  ggml_set_name(expert, "qwen4exp_cpu_expert_outputs");
  ggml_set_name(weights, "qwen4exp_cpu_router_weights");

  ggml_tensor *weighted = ggml_mul(ctx, expert, weights);
  if (!weighted) {
    ggml_free(ctx);
    return false;
  }
  ggml_tensor *sum = nullptr;
  for (int k = 0; k < topk; ++k) {
    ggml_tensor *view =
        ggml_view_2d(ctx, weighted, d_model, 1, weighted->nb[1],
                     static_cast<size_t>(k) * weighted->nb[1]);
    sum = k == 0 ? view : ggml_add(ctx, sum, view);
  }
  if (!sum) {
    ggml_free(ctx);
    return false;
  }
  sum = ggml_cont(ctx, sum);
  ggml_set_output(sum);
  ggml_set_name(sum, "qwen4exp_cpu_gpu_combined");
  ggml_build_forward_expand(graph, sum);

  ggml_backend_buffer_type_t bt =
      ggml_backend_get_default_buffer_type((ggml_backend_t)backend_);
  ggml_gallocr_t ga = ggml_gallocr_new(bt);
  if (!ga || ggml_gallocr_alloc_graph(ga, graph) != GGML_STATUS_SUCCESS) {
    if (ga)
      ggml_gallocr_free(ga);
    ggml_free(ctx);
    return false;
  }

  host_stage_.assign(kStageBuffers, nullptr);
  weight_stage_.assign(kStageBuffers, nullptr);
  stage_events_.assign(kStageBuffers, nullptr);
  stage_used_.assign(kStageBuffers, 0);
  const size_t output_bytes =
      static_cast<size_t>(d_model) * static_cast<size_t>(topk) * sizeof(float);
  const size_t weight_bytes = static_cast<size_t>(topk) * sizeof(float);
  for (int i = 0; i < kStageBuffers; ++i) {
    if (cudaMallocHost(&host_stage_[i], output_bytes) != cudaSuccess ||
        cudaMallocHost(&weight_stage_[i], weight_bytes) != cudaSuccess) {
      ggml_gallocr_free(ga);
      ggml_free(ctx);
      shutdown();
      return false;
    }
    stage_events_[i] = ggml_backend_event_new(
        ggml_backend_get_device((ggml_backend_t)backend_));
    if (!stage_events_[i]) {
      ggml_gallocr_free(ga);
      ggml_free(ctx);
      shutdown();
      return false;
    }
  }

  graph_ctx_ = ctx;
  galloc_ = ga;
  graph_ = graph;
  expert_output_ = expert;
  weights_ = weights;
  output_ = sum;
  device_ = device;
  d_model_ = d_model;
  topk_ = topk;
  stage_next_ = 0;
  graph_ready_ = true;
  LOG_INFO("[GGML-QWEN-COMBINE] GPU post-down combine ready "
           "(GPU=%d D=%d topk=%d staged=%d)\n",
           device, d_model, topk, kStageBuffers);
  return true;
}

bool GgmlQwenDeviceCombineRunner::run(const float *host_expert_output,
                                      const float *host_weights,
                                      const void **device_output) {
  if (device_output)
    *device_output = nullptr;
  if (!graph_ready_ || !host_expert_output || !host_weights ||
      !device_output)
    return false;

  const int slot = stage_next_++ % kStageBuffers;
  if (stage_used_[slot]) {
    cudaError_t ready = cudaEventQuery(
        (cudaEvent_t)((ggml_backend_event_t)stage_events_[slot])->context);
    if (ready == cudaErrorNotReady) {
      // Bounded backpressure is safer than overwriting a live DMA source.
      ggml_backend_event_synchronize(
          (ggml_backend_event_t)stage_events_[slot]);
    } else if (ready != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
  }

  const size_t output_bytes =
      static_cast<size_t>(d_model_) * static_cast<size_t>(topk_) *
      sizeof(float);
  const size_t weight_bytes = static_cast<size_t>(topk_) * sizeof(float);
  std::memcpy(host_stage_[slot], host_expert_output, output_bytes);
  std::memcpy(weight_stage_[slot], host_weights, weight_bytes);
  ggml_backend_tensor_set_async((ggml_backend_t)backend_,
                                (ggml_tensor *)expert_output_,
                                host_stage_[slot], 0, output_bytes);
  ggml_backend_tensor_set_async((ggml_backend_t)backend_,
                                (ggml_tensor *)weights_, weight_stage_[slot],
                                0, weight_bytes);
  if (ggml_backend_graph_compute_async((ggml_backend_t)backend_,
                                       (ggml_cgraph *)graph_) !=
      GGML_STATUS_SUCCESS)
    return false;
  ggml_backend_event_record((ggml_backend_event_t)stage_events_[slot],
                            (ggml_backend_t)backend_);
  stage_used_[slot] = 1;
  *device_output = output_;
  return true;
}
#else  // !MOESTREAM_USE_GGML_CUDA
bool GgmlQwenSharedRunner::compiled_in() { return false; }
GgmlQwenSharedRunner &GgmlQwenSharedRunner::instance() {
  static GgmlQwenSharedRunner r;
  return r;
}
GgmlQwenSharedRunner &GgmlQwenSharedRunner::instance_for_layer(int layer) {
  (void)layer;
  return instance();
}
GgmlQwenSharedRunner::~GgmlQwenSharedRunner() {}
void GgmlQwenSharedRunner::shutdown() {}
bool GgmlQwenSharedRunner::ensure(int, int, int, int, bool, float) {
  return false;
}
bool GgmlQwenSharedRunner::ensure_layer(
    int, const GgmlSharedWeightRef &, const GgmlSharedWeightRef &,
    const GgmlSharedWeightRef &, const GgmlSharedWeightRef *) {
  return false;
}
bool GgmlQwenSharedRunner::run(int, const void *, const void *, const void **) {
  return false;
}
bool GgmlQwenSharedRunner::run_host_routed(int, const void *, const float *,
                                            const void **) {
  return false;
}
bool GgmlQwenSharedRunner::run_shared_only(int, const void *, const void **) {
  return false;
}
bool GgmlQwenSharedRunner::run_host_routed_add(int, const float *,
                                               const void **) {
  return false;
}
bool GgmlQwenSharedRunner::run_device_routed_add(int, const void *,
                                                  const void *, const void **) {
  return false;
}
void *GgmlQwenSharedRunner::backend_stream() const { return nullptr; }
bool GgmlQwenSharedRunner::copy_output_to_host(const void *, float *) {
  return false;
}
bool GgmlQwenSharedRunner::record_output_event() { return false; }

bool GgmlQwenDeviceCombineRunner::compiled_in() { return false; }
GgmlQwenDeviceCombineRunner &GgmlQwenDeviceCombineRunner::instance() {
  static GgmlQwenDeviceCombineRunner r;
  return r;
}
GgmlQwenDeviceCombineRunner::~GgmlQwenDeviceCombineRunner() {}
bool GgmlQwenDeviceCombineRunner::ensure(int, int, int) { return false; }
bool GgmlQwenDeviceCombineRunner::run(const float *, const float *,
                                      const void **) {
  return false;
}
void GgmlQwenDeviceCombineRunner::shutdown() {}
#endif  // MOESTREAM_USE_GGML_CUDA
