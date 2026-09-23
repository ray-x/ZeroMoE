// Qwen4Exp hyper-connections as ggml graphs. See include/ggml_hc_backend.hpp
// for the measurement that motivates this and the math it must reproduce.
#include "ggml_hc_backend.hpp"

#include "ggml_moe_backend.hpp"  // ggml_shared_backend()
#include "logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(MOESTREAM_USE_GGML_CUDA)

#include <cuda_runtime.h>

#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

extern "C" void launch_qwen4exp_hc_q8_pre(
    float *streams, const float *norm, float *xn, float *lo, float *gate,
    float *mixed, float *inject, const void *down, const void *up,
    const void *inject_w, int S, int HC, int D, int LR, float eps,
    int inject_q8, cudaStream_t stream);
extern "C" void launch_qwen4exp_hc_q8_post(
    float *streams, const float *sub_out, const float *inject, int S, int HC,
    int D, cudaStream_t stream);

namespace {
constexpr int kMaxLayers = 256;
bool async_graph_reuse_override = false;
bool qwen_hc_async_readback_override = false;

bool type_ok(uint32_t t) {
  if (t >= (uint32_t)GGML_TYPE_COUNT) return false;
  const ggml_type g = (ggml_type)t;
  return g == GGML_TYPE_F32 || g == GGML_TYPE_F16 || g == GGML_TYPE_BF16 ||
         ggml_is_quantized(g);
}

// ggml's CUDA Q8_0 dot path quantizes an F32 activation to Q8_1 before
// multiplying.  That is a useful general GEMM trade-off, but it is not
// acceptable for Qwen4Exp HC: the low-rank gate feeds the next router and a
// small projection error can change the selected expert IDs.  Dequantize the
// small HC matrices once and let the persistent graph use an F32 matmul,
// matching MoEStream's existing exact-F32-activation Q8_0 dense path.
bool dequant_q8_0_rows(const GgmlHcWeightRef &src,
                       std::vector<float> &dst) {
  if (!src.exact_f32_q8 || src.type != (uint32_t)GGML_TYPE_Q8_0 ||
      !src.host_data || src.ne0 <= 0 || src.ne1 <= 0)
    return false;
  const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, src.ne0);
  const size_t rows = (size_t)src.ne1;
  if (src.nbytes < row_bytes * rows)
    return false;
  dst.resize((size_t)src.ne0 * rows);
  const auto *q = static_cast<const block_q8_0 *>(src.host_data);
  for (size_t row = 0; row < rows; ++row) {
    dequantize_row_q8_0(q + row * (row_bytes / sizeof(block_q8_0)),
                        dst.data() + row * (size_t)src.ne0, src.ne0);
  }
  return true;
}

// All HC graphs use the same shared CUDA backend.  Its main stream is ordered,
// so resetting the allocator and reusing its temporary addresses is safe when
// the next graph is queued after the previous one.  The conservative event
// waits remain the default; this opt-in is used while validating the
// device-resident multi-layer pipeline and makes the dependency model explicit
// without changing the established fallback behaviour.
bool async_graph_reuse() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOESTREAM_GGML_ASYNC_GRAPH_REUSE");
    return v && v[0] == '1';
  }();
  return enabled || async_graph_reuse_override;
}

// Diagnostic only.  The normal path relies on the shared backend stream and
// allocator events.  A full device barrier after each HC graph distinguishes
// a missing dependency/lifetime edge from a numerical problem in the HC math.
bool debug_sync_hc() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOESTREAM_GGML_DEBUG_SYNC_HC");
    return v && v[0] == '1';
  }();
  return qwen_hc_async_readback_override || enabled;
}

bool debug_check_hc_state() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOESTREAM_GGML_DEBUG_CHECK_HC_STATE");
    return v && v[0] == '1';
  }();
  return enabled;
}

// Diagnostic only: force the S=1 HC pre graph to be rebuilt at every bracket.
// This separates a persistent-graph retarget/replay problem from a numerical
// difference in the HC projection itself.  The production path keeps graph
// reuse enabled.
bool debug_rebuild_pre_graph_each_call() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOESTREAM_GGML_HC_REBUILD_EACH_CALL");
    return v && v[0] == '1';
  }();
  return enabled;
}

bool hc_no_external_weights() {
  static const bool disabled = [] {
    const char *e = std::getenv("MOESTREAM_GGML_HC_NO_EXTERNAL");
    return e && e[0] == '1';
  }();
  return disabled;
}

bool debug_external_weight_rebind() {
  static const bool enabled = [] {
    const char *e = std::getenv("MOESTREAM_GGML_HC_DIAG_EXTERNAL");
    return e && e[0] == '1';
  }();
  return enabled;
}

uint64_t fnv64_bytes(const void *data, size_t bytes) {
  const auto *p = static_cast<const uint8_t *>(data);
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

// This is deliberately opt-in while it is being compared against the
// established direct-copy path.  The runner is currently used by Qwen's
// device-HC path; keeping the switch local means other model families retain
// their existing synchronization behavior even if they reuse this class.
bool qwen_hc_async_readback() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOESTREAM_QWEN_HC_ASYNC_READBACK");
    return v && v[0] == '1';
  }();
  return enabled;
}

template <typename T>
size_t count_nonfinite(const std::vector<T> &values) {
  size_t bad = 0;
  for (T v : values) {
    if (!std::isfinite((double)v)) ++bad;
  }
  return bad;
}
}  // namespace

void set_ggml_hc_async_graph_reuse(bool enabled) {
  async_graph_reuse_override = enabled;
}

void set_qwen_hc_async_readback(bool enabled) {
  qwen_hc_async_readback_override = enabled;
}

bool GgmlHcRunner::compiled_in() { return true; }

GgmlHcRunner &GgmlHcRunner::instance() {
  static GgmlHcRunner r;
  return r;
}

GgmlHcRunner::~GgmlHcRunner() { shutdown(); }

void GgmlHcRunner::shutdown() {
  if (device_ >= 0) cudaSetDevice(device_);
  free_pre_graph();
  for (int sb = 0; sb < 2; ++sb) {
    for (int l = 0; l < kMaxLayers; ++l) {
      Bracket &b = brackets_[sb][l];
      if (b.buf) ggml_backend_buffer_free((ggml_backend_buffer_t)b.buf);
      if (b.ctx) ggml_free((ggml_context *)b.ctx);
      b = Bracket{};
    }
  }
  if (post_ctx_) {
    ggml_free((ggml_context *)post_ctx_);
    post_ctx_ = nullptr;
  }
  post_graph_ = nullptr;
  post_input_ = nullptr;
  post_source_view_ = nullptr;
  post_graph_S_ = 0;
  if (state_buf_) {
    ggml_backend_buffer_free((ggml_backend_buffer_t)state_buf_);
    state_buf_ = nullptr;
  }
  if (state_ctx_) {
    ggml_free((ggml_context *)state_ctx_);
    state_ctx_ = nullptr;
  }
  t_streams_ = t_inject_ = t_mixed_ = nullptr;
  s_cap_ = 0;
  inject_valid_S_ = 0;
  pre_graph_workspace_bytes_ = 0;
  post_graph_workspace_bytes_ = 0;
  pre_graph_cache_S_ = 0;
  active_pre_graph_ = nullptr;
  weight_mode_valid_ = false;
  weight_mode_exact_ = false;
  weight_mode_custom_ = false;
  if (custom_xn_) cudaFree(custom_xn_);
  if (custom_lo_) cudaFree(custom_lo_);
  if (custom_gate_) cudaFree(custom_gate_);
  custom_xn_ = custom_lo_ = custom_gate_ = nullptr;
  custom_capacity_ = 0;
  if (readback_done_event_) {
    cudaEventDestroy((cudaEvent_t)readback_done_event_);
    readback_done_event_ = nullptr;
  }
  if (readback_stream_) {
    cudaStreamDestroy((cudaStream_t)readback_stream_);
    readback_stream_ = nullptr;
  }
  if (readback_host_) {
    cudaFreeHost(readback_host_);
    readback_host_ = nullptr;
  }
  readback_capacity_ = 0;
  for (void *&event : pre_done_events_) {
    if (event) {
      ggml_backend_event_free((ggml_backend_event_t)event);
      event = nullptr;
    }
  }
  active_pre_done_event_ = nullptr;
  if (post_done_event_) {
    ggml_backend_event_free((ggml_backend_event_t)post_done_event_);
    post_done_event_ = nullptr;
  }
  if (raw_input_event_) {
    ggml_backend_event_free((ggml_backend_event_t)raw_input_event_);
    raw_input_event_ = nullptr;
  }
  pre_in_flight_ = false;
  post_in_flight_ = false;
  post_fused_in_flight_ = false;
  if (galloc_pre_) {
    ggml_gallocr_free((ggml_gallocr_t)galloc_pre_);
    galloc_pre_ = nullptr;
  }
  if (galloc_post_) {
    ggml_gallocr_free((ggml_gallocr_t)galloc_post_);
    galloc_post_ = nullptr;
  }
  if (external_buf_) {
    ggml_backend_buffer_free((ggml_backend_buffer_t)external_buf_);
    external_buf_ = nullptr;
  }
  backend_ = nullptr;  // shared, not owned
  device_ = -1;
}

bool GgmlHcRunner::ensure(int device, int d_model, int hc_mult,
                          int hc_low_rank, float eps) {
  if (device < 0 || d_model <= 0 || hc_mult <= 1 || hc_low_rank <= 0)
    return false;
  if (backend_ && device_ == device && d_model_ == d_model && hc_ == hc_mult &&
      hc_lr_ == hc_low_rank)
    return true;
  if (backend_) shutdown();
  cudaSetDevice(device);
  backend_ = ggml_shared_backend(device);
  if (!backend_) return false;
  ggml_backend_buffer_type_t bt =
      ggml_backend_get_default_buffer_type((ggml_backend_t)backend_);
  galloc_pre_ = ggml_gallocr_new(bt);
  galloc_post_ = ggml_gallocr_new(bt);
  if (!galloc_pre_ || !galloc_post_) {
    if (galloc_pre_) ggml_gallocr_free((ggml_gallocr_t)galloc_pre_);
    if (galloc_post_) ggml_gallocr_free((ggml_gallocr_t)galloc_post_);
    galloc_pre_ = galloc_post_ = nullptr;
    backend_ = nullptr;
    return false;
  }
  external_buf_ = ggml_backend_alloc_buffer((ggml_backend_t)backend_, 1);
  if (!external_buf_) {
    ggml_gallocr_free((ggml_gallocr_t)galloc_pre_);
    ggml_gallocr_free((ggml_gallocr_t)galloc_post_);
    galloc_pre_ = galloc_post_ = nullptr;
    backend_ = nullptr;
    return false;
  }
  // This marker is only used to tell GGML that an externally-owned pointer is
  // a CUDA tensor.  COMPUTE usage prevents the CUDA buffer initializer from
  // touching quantized padding beyond the marker's one byte.
  ggml_backend_buffer_set_usage(
      (ggml_backend_buffer_t)external_buf_, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
  ggml_backend_dev_t dev = ggml_backend_get_device((ggml_backend_t)backend_);
  post_done_event_ = ggml_backend_event_new(dev);
  raw_input_event_ = ggml_backend_event_new(dev);
  if (!post_done_event_ || !raw_input_event_) {
    if (post_done_event_) ggml_backend_event_free((ggml_backend_event_t)post_done_event_);
    if (raw_input_event_) ggml_backend_event_free((ggml_backend_event_t)raw_input_event_);
    post_done_event_ = nullptr;
    raw_input_event_ = nullptr;
    ggml_gallocr_free((ggml_gallocr_t)galloc_pre_);
    ggml_gallocr_free((ggml_gallocr_t)galloc_post_);
    galloc_pre_ = galloc_post_ = nullptr;
    backend_ = nullptr;
    return false;
  }
  device_ = device;
  d_model_ = d_model;
  hc_ = hc_mult;
  hc_lr_ = hc_low_rank;
  hc_dim_ = hc_mult * d_model;
  eps_ = eps;
  LOG_INFO("[GGML-HC] enabled on GPU %d (D=%d HC=%d low_rank=%d hc_dim=%d)\n",
           device, d_model, hc_mult, hc_low_rank, hc_dim_);
  return true;
}

bool GgmlHcRunner::ensure_state(int S) {
  if (S <= 0) return false;
  if (t_streams_ && S <= s_cap_) return true;
  // Grow-only, with slack: prefill sets the high-water mark and decode (S=1)
  // then fits for free. Freeing and reallocating per call would put a
  // cudaMalloc on the per-layer path.
  if (state_buf_) {
    cudaSetDevice(device_);
    cudaDeviceSynchronize();
    free_pre_graph();
    if (post_ctx_) {
      ggml_free((ggml_context *)post_ctx_);
      post_ctx_ = nullptr;
      post_graph_ = nullptr;
      post_input_ = nullptr;
      post_graph_S_ = 0;
    }
    ggml_backend_buffer_free((ggml_backend_buffer_t)state_buf_);
    state_buf_ = nullptr;
  }
  if (state_ctx_) {
    ggml_free((ggml_context *)state_ctx_);
    state_ctx_ = nullptr;
  }
  const int cap = std::max(S, s_cap_ * 2);
  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 8;
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx) return false;
  ggml_tensor *st =
      ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_model_, hc_, cap);
  ggml_tensor *inj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_, cap);
  ggml_tensor *mixed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model_, cap);
  if (!st || !inj || !mixed) {
    ggml_free(ctx);
    return false;
  }
  ggml_set_name(st, "hc_streams");
  ggml_set_name(inj, "hc_inject");
  ggml_set_name(mixed, "hc_mixed");
  ggml_backend_buffer_t buf =
      ggml_backend_alloc_ctx_tensors(ctx, (ggml_backend_t)backend_);
  if (!buf) {
    LOG_ERROR("[GGML-HC] state alloc failed (S_cap=%d)\n", cap);
    ggml_free(ctx);
    return false;
  }
  state_ctx_ = ctx;
  state_buf_ = buf;
  t_streams_ = st;
  t_inject_ = inj;
  t_mixed_ = mixed;
  s_cap_ = cap;
  LOG_INFO("[GGML-HC-MEM] persistent state: %.2f MiB (S_cap=%d)\n",
           (double)ggml_backend_buffer_get_size(buf) / (1024.0 * 1024.0),
           cap);
  return true;
}

bool GgmlHcRunner::ensure_custom_buffers(int S) {
  if (S <= 0) return false;
  const size_t capacity = (size_t)std::max(S, 1);
  if (custom_xn_ && custom_lo_ && custom_gate_ && custom_capacity_ >= capacity)
    return true;

  // Decode is S=1 today, but use a grow-only allocation so an accidental
  // device-HC prefill call cannot turn this into a per-token cudaMalloc path.
  const size_t xn_bytes = capacity * (size_t)hc_dim_ * sizeof(float);
  const size_t lo_bytes = capacity * (size_t)hc_lr_ * sizeof(float);
  void *xn = nullptr;
  void *lo = nullptr;
  void *gate = nullptr;
  if (cudaMalloc(&xn, xn_bytes) != cudaSuccess ||
      cudaMalloc(&lo, lo_bytes) != cudaSuccess ||
      cudaMalloc(&gate, xn_bytes) != cudaSuccess) {
    if (xn) cudaFree(xn);
    if (lo) cudaFree(lo);
    if (gate) cudaFree(gate);
    cudaGetLastError();
    return false;
  }
  if (custom_xn_) cudaFree(custom_xn_);
  if (custom_lo_) cudaFree(custom_lo_);
  if (custom_gate_) cudaFree(custom_gate_);
  custom_xn_ = xn;
  custom_lo_ = lo;
  custom_gate_ = gate;
  custom_capacity_ = capacity;
  LOG_INFO("[GGML-HC-MEM] custom exact-Q8 workspace: %.2f MiB (S_cap=%d)\n",
           (double)(xn_bytes + lo_bytes + xn_bytes) / (1024.0 * 1024.0),
           S);
  return true;
}

bool GgmlHcRunner::ensure_readback_buffer(size_t bytes) {
  if (bytes == 0) return false;
  if (!readback_stream_) {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
        cudaSuccess)
      return false;
    readback_stream_ = stream;
  }
  if (!readback_done_event_) {
    cudaEvent_t event = nullptr;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) !=
        cudaSuccess)
      return false;
    readback_done_event_ = event;
  }
  if (readback_host_ && readback_capacity_ >= bytes) return true;

  void *host = nullptr;
  if (cudaMallocHost(&host, bytes) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  if (readback_host_) cudaFreeHost(readback_host_);
  readback_host_ = host;
  readback_capacity_ = bytes;
  LOG_INFO("[GGML-HC-MEM] pinned pre_mix readback: %.2f KiB\n",
           (double)bytes / 1024.0);
  return true;
}

void GgmlHcRunner::free_pre_graph() {
  for (int sb = 0; sb < 2; ++sb) {
    for (int l = 0; l < kMaxLayers; ++l) {
      PreGraphSlot &slot = pre_graphs_[sb][l];
      if (slot.ctx) ggml_free((ggml_context *)slot.ctx);
      slot = PreGraphSlot{};
    }
  }
  pre_graph_cache_S_ = 0;
  active_pre_graph_ = nullptr;
}

void GgmlHcRunner::release_completed_weights(int before_layer) {
  if (!backend_ || before_layer <= 0) return;

  // Cached pre graphs keep fixed tensor metadata and data pointers for every
  // bracket. Do not reclaim a bracket whose graph is still cached.
  ggml_backend_synchronize((ggml_backend_t)backend_);
  size_t reclaimed = 0;
  int brackets = 0;
  for (int sb = 0; sb < 2; ++sb) {
    for (int l = 0; l < before_layer; ++l) {
      if (pre_graphs_[sb][l].ctx) continue;
      Bracket &b = brackets_[sb][l];
      if (!b.buf && !b.ctx) continue;
      if (b.buf) {
        reclaimed += ggml_backend_buffer_get_size(
            (ggml_backend_buffer_t)b.buf);
        ggml_backend_buffer_free((ggml_backend_buffer_t)b.buf);
      }
      if (b.ctx) ggml_free((ggml_context *)b.ctx);
      // Keep the host-side exact dequantization cached.  Rebuilding a decode
      // bracket should only allocate/upload its device tensors; repeating the
      // Q8->F32 conversion for every token would move the bottleneck to the
      // CPU as soon as the rolling reclaim policy engages.
      std::vector<float> down_f32 = std::move(b.down_f32);
      std::vector<float> up_f32 = std::move(b.up_f32);
      const bool exact_mode = b.exact_f32_q8;
      const bool custom_mode = b.custom_q8;
      b = Bracket{};
      b.down_f32 = std::move(down_f32);
      b.up_f32 = std::move(up_f32);
      b.exact_f32_q8 = exact_mode;
      b.custom_q8 = custom_mode;
      ++brackets;
    }
  }
  if (brackets > 0) {
    LOG_WARN("[GGML-HC-MEM] allocation-pressure reclaim: released %d "
             "completed-layer brackets (%.2f MiB); cached graph brackets "
             "were retained\n",
             brackets, (double)reclaimed / (1024.0 * 1024.0));
  }
}

void GgmlHcRunner::release_completed_weights_nonblocking(int before_layer) {
  if (!backend_ || before_layer <= 0) return;

  size_t reclaimed = 0;
  int brackets = 0;
  for (int sb = 0; sb < 2; ++sb) {
    for (int l = 0; l < before_layer && l < kMaxLayers; ++l) {
      if (pre_graphs_[sb][l].ctx) continue;
      Bracket &b = brackets_[sb][l];
      if (!b.buf && !b.ctx) continue;

      void *event_ptr = pre_done_events_[sb * 256 + l];
      if (!event_ptr || !((ggml_backend_event_t)event_ptr)->context)
        continue;
      const cudaError_t status = cudaEventQuery(
          (cudaEvent_t)((ggml_backend_event_t)event_ptr)->context);
      if (status == cudaErrorNotReady) {
        // cudaEventQuery leaves the expected not-ready status in the CUDA
        // error slot; clear it so a later unrelated CUDA call is not blamed.
        cudaGetLastError();
        continue;
      }
      if (status != cudaSuccess) {
        cudaGetLastError();
        continue;
      }

      reclaimed += b.buf
                       ? ggml_backend_buffer_get_size(
                             (ggml_backend_buffer_t)b.buf)
                       : 0;
      if (b.buf) {
        ggml_backend_buffer_free((ggml_backend_buffer_t)b.buf);
        b.buf = nullptr;
      }
      if (b.ctx) {
        ggml_free((ggml_context *)b.ctx);
        b.ctx = nullptr;
      }
      std::vector<float> down_f32 = std::move(b.down_f32);
      std::vector<float> up_f32 = std::move(b.up_f32);
      const bool exact_mode = b.exact_f32_q8;
      const bool custom_mode = b.custom_q8;
      b = Bracket{};
      b.down_f32 = std::move(down_f32);
      b.up_f32 = std::move(up_f32);
      b.exact_f32_q8 = exact_mode;
      b.custom_q8 = custom_mode;
      ++brackets;
    }
  }
  if (brackets > 0) {
    LOG_STREAM_INFO()
        << "[GGML-HC-MEM] event reclaim: released " << brackets
        << " completed brackets ("
        << (double)reclaimed / (1024.0 * 1024.0)
        << " MiB) before layer " << before_layer << "\n";
  }
}

bool GgmlHcRunner::ensure_pre_graph(int S, const Bracket &b, int layer,
                                    int sub_block) {
  if (!backend_ || !b.ready || !b.norm || !b.down || !b.up || !b.inject ||
      S <= 0 || S > s_cap_)
    return false;

  const ggml_tensor *bd = (const ggml_tensor *)b.down;
  const ggml_tensor *bu = (const ggml_tensor *)b.up;
  const ggml_tensor *bi = (const ggml_tensor *)b.inject;
  if (debug_rebuild_pre_graph_each_call() ||
      (pre_graph_cache_S_ != 0 && pre_graph_cache_S_ != S)) {
    // The allocator workspace is shared by the cached graphs. Drain the
    // backend before replacing a cache for a different shape, or when the
    // diagnostic explicitly requests a rebuild.
    if (pre_graph_cache_S_ != 0 || active_pre_graph_) {
      ggml_backend_synchronize((ggml_backend_t)backend_);
      free_pre_graph();
    }
  }
  if (pre_graph_cache_S_ == 0) pre_graph_cache_S_ = S;

  PreGraphSlot &slot = pre_graphs_[sub_block][layer];
  const bool matches =
      slot.ctx && slot.graph && slot.S == S &&
      slot.down_type == (uint32_t)bd->type &&
      slot.up_type == (uint32_t)bu->type &&
      slot.inject_type == (uint32_t)bi->type;

  if (!matches) {
    if (slot.ctx) {
      // A bracket's source tensors are immutable for the lifetime of its
      // cached graph. A type/shape change therefore invalidates the complete
      // cache rather than retargeting a captured graph.
      ggml_backend_synchronize((ggml_backend_t)backend_);
      free_pre_graph();
      pre_graph_cache_S_ = S;
    }

    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead() * 96 + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context *ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_cgraph *gf = ggml_new_graph(ctx);
    if (!gf) {
      ggml_free(ctx);
      return false;
    }

    // Keep this bracket's tensor metadata as fixed graph sources. The CUDA
    // graph must never be asked to follow a later data-pointer retarget.
    ggml_tensor *streams = (ggml_tensor *)t_streams_;
    ggml_tensor *x = ggml_view_3d(ctx, streams, d_model_, hc_, S,
                                  streams->nb[1], streams->nb[2], 0);
    ggml_tensor *xn = ggml_rms_norm(ctx, x, eps_);
    ggml_tensor *xn2 = ggml_reshape_2d(ctx, xn, hc_dim_, S);
    xn2 = ggml_mul(ctx, xn2, (ggml_tensor *)b.norm);
    ggml_tensor *lo = ggml_mul_mat(ctx, (ggml_tensor *)b.down, xn2);
    lo = ggml_silu(ctx, ggml_scale(ctx, lo, 1.0f / (float)hc_));
    ggml_tensor *gate =
        ggml_sigmoid(ctx, ggml_mul_mat(ctx, (ggml_tensor *)b.up, lo));
    ggml_tensor *gated = ggml_mul(ctx, xn2, gate);
    ggml_tensor *g3 = ggml_reshape_3d(ctx, gated, d_model_, hc_, S);
    ggml_tensor *mixed = nullptr;
    for (int h = 0; h < hc_; ++h) {
      ggml_tensor *v = ggml_view_2d(ctx, g3, d_model_, S, g3->nb[2],
                                    (size_t)h * g3->nb[1]);
      mixed = (h == 0) ? v : ggml_add(ctx, mixed, v);
    }
    if (hc_ == 1) mixed = ggml_cont(ctx, mixed);
    mixed = ggml_scale(ctx, mixed, 1.0f / (float)hc_);
    ggml_tensor *mixed_view = ggml_view_2d(
        ctx, (ggml_tensor *)t_mixed_, d_model_, S,
        ((ggml_tensor *)t_mixed_)->nb[1], 0);
    ggml_tensor *mixed_store = ggml_cpy(ctx, mixed, mixed_view);

    ggml_tensor *inj_view =
        ggml_view_2d(ctx, (ggml_tensor *)t_inject_, hc_, S,
                     ((ggml_tensor *)t_inject_)->nb[1], 0);
    ggml_tensor *inj = ggml_mul_mat(ctx, (ggml_tensor *)b.inject, xn2);
    ggml_tensor *inj_store = ggml_cpy(ctx, inj, inj_view);
    if (!x || !xn || !xn2 || !lo || !gate || !gated || !g3 || !mixed ||
        !mixed_store || !inj_view || !inj || !inj_store) {
      ggml_free(ctx);
      return false;
    }
    ggml_build_forward_expand(gf, mixed_store);
    ggml_build_forward_expand(gf, inj_store);
    if (!ggml_gallocr_alloc_graph((ggml_gallocr_t)galloc_pre_, gf)) {
      LOG_ERROR("[GGML-HC] persistent pre_mix graph alloc failed (S=%d)\n",
                S);
      ggml_free(ctx);
      return false;
    }
    const size_t graph_bytes =
        ggml_gallocr_get_buffer_size((ggml_gallocr_t)galloc_pre_, 0);
    if (graph_bytes != pre_graph_workspace_bytes_) {
      pre_graph_workspace_bytes_ = graph_bytes;
      LOG_INFO("[GGML-HC-MEM] pre_mix persistent workspace: %.2f MiB (S=%d)\n",
               (double)graph_bytes / (1024.0 * 1024.0), S);
    }
    slot.ctx = ctx;
    slot.graph = gf;
    slot.norm = b.norm;
    slot.down = b.down;
    slot.up = b.up;
    slot.inject = b.inject;
    slot.S = S;
    slot.down_type = (uint32_t)bd->type;
    slot.up_type = (uint32_t)bu->type;
    slot.inject_type = (uint32_t)bi->type;
  }

  active_pre_graph_ = &slot;
  return true;
}

bool GgmlHcRunner::ensure_post_graph(int S) {
  if (!backend_ || !t_streams_ || !t_inject_ || S <= 0 ||
      S > s_cap_)
    return false;
  if (post_graph_ && post_graph_S_ == S)
    return true;

  // A shape change happens at a request boundary in normal use (prefill
  // shape changes, decode remains S=1).  Drain all CUDA work before releasing
  // the old graph/context and allowing GGML's allocator to reuse its storage.
  // This is once per shape, never once per layer.
  if (post_ctx_) {
    cudaSetDevice(device_);
    cudaDeviceSynchronize();
    ggml_free((ggml_context *)post_ctx_);
    post_ctx_ = nullptr;
    post_graph_ = nullptr;
    post_input_ = nullptr;
    post_source_view_ = nullptr;
    post_graph_S_ = 0;
  }

  ggml_init_params ip = {};
  // Keep metadata headroom for the persistent producer view below and the
  // graph's small shape bookkeeping. The old 64-tensor pool was also being
  // consumed by a fresh decode view on every call, eventually turning this
  // harmless shape difference into GGML_ASSERT(obj_new).
  ip.mem_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead();
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx) return false;
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_tensor *so_flat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model_, S);
  if (!gf || !so_flat) {
    ggml_free(ctx);
    return false;
  }
  ggml_set_input(so_flat);
  ggml_set_name(so_flat, "hc_sub_block_out");

  ggml_tensor *streams = (ggml_tensor *)t_streams_;
  // A full-GDN producer keeps a grow-only [D,S_cap] output tensor. Decode
  // consumes only its first row, so creating a fresh view in post_combine_impl
  // would leak one GGML metadata object per token until the context exhausted.
  // Create one [D,S] view here and retarget its data/buffer for each producer.
  ggml_tensor *source_view =
      ggml_view_2d(ctx, streams, d_model_, S, streams->nb[1], 0);
  ggml_tensor *sv = ggml_view_3d(ctx, streams, d_model_, hc_, S,
                                 streams->nb[1], streams->nb[2], 0);
  ggml_tensor *inj = ggml_view_2d(ctx, (ggml_tensor *)t_inject_, hc_, S,
                                  ((ggml_tensor *)t_inject_)->nb[1], 0);
  ggml_tensor *w =
      ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inj, 1.0f / (float)hc_)),
                 2.0f);
  if (!source_view || !sv || !inj || !w) {
    ggml_free(ctx);
    return false;
  }

  for (int h = 0; h < hc_; ++h) {
    ggml_tensor *vh = ggml_view_2d(
        ctx, streams, d_model_, S, streams->nb[2],
        (size_t)h * streams->nb[1]);
    ggml_tensor *wh = ggml_view_2d(ctx, w, 1, S, w->nb[1],
                                   (size_t)h * sizeof(float));
    ggml_tensor *delta_h = ggml_mul(ctx, so_flat, wh);
    ggml_tensor *updated_h = ggml_add(ctx, vh, delta_h);
    ggml_tensor *store_h = ggml_cpy(ctx, updated_h, vh);
    if (!vh || !wh || !delta_h || !updated_h || !store_h) {
      ggml_free(ctx);
      return false;
    }
    ggml_build_forward_expand(gf, store_h);
  }

  if (!ggml_gallocr_alloc_graph((ggml_gallocr_t)galloc_post_, gf)) {
    LOG_ERROR("[GGML-HC] persistent post_combine graph alloc failed (S=%d)\n",
              S);
    ggml_free(ctx);
    return false;
  }
  const size_t graph_bytes =
      ggml_gallocr_get_buffer_size((ggml_gallocr_t)galloc_post_, 0);
  if (graph_bytes != post_graph_workspace_bytes_) {
    post_graph_workspace_bytes_ = graph_bytes;
    LOG_INFO("[GGML-HC-MEM] post_combine persistent workspace: %.2f MiB (S=%d)\n",
             (double)graph_bytes / (1024.0 * 1024.0), S);
  }
  post_ctx_ = ctx;
  post_graph_ = gf;
  post_input_ = so_flat;
  post_source_view_ = source_view;
  post_graph_S_ = S;
  return true;
}

bool GgmlHcRunner::weights_ready(int layer, int sub_block) const {
  if (layer < 0 || layer >= kMaxLayers || sub_block < 0 || sub_block > 1)
    return false;
  return brackets_[sub_block][layer].ready;
}

bool GgmlHcRunner::ensure_weights(int layer, int sub_block,
                                  const GgmlHcWeightRef &norm,
                                  const GgmlHcWeightRef &down,
                                  const GgmlHcWeightRef &up,
                                  const GgmlHcWeightRef &inject) {
  if (!backend_ || layer < 0 || layer >= kMaxLayers || sub_block < 0 ||
      sub_block > 1)
    return false;
  Bracket &b = brackets_[sub_block][layer];
  if (!norm.host_data || !down.host_data || !up.host_data || !inject.host_data)
    return false;
  if (!type_ok(norm.type) || !type_ok(down.type) || !type_ok(up.type) ||
      !type_ok(inject.type))
    return false;
  // Shapes are checked, not assumed: a mismatch here would otherwise be read
  // as valid weights of the wrong layout and produce plausible-but-wrong
  // numbers rather than a clean fallback.
  if (norm.ne0 != hc_dim_ || down.ne0 != hc_dim_ || down.ne1 != hc_lr_ ||
      up.ne0 != hc_lr_ || up.ne1 != hc_dim_ || inject.ne0 != hc_dim_ ||
      inject.ne1 != hc_) {
    LOG_ERROR("[GGML-HC] layer=%d sub=%d unexpected HC weight shapes "
              "(norm %lld, down %lldx%lld, up %lldx%lld, inject %lldx%lld)\n",
              layer, sub_block, (long long)norm.ne0, (long long)down.ne0,
              (long long)down.ne1, (long long)up.ne0, (long long)up.ne1,
              (long long)inject.ne0, (long long)inject.ne1);
    return false;
  }
  const bool custom_mode = down.custom_q8 && up.custom_q8 && inject.custom_q8 &&
                           down.type == (uint32_t)GGML_TYPE_Q8_0 &&
                           up.type == (uint32_t)GGML_TYPE_Q8_0 &&
                           (inject.type == (uint32_t)GGML_TYPE_Q8_0 ||
                            inject.type == (uint32_t)GGML_TYPE_F32);
  const bool exact_mode = down.exact_f32_q8 || up.exact_f32_q8;
  if (weight_mode_valid_ &&
      (weight_mode_exact_ != exact_mode || weight_mode_custom_ != custom_mode)) {
    // All brackets from the previous mode may still be referenced by the
    // persistent graph or by queued CUDA work. One drain makes the whole
    // mode transition safe; later brackets can be replaced without another
    // blocking synchronization.
    ggml_backend_synchronize((ggml_backend_t)backend_);
    free_pre_graph();
  }
  weight_mode_valid_ = true;
  weight_mode_exact_ = exact_mode;
  weight_mode_custom_ = custom_mode;
  if (b.ready && b.exact_f32_q8 == exact_mode && b.custom_q8 == custom_mode) {
    // The dense pool is normally stable, but its owner can rebuild or move a
    // borrowed tensor between request boundaries.  Keep the HC graph's
    // source tensors in sync with the current resident-layer references just
    // like the shared/MoE runners do before each submission.  This is safe for
    // the cached per-bracket graph because these are the same tensor objects;
    // no graph shape or type changes are introduced here.
    auto rebind_if_external = [&](void *opaque, const GgmlHcWeightRef &ref,
                                  const char *name) {
      ggml_tensor *t = (ggml_tensor *)opaque;
      if (!t || t->buffer != (ggml_backend_buffer_t)external_buf_ ||
          !ref.device_data || ref.device_bytes < ggml_nbytes(t))
        return;
      const void *old = t->data;
      t->data = const_cast<void *>(ref.device_data);
      t->buffer = (ggml_backend_buffer_t)external_buf_;
      if (ggml_backend_buffer_init_tensor(t->buffer, t) != GGML_STATUS_SUCCESS)
        return;
      if (debug_external_weight_rebind() && old != t->data) {
        LOG_INFO("[GGML-HC-EXT] rebind layer=%d sub=%d %s old=%p new=%p\n",
                 layer, sub_block, name, old, t->data);
      }
    };
    if (!hc_no_external_weights()) {
      rebind_if_external(b.norm, norm, "norm");
      rebind_if_external(b.down, down, "down");
      rebind_if_external(b.up, up, "up");
      rebind_if_external(b.inject, inject, "inject");
    }
    return true;
  }
  if (b.ready) {
    // The only normal mode transition is the request-boundary switch from
    // prefill's quantized graph to decode's exact Q8 graph.  The forward-level
    // HC collapse has already drained the backend before the next decode
    // request reaches this point, so replacing this bracket is safe.
    if (b.buf) {
      ggml_backend_buffer_free((ggml_backend_buffer_t)b.buf);
      b.buf = nullptr;
    }
    if (b.ctx) {
      ggml_free((ggml_context *)b.ctx);
      b.ctx = nullptr;
    }
    b.norm = b.down = b.up = b.inject = nullptr;
    b.ready = false;
    if (!exact_mode) {
      b.down_f32.clear();
      b.up_f32.clear();
    }
  }
  const bool exact_down_q8 = !custom_mode && down.exact_f32_q8 &&
                             down.type == (uint32_t)GGML_TYPE_Q8_0;
  const bool exact_up_q8 = !custom_mode && up.exact_f32_q8 &&
                           up.type == (uint32_t)GGML_TYPE_Q8_0;
  const size_t down_f32_count = (size_t)down.ne0 * (size_t)down.ne1;
  const size_t up_f32_count = (size_t)up.ne0 * (size_t)up.ne1;
  if (exact_down_q8 &&
      (b.down_f32.size() != down_f32_count) &&
      !dequant_q8_0_rows(down, b.down_f32)) {
    LOG_ERROR("[GGML-HC] layer=%d sub=%d invalid Q8_0 down weight\n", layer,
              sub_block);
    return false;
  }
  if (exact_up_q8 &&
      (b.up_f32.size() != up_f32_count) &&
      !dequant_q8_0_rows(up, b.up_f32)) {
    LOG_ERROR("[GGML-HC] layer=%d sub=%d invalid Q8_0 up weight\n", layer,
              sub_block);
    return false;
  }
  if (exact_mode && !custom_mode) {
    // Exact Q8 HC weights are larger than their on-disk quantized form. Keep
    // completed brackets resident when there is room; otherwise the normal
    // decode path would repeatedly free/re-upload the same HC matrices. The
    // allocation-failure path below still performs a definitive reclaim.
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t mem_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    constexpr size_t kHcReclaimMargin = (size_t)768 * 1024 * 1024;
    if (mem_status == cudaSuccess && free_bytes < kHcReclaimMargin) {
      release_completed_weights_nonblocking(layer);
    } else if (mem_status != cudaSuccess) {
      cudaGetLastError();
    }
  }
  const ggml_type down_ggml_type = exact_down_q8 ? GGML_TYPE_F32
                                                  : (ggml_type)down.type;
  const ggml_type up_ggml_type = exact_up_q8 ? GGML_TYPE_F32
                                              : (ggml_type)up.type;
  cudaSetDevice(device_);
  const bool no_external_weights = hc_no_external_weights();
  ggml_context *ctx = nullptr;
  ggml_tensor *tn = nullptr;
  ggml_tensor *td = nullptr;
  ggml_tensor *tu = nullptr;
  ggml_tensor *ti = nullptr;
  ggml_backend_buffer_t buf = nullptr;
  bool ext_n = false, ext_d = false, ext_u = false, ext_i = false;
  for (int attempt = 0; attempt < 2 && !buf; ++attempt) {
    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead() * 8;
    ip.no_alloc = true;
    ctx = ggml_init(ip);
    if (!ctx) return false;
    tn = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim_);
    td = ggml_new_tensor_2d(ctx, down_ggml_type, hc_dim_, hc_lr_);
    tu = ggml_new_tensor_2d(ctx, up_ggml_type, hc_lr_, hc_dim_);
    ti = ggml_new_tensor_2d(ctx, (ggml_type)inject.type, hc_dim_, hc_);
    if (!tn || !td || !tu || !ti) {
      ggml_free(ctx);
      return false;
    }
    ext_n = !no_external_weights && external_buf_ && norm.device_data &&
            norm.device_bytes >= ggml_nbytes(tn);
    ext_d = (custom_mode || !exact_down_q8) && !no_external_weights && external_buf_ &&
            down.device_data &&
            down.device_bytes >= ggml_nbytes(td);
    ext_u = (custom_mode || !exact_up_q8) && !no_external_weights && external_buf_ &&
            up.device_data &&
            up.device_bytes >= ggml_nbytes(tu);
    ext_i = !no_external_weights && external_buf_ && inject.device_data &&
            inject.device_bytes >= ggml_nbytes(ti);
    auto bind_external = [&](ggml_tensor *t, const void *data) {
      t->data = const_cast<void *>(data);
      t->buffer = (ggml_backend_buffer_t)external_buf_;
      ggml_backend_buffer_init_tensor(t->buffer, t);
    };
    if (ext_n) bind_external(tn, norm.device_data);
    if (ext_d) bind_external(td, down.device_data);
    if (ext_u) bind_external(tu, up.device_data);
    if (ext_i) bind_external(ti, inject.device_data);
    const bool all_external = ext_n && ext_d && ext_u && ext_i;
    if (all_external) {
      // No private weight allocation is needed.  `ctx` remains alive because
      // its tensor metadata is still used by the persistent pre graph.
      if (debug_external_weight_rebind() && layer == 0 && sub_block == 1) {
        auto log_external_hash = [&](const char *name, const GgmlHcWeightRef &ref,
                                     ggml_tensor *t) {
          const size_t bytes = ggml_nbytes(t);
          std::vector<uint8_t> copy(bytes);
          const cudaError_t e = cudaMemcpy(copy.data(), t->data, bytes,
                                           cudaMemcpyDeviceToHost);
          if (e != cudaSuccess) {
            LOG_INFO("[GGML-HC-EXT] %s copy failed: %s\n", name,
                     cudaGetErrorString(e));
            cudaGetLastError();
            return;
          }
          LOG_INFO("[GGML-HC-EXT] layer=0 sub=1 %s ptr=%p bytes=%zu "
                   "host_fnv=0x%016llx device_fnv=0x%016llx\n",
                   name, t->data, bytes,
                   (unsigned long long)fnv64_bytes(ref.host_data,
                                                   std::min(ref.nbytes, bytes)),
                   (unsigned long long)fnv64_bytes(copy.data(), bytes));
        };
        log_external_hash("norm", norm, tn);
        log_external_hash("down", down, td);
        log_external_hash("up", up, tu);
        log_external_hash("inject", inject, ti);
      }
      buf = nullptr;
      break;
    }
    buf = ggml_backend_alloc_ctx_tensors(ctx, (ggml_backend_t)backend_);
    if (buf) break;

    ggml_free(ctx);
    ctx = nullptr;
    if (attempt == 0) {
      LOG_WARN("[GGML-HC-MEM] layer=%d sub=%d allocation pressure; "
               "reclaiming completed HC layers and retrying\n",
               layer, sub_block);
      release_completed_weights(layer);
    }
  }
  const bool uses_external = ext_n && ext_d && ext_u && ext_i;
  if (!buf && !uses_external) {
    LOG_ERROR("[GGML-HC] weight alloc failed for layer=%d sub=%d\n", layer,
              sub_block);
    return false;
  }
  // norm is read by the engine as F32 regardless of its on-disk type (see
  // find_resident_float_anytype), so it is uploaded as F32 here too.
  if (!ext_n)
    ggml_backend_tensor_set(tn, norm.host_data, 0,
                            (size_t)hc_dim_ * sizeof(float));
  if (!ext_d)
    ggml_backend_tensor_set(
        td, exact_down_q8 ? static_cast<const void *>(b.down_f32.data())
                          : down.host_data,
        0, exact_down_q8 ? b.down_f32.size() * sizeof(float)
                         : std::min(down.nbytes, ggml_nbytes(td)));
  if (!ext_u)
    ggml_backend_tensor_set(
        tu, exact_up_q8 ? static_cast<const void *>(b.up_f32.data())
                        : up.host_data,
        0, exact_up_q8 ? b.up_f32.size() * sizeof(float)
                       : std::min(up.nbytes, ggml_nbytes(tu)));
  if (!ext_i)
    ggml_backend_tensor_set(ti, inject.host_data, 0,
                            std::min(inject.nbytes, ggml_nbytes(ti)));
  b.ctx = ctx;
  b.buf = buf;
  b.norm = tn;
  b.down = td;
  b.up = tu;
  b.inject = ti;
  b.exact_f32_q8 = exact_mode;
  b.custom_q8 = custom_mode;
  b.ready = true;
  const size_t weight_bytes = ggml_nbytes(tn) + ggml_nbytes(td) +
                              ggml_nbytes(tu) + ggml_nbytes(ti);
  static const bool log_types = [] {
    const char *e = std::getenv("MOESTREAM_GGML_HC_DEBUG_TYPES");
    return e && e[0] == '1';
  }();
  if (log_types) {
    LOG_INFO("[GGML-HC-MEM] types: layer=%d sub=%d norm=%u down=%u up=%u "
             "inject=%u external=%s\n",
             layer, sub_block, norm.type, (unsigned)down_ggml_type,
             (unsigned)up_ggml_type, inject.type,
             uses_external ? "yes" : "no");
  }
  LOG_INFO("[GGML-HC-MEM] weights: layer=%d sub=%d %.2f MiB (%s)\n", layer,
           sub_block, (double)weight_bytes / (1024.0 * 1024.0),
           uses_external ? "borrowed dense pool" : "private buffer");
  return true;
}

bool GgmlHcRunner::upload_streams(const float *host, int S) {
  if (!backend_ || !host || !ensure_state(S)) return false;
  cudaSetDevice(device_);
  ggml_backend_tensor_set((ggml_tensor *)t_streams_, host, 0,
                          (size_t)S * hc_ * d_model_ * sizeof(float));
  return true;
}

bool GgmlHcRunner::download_streams(float *host, int S) {
  if (!backend_ || !host || !t_streams_ || S > s_cap_) return false;
  cudaSetDevice(device_);
  if (post_fused_in_flight_ && post_done_event_ &&
      ((ggml_backend_event_t)post_done_event_)->context)
    cudaEventSynchronize(
        (cudaEvent_t)((ggml_backend_event_t)post_done_event_)->context);
  ggml_backend_synchronize((ggml_backend_t)backend_);
  ggml_backend_tensor_get((ggml_tensor *)t_streams_, host, 0,
                          (size_t)S * hc_ * d_model_ * sizeof(float));
  return true;
}

bool GgmlHcRunner::download_inject(float *host, int S) {
  if (!backend_ || !host || !t_inject_ || S <= 0 || S > s_cap_ ||
      inject_valid_S_ != S)
    return false;
  cudaSetDevice(device_);
  ggml_backend_synchronize((ggml_backend_t)backend_);
  // t_inject_ is [HC,S], whose contiguous ggml order is the caller's
  // row-major [S,HC] layout: h + s*HC.
  ggml_backend_tensor_get((ggml_tensor *)t_inject_, host, 0,
                          (size_t)S * hc_ * sizeof(float));
  return true;
}

bool GgmlHcRunner::pre_mix(int layer, int sub_block, int S, float *out_hidden,
                            const void **device_output) {
  if (!backend_ || (!out_hidden && !device_output) ||
      !weights_ready(layer, sub_block))
    return false;
  if (!ensure_state(S)) return false;
  if (layer < 0 || layer >= 256 || sub_block < 0 || sub_block > 1)
    return false;
  // Build the matching post graph before the first attention/MoE graph of
  // this request.  This keeps graph allocation out of the producer->HC
  // handoff and lets all later layers use a pure queued device dependency.
  if (!ensure_post_graph(S)) return false;
  const Bracket &b = brackets_[sub_block][layer];
  cudaSetDevice(device_);
  if (debug_external_weight_rebind() && layer == 0 && sub_block == 1 &&
      S == 1 && b.norm && b.down && b.up && b.inject) {
    auto log_replay_hash = [&](const char *name, void *opaque) {
      ggml_tensor *t = (ggml_tensor *)opaque;
      if (!t || t->buffer != (ggml_backend_buffer_t)external_buf_ ||
          !t->data)
        return;
      const size_t bytes = ggml_nbytes(t);
      std::vector<uint8_t> copy(bytes);
      const cudaError_t sync = cudaDeviceSynchronize();
      const cudaError_t e = sync == cudaSuccess
                                ? cudaMemcpy(copy.data(), t->data, bytes,
                                             cudaMemcpyDeviceToHost)
                                : sync;
      if (e == cudaSuccess) {
        LOG_INFO("[GGML-HC-EXT] replay layer=0 sub=1 %s ptr=%p bytes=%zu "
                 "fnv=0x%016llx nb1=%zu alloc=%zu\n",
                 name, t->data, bytes,
                 (unsigned long long)fnv64_bytes(copy.data(), bytes),
                 t->nb[1], ggml_backend_buft_get_alloc_size(t->buffer->buft, t));
      } else {
        LOG_INFO("[GGML-HC-EXT] replay layer=0 sub=1 %s copy failed: %s\n",
                 name, cudaGetErrorString(e));
        cudaGetLastError();
      }
    };
    log_replay_hash("norm", b.norm);
    log_replay_hash("down", b.down);
    log_replay_hash("up", b.up);
    log_replay_hash("inject", b.inject);
  }
  if (S == 1) {
    // In Qwen device-HC decode, GgmlMoeRunner's direct-input optimization
    // aliases its reusable graph input to t_mixed_.  The previous layer's
    // MoE graph may still be reading that tensor when this pre-mix graph is
    // about to overwrite it.  Publish the dependency on the shared backend
    // stream instead of synchronizing the host at every layer.
    GgmlMoeRunner::queue_input_reuse_dependency();
  }
  // A Qwen CPU-routed layer may have published the previous HC update from
  // CUDA stream 0 (the fused post-combine path below). Import that fence into
  // the shared GGML stream before a normal pre graph reads t_streams_.
  if (post_fused_in_flight_ && post_done_event_ &&
      ((ggml_backend_event_t)post_done_event_)->context)
    ggml_backend_event_wait((ggml_backend_t)backend_,
                            (ggml_backend_event_t)post_done_event_);
  if (b.custom_q8) {
    if (!ensure_custom_buffers(S)) return false;
    // The previous post graph updates t_streams_ on GGML's nonblocking
    // backend stream. Import that completion before the custom kernels read
    // the persistent state on CUDA's legacy stream.
    if (post_in_flight_ && post_done_event_ &&
        ((ggml_backend_event_t)post_done_event_)->context) {
      cudaStreamWaitEvent(
          0, (cudaEvent_t)((ggml_backend_event_t)post_done_event_)->context,
          0);
    }
    const auto *streams = (const ggml_tensor *)t_streams_;
    const auto *norm = (const ggml_tensor *)b.norm;
    const auto *down = (const ggml_tensor *)b.down;
    const auto *up = (const ggml_tensor *)b.up;
    const auto *inject = (const ggml_tensor *)b.inject;
    launch_qwen4exp_hc_q8_pre(
        static_cast<float *>(streams->data),
        static_cast<const float *>(norm->data),
        static_cast<float *>(custom_xn_), static_cast<float *>(custom_lo_),
        static_cast<float *>(custom_gate_),
        static_cast<float *>(((ggml_tensor *)t_mixed_)->data),
        static_cast<float *>(((ggml_tensor *)t_inject_)->data), down->data,
        up->data,
        inject->data, S, hc_, d_model_, hc_lr_, eps_,
        inject->type == GGML_TYPE_Q8_0 ? 1 : 0, 0);
  } else {
    // The persistent graph is allocated once per request shape. Its stable
    // source metadata has just been rebound to this layer's HC weights, so
    // submitting it does not reset/reuse a temporary allocator while the
    // prior layer is still queued.
    if (!ensure_pre_graph(S, b, layer, sub_block)) return false;
    if (!active_pre_graph_ || !active_pre_graph_->graph) {
      LOG_ERROR("[GGML-HC] active pre_mix graph missing (layer=%d sub=%d S=%d)\n",
                layer, sub_block, S);
      return false;
    }
    ggml_cgraph *gf = (ggml_cgraph *)active_pre_graph_->graph;
    const ggml_status st =
        ggml_backend_graph_compute_async((ggml_backend_t)backend_, gf);
    if (st != GGML_STATUS_SUCCESS) {
      LOG_ERROR("[GGML-HC] pre_mix compute failed (status=%d)\n", (int)st);
      return false;
    }
  }
  const int event_index = sub_block * 256 + layer;
  if (!pre_done_events_[event_index]) {
    pre_done_events_[event_index] = ggml_backend_event_new(
        ggml_backend_get_device((ggml_backend_t)backend_));
    if (!pre_done_events_[event_index]) {
      LOG_ERROR("[GGML-HC] pre_mix completion event allocation failed "
                "(layer=%d sub=%d)\n",
                layer, sub_block);
      return false;
    }
  }
  active_pre_done_event_ = pre_done_events_[event_index];
  if (b.custom_q8) {
    cudaEventRecord(
        (cudaEvent_t)((ggml_backend_event_t)active_pre_done_event_)->context,
        0);
  } else {
    ggml_backend_event_record((ggml_backend_event_t)active_pre_done_event_,
                              (ggml_backend_t)backend_);
  }
  pre_in_flight_ = true;
  if (debug_sync_hc())
    ggml_backend_synchronize((ggml_backend_t)backend_);
  if (debug_check_hc_state()) {
    cudaDeviceSynchronize();
    static thread_local std::vector<float> check;
    const size_t n = (size_t)S * d_model_;
    check.resize(n);
    const cudaError_t e = cudaMemcpy(
        check.data(), ((ggml_tensor *)t_mixed_)->data,
        n * sizeof(float), cudaMemcpyDeviceToHost);
    double max_abs = 0.0;
    if (e == cudaSuccess) {
      for (float v : check) {
        if (std::isfinite(v)) max_abs = std::max(max_abs, std::abs((double)v));
      }
    }
    LOG_INFO("[GGML-HC-DEBUG] pre layer=%d sub=%d finite=%zu/%zu bad=%zu\n",
             layer, sub_block, e == cudaSuccess ? n - count_nonfinite(check) : 0,
             n, e == cudaSuccess ? count_nonfinite(check) : n);
    LOG_INFO("[GGML-HC-DEBUG] pre layer=%d sub=%d max_abs=%g\n", layer,
             sub_block, max_abs);
    static thread_local std::vector<float> inject_check;
    const size_t inject_n = (size_t)S * hc_;
    inject_check.resize(inject_n);
    const cudaError_t inject_e = cudaMemcpy(
        inject_check.data(), ((ggml_tensor *)t_inject_)->data,
        inject_n * sizeof(float), cudaMemcpyDeviceToHost);
    LOG_INFO("[GGML-HC-DEBUG] inject layer=%d sub=%d finite=%zu/%zu bad=%zu\n",
             layer, sub_block,
             inject_e == cudaSuccess ? inject_n - count_nonfinite(inject_check) : 0,
             inject_n,
             inject_e == cudaSuccess ? count_nonfinite(inject_check) : inject_n);
    if (e != cudaSuccess) {
      return false;
    }
  }
  if (out_hidden) {
    const size_t bytes = (size_t)S * d_model_ * sizeof(float);
    if (qwen_hc_async_readback() &&
        ensure_readback_buffer(bytes) && readback_stream_ &&
        readback_done_event_ && active_pre_done_event_ &&
        ((ggml_backend_event_t)active_pre_done_event_)->context) {
      // Do not synchronize the producer event on the host.  The copy stream
      // waits for the exact HC completion event, performs a D2H copy into
      // pinned memory, and publishes a second event for the one unavoidable
      // CPU consumer wait.  This removes the pageable cudaMemcpy from the
      // per-layer path while retaining the same dependency edge.
      const cudaError_t wait_status = cudaStreamWaitEvent(
          (cudaStream_t)readback_stream_,
          (cudaEvent_t)((ggml_backend_event_t)active_pre_done_event_)->context,
          0);
      const cudaError_t copy_status =
          wait_status == cudaSuccess
              ? cudaMemcpyAsync(readback_host_, ((ggml_tensor *)t_mixed_)->data,
                                bytes, cudaMemcpyDeviceToHost,
                                (cudaStream_t)readback_stream_)
              : wait_status;
      const cudaError_t record_status =
          copy_status == cudaSuccess
              ? cudaEventRecord((cudaEvent_t)readback_done_event_,
                                (cudaStream_t)readback_stream_)
              : copy_status;
      if (record_status != cudaSuccess) {
        LOG_ERROR("[GGML-HC] async pre_mix readback failed: %s\n",
                  cudaGetErrorString(record_status));
        return false;
      }
      const cudaError_t sync_status =
          cudaEventSynchronize((cudaEvent_t)readback_done_event_);
      if (sync_status != cudaSuccess) {
        LOG_ERROR("[GGML-HC] async pre_mix readback wait failed: %s\n",
                  cudaGetErrorString(sync_status));
        return false;
      }
      std::memcpy(out_hidden, readback_host_, bytes);
    } else if (b.custom_q8) {
      cudaEventSynchronize(
          (cudaEvent_t)((ggml_backend_event_t)active_pre_done_event_)->context);
      cudaMemcpy(out_hidden, ((ggml_tensor *)t_mixed_)->data, bytes,
                 cudaMemcpyDeviceToHost);
    } else {
      // The completion event is recorded immediately after this graph was
      // queued on the shared backend stream. Waiting on that event is the
      // only dependency needed to read t_mixed_; a backend-wide synchronize
      // also drains unrelated router/MoE work queued on the same backend.
      // tensor_get() additionally routes the small readback through GGML's
      // copy stream, which adds another stream handoff at every CPU-affinity
      // layer. Use the event followed by one direct copy instead. The tensor
      // is the persistent F32 [D,S] buffer, whose contiguous layout is
      // identical to the host [S,D] row-major layout used by callers.
      if (active_pre_done_event_) {
        ggml_backend_event_synchronize(
            (ggml_backend_event_t)active_pre_done_event_);
      } else {
        // Defensive fallback for an incomplete runner initialization. The
        // normal path always creates/records the event above.
        ggml_backend_synchronize((ggml_backend_t)backend_);
      }
      const cudaError_t status = cudaMemcpy(
          out_hidden, ((ggml_tensor *)t_mixed_)->data, bytes,
          cudaMemcpyDeviceToHost);
      if (status != cudaSuccess) {
        LOG_ERROR("[GGML-HC] direct pre_mix readback failed: %s\n",
                  cudaGetErrorString(status));
        return false;
      }
    }
  }
  if (device_output) *device_output = t_mixed_;
  inject_valid_S_ = S;
  return true;
}

void GgmlHcRunner::queue_pre_mix_dependency() {
  if (!backend_ || !active_pre_done_event_ || !pre_in_flight_) return;
  auto *event = (ggml_backend_event_t)active_pre_done_event_;
  // GGML consumers use the backend's stream. The explicit wait documents and
  // enforces the producer edge even when the backend implementation changes
  // its internal stream policy.
  ggml_backend_event_wait((ggml_backend_t)backend_, event);
  // Qwen4Exp's direct attention projections use CUDA's legacy stream (0),
  // which is independent of GGML's nonblocking backend stream.
  cudaSetDevice(device_);
  cudaStreamWaitEvent(0, (cudaEvent_t)event->context, 0);
}

bool GgmlHcRunner::post_combine(int layer, int sub_block, int S,
                                const float *sub_block_out) {
  return post_combine_impl(layer, sub_block, S, sub_block_out, nullptr);
}

bool GgmlHcRunner::post_combine_device(int layer, int sub_block, int S,
                                       const void *device_sub_block_out) {
  return post_combine_impl(layer, sub_block, S, nullptr,
                           device_sub_block_out);
}

bool GgmlHcRunner::post_combine_device_fused(
    int layer, int sub_block, int S, const void *device_sub_block_out,
    const void *producer_event) {
  (void)layer;
  if (!backend_ || !t_streams_ || !t_inject_ || !device_sub_block_out ||
      !producer_event || sub_block != 0 || S <= 0 || S > s_cap_ ||
      inject_valid_S_ != S || !post_done_event_)
    return false;

  const ggml_tensor *src = (const ggml_tensor *)device_sub_block_out;
  const ggml_backend_event_t producer =
      (ggml_backend_event_t)const_cast<void *>(producer_event);
  const ggml_tensor *streams = (const ggml_tensor *)t_streams_;
  const ggml_tensor *inject = (const ggml_tensor *)t_inject_;
  if (!src || !src->data || !src->buffer || src->type != GGML_TYPE_F32 ||
      src->ne[0] != d_model_ || src->ne[1] < S || src->ne[2] != 1 ||
      src->ne[3] != 1 || src->nb[0] != sizeof(float) ||
      src->nb[1] != (size_t)d_model_ * sizeof(float) || !streams->data ||
      !inject->data || !producer->context ||
      !((ggml_backend_event_t)post_done_event_)->context)
    return false;

  cudaSetDevice(device_);
  // The shared expert is queued on GGML's nonblocking CUDA stream. The fused
  // HC update runs on stream 0 because the existing Qwen dense/attention
  // bridge uses that stream. Bridge both producer and previous-state fences
  // explicitly; no host synchronization is needed here.
  if (post_fused_in_flight_)
    cudaStreamWaitEvent(
        0, (cudaEvent_t)((ggml_backend_event_t)post_done_event_)->context, 0);
  cudaStreamWaitEvent(0, (cudaEvent_t)producer->context, 0);
  launch_qwen4exp_hc_q8_post(
      static_cast<float *>(streams->data),
      static_cast<const float *>(src->data),
      static_cast<const float *>(inject->data), S, hc_, d_model_, 0);
  if (cudaEventRecord(
          (cudaEvent_t)((ggml_backend_event_t)post_done_event_)->context,
          0) != cudaSuccess)
    return false;
  post_in_flight_ = true;
  post_fused_in_flight_ = true;
  return true;
}

bool GgmlHcRunner::post_combine_device_raw(int layer, int sub_block, int S,
                                            const void *device_data,
                                            size_t bytes) {
  return post_combine_impl(layer, sub_block, S, nullptr, device_data, bytes);
}

bool GgmlHcRunner::copy_pre_mix_to_host(const void *device_data, float *host,
                                        int S) {
  if (!backend_ || !device_data || !host || !t_mixed_ || S <= 0 ||
      S > s_cap_ || (device_data != t_mixed_ &&
                     device_data != ((ggml_tensor *)t_mixed_)->data))
    return false;
  cudaSetDevice(device_);
  ggml_backend_synchronize((ggml_backend_t)backend_);
  ggml_backend_tensor_get((ggml_tensor *)t_mixed_, host, 0,
                          (size_t)S * d_model_ * sizeof(float));
  return true;
}

const void *GgmlHcRunner::device_data(const void *device_output) const {
  if (!backend_ || !device_output || !t_mixed_ ||
      (device_output != t_mixed_ &&
       device_output != ((ggml_tensor *)t_mixed_)->data))
    return nullptr;
  return ((ggml_tensor *)t_mixed_)->data;
}

bool GgmlHcRunner::post_combine_impl(int layer, int sub_block, int S,
                                     const float *host_sub_block_out,
                                     const void *device_sub_block_out,
                                     size_t device_bytes) {
  if (!backend_ || !t_streams_ || !t_inject_ ||
      (!!host_sub_block_out == !!device_sub_block_out))
    return false;
  if (S > s_cap_ || inject_valid_S_ != S) return false;
  cudaSetDevice(device_);
  if (!ensure_post_graph(S)) return false;

  // The graph and its [D,S] input are persistent for the whole request shape.
  // In particular, do not call ggml_gallocr_alloc_graph() here: doing so at
  // the MoE->HC boundary can recycle temporary storage while the MoE graph's
  // auxiliary CUDA streams are still finishing.
  ggml_tensor *so_flat = (ggml_tensor *)post_input_;
  ggml_cgraph *gf = (ggml_cgraph *)post_graph_;
  if (host_sub_block_out) {
    ggml_backend_tensor_set(so_flat, host_sub_block_out, 0,
                            (size_t)S * d_model_ * sizeof(float));
  } else if (device_bytes != 0) {
    const size_t bytes = (size_t)S * d_model_ * sizeof(float);
    if (!device_sub_block_out || device_bytes < bytes || !so_flat->data) {
      LOG_ERROR("[GGML-HC] post_combine raw device input has unexpected size "
                "(layer=%d sub=%d S=%d bytes=%zu expected=%zu)\n",
                layer, sub_block, S, device_bytes, bytes);
      return false;
    }
    if (!raw_input_event_ ||
        !((ggml_backend_event_t)raw_input_event_)->context) {
      LOG_ERROR("[GGML-HC] raw post-combine dependency event unavailable\n");
      return false;
    }
    // Full-attention's producer kernels run on CUDA stream 0. Queue the
    // device-to-device handoff on that same stream, then bridge it to the
    // shared GGML stream with one event. The old blocking cudaMemcpy forced
    // the CPU to wait for every attention layer before HC could be queued.
    const cudaError_t copy_status = cudaMemcpyAsync(
        so_flat->data, device_sub_block_out, bytes, cudaMemcpyDeviceToDevice,
        0);
    const cudaError_t event_status =
        copy_status == cudaSuccess
            ? cudaEventRecord(
                  (cudaEvent_t)((ggml_backend_event_t)raw_input_event_)->context,
                  0)
            : copy_status;
    if (copy_status != cudaSuccess || event_status != cudaSuccess) {
      LOG_ERROR("[GGML-HC] post_combine raw device copy failed (layer=%d "
                "sub=%d S=%d): %s\n",
                layer, sub_block, S, cudaGetErrorString(
                    copy_status != cudaSuccess ? copy_status : event_status));
      return false;
    }
    ggml_backend_event_wait((ggml_backend_t)backend_,
                            (ggml_backend_event_t)raw_input_event_);
  } else {
    // GgmlMoeRunner's graph output is [D,S] F32 contiguous and `so` now has
    // the exact same layout. Both runners borrow ggml_shared_backend(device),
    // so the public async backend copy queues a same-stream D2D operation. It
    // also preserves ordering when the producer graph was submitted with
    // ggml_backend_graph_compute_async above.
    ggml_tensor *src = (ggml_tensor *)device_sub_block_out;
    if (!src || src->type != GGML_TYPE_F32 || src->ne[0] != d_model_ ||
        src->ne[1] < S || src->ne[2] != 1 || src->ne[3] != 1 ||
        src->nb[0] != sizeof(float) ||
        src->nb[1] != (size_t)d_model_ * sizeof(float) || !src->buffer ||
        !src->data || !so_flat->buffer || !so_flat->data) {
        LOG_ERROR("[GGML-HC] post_combine device input has unexpected layout "
                  "(layer=%d sub=%d S=%d)\n",
                  layer, sub_block, S);
      return false;
    }
    ggml_tensor *src_view = src;
    if (src->ne[1] != S) {
      // Reuse the one metadata view built by ensure_post_graph().  Retargeting
      // the CUDA pointer is safe here because the producer tensor is already
      // complete/ordered by the shared backend dependency; allocating a new
      // view on every decode token would eventually exhaust post_ctx_.
      src_view = (ggml_tensor *)post_source_view_;
      if (!src_view) return false;
      src_view->data = src->data;
      src_view->buffer = src->buffer;
    }
    if (!src_view) {
      return false;
    }
    if (debug_check_hc_state()) {
      cudaDeviceSynchronize();
      static thread_local std::vector<float> source_check;
      const size_t n = (size_t)S * d_model_;
      source_check.resize(n);
      const cudaError_t e = cudaMemcpy(
          source_check.data(), src_view->data, n * sizeof(float),
          cudaMemcpyDeviceToHost);
      LOG_INFO("[GGML-HC-DEBUG] input layer=%d sub=%d finite=%zu/%zu bad=%zu\n",
               layer, sub_block,
               e == cudaSuccess ? n - count_nonfinite(source_check) : 0,
               n, e == cudaSuccess ? count_nonfinite(source_check) : n);
      if (e != cudaSuccess) {
        return false;
      }
    }
    ggml_backend_tensor_copy_async((ggml_backend_t)backend_,
                                   (ggml_backend_t)backend_, src_view,
                                   so_flat);
  }
  const ggml_status st =
      ggml_backend_graph_compute_async((ggml_backend_t)backend_, gf);
  if (st != GGML_STATUS_SUCCESS) {
    LOG_ERROR("[GGML-HC] post_combine compute failed (status=%d)\n", (int)st);
    return false;
  }
  ggml_backend_event_record((ggml_backend_event_t)post_done_event_,
                            (ggml_backend_t)backend_);
  post_in_flight_ = true;
  post_fused_in_flight_ = false;
  if (debug_sync_hc())
    ggml_backend_synchronize((ggml_backend_t)backend_);
  if (debug_check_hc_state()) {
    cudaDeviceSynchronize();
    static thread_local std::vector<float> check;
    const size_t n = (size_t)S * hc_ * d_model_;
    check.resize(n);
    const cudaError_t e = cudaMemcpy(
        check.data(), ((ggml_tensor *)t_streams_)->data,
        n * sizeof(float), cudaMemcpyDeviceToHost);
    LOG_INFO("[GGML-HC-DEBUG] post layer=%d sub=%d finite=%zu/%zu bad=%zu\n",
             layer, sub_block, e == cudaSuccess ? n - count_nonfinite(check) : 0,
             n, e == cudaSuccess ? count_nonfinite(check) : n);
    if (e != cudaSuccess) {
      return false;
    }
  }
  return true;
}

#else  // !MOESTREAM_USE_GGML_CUDA

void set_ggml_hc_async_graph_reuse(bool) {}
void set_qwen_hc_async_readback(bool) {}

bool GgmlHcRunner::compiled_in() { return false; }
GgmlHcRunner &GgmlHcRunner::instance() {
  static GgmlHcRunner r;
  return r;
}
GgmlHcRunner::~GgmlHcRunner() {}
void GgmlHcRunner::shutdown() {}
bool GgmlHcRunner::ensure(int, int, int, int, float) { return false; }
bool GgmlHcRunner::ensure_state(int) { return false; }
bool GgmlHcRunner::ensure_readback_buffer(size_t) { return false; }
bool GgmlHcRunner::weights_ready(int, int) const { return false; }
void GgmlHcRunner::release_completed_weights_nonblocking(int) {}
bool GgmlHcRunner::ensure_weights(int, int, const GgmlHcWeightRef &,
                                  const GgmlHcWeightRef &,
                                  const GgmlHcWeightRef &,
                                  const GgmlHcWeightRef &) {
  return false;
}
bool GgmlHcRunner::upload_streams(const float *, int) { return false; }
bool GgmlHcRunner::download_streams(float *, int) { return false; }
bool GgmlHcRunner::download_inject(float *, int) { return false; }
bool GgmlHcRunner::pre_mix(int, int, int, float *, const void **) { return false; }
bool GgmlHcRunner::post_combine(int, int, int, const float *) { return false; }
bool GgmlHcRunner::post_combine_device(int, int, int, const void *) {
  return false;
}
bool GgmlHcRunner::post_combine_device_fused(int, int, int, const void *,
                                             const void *) {
  return false;
}
bool GgmlHcRunner::post_combine_device_raw(int, int, int, const void *,
                                            size_t) {
  return false;
}
bool GgmlHcRunner::copy_pre_mix_to_host(const void *, float *, int) {
  return false;
}
const void *GgmlHcRunner::device_data(const void *) const { return nullptr; }

void GgmlHcRunner::queue_pre_mix_dependency() {}

#endif
