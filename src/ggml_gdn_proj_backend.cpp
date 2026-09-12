// Qwen4Exp GDN's shared-input projection fusion. See
// include/ggml_gdn_proj_backend.hpp for the measurement that motivates this
// and its deliberately narrow scope.
#include "ggml_gdn_proj_backend.hpp"

#include "ggml_moe_backend.hpp" // ggml_shared_backend()
#include "logger.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#if defined(MOESTREAM_USE_GGML_CUDA)

#include <cuda_runtime.h>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

namespace {
bool type_ok(uint32_t t) {
  if (t >= (uint32_t)GGML_TYPE_COUNT)
    return false;
  const ggml_type g = (ggml_type)t;
  return g == GGML_TYPE_F32 || g == GGML_TYPE_F16 || g == GGML_TYPE_BF16 ||
         ggml_is_quantized(g);
}

// Diagnostic only.  The device handoff path is intended to remain fully
// asynchronous, but this switch lets us distinguish a graph lifetime/reuse
// race from a bad GDN graph or HC consumer without changing the default
// runtime behaviour.
bool debug_sync_full_gdn() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOESTREAM_GGML_DEBUG_SYNC_FULL_GDN");
    return v && v[0] == '1';
  }();
  return enabled;
}

bool debug_device_sync_full_gdn() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOESTREAM_GGML_DEBUG_DEVICE_SYNC_FULL_GDN");
    return v && v[0] == '1';
  }();
  return enabled;
}
} // namespace

bool GgmlGdnProjRunner::compiled_in() { return true; }

GgmlGdnProjRunner &GgmlGdnProjRunner::instance() {
  static GgmlGdnProjRunner r;
  return r;
}

GgmlGdnProjRunner::~GgmlGdnProjRunner() { shutdown(); }

void GgmlGdnProjRunner::release_completed_weights(int before_layer) {
  if (!backend_ || before_layer <= 0)
    return;

  // Every completed-layer GDN graph has already been submitted to this
  // backend.  One barrier makes its projection/conv/full weight buffers safe
  // to return to CUDA.  Keep FullState::history/state/output: recurrent state
  // is required by decode, and the output/state objects are separate from the
  // weight buffers being reclaimed here.
  ggml_backend_synchronize((ggml_backend_t)backend_);
  size_t reclaimed = 0;
  int layers = 0;
  for (int l = 0; l < before_layer; ++l) {
    Bracket &b = brackets_[l];
    if (!b.buf && !b.ctx && !b.conv_buf && !b.conv_ctx && !b.full_buf &&
        !b.full_ctx)
      continue;
    if (b.buf) {
      reclaimed += ggml_backend_buffer_get_size((ggml_backend_buffer_t)b.buf);
      ggml_backend_buffer_free((ggml_backend_buffer_t)b.buf);
    }
    if (b.ctx)
      ggml_free((ggml_context *)b.ctx);
    if (b.conv_buf) {
      reclaimed +=
          ggml_backend_buffer_get_size((ggml_backend_buffer_t)b.conv_buf);
      ggml_backend_buffer_free((ggml_backend_buffer_t)b.conv_buf);
    }
    if (b.conv_ctx)
      ggml_free((ggml_context *)b.conv_ctx);
    if (b.full_buf) {
      reclaimed +=
          ggml_backend_buffer_get_size((ggml_backend_buffer_t)b.full_buf);
      ggml_backend_buffer_free((ggml_backend_buffer_t)b.full_buf);
    }
    if (b.full_ctx)
      ggml_free((ggml_context *)b.full_ctx);
    b = Bracket{};
    ++layers;
  }
  if (layers > 0) {
    LOG_WARN("[GGML-GDN-MEM] allocation-pressure reclaim: released %d "
             "completed-layer weight sets (%.2f MiB); recurrent state kept\n",
             layers, (double)reclaimed / (1024.0 * 1024.0));
  }
}

void GgmlGdnProjRunner::shutdown() {
  if (device_ >= 0)
    cudaSetDevice(device_);
  if (backend_)
    ggml_backend_synchronize((ggml_backend_t)backend_);
  for (int l = 0; l < kMaxLayers; ++l) {
    Bracket &b = brackets_[l];
    if (b.buf)
      ggml_backend_buffer_free((ggml_backend_buffer_t)b.buf);
    if (b.ctx)
      ggml_free((ggml_context *)b.ctx);
    if (b.conv_buf)
      ggml_backend_buffer_free((ggml_backend_buffer_t)b.conv_buf);
    if (b.conv_ctx)
      ggml_free((ggml_context *)b.conv_ctx);
    if (b.full_buf)
      ggml_backend_buffer_free((ggml_backend_buffer_t)b.full_buf);
    if (b.full_ctx)
      ggml_free((ggml_context *)b.full_ctx);
    b = Bracket{};
  }
  for (FullState &s : full_states_) {
    if (s.output_buf) {
      ggml_backend_buffer_free((ggml_backend_buffer_t)s.output_buf);
      s.output_buf = nullptr;
    }
    if (s.output_ctx) {
      ggml_free((ggml_context *)s.output_ctx);
      s.output_ctx = nullptr;
    }
    if (s.buf)
      ggml_backend_buffer_free((ggml_backend_buffer_t)s.buf);
    if (s.ctx)
      ggml_free((ggml_context *)s.ctx);
    s = FullState{};
  }
  for (FullDeviceSlot &slot : full_device_slots_) {
    if (slot.done_event) {
      ggml_backend_event_free((ggml_backend_event_t)slot.done_event);
      slot.done_event = nullptr;
    }
    if (slot.galloc) {
      ggml_gallocr_free((ggml_gallocr_t)slot.galloc);
      slot.galloc = nullptr;
    }
    slot.in_flight = false;
  }
  full_device_workspace_bytes_ = 0;
  if (full_output_buf_) {
    ggml_backend_buffer_free((ggml_backend_buffer_t)full_output_buf_);
    full_output_buf_ = nullptr;
  }
  if (full_output_ctx_) {
    ggml_free((ggml_context *)full_output_ctx_);
    full_output_ctx_ = nullptr;
  }
  t_full_output_ = nullptr;
  full_output_cap_ = 0;
  if (galloc_) {
    ggml_gallocr_free((ggml_gallocr_t)galloc_);
    galloc_ = nullptr;
  }
  if (external_buf_) {
    ggml_backend_buffer_free((ggml_backend_buffer_t)external_buf_);
    external_buf_ = nullptr;
  }
  backend_ = nullptr; // shared, not owned
  device_ = -1;
}

bool GgmlGdnProjRunner::ensure(int device, int D, int conv_dim, int value_dim,
                               int H_v) {
  if (device < 0 || D <= 0 || conv_dim <= 0 || value_dim <= 0 || H_v <= 0)
    return false;
  if (backend_ && device_ == device && D_ == D && conv_dim_ == conv_dim &&
      value_dim_ == value_dim && H_v_ == H_v)
    return true;
  if (backend_)
    shutdown();
  cudaSetDevice(device);
  backend_ = ggml_shared_backend(device);
  if (!backend_)
    return false;
  galloc_ = ggml_gallocr_new(
      ggml_backend_get_default_buffer_type((ggml_backend_t)backend_));
  if (!galloc_) {
    backend_ = nullptr;
    return false;
  }
  external_buf_ = ggml_backend_alloc_buffer((ggml_backend_t)backend_, 1);
  if (!external_buf_) {
    ggml_gallocr_free((ggml_gallocr_t)galloc_);
    galloc_ = nullptr;
    backend_ = nullptr;
    return false;
  }
  ggml_backend_buffer_set_usage((ggml_backend_buffer_t)external_buf_,
                                GGML_BACKEND_BUFFER_USAGE_COMPUTE);
  device_ = device;
  D_ = D;
  conv_dim_ = conv_dim;
  value_dim_ = value_dim;
  H_v_ = H_v;
  const ggml_backend_buffer_type_t bt =
      ggml_backend_get_default_buffer_type((ggml_backend_t)backend_);
  const ggml_backend_dev_t dev =
      ggml_backend_get_device((ggml_backend_t)backend_);
  for (FullDeviceSlot &slot : full_device_slots_) {
    slot.galloc = ggml_gallocr_new(bt);
    slot.done_event = (void *)ggml_backend_event_new(dev);
    if (!slot.galloc || !slot.done_event) {
      shutdown();
      return false;
    }
  }
  LOG_INFO("[GGML-GDN-PROJ] enabled on GPU %d (D=%d conv_dim=%d value_dim=%d "
           "H_v=%d)\n",
           device, D, conv_dim, value_dim, H_v);
  return true;
}

bool GgmlGdnProjRunner::weights_ready(int layer) const {
  if (layer < 0 || layer >= kMaxLayers)
    return false;
  return brackets_[layer].ready;
}

bool GgmlGdnProjRunner::full_weights_ready(int layer) const {
  if (layer < 0 || layer >= kMaxLayers)
    return false;
  return brackets_[layer].full_ready;
}

bool GgmlGdnProjRunner::ensure_full_output(int layer, int S) {
  // Decode uses the same persistent per-layer output as prefill.  The old
  // S<=1 guard silently disabled the device-resident full-GDN graph for every
  // single-token step, even though the graph itself and its state tensors
  // support S=1.
  if (!backend_ || layer < 0 || layer >= kMaxLayers || S <= 0)
    return false;
  FullState &state = full_states_[layer];
  if (state.output && S <= state.output_cap)
    return true;

  cudaSetDevice(device_);
  // A request can change S between calls.  The previous request's
  // finish_full_device_updates() normally drained the backend already, but
  // keep this local guard so a direct runner user cannot free an output that
  // is still referenced by a queued HC graph.
  if (state.output_buf) {
    ggml_backend_synchronize((ggml_backend_t)backend_);
    ggml_backend_buffer_free((ggml_backend_buffer_t)state.output_buf);
    state.output_buf = nullptr;
  }
  if (state.output_ctx) {
    ggml_free((ggml_context *)state.output_ctx);
    state.output_ctx = nullptr;
  }

  const int cap = std::max(S, state.output_cap * 2);
  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 2;
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx)
    return false;
  ggml_tensor *out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_, cap);
  if (!out) {
    ggml_free(ctx);
    return false;
  }
  ggml_backend_buffer_t buf =
      ggml_backend_alloc_ctx_tensors(ctx, (ggml_backend_t)backend_);
  if (!buf) {
    LOG_WARN("[GGML-GDN-FULL] persistent output allocation failed (S=%d)\n", S);
    ggml_free(ctx);
    return false;
  }
  ggml_set_name(out, "gdn_full_output");
  state.output_ctx = ctx;
  state.output_buf = buf;
  state.output = out;
  state.output_cap = cap;
  LOG_INFO("[GGML-GDN-FULL-MEM] persistent output: layer=%d %.2f MiB "
           "(S_cap=%d)\n",
           layer, (double)ggml_backend_buffer_get_size(buf) / (1024.0 * 1024.0),
           cap);
  return true;
}

bool GgmlGdnProjRunner::ensure_full_state(int layer, int d_conv, int hd) {
  if (!backend_ || layer < 0 || layer >= kMaxLayers || d_conv <= 1 || hd <= 0)
    return false;
  const int hist = d_conv - 1;
  FullState &s = full_states_[layer];
  const size_t history_bytes = (size_t)conv_dim_ * hist * sizeof(float);
  const size_t state_bytes = (size_t)H_v_ * hd * hd * sizeof(float);
  if (s.ctx && s.buf && s.history && s.state &&
      s.history_bytes == history_bytes && s.state_bytes == state_bytes)
    return true;

  if (s.ctx || s.buf) {
    // A shape change is rare (only a different model or convolution width),
    // but the old persistent tensors may still be referenced by the queued
    // graph.  The shared backend barrier makes freeing them safe.
    ggml_backend_synchronize((ggml_backend_t)backend_);
    if (s.output_buf)
      ggml_backend_buffer_free((ggml_backend_buffer_t)s.output_buf);
    if (s.output_ctx)
      ggml_free((ggml_context *)s.output_ctx);
    if (s.buf)
      ggml_backend_buffer_free((ggml_backend_buffer_t)s.buf);
    if (s.ctx)
      ggml_free((ggml_context *)s.ctx);
    s = FullState{};
  }

  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 4;
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx)
    return false;
  ggml_tensor *history =
      ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hist, conv_dim_, 1);
  ggml_tensor *state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, hd, H_v_, 1);
  if (!history || !state) {
    ggml_free(ctx);
    return false;
  }
  ggml_backend_buffer_t buf =
      ggml_backend_alloc_ctx_tensors(ctx, (ggml_backend_t)backend_);
  if (!buf) {
    ggml_free(ctx);
    return false;
  }
  ggml_set_name(history, "gdn_full_history_state");
  ggml_set_name(state, "gdn_full_recurrent_state");
  s.ctx = ctx;
  s.buf = buf;
  s.history = history;
  s.state = state;
  s.history_bytes = history_bytes;
  s.state_bytes = state_bytes;
  // cudaMalloc-backed GGML state is intentionally initialized here.  This is
  // the only zeroing needed for a first S=1 sequence; subsequent requests use
  // reset_full_device_states() before their first decode graph.
  cudaSetDevice(device_);
  if (cudaMemset(((ggml_tensor *)history)->data, 0, history_bytes) !=
          cudaSuccess ||
      cudaMemset(((ggml_tensor *)state)->data, 0, state_bytes) != cudaSuccess) {
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    s = FullState{};
    return false;
  }
  LOG_INFO("[GGML-GDN-FULL-MEM] deferred state layer=%d: %.2f MiB\n", layer,
           (double)ggml_backend_buffer_get_size(buf) / (1024.0 * 1024.0));
  return true;
}

void GgmlGdnProjRunner::finish_full_device_updates(bool copy_to_host) {
  if (!backend_)
    return;
  if (!copy_to_host) {
    // HC collapse/download already provides the forward-level device barrier.
    // Do not enqueue state D2H copies for S=1 decode; the next graph on the
    // same backend stream consumes these persistent tensors directly.
    for (FullState &s : full_states_) {
      if (s.pending) {
        s.pending = false;
        s.device_state_resident = true;
      }
    }
    return;
  }
  bool queued = false;
  for (FullState &s : full_states_) {
    if (!s.pending || !s.host_history || !s.host_state)
      continue;
    ggml_backend_tensor_get_async((ggml_backend_t)backend_,
                                  (ggml_tensor *)s.history, s.host_history, 0,
                                  s.history_bytes);
    ggml_backend_tensor_get_async((ggml_backend_t)backend_,
                                  (ggml_tensor *)s.state, s.host_state, 0,
                                  s.state_bytes);
    queued = true;
    s.pending = false;
    s.device_state_resident = false;
  }
  if (queued)
    ggml_backend_synchronize((ggml_backend_t)backend_);
}

void GgmlGdnProjRunner::sync_full_device_states_to_host() {
  if (!backend_)
    return;
  bool queued = false;
  for (FullState &s : full_states_) {
    if ((!s.pending && !s.device_state_resident) || !s.host_history ||
        !s.host_state || !s.history || !s.state)
      continue;
    ggml_backend_tensor_get_async((ggml_backend_t)backend_,
                                  (ggml_tensor *)s.history, s.host_history, 0,
                                  s.history_bytes);
    ggml_backend_tensor_get_async((ggml_backend_t)backend_,
                                  (ggml_tensor *)s.state, s.host_state, 0,
                                  s.state_bytes);
    s.pending = false;
    s.device_state_resident = false;
    queued = true;
  }
  if (queued)
    ggml_backend_synchronize((ggml_backend_t)backend_);
}

void GgmlGdnProjRunner::reset_full_device_states() {
  if (!backend_)
    return;
  ggml_backend_synchronize((ggml_backend_t)backend_);
  cudaSetDevice(device_);
  for (FullState &s : full_states_) {
    if (!s.history || !s.state)
      continue;
    if (cudaMemset(((ggml_tensor *)s.history)->data, 0, s.history_bytes) !=
            cudaSuccess ||
        cudaMemset(((ggml_tensor *)s.state)->data, 0, s.state_bytes) !=
            cudaSuccess) {
      LOG_ERROR("[GGML-GDN-FULL] failed to reset persistent device state\n");
      break;
    }
    s.pending = false;
    s.device_state_resident = true;
  }
}

bool GgmlGdnProjRunner::ensure_full_weights(int layer,
                                            const GgmlGdnProjWeightRef &norm,
                                            const GgmlGdnProjWeightRef &out) {
  if (!backend_ || layer < 0 || layer >= kMaxLayers)
    return false;
  Bracket &b = brackets_[layer];
  if (b.full_ready)
    return true;
  if (!b.ready || !norm.host_data || !out.host_data || !type_ok(norm.type) ||
      !type_ok(out.type) || value_dim_ % H_v_ != 0 ||
      norm.ne0 != value_dim_ / H_v_ || norm.ne1 != 1 || out.ne0 != value_dim_ ||
      out.ne1 != D_) {
    LOG_WARN("[GGML-GDN-FULL] layer=%d weight precondition/shape check failed "
             "(norm=%lldx%lld out=%lldx%lld expected norm=%dx1 out=%dx%d)\n",
             layer, (long long)norm.ne0, (long long)norm.ne1,
             (long long)out.ne0, (long long)out.ne1, value_dim_ / H_v_, 1,
             value_dim_, D_);
    return false;
  }

  cudaSetDevice(device_);
  ggml_context *ctx = nullptr;
  ggml_tensor *tnorm = nullptr;
  ggml_tensor *tout = nullptr;
  ggml_backend_buffer_t buf = nullptr;
  bool ext_norm = false, ext_out = false;
  for (int attempt = 0; attempt < 2 && !buf; ++attempt) {
    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead() * 4;
    ip.no_alloc = true;
    ctx = ggml_init(ip);
    if (!ctx)
      return false;
    tnorm = ggml_new_tensor_2d(ctx, (ggml_type)norm.type, norm.ne0, norm.ne1);
    tout = ggml_new_tensor_2d(ctx, (ggml_type)out.type, out.ne0, out.ne1);
    if (!tnorm || !tout) {
      ggml_free(ctx);
      return false;
    }
    ext_norm = external_buf_ && norm.device_data &&
               norm.device_bytes >= ggml_nbytes(tnorm);
    ext_out = external_buf_ && out.device_data &&
              out.device_bytes >= ggml_nbytes(tout);
    if (ext_norm) {
      tnorm->data = const_cast<void *>(norm.device_data);
      tnorm->buffer = (ggml_backend_buffer_t)external_buf_;
      ggml_backend_buffer_init_tensor(tnorm->buffer, tnorm);
    }
    if (ext_out) {
      tout->data = const_cast<void *>(out.device_data);
      tout->buffer = (ggml_backend_buffer_t)external_buf_;
      ggml_backend_buffer_init_tensor(tout->buffer, tout);
    }
    if (ext_norm && ext_out) {
      // Both tensors are already in the engine's dense pool; keep only the
      // GGML metadata context and borrow the marker buffer.
      break;
    }
    buf = ggml_backend_alloc_ctx_tensors(ctx, (ggml_backend_t)backend_);
    if (buf)
      break;
    ggml_free(ctx);
    ctx = nullptr;
    if (attempt == 0) {
      LOG_WARN("[GGML-GDN-FULL-MEM] layer=%d allocation pressure; "
               "reclaiming completed GDN layers and retrying\n",
               layer);
      release_completed_weights(layer);
    }
  }
  if (!buf && !(ext_norm && ext_out)) {
    LOG_WARN("[GGML-GDN-FULL] layer=%d full weight allocation failed\n", layer);
    return false;
  }
  if (!ext_norm)
    ggml_backend_tensor_set(tnorm, norm.host_data, 0,
                            std::min(norm.nbytes, ggml_nbytes(tnorm)));
  if (!ext_out)
    ggml_backend_tensor_set(tout, out.host_data, 0,
                            std::min(out.nbytes, ggml_nbytes(tout)));
  b.full_ctx = ctx;
  b.full_buf = buf;
  b.ssm_norm = tnorm;
  b.ssm_out = tout;
  b.full_ready = true;
  const size_t weight_bytes = ggml_nbytes(tnorm) + ggml_nbytes(tout);
  LOG_INFO("[GGML-GDN-FULL-MEM] weights: layer=%d %.2f MiB (%s)\n", layer,
           (double)weight_bytes / (1024.0 * 1024.0),
           (ext_norm && ext_out) ? "borrowed dense pool" : "private buffer");
  return true;
}

bool GgmlGdnProjRunner::ensure_weights(int layer,
                                       const GgmlGdnProjWeightRef &qkv,
                                       const GgmlGdnProjWeightRef &gate,
                                       const GgmlGdnProjWeightRef &alpha,
                                       const GgmlGdnProjWeightRef &beta,
                                       const float *dt_bias_host,
                                       const float *ssm_a_host) {
  if (!backend_ || layer < 0 || layer >= kMaxLayers)
    return false;
  Bracket &b = brackets_[layer];
  if (b.ready)
    return true;
  if (!qkv.host_data || !gate.host_data || !alpha.host_data ||
      !beta.host_data || !dt_bias_host || !ssm_a_host)
    return false;
  if (!type_ok(qkv.type) || !type_ok(gate.type) || !type_ok(alpha.type) ||
      !type_ok(beta.type))
    return false;
  // Shape check, not assumption: a mismatch here would otherwise silently
  // read one tensor's bytes as another's shape.
  if (qkv.ne0 != D_ || qkv.ne1 != conv_dim_ || gate.ne0 != D_ ||
      gate.ne1 != value_dim_ || alpha.ne0 != D_ || alpha.ne1 != H_v_ ||
      beta.ne0 != D_ || beta.ne1 != H_v_) {
    LOG_ERROR(
        "[GGML-GDN-PROJ] layer=%d unexpected weight shapes (qkv %lldx%lld, "
        "gate %lldx%lld, alpha %lldx%lld, beta %lldx%lld)\n",
        layer, (long long)qkv.ne0, (long long)qkv.ne1, (long long)gate.ne0,
        (long long)gate.ne1, (long long)alpha.ne0, (long long)alpha.ne1,
        (long long)beta.ne0, (long long)beta.ne1);
    return false;
  }
  cudaSetDevice(device_);
  ggml_context *ctx = nullptr;
  ggml_tensor *tqkv = nullptr;
  ggml_tensor *tgate = nullptr;
  ggml_tensor *talpha = nullptr;
  ggml_tensor *tbeta = nullptr;
  ggml_tensor *tdt = nullptr;
  ggml_tensor *tssma = nullptr;
  ggml_backend_buffer_t buf = nullptr;
  bool ext_qkv = false, ext_gate = false, ext_alpha = false, ext_beta = false;
  for (int attempt = 0; attempt < 2 && !buf; ++attempt) {
    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead() * 8;
    ip.no_alloc = true;
    ctx = ggml_init(ip);
    if (!ctx)
      return false;
    tqkv = ggml_new_tensor_2d(ctx, (ggml_type)qkv.type, D_, conv_dim_);
    tgate = ggml_new_tensor_2d(ctx, (ggml_type)gate.type, D_, value_dim_);
    talpha = ggml_new_tensor_2d(ctx, (ggml_type)alpha.type, D_, H_v_);
    tbeta = ggml_new_tensor_2d(ctx, (ggml_type)beta.type, D_, H_v_);
    tdt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H_v_);
    tssma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H_v_);
    if (!tqkv || !tgate || !talpha || !tbeta || !tdt || !tssma) {
      ggml_free(ctx);
      return false;
    }
    ext_qkv = external_buf_ && qkv.device_data &&
              qkv.device_bytes >= ggml_nbytes(tqkv);
    ext_gate = external_buf_ && gate.device_data &&
               gate.device_bytes >= ggml_nbytes(tgate);
    ext_alpha = external_buf_ && alpha.device_data &&
                alpha.device_bytes >= ggml_nbytes(talpha);
    ext_beta = external_buf_ && beta.device_data &&
               beta.device_bytes >= ggml_nbytes(tbeta);
    auto bind_external = [&](ggml_tensor *t, const void *data) {
      t->data = const_cast<void *>(data);
      t->buffer = (ggml_backend_buffer_t)external_buf_;
      ggml_backend_buffer_init_tensor(t->buffer, t);
    };
    if (ext_qkv)
      bind_external(tqkv, qkv.device_data);
    if (ext_gate)
      bind_external(tgate, gate.device_data);
    if (ext_alpha)
      bind_external(talpha, alpha.device_data);
    if (ext_beta)
      bind_external(tbeta, beta.device_data);
    buf = ggml_backend_alloc_ctx_tensors(ctx, (ggml_backend_t)backend_);
    if (buf)
      break;
    ggml_free(ctx);
    ctx = nullptr;
    if (attempt == 0) {
      LOG_WARN("[GGML-GDN-PROJ-MEM] layer=%d allocation pressure; "
               "reclaiming completed GDN layers and retrying\n",
               layer);
      release_completed_weights(layer);
    }
  }
  if (!buf) {
    LOG_ERROR("[GGML-GDN-PROJ] weight alloc failed for layer=%d\n", layer);
    return false;
  }
  if (!ext_qkv)
    ggml_backend_tensor_set(tqkv, qkv.host_data, 0,
                            std::min(qkv.nbytes, ggml_nbytes(tqkv)));
  if (!ext_gate)
    ggml_backend_tensor_set(tgate, gate.host_data, 0,
                            std::min(gate.nbytes, ggml_nbytes(tgate)));
  if (!ext_alpha)
    ggml_backend_tensor_set(talpha, alpha.host_data, 0,
                            std::min(alpha.nbytes, ggml_nbytes(talpha)));
  if (!ext_beta)
    ggml_backend_tensor_set(tbeta, beta.host_data, 0,
                            std::min(beta.nbytes, ggml_nbytes(tbeta)));
  ggml_backend_tensor_set(tdt, dt_bias_host, 0, (size_t)H_v_ * sizeof(float));
  ggml_backend_tensor_set(tssma, ssm_a_host, 0, (size_t)H_v_ * sizeof(float));
  b.ctx = ctx;
  b.buf = buf;
  b.w_qkv = tqkv;
  b.w_gate = tgate;
  b.w_alpha = talpha;
  b.w_beta = tbeta;
  b.dt_bias = tdt;
  b.ssm_a = tssma;
  b.ready = true;
  const size_t weight_bytes = ggml_nbytes(tqkv) + ggml_nbytes(tgate) +
                              ggml_nbytes(talpha) + ggml_nbytes(tbeta) +
                              ggml_nbytes(tdt) + ggml_nbytes(tssma);
  LOG_INFO("[GGML-GDN-PROJ-MEM] weights: layer=%d %.2f MiB (%s)\n", layer,
           (double)weight_bytes / (1024.0 * 1024.0),
           (ext_qkv || ext_gate || ext_alpha || ext_beta)
               ? "borrowed dense pool"
               : "private buffer");
  return true;
}

bool GgmlGdnProjRunner::run(int layer, const float *hidden_states, int S,
                            float *qkv_out, float *gate_out, float *g_log_out,
                            float *beta_out) {
  if (!backend_ || layer < 0 || layer >= kMaxLayers || !hidden_states ||
      !qkv_out || !gate_out || !g_log_out || !beta_out)
    return false;
  const Bracket &b = brackets_[layer];
  if (!b.ready)
    return false;
  cudaSetDevice(device_);

  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx)
    return false;
  ggml_cgraph *gf = ggml_new_graph(ctx);
  if (!gf) {
    ggml_free(ctx);
    return false;
  }

  ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_, S);
  ggml_set_input(x);
  ggml_set_name(x, "gdn_proj_x");

  ggml_tensor *qkv = ggml_mul_mat(ctx, (ggml_tensor *)b.w_qkv, x);
  ggml_tensor *gate = ggml_mul_mat(ctx, (ggml_tensor *)b.w_gate, x);
  ggml_tensor *alpha_raw = ggml_mul_mat(ctx, (ggml_tensor *)b.w_alpha, x);
  ggml_tensor *beta_raw = ggml_mul_mat(ctx, (ggml_tensor *)b.w_beta, x);

  // decay_log = softplus(alpha_raw + dt_bias) * ssm_a -- dt_bias/ssm_a are
  // [H_v], broadcasting over the S axis (ggml_can_repeat requires
  // alpha_raw->ne[i] % bias->ne[i] == 0, true here: ne1=S % 1 == 0).
  ggml_tensor *alpha_biased =
      ggml_add(ctx, alpha_raw, (ggml_tensor *)b.dt_bias);
  ggml_tensor *sp = ggml_softplus(ctx, alpha_biased);
  ggml_tensor *g_log = ggml_mul(ctx, sp, (ggml_tensor *)b.ssm_a);
  ggml_tensor *beta = ggml_sigmoid(ctx, beta_raw);

  ggml_set_output(qkv);
  ggml_set_output(gate);
  ggml_set_output(g_log);
  ggml_set_output(beta);
  ggml_build_forward_expand(gf, qkv);
  ggml_build_forward_expand(gf, gate);
  ggml_build_forward_expand(gf, g_log);
  ggml_build_forward_expand(gf, beta);

  if (!ggml_gallocr_alloc_graph((ggml_gallocr_t)galloc_, gf)) {
    LOG_ERROR("[GGML-GDN-PROJ] graph alloc failed (layer=%d S=%d)\n", layer, S);
    ggml_free(ctx);
    return false;
  }
  const size_t graph_bytes =
      ggml_gallocr_get_buffer_size((ggml_gallocr_t)galloc_, 0);
  if (graph_bytes != graph_workspace_bytes_) {
    graph_workspace_bytes_ = graph_bytes;
    LOG_INFO("[GGML-GDN-PROJ-MEM] workspace: %.2f MiB (S=%d)\n",
             (double)graph_bytes / (1024.0 * 1024.0), S);
  }
  ggml_backend_tensor_set(x, hidden_states, 0, (size_t)D_ * S * sizeof(float));

  const ggml_status st =
      ggml_backend_graph_compute((ggml_backend_t)backend_, gf);
  if (st != GGML_STATUS_SUCCESS) {
    LOG_ERROR("[GGML-GDN-PROJ] graph compute failed (status=%d)\n", (int)st);
    ggml_free(ctx);
    return false;
  }

  // Each output is [O, S] row-major-per-column (ggml's own column-major
  // convention over ne0), i.e. contiguous per token in the SAME [S, O]
  // layout the CPU callers already expect from dense_proj (row-major [S,O]
  // with row stride O) -- ggml's ne0-fastest layout IS that row layout when
  // read back as a flat buffer, so no transpose is needed either direction.
  ggml_backend_tensor_get(qkv, qkv_out, 0,
                          (size_t)conv_dim_ * S * sizeof(float));
  ggml_backend_tensor_get(gate, gate_out, 0,
                          (size_t)value_dim_ * S * sizeof(float));
  ggml_backend_tensor_get(g_log, g_log_out, 0,
                          (size_t)H_v_ * S * sizeof(float));
  ggml_backend_tensor_get(beta, beta_out, 0, (size_t)H_v_ * S * sizeof(float));

  ggml_free(ctx);
  return true;
}

bool GgmlGdnProjRunner::conv_weight_ready(int layer) const {
  if (layer < 0 || layer >= kMaxLayers)
    return false;
  return brackets_[layer].conv_ready;
}

bool GgmlGdnProjRunner::ensure_conv_weight(int layer,
                                           const GgmlGdnProjWeightRef &conv_w) {
  if (!backend_ || layer < 0 || layer >= kMaxLayers)
    return false;
  Bracket &b = brackets_[layer];
  if (b.conv_ready)
    return true;
  if (!conv_w.host_data || conv_w.type != (uint32_t)GGML_TYPE_F32)
    return false;
  // REAL, hard constraint found via Test 29, not documented in ggml_ssm_conv's
  // own header: ggml-cuda's ssm_conv kernel (ggml-cuda/ssm-conv.cu) hardcodes
  // a 128-thread block and asserts `nr % 128 == 0` where nr is the CHANNEL
  // count (conv_dim_) -- NOT a graceful fallback, a fatal GGML_ASSERT that
  // would abort the whole process. qwen4exp's real conv_dim (2*key_dim+
  // value_dim, 10240 at this model's dims) IS a multiple of 128, but a
  // different qwen4exp-family checkpoint's dims might not be -- checked here,
  // not left to crash, so the caller's normal "declines -> fall back" path
  // handles it instead.
  if (conv_dim_ % 128 != 0) {
    LOG_WARN("[GGML-GDN-PROJ] conv_dim=%d is not a multiple of 128 -- "
             "ggml-cuda's ssm_conv kernel requires this and would abort "
             "(GGML_ASSERT), not fail gracefully. Declining conv fusion for "
             "this model; the 4-projection fusion and/or the separate GPU "
             "conv1d path are unaffected.\n",
             conv_dim_);
    return false;
  }
  // ne0=d_conv (kernel taps), ne1=conv_dim_ (channels) -- ggml_ssm_conv's own
  // `c` parameter convention. Any real d_conv works (checked against conv_dim_
  // only, since d_conv itself isn't fixed at ensure() time).
  if (conv_w.ne1 != conv_dim_ || conv_w.ne0 <= 0) {
    LOG_ERROR(
        "[GGML-GDN-PROJ] layer=%d unexpected conv weight shape (%lldx%lld, "
        "expected ne1=%d)\n",
        layer, (long long)conv_w.ne0, (long long)conv_w.ne1, conv_dim_);
    return false;
  }
  cudaSetDevice(device_);
  ggml_context *ctx = nullptr;
  ggml_tensor *tconv = nullptr;
  ggml_backend_buffer_t buf = nullptr;
  for (int attempt = 0; attempt < 2 && !buf; ++attempt) {
    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead() * 4;
    ip.no_alloc = true;
    ctx = ggml_init(ip);
    if (!ctx)
      return false;
    tconv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, conv_w.ne0, conv_dim_);
    if (!tconv) {
      ggml_free(ctx);
      return false;
    }
    buf = ggml_backend_alloc_ctx_tensors(ctx, (ggml_backend_t)backend_);
    if (buf)
      break;
    ggml_free(ctx);
    ctx = nullptr;
    if (attempt == 0) {
      LOG_WARN("[GGML-GDN-PROJ-MEM] layer=%d conv allocation pressure; "
               "reclaiming completed GDN layers and retrying\n",
               layer);
      release_completed_weights(layer);
    }
  }
  if (!buf) {
    LOG_ERROR("[GGML-GDN-PROJ] conv weight alloc failed for layer=%d\n", layer);
    return false;
  }
  ggml_backend_tensor_set(tconv, conv_w.host_data, 0,
                          std::min(conv_w.nbytes, ggml_nbytes(tconv)));
  b.conv_ctx = ctx;
  b.conv_buf = buf;
  b.conv_w = tconv;
  b.conv_ready = true;
  LOG_INFO("[GGML-GDN-PROJ-MEM] conv weight: layer=%d %.2f MiB\n", layer,
           (double)ggml_backend_buffer_get_size(buf) / (1024.0 * 1024.0));
  return true;
}

// See the header's own comment for the full rationale. Layout derivation,
// confirmed against Llama.cpp's own build_conv_state_at/build_layer_attn_linear
// (src/models/qwen4exp.cpp) rather than reverse-engineered from ggml_ssm_conv's
// assert()s alone:
//   - qkv (this graph's own mul_mat result) is ne=[conv_dim_, S] -- channels
//     fastest, matching this codebase's [S,O] row-major convention exactly
//     (see run()'s own comment). ggml_ssm_conv's `sx` wants the OPPOSITE
//     orientation (ne0=time, ne1=channels), so a ggml_transpose (a view, no
//     copy) is required going in.
//   - history (host, [conv_dim_, hist] row-major) is byte-identical to a
//     ggml ne=[hist, conv_dim_, 1] tensor: row-major (c,j) is at offset
//     c*hist+j; ggml ne0-fastest (j,c) is at offset c*hist+j. Same formula,
//     so it uploads as-is, no transform, exactly like the conv weight does.
//   - ggml_concat(history, transpose(qkv), dim=0) prepends history along
//     the time axis, giving sx=[hist+S, conv_dim_, 1] -- exactly
//     ggml_ssm_conv's required shape.
//   - ggml_ssm_conv's OUTPUT comes back as ne=[conv_dim_, S, 1] -- the SAME
//     orientation as qkv itself, so the final download needs no transpose
//     either; only the intermediate `sx` does.
//   - The new history is the LAST `hist` time-positions of the same
//     concatenated sx (not re-derived from qkv separately) -- a strided
//     view (ggml_view_3d) is made genuinely contiguous via ggml_cont before
//     downloading, since ggml_backend_tensor_get only copies a flat byte
//     range and can't itself skip strides.
bool GgmlGdnProjRunner::run_with_conv(int layer, const float *hidden_states,
                                      int S, float *conv_out, float *gate_out,
                                      float *g_log_out, float *beta_out,
                                      float *history, int d_conv) {
  if (!backend_ || layer < 0 || layer >= kMaxLayers || !hidden_states ||
      !conv_out || !gate_out || !g_log_out || !beta_out || !history)
    return false;
  if (d_conv <= 1)
    return false;
  const int hist = d_conv - 1;
  const Bracket &b = brackets_[layer];
  if (!b.ready || !b.conv_ready)
    return false;
  cudaSetDevice(device_);

  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 48 + ggml_graph_overhead();
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx)
    return false;
  ggml_cgraph *gf = ggml_new_graph(ctx);
  if (!gf) {
    ggml_free(ctx);
    return false;
  }

  ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_, S);
  ggml_set_input(x);
  ggml_set_name(x, "gdn_proj_x");

  ggml_tensor *qkv = ggml_mul_mat(ctx, (ggml_tensor *)b.w_qkv, x);
  ggml_tensor *gate = ggml_mul_mat(ctx, (ggml_tensor *)b.w_gate, x);
  ggml_tensor *alpha_raw = ggml_mul_mat(ctx, (ggml_tensor *)b.w_alpha, x);
  ggml_tensor *beta_raw = ggml_mul_mat(ctx, (ggml_tensor *)b.w_beta, x);

  ggml_tensor *alpha_biased =
      ggml_add(ctx, alpha_raw, (ggml_tensor *)b.dt_bias);
  ggml_tensor *sp = ggml_softplus(ctx, alpha_biased);
  ggml_tensor *g_log = ggml_mul(ctx, sp, (ggml_tensor *)b.ssm_a);
  ggml_tensor *beta = ggml_sigmoid(ctx, beta_raw);

  ggml_tensor *hist_in =
      ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hist, conv_dim_, 1);
  ggml_set_input(hist_in);
  ggml_set_name(hist_in, "gdn_conv_hist_in");

  ggml_tensor *conv_input =
      ggml_concat(ctx, hist_in, ggml_transpose(ctx, qkv), 0);
  ggml_tensor *conv_raw =
      ggml_ssm_conv(ctx, conv_input, (ggml_tensor *)b.conv_w);
  ggml_tensor *conv_silu = ggml_silu(ctx, conv_raw);

  ggml_tensor *tail_view = ggml_view_3d(
      ctx, conv_input, hist, conv_dim_, 1, conv_input->nb[1], conv_input->nb[2],
      ggml_row_size(conv_input->type, conv_input->ne[0] - hist));
  ggml_tensor *tail_cont = ggml_cont(ctx, tail_view);

  ggml_set_output(conv_silu);
  ggml_set_output(gate);
  ggml_set_output(g_log);
  ggml_set_output(beta);
  ggml_set_output(tail_cont);
  ggml_build_forward_expand(gf, conv_silu);
  ggml_build_forward_expand(gf, gate);
  ggml_build_forward_expand(gf, g_log);
  ggml_build_forward_expand(gf, beta);
  ggml_build_forward_expand(gf, tail_cont);

  if (!ggml_gallocr_alloc_graph((ggml_gallocr_t)galloc_, gf)) {
    LOG_ERROR("[GGML-GDN-PROJ] fused-conv graph alloc failed (layer=%d S=%d)\n",
              layer, S);
    ggml_free(ctx);
    return false;
  }
  const size_t graph_bytes =
      ggml_gallocr_get_buffer_size((ggml_gallocr_t)galloc_, 0);
  if (graph_bytes != graph_workspace_bytes_) {
    graph_workspace_bytes_ = graph_bytes;
    LOG_INFO("[GGML-GDN-PROJ-MEM] fused-conv workspace: %.2f MiB (S=%d)\n",
             (double)graph_bytes / (1024.0 * 1024.0), S);
  }
  ggml_backend_tensor_set(x, hidden_states, 0, (size_t)D_ * S * sizeof(float));
  ggml_backend_tensor_set(hist_in, history, 0,
                          (size_t)conv_dim_ * hist * sizeof(float));

  const ggml_status st =
      ggml_backend_graph_compute((ggml_backend_t)backend_, gf);
  if (st != GGML_STATUS_SUCCESS) {
    LOG_ERROR("[GGML-GDN-PROJ] fused-conv graph compute failed (status=%d)\n",
              (int)st);
    ggml_free(ctx);
    return false;
  }

  ggml_backend_tensor_get(conv_silu, conv_out, 0,
                          (size_t)conv_dim_ * S * sizeof(float));
  ggml_backend_tensor_get(gate, gate_out, 0,
                          (size_t)value_dim_ * S * sizeof(float));
  ggml_backend_tensor_get(g_log, g_log_out, 0,
                          (size_t)H_v_ * S * sizeof(float));
  ggml_backend_tensor_get(beta, beta_out, 0, (size_t)H_v_ * S * sizeof(float));
  // Small (conv_dim_*hist floats, e.g. ~123 KB at qwen4exp's real dims)
  // update, in place -- the only "download" this path pays beyond the
  // outputs it would have needed anyway.
  ggml_backend_tensor_get(tail_cont, history, 0,
                          (size_t)conv_dim_ * hist * sizeof(float));

  ggml_free(ctx);
  return true;
}

bool GgmlGdnProjRunner::run_full(int layer, const float *hidden_states, int S,
                                 float *out_context, float *history, int d_conv,
                                 float *state, float eps) {
  if (!backend_ || layer < 0 || layer >= kMaxLayers || !hidden_states ||
      !out_context || !history || !state || S <= 0 || d_conv <= 1)
    return false;
  const Bracket &b = brackets_[layer];
  if (!b.ready || !b.conv_ready || !b.full_ready || H_v_ <= 0 ||
      value_dim_ % H_v_ != 0)
    return false;

  const int hist = d_conv - 1;
  const int hd = value_dim_ / H_v_;
  // Qwen4Exp's model metadata gives 16 query/key heads and 48 value heads.
  // The GGML GDN CUDA kernel supports this grouped layout directly, using
  // q/k head h % H_k, so no materialized repeat is needed here.
  constexpr int kQwen4ExpKeyHeads = 16;
  if (conv_dim_ <= 2 * hd || conv_dim_ % hd != 0 ||
      (conv_dim_ - value_dim_) % (2 * hd) != 0)
    return false;
  const int key_dim = (conv_dim_ - value_dim_) / 2;
  if (key_dim <= 0 || key_dim % hd != 0 || key_dim / hd != kQwen4ExpKeyHeads)
    return false;

  cudaSetDevice(device_);
  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 96 + ggml_graph_overhead();
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx)
    return false;
  ggml_cgraph *gf = ggml_new_graph(ctx);
  if (!gf) {
    ggml_free(ctx);
    return false;
  }

  ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_, S);
  ggml_tensor *hist_in =
      ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hist, conv_dim_, 1);
  ggml_tensor *state_in =
      ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, hd, H_v_, 1);
  if (!x || !hist_in || !state_in) {
    ggml_free(ctx);
    return false;
  }
  ggml_set_input(x);
  ggml_set_input(hist_in);
  ggml_set_input(state_in);
  ggml_set_name(x, "gdn_full_x");
  ggml_set_name(hist_in, "gdn_full_conv_hist_in");
  ggml_set_name(state_in, "gdn_full_state_in");

  ggml_tensor *qkv = ggml_mul_mat(ctx, (ggml_tensor *)b.w_qkv, x);
  ggml_tensor *z = ggml_mul_mat(ctx, (ggml_tensor *)b.w_gate, x);
  ggml_tensor *alpha_raw = ggml_mul_mat(ctx, (ggml_tensor *)b.w_alpha, x);
  ggml_tensor *beta_raw = ggml_mul_mat(ctx, (ggml_tensor *)b.w_beta, x);
  if (!qkv || !z || !alpha_raw || !beta_raw) {
    ggml_free(ctx);
    return false;
  }
  ggml_tensor *alpha_biased =
      ggml_add(ctx, alpha_raw, (ggml_tensor *)b.dt_bias);
  ggml_tensor *g =
      ggml_mul(ctx, ggml_softplus(ctx, alpha_biased), (ggml_tensor *)b.ssm_a);
  ggml_tensor *beta = ggml_sigmoid(ctx, beta_raw);

  // concat() materializes the transposed qkv into the contiguous
  // [time,channels] layout required by ggml_ssm_conv's CUDA kernel.
  ggml_tensor *conv_input =
      ggml_concat(ctx, hist_in, ggml_transpose(ctx, qkv), 0);
  ggml_tensor *conv_raw =
      ggml_ssm_conv(ctx, conv_input, (ggml_tensor *)b.conv_w);
  ggml_tensor *conv_silu = ggml_silu(ctx, conv_raw);

  const size_t qkv_row = ggml_row_size(GGML_TYPE_F32, conv_dim_);
  ggml_tensor *q =
      ggml_view_4d(ctx, conv_silu, hd, key_dim / hd, S, 1,
                   ggml_row_size(GGML_TYPE_F32, hd), qkv_row, qkv_row * S, 0);
  ggml_tensor *k = ggml_view_4d(
      ctx, conv_silu, hd, key_dim / hd, S, 1, ggml_row_size(GGML_TYPE_F32, hd),
      qkv_row, qkv_row * S, ggml_row_size(GGML_TYPE_F32, key_dim));
  ggml_tensor *v = ggml_view_4d(
      ctx, conv_silu, hd, H_v_, S, 1, ggml_row_size(GGML_TYPE_F32, hd), qkv_row,
      qkv_row * S, ggml_row_size(GGML_TYPE_F32, 2 * key_dim));
  q = ggml_l2_norm(ctx, q, eps);
  k = ggml_l2_norm(ctx, k, eps);

  g = ggml_reshape_4d(ctx, g, 1, H_v_, S, 1);
  beta = ggml_reshape_4d(ctx, beta, 1, H_v_, S, 1);
  ggml_tensor *gdn = ggml_gated_delta_net(ctx, q, k, v, g, beta, state_in, 1);
  if (!gdn) {
    ggml_free(ctx);
    return false;
  }

  const size_t hd_row = ggml_row_size(GGML_TYPE_F32, hd);
  const size_t value_row = ggml_row_size(GGML_TYPE_F32, value_dim_);
  const size_t state_row = ggml_row_size(GGML_TYPE_F32, hd * hd);
  const size_t score_bytes = (size_t)value_dim_ * S * sizeof(float);
  ggml_tensor *core = ggml_view_4d(ctx, gdn, hd, H_v_, S, 1, hd_row, value_row,
                                   value_row * S, 0);
  ggml_tensor *new_state =
      ggml_view_4d(ctx, gdn, hd, hd, H_v_, 1, hd_row, state_row,
                   state_row * H_v_, score_bytes);

  // build_norm_gated() in Llama.cpp is RMSNorm(ssm_norm) * sigmoid(z).
  ggml_tensor *normed = ggml_rms_norm(ctx, core, eps);
  normed = ggml_mul(ctx, normed, (ggml_tensor *)b.ssm_norm);
  ggml_tensor *z4 = ggml_reshape_4d(ctx, z, hd, H_v_, S, 1);
  ggml_tensor *gated = ggml_mul(ctx, normed, ggml_sigmoid(ctx, z4));
  ggml_tensor *final_input = ggml_reshape_2d(ctx, gated, value_dim_, S);
  ggml_tensor *final_output =
      ggml_mul_mat(ctx, (ggml_tensor *)b.ssm_out, final_input);

  ggml_tensor *tail_view = ggml_view_3d(
      ctx, conv_input, hist, conv_dim_, 1, conv_input->nb[1], conv_input->nb[2],
      ggml_row_size(conv_input->type, conv_input->ne[0] - hist));
  ggml_tensor *tail_cont = ggml_cont(ctx, tail_view);
  if (!final_output || !tail_cont) {
    ggml_free(ctx);
    return false;
  }

  ggml_set_output(final_output);
  ggml_set_output(new_state);
  ggml_set_output(tail_cont);
  ggml_build_forward_expand(gf, final_output);
  ggml_build_forward_expand(gf, new_state);
  ggml_build_forward_expand(gf, tail_cont);

  if (!ggml_gallocr_alloc_graph((ggml_gallocr_t)galloc_, gf)) {
    LOG_ERROR("[GGML-GDN-FULL] graph alloc failed (layer=%d S=%d)\n", layer, S);
    ggml_free(ctx);
    return false;
  }
  const size_t graph_bytes =
      ggml_gallocr_get_buffer_size((ggml_gallocr_t)galloc_, 0);
  if (graph_bytes != graph_workspace_bytes_) {
    graph_workspace_bytes_ = graph_bytes;
    LOG_INFO("[GGML-GDN-FULL-MEM] workspace: %.2f MiB (S=%d)\n",
             (double)graph_bytes / (1024.0 * 1024.0), S);
  }

  const size_t x_bytes = (size_t)D_ * S * sizeof(float);
  const size_t history_bytes = (size_t)conv_dim_ * hist * sizeof(float);
  const size_t state_bytes = (size_t)H_v_ * hd * hd * sizeof(float);
  // Keep the host-output path ordered, but let GGML queue the three input
  // uploads on its copy stream.  The caller does not mutate these buffers
  // until this function returns, so one backend barrier below is sufficient
  // to cover their lifetimes.
  ggml_backend_tensor_set_async((ggml_backend_t)backend_, x, hidden_states, 0,
                                x_bytes);
  ggml_backend_tensor_set_async((ggml_backend_t)backend_, hist_in, history, 0,
                                history_bytes);
  ggml_backend_tensor_set_async((ggml_backend_t)backend_, state_in, state, 0,
                                state_bytes);

  const ggml_status st =
      ggml_backend_graph_compute_async((ggml_backend_t)backend_, gf);
  if (st != GGML_STATUS_SUCCESS) {
    LOG_ERROR("[GGML-GDN-FULL] graph compute failed (status=%d)\n", (int)st);
    ggml_free(ctx);
    return false;
  }
  // Queue all three outputs before waiting.  The old synchronous gets could
  // each force a separate CUDA stream wait/copy; one barrier preserves the
  // host-output contract while removing those redundant CPU stalls.
  ggml_backend_tensor_get_async((ggml_backend_t)backend_, final_output,
                                out_context, 0, x_bytes);
  ggml_backend_tensor_get_async((ggml_backend_t)backend_, new_state, state, 0,
                                state_bytes);
  ggml_backend_tensor_get_async((ggml_backend_t)backend_, tail_cont, history, 0,
                                history_bytes);
  ggml_backend_synchronize((ggml_backend_t)backend_);
  ggml_free(ctx);
  return true;
}

bool GgmlGdnProjRunner::run_full_device_impl(
    int layer, const void *device_hidden, const void *device_hidden_tensor,
    int S, const void **device_output, float *history, int d_conv, float *state,
    float eps, bool device_state_resident) {
  if (device_output)
    *device_output = nullptr;
  if (!backend_ || layer < 0 || layer >= kMaxLayers ||
      (!device_hidden && !device_hidden_tensor) || !device_output || !history ||
      !state || S <= 0 || d_conv <= 1)
    return false;
  const Bracket &b = brackets_[layer];
  if (!b.ready || !b.conv_ready || !b.full_ready || H_v_ <= 0 ||
      value_dim_ % H_v_ != 0 || !ensure_full_output(layer, S))
    return false;

  const int hist = d_conv - 1;
  const int hd = value_dim_ / H_v_;
  constexpr int kQwen4ExpKeyHeads = 16;
  if (conv_dim_ <= 2 * hd || conv_dim_ % hd != 0 ||
      (conv_dim_ - value_dim_) % (2 * hd) != 0)
    return false;
  const int key_dim = (conv_dim_ - value_dim_) / 2;
  if (key_dim <= 0 || key_dim % hd != 0 || key_dim / hd != kQwen4ExpKeyHeads)
    return false;
  if (!ensure_full_state(layer, d_conv, hd))
    return false;
  FullState &full_state = full_states_[layer];
  if (device_state_resident) {
    static bool logged_decode_state = false;
    if (!logged_decode_state) {
      logged_decode_state = true;
      LOG_INFO(
          "[GGML-GDN-FULL] decode recurrent state/history remain on CUDA\n");
    }
  }
  if (full_state.pending) {
    if (device_state_resident) {
      // The previous decode token's state update is ordered before this graph
      // on the shared backend stream.  It must not be downloaded merely to
      // make the next token visible to the host.
      full_state.pending = false;
      full_state.device_state_resident = true;
    } else {
      finish_full_device_updates();
    }
  }
  if (!full_state.output || !full_device_slots_[0].galloc)
    return false;

  cudaSetDevice(device_);
  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 96 + ggml_graph_overhead();
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx)
    return false;
  ggml_cgraph *gf = ggml_new_graph(ctx);
  if (!gf) {
    ggml_free(ctx);
    return false;
  }

  ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_, S);
  ggml_tensor *hist_in =
      device_state_resident
          ? (ggml_tensor *)full_state.history
          : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hist, conv_dim_, 1);
  ggml_tensor *state_in =
      device_state_resident
          ? (ggml_tensor *)full_state.state
          : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, hd, H_v_, 1);
  if (!x || !hist_in || !state_in) {
    ggml_free(ctx);
    return false;
  }
  ggml_set_input(x);
  if (!device_state_resident) {
    ggml_set_input(hist_in);
    ggml_set_input(state_in);
  }

  ggml_tensor *qkv = ggml_mul_mat(ctx, (ggml_tensor *)b.w_qkv, x);
  ggml_tensor *z = ggml_mul_mat(ctx, (ggml_tensor *)b.w_gate, x);
  ggml_tensor *alpha_raw = ggml_mul_mat(ctx, (ggml_tensor *)b.w_alpha, x);
  ggml_tensor *beta_raw = ggml_mul_mat(ctx, (ggml_tensor *)b.w_beta, x);
  if (!qkv || !z || !alpha_raw || !beta_raw) {
    ggml_free(ctx);
    return false;
  }
  ggml_tensor *alpha_biased =
      ggml_add(ctx, alpha_raw, (ggml_tensor *)b.dt_bias);
  ggml_tensor *g =
      ggml_mul(ctx, ggml_softplus(ctx, alpha_biased), (ggml_tensor *)b.ssm_a);
  ggml_tensor *beta = ggml_sigmoid(ctx, beta_raw);
  ggml_tensor *conv_input =
      ggml_concat(ctx, hist_in, ggml_transpose(ctx, qkv), 0);
  ggml_tensor *conv_raw =
      ggml_ssm_conv(ctx, conv_input, (ggml_tensor *)b.conv_w);
  ggml_tensor *conv_silu = ggml_silu(ctx, conv_raw);
  const size_t qkv_row = ggml_row_size(GGML_TYPE_F32, conv_dim_);
  ggml_tensor *q =
      ggml_view_4d(ctx, conv_silu, hd, key_dim / hd, S, 1,
                   ggml_row_size(GGML_TYPE_F32, hd), qkv_row, qkv_row * S, 0);
  ggml_tensor *k = ggml_view_4d(
      ctx, conv_silu, hd, key_dim / hd, S, 1, ggml_row_size(GGML_TYPE_F32, hd),
      qkv_row, qkv_row * S, ggml_row_size(GGML_TYPE_F32, key_dim));
  ggml_tensor *v = ggml_view_4d(
      ctx, conv_silu, hd, H_v_, S, 1, ggml_row_size(GGML_TYPE_F32, hd), qkv_row,
      qkv_row * S, ggml_row_size(GGML_TYPE_F32, 2 * key_dim));
  q = ggml_l2_norm(ctx, q, eps);
  k = ggml_l2_norm(ctx, k, eps);
  g = ggml_reshape_4d(ctx, g, 1, H_v_, S, 1);
  beta = ggml_reshape_4d(ctx, beta, 1, H_v_, S, 1);
  ggml_tensor *gdn = ggml_gated_delta_net(ctx, q, k, v, g, beta, state_in, 1);
  if (!gdn) {
    ggml_free(ctx);
    return false;
  }
  const size_t hd_row = ggml_row_size(GGML_TYPE_F32, hd);
  const size_t value_row = ggml_row_size(GGML_TYPE_F32, value_dim_);
  const size_t state_row = ggml_row_size(GGML_TYPE_F32, hd * hd);
  const size_t score_bytes = (size_t)value_dim_ * S * sizeof(float);
  ggml_tensor *core = ggml_view_4d(ctx, gdn, hd, H_v_, S, 1, hd_row, value_row,
                                   value_row * S, 0);
  ggml_tensor *new_state =
      ggml_view_4d(ctx, gdn, hd, hd, H_v_, 1, hd_row, state_row,
                   state_row * H_v_, score_bytes);
  ggml_tensor *normed = ggml_rms_norm(ctx, core, eps);
  normed = ggml_mul(ctx, normed, (ggml_tensor *)b.ssm_norm);
  ggml_tensor *z4 = ggml_reshape_4d(ctx, z, hd, H_v_, S, 1);
  ggml_tensor *gated = ggml_mul(ctx, normed, ggml_sigmoid(ctx, z4));
  ggml_tensor *final_input = ggml_reshape_2d(ctx, gated, value_dim_, S);
  ggml_tensor *final_output =
      ggml_mul_mat(ctx, (ggml_tensor *)b.ssm_out, final_input);
  ggml_tensor *tail_view = ggml_view_3d(
      ctx, conv_input, hist, conv_dim_, 1, conv_input->nb[1], conv_input->nb[2],
      ggml_row_size(conv_input->type, conv_input->ne[0] - hist));
  ggml_tensor *tail_cont = ggml_cont(ctx, tail_view);
  ggml_tensor *state_store =
      ggml_cpy(ctx, new_state, (ggml_tensor *)full_state.state);
  ggml_tensor *history_store =
      ggml_cpy(ctx, tail_cont, (ggml_tensor *)full_state.history);
  // Each layer owns its output buffer.  A single shared output used to be
  // overwritten by the next GDN graph before HC's post-combine graph had
  // consumed it, which is the source of the old asynchronous NaNs.
  ggml_tensor *out_view =
      ggml_view_2d(ctx, (ggml_tensor *)full_state.output, D_, S,
                   ((ggml_tensor *)full_state.output)->nb[1], 0);
  ggml_tensor *out_store = ggml_cpy(ctx, final_output, out_view);
  if (!final_output || !tail_cont || !state_store || !history_store ||
      !out_store) {
    ggml_free(ctx);
    return false;
  }
  ggml_set_output(out_store);
  ggml_set_output(state_store);
  ggml_set_output(history_store);
  ggml_build_forward_expand(gf, out_store);
  ggml_build_forward_expand(gf, state_store);
  ggml_build_forward_expand(gf, history_store);
  FullDeviceSlot &slot = full_device_slots_[layer % kFullDeviceSlots];
  if (slot.in_flight) {
    // This is the only host wait in the ring. With eight slots it can happen
    // at most once per eight GDN layers, allowing the device pipeline to fill
    // instead of draining at every layer boundary.
    if (debug_device_sync_full_gdn()) {
      const cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) {
        LOG_ERROR("[GGML-GDN-FULL] diagnostic device synchronize failed: %s\n",
                  cudaGetErrorString(e));
        ggml_free(ctx);
        return false;
      }
    } else {
      ggml_backend_event_synchronize((ggml_backend_event_t)slot.done_event);
    }
    slot.in_flight = false;
  }
  if (!slot.galloc ||
      !ggml_gallocr_alloc_graph((ggml_gallocr_t)slot.galloc, gf)) {
    LOG_ERROR("[GGML-GDN-FULL] device graph alloc failed (layer=%d S=%d)\n",
              layer, S);
    ggml_free(ctx);
    return false;
  }
  const size_t graph_bytes =
      ggml_gallocr_get_buffer_size((ggml_gallocr_t)slot.galloc, 0);
  if (graph_bytes != full_device_workspace_bytes_) {
    full_device_workspace_bytes_ = graph_bytes;
    full_state.workspace_bytes = graph_bytes;
    LOG_INFO("[GGML-GDN-FULL-MEM] workspace: %.2f MiB (S=%d)\n",
             (double)graph_bytes / (1024.0 * 1024.0), S);
  }

  const size_t x_bytes = (size_t)D_ * S * sizeof(float);
  if (device_hidden_tensor) {
    ggml_tensor *src = (ggml_tensor *)device_hidden_tensor;
    if (src->type != GGML_TYPE_F32 || src->ne[0] != D_ || src->ne[1] < S ||
        src->ne[2] != 1 || src->ne[3] != 1 || src->nb[0] != sizeof(float) ||
        src->nb[1] != (size_t)D_ * sizeof(float) || !src->buffer ||
        !src->data) {
      LOG_ERROR("[GGML-GDN-FULL] device tensor input has unexpected layout\n");
      ggml_free(ctx);
      return false;
    }
    ggml_tensor *src_view =
        src->ne[1] == S ? src : ggml_view_2d(ctx, src, D_, S, src->nb[1], 0);
    if (!src_view) {
      ggml_free(ctx);
      return false;
    }
    ggml_backend_tensor_copy_async((ggml_backend_t)backend_,
                                   (ggml_backend_t)backend_, src_view, x);
  } else if (cudaMemcpy(x->data, device_hidden, x_bytes,
                        cudaMemcpyDeviceToDevice) != cudaSuccess) {
    ggml_free(ctx);
    return false;
  }
  if (!device_state_resident) {
    ggml_backend_tensor_set_async((ggml_backend_t)backend_, hist_in, history, 0,
                                  (size_t)conv_dim_ * hist * sizeof(float));
    ggml_backend_tensor_set_async((ggml_backend_t)backend_, state_in, state, 0,
                                  (size_t)H_v_ * hd * hd * sizeof(float));
  }
  const ggml_status st =
      ggml_backend_graph_compute_async((ggml_backend_t)backend_, gf);
  if (st != GGML_STATUS_SUCCESS) {
    LOG_ERROR("[GGML-GDN-FULL] device graph compute failed (status=%d)\n",
              (int)st);
    ggml_free(ctx);
    return false;
  }
  ggml_backend_event_record((ggml_backend_event_t)slot.done_event,
                            (ggml_backend_t)backend_);
  slot.in_flight = true;
  if (debug_sync_full_gdn()) {
    ggml_backend_synchronize((ggml_backend_t)backend_);
    slot.in_flight = false;
  }
  if (debug_device_sync_full_gdn()) {
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      LOG_ERROR("[GGML-GDN-FULL] diagnostic device synchronize failed: %s\n",
                cudaGetErrorString(e));
      ggml_free(ctx);
      return false;
    }
    slot.in_flight = false;
  }
  full_state.host_history = history;
  full_state.host_state = state;
  full_state.pending = true;
  full_state.device_state_resident = device_state_resident;
  *device_output = full_state.output;
  ggml_free(ctx);
  return true;
}

bool GgmlGdnProjRunner::run_full_device(int layer, const void *device_hidden,
                                        int S, const void **device_output,
                                        float *history, int d_conv,
                                        float *state, float eps,
                                        bool device_state_resident) {
  return run_full_device_impl(layer, device_hidden, nullptr, S, device_output,
                              history, d_conv, state, eps,
                              device_state_resident);
}

bool GgmlGdnProjRunner::run_full_device_tensor(
    int layer, const void *device_hidden_tensor, int S,
    const void **device_output, float *history, int d_conv, float *state,
    float eps, bool device_state_resident) {
  return run_full_device_impl(layer, nullptr, device_hidden_tensor, S,
                              device_output, history, d_conv, state, eps,
                              device_state_resident);
}

bool GgmlGdnProjRunner::copy_full_output_to_host(const void *device_output,
                                                 float *host, int S) {
  if (!backend_ || !device_output || !host || S <= 0)
    return false;
  ggml_tensor *out = nullptr;
  if (t_full_output_ &&
      (device_output == t_full_output_ ||
       device_output == ((ggml_tensor *)t_full_output_)->data) &&
      S <= full_output_cap_) {
    out = (ggml_tensor *)t_full_output_;
  }
  for (const FullState &state : full_states_) {
    if (!state.output || S > state.output_cap)
      continue;
    if (device_output == state.output ||
        device_output == ((ggml_tensor *)state.output)->data) {
      out = (ggml_tensor *)state.output;
      break;
    }
  }
  if (!out)
    return false;
  cudaSetDevice(device_);
  ggml_backend_tensor_get(out, host, 0, (size_t)D_ * S * sizeof(float));
  return true;
}

#else // !MOESTREAM_USE_GGML_CUDA

bool GgmlGdnProjRunner::compiled_in() { return false; }
GgmlGdnProjRunner &GgmlGdnProjRunner::instance() {
  static GgmlGdnProjRunner r;
  return r;
}
GgmlGdnProjRunner::~GgmlGdnProjRunner() {}
void GgmlGdnProjRunner::shutdown() {}
bool GgmlGdnProjRunner::ensure(int, int, int, int, int) { return false; }
bool GgmlGdnProjRunner::weights_ready(int) const { return false; }
bool GgmlGdnProjRunner::full_weights_ready(int) const { return false; }
bool GgmlGdnProjRunner::ensure_weights(int, const GgmlGdnProjWeightRef &,
                                       const GgmlGdnProjWeightRef &,
                                       const GgmlGdnProjWeightRef &,
                                       const GgmlGdnProjWeightRef &,
                                       const float *, const float *) {
  return false;
}
bool GgmlGdnProjRunner::ensure_full_weights(int, const GgmlGdnProjWeightRef &,
                                            const GgmlGdnProjWeightRef &) {
  return false;
}
bool GgmlGdnProjRunner::run(int, const float *, int, float *, float *, float *,
                            float *) {
  return false;
}
bool GgmlGdnProjRunner::conv_weight_ready(int) const { return false; }
bool GgmlGdnProjRunner::ensure_conv_weight(int, const GgmlGdnProjWeightRef &) {
  return false;
}
bool GgmlGdnProjRunner::run_with_conv(int, const float *, int, float *, float *,
                                      float *, float *, float *, int) {
  return false;
}
bool GgmlGdnProjRunner::run_full(int, const float *, int, float *, float *, int,
                                 float *, float) {
  return false;
}
bool GgmlGdnProjRunner::run_full_device(int, const void *, int, const void **,
                                        float *, int, float *, float, bool) {
  return false;
}
bool GgmlGdnProjRunner::run_full_device_tensor(int, const void *, int,
                                               const void **, float *, int,
                                               float *, float, bool) {
  return false;
}
void GgmlGdnProjRunner::finish_full_device_updates(bool) {}
void GgmlGdnProjRunner::sync_full_device_states_to_host() {}
void GgmlGdnProjRunner::reset_full_device_states() {}
bool GgmlGdnProjRunner::copy_full_output_to_host(const void *, float *, int) {
  return false;
}

#endif
