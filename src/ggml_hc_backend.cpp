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
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

namespace {
constexpr int kMaxLayers = 256;
bool async_graph_reuse_override = false;

bool type_ok(uint32_t t) {
  if (t >= (uint32_t)GGML_TYPE_COUNT) return false;
  const ggml_type g = (ggml_type)t;
  return g == GGML_TYPE_F32 || g == GGML_TYPE_F16 || g == GGML_TYPE_BF16 ||
         ggml_is_quantized(g);
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
  return enabled;
}

bool debug_check_hc_state() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOESTREAM_GGML_DEBUG_CHECK_HC_STATE");
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
  pre_source_layer_ = -1;
  pre_source_sub_block_ = -1;
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
  pre_in_flight_ = false;
  post_in_flight_ = false;
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
  if (!post_done_event_) {
    if (post_done_event_) ggml_backend_event_free((ggml_backend_event_t)post_done_event_);
    post_done_event_ = nullptr;
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

void GgmlHcRunner::free_pre_graph() {
  if (pre_ctx_) {
    ggml_free((ggml_context *)pre_ctx_);
    pre_ctx_ = nullptr;
  }
  pre_graph_ = nullptr;
  pre_norm_ = nullptr;
  pre_down_ = nullptr;
  pre_up_ = nullptr;
  pre_inject_ = nullptr;
  pre_graph_S_ = 0;
  pre_down_type_ = 0;
  pre_up_type_ = 0;
  pre_inject_type_ = 0;
}

void GgmlHcRunner::release_completed_weights(int before_layer) {
  if (!backend_ || before_layer <= 0) return;

  // The persistent pre graph keeps tensor metadata from its first bracket as
  // stable source nodes and retargets only their data pointers.  Leave that
  // tiny source bracket allocated; all other completed-layer buffers can be
  // returned to CUDA after the one barrier below.
  ggml_backend_synchronize((ggml_backend_t)backend_);
  size_t reclaimed = 0;
  int brackets = 0;
  for (int sb = 0; sb < 2; ++sb) {
    for (int l = 0; l < before_layer; ++l) {
      if (l == pre_source_layer_ && sb == pre_source_sub_block_) continue;
      Bracket &b = brackets_[sb][l];
      if (!b.buf && !b.ctx) continue;
      if (b.buf) {
        reclaimed += ggml_backend_buffer_get_size(
            (ggml_backend_buffer_t)b.buf);
        ggml_backend_buffer_free((ggml_backend_buffer_t)b.buf);
      }
      if (b.ctx) ggml_free((ggml_context *)b.ctx);
      b = Bracket{};
      ++brackets;
    }
  }
  if (brackets > 0) {
    LOG_WARN("[GGML-HC-MEM] allocation-pressure reclaim: released %d "
             "completed-layer brackets (%.2f MiB), keeping source layer=%d "
             "sub=%d\n",
             brackets, (double)reclaimed / (1024.0 * 1024.0),
             pre_source_layer_, pre_source_sub_block_);
  }
}

bool GgmlHcRunner::ensure_pre_graph(int S, const Bracket &b, int layer,
                                    int sub_block) {
  if (!backend_ || !b.ready || !b.norm || !b.down || !b.up || !b.inject ||
      S <= 0 || S > s_cap_)
    return false;

  const ggml_tensor *bn = (const ggml_tensor *)b.norm;
  const ggml_tensor *bd = (const ggml_tensor *)b.down;
  const ggml_tensor *bu = (const ggml_tensor *)b.up;
  const ggml_tensor *bi = (const ggml_tensor *)b.inject;
  const bool matches =
      pre_ctx_ && pre_graph_ && pre_graph_S_ == S &&
      pre_down_type_ == (uint32_t)bd->type &&
      pre_up_type_ == (uint32_t)bu->type &&
      pre_inject_type_ == (uint32_t)bi->type;

  if (!matches) {
    // A shape/type change is a request-boundary event in normal use.  The
    // old graph owns temporary addresses from galloc_pre_, so drain the
    // shared CUDA stream before releasing it and asking the allocator for a
    // new plan.
    if (pre_ctx_) {
      ggml_backend_synchronize((ggml_backend_t)backend_);
      free_pre_graph();
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

    // Keep the first layer's tensor metadata as stable graph sources.  On
    // later calls bind_pre_weights() changes only their CUDA data pointers;
    // all layers have the same HC shapes and the type check above prevents a
    // quantization-kernel mismatch from being hidden.
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
    pre_ctx_ = ctx;
    pre_graph_ = gf;
    pre_norm_ = b.norm;
    pre_down_ = b.down;
    pre_up_ = b.up;
    pre_inject_ = b.inject;
    pre_graph_S_ = S;
    pre_down_type_ = (uint32_t)bd->type;
    pre_up_type_ = (uint32_t)bu->type;
    pre_inject_type_ = (uint32_t)bi->type;
    pre_source_layer_ = layer;
    pre_source_sub_block_ = sub_block;
  }

  // The graph is already queued or complete when this call returns.  CUDA
  // kernels receive the current data pointers as launch arguments, so it is
  // safe to retarget the stable source metadata before the next submission.
  ((ggml_tensor *)pre_norm_)->data = bn->data;
  ((ggml_tensor *)pre_down_)->data = bd->data;
  ((ggml_tensor *)pre_up_)->data = bu->data;
  ((ggml_tensor *)pre_inject_)->data = bi->data;
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
  if (b.ready) return true;
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
  cudaSetDevice(device_);
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
    td = ggml_new_tensor_2d(ctx, (ggml_type)down.type, hc_dim_, hc_lr_);
    tu = ggml_new_tensor_2d(ctx, (ggml_type)up.type, hc_lr_, hc_dim_);
    ti = ggml_new_tensor_2d(ctx, (ggml_type)inject.type, hc_dim_, hc_);
    if (!tn || !td || !tu || !ti) {
      ggml_free(ctx);
      return false;
    }
    ext_n = external_buf_ && norm.device_data &&
            norm.device_bytes >= ggml_nbytes(tn);
    ext_d = external_buf_ && down.device_data &&
            down.device_bytes >= ggml_nbytes(td);
    ext_u = external_buf_ && up.device_data &&
            up.device_bytes >= ggml_nbytes(tu);
    ext_i = external_buf_ && inject.device_data &&
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
  const bool uses_external = norm.device_data && down.device_data &&
                             up.device_data && inject.device_data &&
                             external_buf_;
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
    ggml_backend_tensor_set(td, down.host_data, 0,
                            std::min(down.nbytes, ggml_nbytes(td)));
  if (!ext_u)
    ggml_backend_tensor_set(tu, up.host_data, 0,
                            std::min(up.nbytes, ggml_nbytes(tu)));
  if (!ext_i)
    ggml_backend_tensor_set(ti, inject.host_data, 0,
                            std::min(inject.nbytes, ggml_nbytes(ti)));
  b.ctx = ctx;
  b.buf = buf;
  b.norm = tn;
  b.down = td;
  b.up = tu;
  b.inject = ti;
  b.ready = true;
  const size_t weight_bytes = ggml_nbytes(tn) + ggml_nbytes(td) +
                              ggml_nbytes(tu) + ggml_nbytes(ti);
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
  if (!ensure_pre_graph(S, b, layer, sub_block)) return false;
  // The persistent graph is allocated once per request shape.  Its stable
  // source metadata has just been rebound to this layer's HC weights, so
  // submitting it does not reset/reuse a temporary allocator while the prior
  // layer is still queued.
  ggml_cgraph *gf = (ggml_cgraph *)pre_graph_;
  const ggml_status st =
      ggml_backend_graph_compute_async((ggml_backend_t)backend_, gf);
  if (st != GGML_STATUS_SUCCESS) {
    LOG_ERROR("[GGML-HC] pre_mix compute failed (status=%d)\n", (int)st);
    return false;
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
  ggml_backend_event_record((ggml_backend_event_t)active_pre_done_event_,
                            (ggml_backend_t)backend_);
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
    // tensor_get synchronizes GGML's copy stream, not the shared compute
    // stream on which pre_mix was queued.
    ggml_backend_synchronize((ggml_backend_t)backend_);
    ggml_backend_tensor_get((ggml_tensor *)t_mixed_, out_hidden, 0,
                            (size_t)S * d_model_ * sizeof(float));
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
    const cudaError_t copy_status = cudaMemcpy(
        so_flat->data, device_sub_block_out, bytes, cudaMemcpyDeviceToDevice);
    if (copy_status != cudaSuccess) {
      LOG_ERROR("[GGML-HC] post_combine raw device copy failed (layer=%d "
                "sub=%d S=%d): %s\n",
                layer, sub_block, S, cudaGetErrorString(copy_status));
      return false;
    }
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

bool GgmlHcRunner::compiled_in() { return false; }
GgmlHcRunner &GgmlHcRunner::instance() {
  static GgmlHcRunner r;
  return r;
}
GgmlHcRunner::~GgmlHcRunner() {}
void GgmlHcRunner::shutdown() {}
bool GgmlHcRunner::ensure(int, int, int, int, float) { return false; }
bool GgmlHcRunner::ensure_state(int) { return false; }
bool GgmlHcRunner::weights_ready(int, int) const { return false; }
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
