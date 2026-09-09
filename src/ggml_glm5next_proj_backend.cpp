#include "ggml_glm5next_proj_backend.hpp"

#include "ggml_moe_backend.hpp"  // ggml_shared_backend()
#include "logger.hpp"

#include <algorithm>
#include <cstring>
#include <iterator>

#if defined(MOESTREAM_USE_GGML_CUDA)

#include <cuda_runtime.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

namespace {
bool type_ok(uint32_t type) {
  if (type >= (uint32_t)GGML_TYPE_COUNT) return false;
  const ggml_type t = (ggml_type)type;
  return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 ||
         ggml_is_quantized(t);
}
}

bool GgmlGlm5nextKdaProjRunner::compiled_in() { return true; }

GgmlGlm5nextKdaProjRunner &GgmlGlm5nextKdaProjRunner::instance() {
  static GgmlGlm5nextKdaProjRunner runner;
  return runner;
}

GgmlGlm5nextKdaProjRunner::~GgmlGlm5nextKdaProjRunner() { shutdown(); }

void GgmlGlm5nextKdaProjRunner::destroy_graph() {
  if (backend_)
    ggml_backend_synchronize((ggml_backend_t)backend_);
  if (graph_ctx_) ggml_free((ggml_context *)graph_ctx_);
  graph_ctx_ = nullptr;
  graph_ = nullptr;
  x_ = nullptr;
  std::fill(std::begin(outputs_), std::end(outputs_), nullptr);
  graph_ready_ = false;
}

void GgmlGlm5nextKdaProjRunner::shutdown() {
  if (device_ >= 0) cudaSetDevice(device_);
  destroy_graph();
  if (weight_buf_) {
    ggml_backend_buffer_free((ggml_backend_buffer_t)weight_buf_);
    weight_buf_ = nullptr;
  }
  if (weight_ctx_) {
    ggml_free((ggml_context *)weight_ctx_);
    weight_ctx_ = nullptr;
  }
  std::fill(std::begin(weights_), std::end(weights_), nullptr);
  if (galloc_) {
    ggml_gallocr_free((ggml_gallocr_t)galloc_);
    galloc_ = nullptr;
  }
  backend_ = nullptr;  // shared, not owned
  std::fill(std::begin(weight_types_), std::end(weight_types_), 0);
  std::fill(std::begin(weight_ne0_), std::end(weight_ne0_), 0);
  std::fill(std::begin(weight_ne1_), std::end(weight_ne1_), 0);
  device_ = -1;
  D_ = d_inner_ = hd_ = H_ = S_ = 0;
  current_layer_ = -1;
  weights_ready_ = false;
}

bool GgmlGlm5nextKdaProjRunner::ensure(int device, int D, int d_inner, int hd,
                                       int H, int S) {
  if (device < 0 || D <= 0 || d_inner <= 0 || hd <= 0 || H <= 0 || S <= 1)
    return false;
  if (backend_ && device_ == device && D_ == D && d_inner_ == d_inner &&
      hd_ == hd && H_ == H) {
    if (S_ != S) {
      destroy_graph();
      S_ = S;
    }
    return true;
  }

  if (backend_) shutdown();
  cudaSetDevice(device);
  backend_ = ggml_shared_backend(device);
  if (!backend_) return false;
  galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(
      (ggml_backend_t)backend_));
  if (!galloc_) {
    backend_ = nullptr;
    return false;
  }
  device_ = device;
  D_ = D;
  d_inner_ = d_inner;
  hd_ = hd;
  H_ = H;
  S_ = S;
  LOG_INFO("[GLM5NEXT-DENSE] KDA projection graph available on GPU %d "
           "(D=%d d_inner=%d hd=%d H=%d S=%d)\n",
           device, D, d_inner, hd, H, S);
  return true;
}

bool GgmlGlm5nextKdaProjRunner::weight_signature_matches(
    const GgmlGlm5nextProjWeightRef refs[kWeightCount]) const {
  for (int i = 0; i < kWeightCount; ++i) {
    if (weight_types_[i] != refs[i].type || weight_ne0_[i] != refs[i].ne0 ||
        weight_ne1_[i] != refs[i].ne1)
      return false;
  }
  return true;
}

bool GgmlGlm5nextKdaProjRunner::allocate_weights(
    const GgmlGlm5nextProjWeightRef refs[kWeightCount]) {
  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * kWeightCount;
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx) return false;

  for (int i = 0; i < kWeightCount; ++i) {
    weights_[i] = ggml_new_tensor_2d(ctx, (ggml_type)refs[i].type,
                                     refs[i].ne0, refs[i].ne1);
    if (!weights_[i]) {
      ggml_free(ctx);
      return false;
    }
  }
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(
      ctx, (ggml_backend_t)backend_);
  if (!buf) {
    ggml_free(ctx);
    return false;
  }
  weight_ctx_ = ctx;
  weight_buf_ = buf;
  for (int i = 0; i < kWeightCount; ++i) {
    weight_types_[i] = refs[i].type;
    weight_ne0_[i] = refs[i].ne0;
    weight_ne1_[i] = refs[i].ne1;
  }
  return true;
}

bool GgmlGlm5nextKdaProjRunner::build_graph() {
  if (!backend_ || !galloc_ || !weights_ready_ || S_ <= 1) return false;
  destroy_graph();

  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx) return false;
  ggml_cgraph *graph = ggml_new_graph(ctx);
  ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_, S_);
  if (!graph || !x) {
    ggml_free(ctx);
    return false;
  }
  ggml_set_input(x);
  ggml_set_name(x, "glm5next_kda_proj_x");

  // Weight order: q, k, v, f_a, f_b, beta, g_a, g_b.
  ggml_tensor *q = ggml_mul_mat(ctx, (ggml_tensor *)weights_[0], x);
  ggml_tensor *k = ggml_mul_mat(ctx, (ggml_tensor *)weights_[1], x);
  ggml_tensor *v = ggml_mul_mat(ctx, (ggml_tensor *)weights_[2], x);
  ggml_tensor *fa = ggml_mul_mat(ctx, (ggml_tensor *)weights_[3], x);
  ggml_tensor *fb = ggml_mul_mat(ctx, (ggml_tensor *)weights_[4], fa);
  ggml_tensor *beta = ggml_mul_mat(ctx, (ggml_tensor *)weights_[5], x);
  ggml_tensor *ga = ggml_mul_mat(ctx, (ggml_tensor *)weights_[6], x);
  ggml_tensor *gb = ggml_mul_mat(ctx, (ggml_tensor *)weights_[7], ga);
  if (!q || !k || !v || !fa || !fb || !beta || !ga || !gb) {
    LOG_WARN("[GLM5NEXT-DENSE] KDA projection graph node creation failed "
             "(S=%d) -- using legacy dense dispatch\n",
             S_);
    ggml_free(ctx);
    return false;
  }
  ggml_tensor *outs[6] = {q, k, v, fb, beta, gb};
  for (ggml_tensor *out : outs) {
    if (!out) {
      ggml_free(ctx);
      return false;
    }
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);
  }

  if (!ggml_gallocr_alloc_graph((ggml_gallocr_t)galloc_, graph)) {
    LOG_WARN("[GLM5NEXT-DENSE] KDA projection graph allocation failed "
             "(S=%d) -- using legacy dense dispatch\n",
             S_);
    ggml_free(ctx);
    return false;
  }
  graph_ctx_ = ctx;
  graph_ = graph;
  x_ = x;
  for (int i = 0; i < 6; ++i) outputs_[i] = outs[i];
  graph_ready_ = true;
  LOG_INFO("[GLM5NEXT-DENSE] KDA projection fusion active: 8 dense "
           "projections -> 1 CUDA graph (S=%d)\n",
           S_);
  return true;
}

bool GgmlGlm5nextKdaProjRunner::ensure_layer(
    int layer, const GgmlGlm5nextProjWeightRef &q,
    const GgmlGlm5nextProjWeightRef &k, const GgmlGlm5nextProjWeightRef &v,
    const GgmlGlm5nextProjWeightRef &f_a,
    const GgmlGlm5nextProjWeightRef &f_b,
    const GgmlGlm5nextProjWeightRef &beta,
    const GgmlGlm5nextProjWeightRef &g_a,
    const GgmlGlm5nextProjWeightRef &g_b) {
  if (!backend_ || layer < 0 || !q.host_data || !k.host_data || !v.host_data ||
      !f_a.host_data || !f_b.host_data || !beta.host_data || !g_a.host_data ||
      !g_b.host_data)
    return false;
  const GgmlGlm5nextProjWeightRef refs[kWeightCount] = {
      q, k, v, f_a, f_b, beta, g_a, g_b};
  for (int i = 0; i < kWeightCount; ++i) {
    if (!type_ok(refs[i].type) || refs[i].ne0 <= 0 || refs[i].ne1 <= 0)
      return false;
  }
  if (q.ne0 != D_ || q.ne1 != d_inner_ || k.ne0 != D_ || k.ne1 != d_inner_ ||
      v.ne0 != D_ || v.ne1 != d_inner_ || f_a.ne0 != D_ || f_a.ne1 != hd_ ||
      f_b.ne0 != hd_ || f_b.ne1 != d_inner_ || beta.ne0 != D_ ||
      beta.ne1 != H_ || g_a.ne0 != D_ || g_a.ne1 != hd_ || g_b.ne0 != hd_ ||
      g_b.ne1 != d_inner_)
    return false;

  cudaSetDevice(device_);
  if (weights_ready_ && current_layer_ == layer &&
      weight_signature_matches(refs))
    return graph_ready_ || build_graph();

  if (!weights_ready_ || !weight_signature_matches(refs)) {
    destroy_graph();
    if (weight_buf_) ggml_backend_buffer_free((ggml_backend_buffer_t)weight_buf_);
    if (weight_ctx_) ggml_free((ggml_context *)weight_ctx_);
    weight_buf_ = nullptr;
    weight_ctx_ = nullptr;
    std::fill(std::begin(weights_), std::end(weights_), nullptr);
    weights_ready_ = false;
    if (!allocate_weights(refs)) {
      LOG_WARN("[GLM5NEXT-DENSE] layer=%d KDA weight allocation failed -- "
               "using legacy dense dispatch\n",
               layer);
      return false;
    }
  } else {
    // The graph is synchronous today, but the barrier makes this safe if the
    // shared backend later grows an asynchronous graph mode.
    ggml_backend_synchronize((ggml_backend_t)backend_);
  }

  for (int i = 0; i < kWeightCount; ++i) {
    ggml_tensor *t = (ggml_tensor *)weights_[i];
    const size_t bytes = ggml_nbytes(t);
    if (refs[i].nbytes < bytes) {
      weights_ready_ = false;
      current_layer_ = -1;
      LOG_WARN("[GLM5NEXT-DENSE] layer=%d weight %d is truncated "
               "(%zu < %zu bytes) -- using legacy dense dispatch\n",
               layer, i, refs[i].nbytes, bytes);
      return false;
    }
    ggml_backend_tensor_set(t, refs[i].host_data, 0, bytes);
  }
  weights_ready_ = true;
  current_layer_ = layer;
  return build_graph();
}

bool GgmlGlm5nextKdaProjRunner::run(
    int layer, const float *hidden_states, int S, float *q_out, float *k_out,
    float *v_out, float *f_b_out, float *beta_out, float *g_b_out) {
  if (!backend_ || !graph_ready_ || !weights_ready_ || layer != current_layer_ ||
      !hidden_states || !q_out || !k_out || !v_out || !f_b_out || !beta_out ||
      !g_b_out || S != S_)
    return false;
  cudaSetDevice(device_);
  ggml_backend_tensor_set((ggml_tensor *)x_, hidden_states, 0,
                          (size_t)D_ * S_ * sizeof(float));
  const ggml_status st = ggml_backend_graph_compute(
      (ggml_backend_t)backend_, (ggml_cgraph *)graph_);
  if (st != GGML_STATUS_SUCCESS) {
    LOG_WARN("[GLM5NEXT-DENSE] KDA projection graph compute failed "
             "(layer=%d status=%d) -- using legacy dense dispatch\n",
             layer, (int)st);
    return false;
  }
  const size_t inner_bytes = (size_t)d_inner_ * S_ * sizeof(float);
  const size_t beta_bytes = (size_t)H_ * S_ * sizeof(float);
  ggml_backend_tensor_get((ggml_tensor *)outputs_[0], q_out, 0, inner_bytes);
  ggml_backend_tensor_get((ggml_tensor *)outputs_[1], k_out, 0, inner_bytes);
  ggml_backend_tensor_get((ggml_tensor *)outputs_[2], v_out, 0, inner_bytes);
  ggml_backend_tensor_get((ggml_tensor *)outputs_[3], f_b_out, 0, inner_bytes);
  ggml_backend_tensor_get((ggml_tensor *)outputs_[4], beta_out, 0, beta_bytes);
  ggml_backend_tensor_get((ggml_tensor *)outputs_[5], g_b_out, 0, inner_bytes);
  return true;
}

#else

bool GgmlGlm5nextKdaProjRunner::compiled_in() { return false; }
GgmlGlm5nextKdaProjRunner &GgmlGlm5nextKdaProjRunner::instance() {
  static GgmlGlm5nextKdaProjRunner runner;
  return runner;
}
GgmlGlm5nextKdaProjRunner::~GgmlGlm5nextKdaProjRunner() = default;
bool GgmlGlm5nextKdaProjRunner::ensure(int, int, int, int, int, int) {
  return false;
}
bool GgmlGlm5nextKdaProjRunner::ensure_layer(
    int, const GgmlGlm5nextProjWeightRef &, const GgmlGlm5nextProjWeightRef &,
    const GgmlGlm5nextProjWeightRef &, const GgmlGlm5nextProjWeightRef &,
    const GgmlGlm5nextProjWeightRef &, const GgmlGlm5nextProjWeightRef &,
    const GgmlGlm5nextProjWeightRef &, const GgmlGlm5nextProjWeightRef &) {
  return false;
}
bool GgmlGlm5nextKdaProjRunner::run(int, const float *, int, float *, float *,
                                    float *, float *, float *, float *) {
  return false;
}
void GgmlGlm5nextKdaProjRunner::shutdown() {}

#endif
