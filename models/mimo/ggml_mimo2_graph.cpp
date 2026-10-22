#include "ggml_mimo2_graph.hpp"
#include "aliased_expert_placement.hpp"
#include "cache_backed_moe.hpp"
#include "full_graph_model.hpp"
#include "ggml_expert_copy.hpp"
#include "moe_block.hpp"
#include "moe_routed_ggml.hpp"
#include "offload_expert_runtime.hpp"
#include "graph_core.hpp"
#include "graph_config.hpp"
#include "ggml-cuda.h"
#include "ggml_moe_backend.hpp"
#include "mimo2_spec.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#if defined(MOESTREAM_USE_GGML_CUDA)
namespace {
using namespace moestream::graph;
constexpr int kNodes = 16384;
constexpr size_t kContextBytes = 48ull << 20;

struct ModelState {
  StreamingEngine *engine = nullptr;
  ggml_backend_t cuda = nullptr; // borrowed from the shared backend
  ggml_context *ctx = nullptr;
  ggml_backend_buffer_t marker = nullptr;
  ggml_backend_buffer_t kv_buffer = nullptr;
  std::vector<ggml_backend_buffer_t> private_weights;
  std::unordered_map<std::string, ggml_tensor *> weights;
  std::vector<ggml_tensor *> keys, values;
  CacheBackedMoeRuntime runtime;
  // In-graph routed MoE (hot experts alias the VRAM cache, cold experts come
  // from the DDR/SSD offload tier). Unset -> the synchronous cache-backed callback.
  bool ingraph_moe = false;
  GpuTopology *topology = nullptr;
  AliasedExpertPlacement placement;
  OffloadExpertRuntime offload;
  GpuAssistConfig assist;
  ggml_backend_t gpu1 = nullptr;              // borrowed; second GPU's hot experts
  ggml_backend_buffer_t desc1 = nullptr;      // 1 MiB descriptor tagging aliased GPU1 tensors
  int capacity = 0;
  size_t reservation = 0;
  ~ModelState() {
    if (cuda) ggml_backend_synchronize(cuda);
    for (auto *b : private_weights) ggml_backend_buffer_free(b);
    if (desc1) ggml_backend_buffer_free(desc1);
    if (kv_buffer) ggml_backend_buffer_free(kv_buffer);
    if (marker) ggml_backend_buffer_free(marker);
    if (ctx) ggml_free(ctx);
    MemoryGuard::release_host_bytes(reservation);
  }
};
struct Graph;
struct MoeSite { Graph *graph = nullptr; int layer = -1; };
struct Graph : GraphCore {
  ModelState *model = nullptr;
  std::vector<MoeSite> sites;
  std::vector<int> token_ids;
  ggml_tensor *full_mask = nullptr, *swa_mask = nullptr;
  std::unordered_map<int, ggml_tensor *> indices;
  std::string callback_failure;
  moestream::ggmlcopy::ExpertCopy expert_copy; // selective host->GPU copies for op-offloaded prefill
  bool op_offload = false;
  int position = 0;
  bool owns_cpu = false;
  ~Graph() {
    if (sched) { ggml_backend_sched_synchronize(sched); ggml_backend_sched_free(sched); sched = nullptr; }
    early_ready.clear(); // lanes reference the offload store and their own CPU pools
    if (owns_cpu && cpu) ggml_backend_free(cpu);
    release_core();
  }
};

std::unique_ptr<ModelState> g_model;
std::unique_ptr<Graph> g_decode, g_prefill;
CacheBackedMoeRuntime g_runtime;
RAMSafetyManager *g_ram = nullptr;
GpuTopology *g_topology = nullptr;

void moe_node(ggml_tensor *dst, const ggml_tensor *src, int ith, int, void *ud) {
  if (ith) return;
  const auto &site = *static_cast<MoeSite *>(ud);
  auto &g = *site.graph;
  // Never let a C++ exception cross ggml's C/OpenMP callback boundary. Finish
  // the scheduled graph with zeros, then fail the request on its calling thread.
  try {
    if (!g.callback_failure.empty()) {
      if (dst && dst->data) std::memset(dst->data, 0, ggml_nbytes(dst));
      return;
    }
    if (!src || !dst || !src->data || !dst->data ||
        ggml_nelements(src) != int64_t(g.D) * g.S || ggml_nelements(dst) != int64_t(g.D) * g.S)
      throw std::runtime_error("MiMo cache-backed MoE tensor shape mismatch");
    g.model->runtime.run(site.layer, static_cast<const float *>(src->data), g.S,
                         static_cast<float *>(dst->data), g.token_ids.data());
  } catch (const std::exception &ex) {
    g.callback_failure = ex.what();
    if (dst && dst->data) std::memset(dst->data, 0, ggml_nbytes(dst));
  } catch (...) {
    g.callback_failure = "unknown native MoE callback failure";
    if (dst && dst->data) std::memset(dst->data, 0, ggml_nbytes(dst));
  }
}

ggml_tensor *weight(ModelState &m, const std::string &name) {
  const auto it = m.weights.find(name);
  return it == m.weights.end() ? nullptr : it->second;
}

// Environment -> plain profile. MiMo has no private DDR tier of its own: the
// shared provider builds it from the engine's expert geometry.
bool configure_ingraph_moe(StreamingEngine &e, ModelState &m) {
  namespace gc = moestream::graphcfg;
  if (!gc::flag_unless_off("MOESTREAM_MIMO2_INGRAPH_MOE")) return false;
  if (!g_topology) {
    LOG_ERROR("[MIMO2-GRAPH] in-graph MoE needs the VRAM expert cache (moe_cache_gb > 0)\n");
    throw std::runtime_error("no VRAM expert cache for in-graph MoE");
  }
  const auto &c = e.model_file.architecture_config;
  m.topology = g_topology;
  AliasedPlacementOptions opt;
  opt.device = e.gpu->get_core_device();
  const int dev1 = e.gpu->get_core_device() == 0 ? 1 : 0;
  m.gpu1 = gc::flag_unless_off("MOESTREAM_MIMO2_GPU1")
               ? static_cast<ggml_backend_t>(ggml_shared_backend(dev1)) : nullptr;
  if (m.gpu1) {
    m.desc1 = ggml_backend_alloc_buffer(m.gpu1, 1 << 20);
    if (!m.desc1) throw std::runtime_error("GPU1 descriptor buffer allocation failed");
    ggml_backend_buffer_set_usage(m.desc1, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  }
  opt.secondary_device = m.gpu1 ? dev1 : -1; // second GPU's cache is a second hot set
  if (!m.placement.configure(e, *m.topology, opt))
    throw std::runtime_error("expert placement configuration failed");
  if (gc::flag_unless_off("MOESTREAM_MIMO2_VERIFY_ALIAS") &&
      m.placement.verify_against_model(e) != 0)
    throw std::runtime_error("aliased VRAM experts differ from the model file");
  OffloadProfile prof;
  prof.first_layer = c.first_dense;
  prof.log_tag = "[MIMO2-OFFLOAD]";
  prof.model_shards = e.model_file.shard_paths;
  const std::string ddr = gc::env("MOESTREAM_MIMO2_OFFLOAD_DDR_GIB") ? gc::env("MOESTREAM_MIMO2_OFFLOAD_DDR_GIB") : "auto";
  prof.ddr_gib = ddr == "auto" ? auto_ddr_gib() : std::atof(ddr.c_str());
  prof.pinned_fraction = double(gc::integer("MOESTREAM_MIMO2_OFFLOAD_PINNED_PCT", 50)) / 100.0;
  prof.byte_capacity = true;
  prof.async_loads = true;
  const int assist_slots = int(gc::integer("MOESTREAM_MIMO2_GPU_ASSIST", 0));
  // Page-locked arena slots let the DMA engine read host experts directly: needed for the GPU
  // assist ring and for op-offloaded prefill copies. MOESTREAM_MIMO2_OFFLOAD_PIN=0 disables it.
  if (assist_slots > 0 || gc::flag_unless_off("MOESTREAM_MIMO2_OFFLOAD_PIN")) {
    prof.pin_for_dma = true;
    prof.pin_fn = [](void *ptr, size_t bytes) {
      if (cudaHostRegister(ptr, bytes, cudaHostRegisterPortable) == cudaSuccess) return true;
      cudaGetLastError();
      return false;
    };
    prof.unpin_fn = [](void *ptr) { cudaHostUnregister(ptr); };
  }
  SeedExclusions excl;
  excl.resident.assign(size_t(c.n_layers), {});
  excl.deferred.assign(size_t(c.n_layers), {});
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    const AliasedLayerPlacement *lp = m.placement.layer(l);
    excl.resident[size_t(l)].assign(size_t(c.n_experts), 0);
    for (int x : lp->primary.experts) { excl.resident[size_t(l)][size_t(x)] = 1; ++excl.excluded; }
    for (int x : lp->secondary.experts) { excl.resident[size_t(l)][size_t(x)] = 1; ++excl.excluded; }
  }
  const char *usage = gc::env("MOESTREAM_MIMO2_OFFLOAD_USAGE");
  std::string err;
  if (!m.offload.create(e, prof, usage ? usage : "", &excl, &err))
    throw std::runtime_error("offload tier: " + err);
  m.assist.slots = assist_slots;
  m.assist.device = e.gpu->get_core_device();
  m.assist.layer_ok = [](int) { return true; };
  LOG_INFO("[MIMO2-GRAPH] in-graph MoE: %zu + %zu VRAM-aliased experts (GPU0 + GPU1), DDR tier from layer %d\n",
           m.placement.primary_records(), m.placement.secondary_records(), c.first_dense);
  return true;
}

// A small constant graph input resident on the GPU (route maps / masks).
ggml_tensor *device_const(Graph &g, ggml_type type, int64_t n_experts,
                          const void *data, size_t bytes, const std::string &name) {
  ggml_tensor *t = ggml_new_tensor_3d(g.ctx, type, 1, n_experts, g.S);
  if (!t || ggml_nbytes(t) != bytes) throw std::runtime_error("route map tensor shape mismatch");
  ggml_backend_buffer_t b = ggml_backend_alloc_buffer(g.cuda, bytes);
  if (!b) throw std::runtime_error("route map buffer allocation failed");
  g.host_buffers.push_back(b);
  ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  if (ggml_backend_tensor_alloc(b, t, ggml_backend_buffer_get_base(b)) != GGML_STATUS_SUCCESS)
    throw std::runtime_error("route map tensor placement failed");
  ggml_backend_tensor_set(t, data, 0, bytes);
  ggml_set_name(t, name.c_str());
  return t;
}

void branch_maps(Graph &g, const moe_partial::ExpertBranchMap &bm, int layer,
                 const char *tag, MoEBranch &out) {
  const int64_t n = g.engine->model_file.architecture_config.n_experts;
  const std::string base = std::string("mimo_") + tag + "_";
  out.route_map = device_const(g, GGML_TYPE_I32, n, bm.route_ids().data(),
                               bm.route_ids().size() * sizeof(int32_t), base + "map_" + std::to_string(layer));
  out.route_mask = device_const(g, GGML_TYPE_F32, n, bm.route_mask().data(),
                                bm.route_mask().size() * sizeof(float), base + "mask_" + std::to_string(layer));
}

ggml_tensor *moe_ingraph(Graph &g, ggml_tensor *cur, int l);
ggml_tensor *on_gpu(Graph &g, ggml_tensor *t);
ggml_tensor *norm(Graph &g, ggml_tensor *x, const std::string &name);
ggml_tensor *weight(ModelState &m, const std::string &name);

bool create_state(StreamingEngine &e) {
  const auto &c = e.model_file.architecture_config;
  auto m = std::make_unique<ModelState>();
  m->engine = &e;
  const int max_heads = int(*std::max_element(c.mimo2_head_count_kv.begin(),
                                            c.mimo2_head_count_kv.end()));
  if (e.max_t <= 0 || e.max_t > std::numeric_limits<int>::max() - 255 ||
      int64_t((e.max_t + 255) / 256) * 256 * max_heads > std::numeric_limits<int32_t>::max())
    return false; // ggml_set_rows indices are I32, including the head offset.
  m->capacity = ((e.max_t + 255) / 256) * 256;
  m->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(e.gpu->get_core_device()));
  if (!m->cuda || m->capacity <= 0) return false;
  constexpr size_t meta = 4ull << 20;
  if (!MemoryGuard::try_reserve_host_bytes(meta)) return false;
  m->reservation = meta;
  m->ctx = ggml_init({meta, nullptr, true});
  if (!m->ctx) return false;
  size_t private_bytes = 0;
  for (const auto &[name, q] : e.resident_layers)
    if (q.host_ptr && !q.device_ptr && name.find("_exps.") == std::string::npos)
      private_bytes += size_t(q.meta.size_bytes) + 512;
  uint64_t kv_bytes = 0;
  for (int64_t h : c.mimo2_head_count_kv)
    kv_bytes += uint64_t(m->capacity) * uint64_t(h) * uint64_t(c.head_dim + c.v_head) * 2;
  size_t free_bytes = 0, total_bytes = 0;
  cudaSetDevice(e.gpu->get_core_device());
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      kv_bytes + private_bytes + (512ull << 20) > free_bytes) {
    LOG_ERROR("[MIMO2-GRAPH] KV/private weights need %.2f GiB plus workspace; reduce context/cache budgets\n",
               double(kv_bytes + private_bytes) / double(1ull << 30));
    return false;
  }
  // As in the existing native-cache graph, this small buffer tags non-owning
  // tensors with the CUDA backend; the engine owns their actual dense pool.
  m->marker = ggml_backend_alloc_buffer(m->cuda, 256);
  if (!m->marker) return false;
  ggml_backend_buffer_set_usage(m->marker, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  for (const auto &[name, q] : e.resident_layers) {
    if (!q.host_ptr || name.find("_exps.") != std::string::npos) continue;
    std::vector<int64_t> shape(q.meta.shape.begin(), q.meta.shape.end());
    auto *t = new_tensor(m->ctx, static_cast<ggml_type>(q.meta.type), shape);
    if (!t || ggml_nbytes(t) > q.meta.size_bytes) return false;
    ggml_set_name(t, name.c_str());
    if (q.device_ptr) {
      t->data = q.device_ptr;
      t->buffer = m->marker;
    } else {
      auto *b = ggml_backend_alloc_buffer(m->cuda, ggml_nbytes(t));
      if (!b) return false;
      m->private_weights.push_back(b);
      ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
      if (ggml_backend_tensor_alloc(b, t, ggml_backend_buffer_get_base(b)) != GGML_STATUS_SUCCESS)
        return false;
      ggml_backend_tensor_set(t, q.host_ptr, 0, ggml_nbytes(t));
    }
    m->weights.emplace(name, t);
  }
  // Allocate KV separately so allocating context tensors never duplicates borrowed weights.
  ggml_context *kvctx = ggml_init({meta, nullptr, true});
  if (!kvctx) return false;
  for (int l = 0; l < c.n_layers; ++l) {
    const int h = int(c.mimo2_head_count_kv[l]);
    m->keys.push_back(ggml_new_tensor_2d(kvctx, GGML_TYPE_F16, c.head_dim, int64_t(m->capacity) * h));
    m->values.push_back(ggml_new_tensor_2d(kvctx, GGML_TYPE_F16, c.v_head, int64_t(m->capacity) * h));
  }
  m->kv_buffer = ggml_backend_alloc_ctx_tensors(kvctx, m->cuda);
  if (!m->kv_buffer) { ggml_free(kvctx); return false; }
  ggml_backend_buffer_clear(m->kv_buffer, 0);
  // Copy the persistent metadata into the state context before releasing the temporary context.
  for (auto &t : m->keys) {
    auto *copy = ggml_dup_tensor(m->ctx, t); copy->data = t->data; copy->buffer = t->buffer; t = copy;
  }
  for (auto &t : m->values) {
    auto *copy = ggml_dup_tensor(m->ctx, t); copy->data = t->data; copy->buffer = t->buffer; t = copy;
  }
  ggml_free(kvctx);
  m->runtime = g_runtime;
  m->ingraph_moe = configure_ingraph_moe(e, *m);
  LOG_INFO("[MIMO2-GRAPH] weights=%zu KV=%.3f GiB capacity=%d; expert bytes owned by graph=0\n",
           m->weights.size(), ggml_backend_buffer_get_size(m->kv_buffer) / double(1ull << 30), m->capacity);
  g_model = std::move(m);
  return true;
}

ggml_tensor *on_gpu(Graph &g, ggml_tensor *t) {
  if (!t) throw std::runtime_error("MiMo graph tensor construction failed");
  ggml_backend_sched_set_tensor_backend(g.sched, t, g.cuda);
  return t;
}

ggml_tensor *norm(Graph &g, ggml_tensor *x, const std::string &name) {
  auto *w = weight(*g.model, name);
  if (!w) throw std::runtime_error("missing MiMo norm: " + name);
  return on_gpu(g, ggml_mul(g.ctx, ggml_rms_norm(g.ctx, x,
      g.engine->model_file.architecture_config.eps), w));
}

ggml_tensor *attention(Graph &g, ggml_tensor *x, int layer) {
  const auto &c = g.engine->model_file.architecture_config;
  const std::string p = "blk." + std::to_string(layer) + ".";
  const int h = int(c.mimo2_head_count_kv[layer]);
  const bool swa = c.mimo2_sliding_window_pattern[layer] != 0;
  auto *cur = norm(g, x, p + "attn_norm.weight");
  ggml_tensor *q, *k, *v;
  if (auto *w = weight(*g.model, p + "attn_qkv.weight")) {
    auto *qkv = on_gpu(g, ggml_mul_mat(g.ctx, w, cur));
    const size_t qbytes = size_t(c.head_dim) * c.n_heads * sizeof(float);
    const size_t kbytes = size_t(c.head_dim) * h * sizeof(float);
    q = ggml_view_3d(g.ctx, qkv, c.head_dim, c.n_heads, g.S,
                      size_t(c.head_dim) * sizeof(float), qkv->nb[1], 0);
    k = ggml_view_3d(g.ctx, qkv, c.head_dim, h, g.S,
                      size_t(c.head_dim) * sizeof(float), qkv->nb[1], qbytes);
    v = ggml_view_3d(g.ctx, qkv, c.v_head, h, g.S,
                      size_t(c.v_head) * sizeof(float), qkv->nb[1], qbytes + kbytes);
  } else {
    q = ggml_reshape_3d(g.ctx, ggml_mul_mat(g.ctx, weight(*g.model, p + "attn_q.weight"), cur), c.head_dim, c.n_heads, g.S);
    k = ggml_reshape_3d(g.ctx, ggml_mul_mat(g.ctx, weight(*g.model, p + "attn_k.weight"), cur), c.head_dim, h, g.S);
    v = ggml_reshape_3d(g.ctx, ggml_mul_mat(g.ctx, weight(*g.model, p + "attn_v.weight"), cur), c.v_head, h, g.S);
  }
  const float base = swa ? c.mimo2_rope_freq_base_swa :
      g.engine->model_file.get_config_float("rope.freq_base", 10000.0f);
  const int rotary = g.engine->model_file.get_config_int("rope.dimension_count", c.head_dim);
  q = on_gpu(g, ggml_rope_ext(g.ctx, q, g.positions, nullptr, rotary, GGML_ROPE_TYPE_NEOX,
      c.rope_n_ctx_orig, base, c.rope_freq_scale, c.rope_ext_factor, c.rope_attn_factor, 32.0f, 1.0f));
  k = on_gpu(g, ggml_rope_ext(g.ctx, k, g.positions, nullptr, rotary, GGML_ROPE_TYPE_NEOX,
      c.rope_n_ctx_orig, base, c.rope_freq_scale, c.rope_ext_factor, c.rope_attn_factor, 32.0f, 1.0f));
  auto *krows = ggml_reshape_2d(g.ctx, ggml_cont(g.ctx, k), c.head_dim, int64_t(h) * g.S);
  auto *vrows = ggml_reshape_2d(g.ctx, ggml_cont(g.ctx, v), c.v_head, int64_t(h) * g.S);
  auto *ks = on_gpu(g, ggml_set_rows(g.ctx, g.model->keys[layer], krows, g.indices.at(h)));
  auto *vs = on_gpu(g, ggml_set_rows(g.ctx, g.model->values[layer], vrows, g.indices.at(h)));
  ggml_build_forward_expand(g.graph, ks);
  ggml_build_forward_expand(g.graph, vs);
  auto *kc = ggml_reshape_4d(g.ctx, ks, c.head_dim, g.kv_capacity, h, 1);
  auto *vc = ggml_reshape_4d(g.ctx, vs, c.v_head, g.kv_capacity, h, 1);
  q = ggml_permute(g.ctx, q, 0, 2, 1, 3);
  auto *attn = ggml_flash_attn_ext(g.ctx, q, kc, vc, swa ? g.swa_mask : g.full_mask,
                                  1.0f / std::sqrt(float(c.head_dim)), 0.0f, 0.0f);
  if (auto *sinks = weight(*g.model, p + "attn_sinks.weight"))
    ggml_flash_attn_ext_add_sinks(attn, sinks);
  if (!ggml_backend_supports_op(g.cuda, attn))
    throw std::runtime_error("linked ggml CUDA does not support MiMo attention (K=192,V=128,sinks)");
  on_gpu(g, attn);
  auto *out = on_gpu(g, ggml_mul_mat(g.ctx, weight(*g.model, p + "attn_output.weight"),
                                    ggml_reshape_2d(g.ctx, attn, int64_t(c.v_head) * c.n_heads, g.S)));
  if (c.mimo2_value_scale != 0.0f && c.mimo2_value_scale != 1.0f)
    out = on_gpu(g, ggml_scale(g.ctx, out, c.mimo2_value_scale));
  return out;
}

ggml_tensor *moe_ingraph(Graph &g, ggml_tensor *cur, int l) {
  ModelState &m = *g.model;
  const auto &c = g.engine->model_file.architecture_config;
  const std::string p = "blk." + std::to_string(l) + ".";
  auto *x = norm(g, cur, p + "ffn_norm.weight");
  auto *router = weight(m, p + "ffn_gate_inp.weight");
  if (!router) throw std::runtime_error("missing MiMo router: " + p + "ffn_gate_inp.weight");
  auto *bias = weight(m, p + "exp_probs_b.bias");
  const auto rspec = family_router_spec(ModelFamily::MiMo2, c, g.S);
  auto *probs = on_gpu(g, moe_routed::gate_probs(g.ctx, rspec, router, x));
  auto *ids = moe_routed::select_ids(g.ctx, rspec, probs, bias);
  auto *weights = moe_routed::route_weights(g.ctx, rspec, probs, ids);
  ggml_build_forward_expand(g.graph, weights);

  const AliasedLayerPlacement *lp = m.placement.layer(l);
  if (!lp) throw std::runtime_error("no expert placement for MoE layer");
  const int device = g.engine->gpu->get_core_device();
  const bool decode = g.S == 1;
  AliasedMapRequest req;
  req.n_experts = c.n_experts;
  req.n_tokens = g.S;
  req.decode = decode;
  req.full_ddr_view = true; // the offload arena is indexed by expert id
  req.secondary_active = m.gpu1 != nullptr;
  req.policy = (g.S > 1 && layer_experts_take_mmq(*g.engine, l)) ? moe_partial::SentinelPolicy::AutoBatched
                                                              : moe_partial::SentinelPolicy::Auto;

  MoEBranch hot, cold;
  if (!lp->primary.empty()) {
    hot.weights.gate = make_alias_weight(*g.engine, g.ctx, l, "ffn_gate_exps.weight", lp->primary, m.marker, device, false);
    hot.weights.up = make_alias_weight(*g.engine, g.ctx, l, "ffn_up_exps.weight", lp->primary, m.marker, device, false);
    hot.weights.down = make_alias_weight(*g.engine, g.ctx, l, "ffn_down_exps.weight", lp->primary, m.marker, device, false);
    if (!hot.weights.valid()) throw std::runtime_error("cannot alias VRAM experts of layer " + std::to_string(l));
    moe_partial::ExpertBranchMap bm;
    if (!build_primary_map(*lp, req, &bm)) throw std::runtime_error("hot route map failed");
    branch_maps(g, bm, l, "hot", hot);
    hot.present = true;
  }
  std::vector<uint8_t> not_cold(size_t(c.n_experts), 0);
  for (int e : lp->primary.experts) not_cold[size_t(e)] = 1;
  if (m.gpu1) for (int e : lp->secondary.experts) not_cold[size_t(e)] = 1;
  if (!lp->cold.empty()) {
    const auto any_type = [](GGUFType) { return true; };
    cold.weights.gate = m.offload.plane(*g.engine, g.ctx, l, "ffn_gate_exps.weight", any_type, &g.host_buffers);
    cold.weights.up = m.offload.plane(*g.engine, g.ctx, l, "ffn_up_exps.weight", any_type, &g.host_buffers);
    cold.weights.down = m.offload.plane(*g.engine, g.ctx, l, "ffn_down_exps.weight", any_type, &g.host_buffers);
    if (!cold.weights.valid()) throw std::runtime_error("cannot map DDR experts of layer " + std::to_string(l));
    moe_partial::ExpertBranchMap bm;
    if (!build_cold_map(*lp, req, &bm)) throw std::runtime_error("cold route map failed");
    branch_maps(g, bm, l, "cold", cold);
    cold.present = true;
  }
  if (!hot.present && !cold.present) throw std::runtime_error("layer has no routed experts");

  // The executor is chosen from the placement and what the runtime can do.
  MoEPlacementSummary ps;
  ps.placement = hot.present && cold.present ? ExpertPlacementClass::MixedVramDdr
               : cold.present ? ExpertPlacementClass::StorageBacked
                              : ExpertPlacementClass::AllVram;
  MoEExecutionCapabilities caps;
  caps.early_ready = decode && ggml_backend_sched_early_cpu_supported &&
                     ggml_backend_sched_early_cpu_supported() == 1;
  caps.gpu_assist = m.assist.slots > 0;
  const ExecutionPolicyDecision decision = resolve_execution_policy(
      ps, caps, decode ? ModelGraphPhase::Decode : ModelGraphPhase::Prefill);

  MoEBlockGraph bg;
  bg.ctx = g.ctx; bg.graph = g.graph; bg.sched = g.sched; bg.cuda = g.cuda; bg.cpu = g.cpu;
  bg.device = device; bg.decode = decode; bg.n_tokens = g.S; bg.name_prefix = "mimo2";
  bg.pin_prefill_branch_sums = true;
  MoEBlockLayer bl;
  bl.layer = l; bl.input = x; bl.route_ids = ids; bl.route_weights = weights;
  MoEBlockHooks hooks;
  hooks.adjust_cold_inputs = [&](moe_partial::BranchInputs &in) {
    in.ids = m.offload.miss_node(g.ctx, l, in.ids, !decode, g.S, not_cold);
    return in.ids != nullptr;
  };
  hooks.adopt_lane = [&](std::unique_ptr<EarlyReadyLane> lane) {
    return m.offload.adopt_lane(l, cold.weights, std::move(lane), m.assist, true, g.early_ready);
  };
  EarlyReadyLane *adopted = nullptr;
  const auto adopt_inner = hooks.adopt_lane;
  hooks.adopt_lane = [&, adopt_inner](std::unique_ptr<EarlyReadyLane> lane) {
    adopted = adopt_inner(std::move(lane));
    return adopted;
  };
  if (m.gpu1 && !lp->secondary.empty()) {
    hooks.extra_branches = [&](ggml_tensor *out_in) -> ggml_tensor * {
      MoEBranch h1;
      h1.weights.gate = make_alias_weight(*g.engine, g.ctx, l, "ffn_gate_exps.weight", lp->secondary, m.desc1, 1, true);
      h1.weights.up = make_alias_weight(*g.engine, g.ctx, l, "ffn_up_exps.weight", lp->secondary, m.desc1, 1, true);
      h1.weights.down = make_alias_weight(*g.engine, g.ctx, l, "ffn_down_exps.weight", lp->secondary, m.desc1, 1, true);
      if (!h1.weights.valid()) throw std::runtime_error("cannot alias GPU1 experts of layer " + std::to_string(l));
      moe_partial::ExpertBranchMap bm;
      if (!build_secondary_map(*lp, req, &bm)) throw std::runtime_error("GPU1 route map failed");
      branch_maps(g, bm, l, "hot1", h1);
      const auto in1 = moe_partial::make_branch_inputs(g.ctx, h1.route_map, h1.route_mask, ids, weights, c.topk, g.S);
      ggml_tensor *src = adopted ? ggml_reshape_3d(g.ctx, adopted->gpu_input, c.hidden, 1, g.S)
                                 : ggml_reshape_3d(g.ctx, x, c.hidden, 1, g.S);
      ggml_tensor *b1 = build_routed_branch(moe_block_spec(c), bg, bl, src, h1.weights, in1.ids, in1.weights, m.gpu1);
      return b1 ? ggml_add(g.ctx, out_in, b1) : nullptr;
    };
  }
  const MoEBlockResult r = build_moe_block(moe_block_spec(c), bg, bl, hot, cold,
                                           cold.present && decode, decision, hooks);
  if (!r.out) throw std::runtime_error("MoE block construction failed at layer " + std::to_string(l));
  if (decode && cold.present && !r.lane && caps.early_ready)
    LOG_WARN("[MIMO2-GRAPH] layer %d: early-ready lane unavailable; cold experts run synchronously\n", l);
  return r.out;
}

std::unique_ptr<Graph> create_graph(StreamingEngine &e, int tokens) {
  const auto &c = e.model_file.architecture_config;
  auto g = std::make_unique<Graph>();
  g->engine = &e; g->model = g_model.get(); g->S = tokens; g->D = c.hidden;
  g->cuda = g_model->cuda; g->kv_capacity = g_model->capacity;
  g->cpu = ggml_backend_cpu_init(); g->owns_cpu = true;
  if (!g->cpu || !MemoryGuard::try_reserve_host_bytes(kContextBytes)) return nullptr;
  // The early-ready lane runs the cold experts on this backend, so decode needs real threads too.
  ggml_backend_cpu_set_n_threads(g->cpu, int(moestream::graphcfg::integer("MOESTREAM_MIMO2_CPU_THREADS", 10)));
  g->context_reservation_bytes = kContextBytes;
  g->ctx = ggml_init({kContextBytes, nullptr, true});
  if (!g->ctx) return nullptr;
  g->graph = ggml_new_graph_custom(g->ctx, kNodes, false);
  ggml_backend_t backends[] = {g->cuda, g->model->gpu1 ? g->model->gpu1 : g->cpu, g->cpu};
  // Batched prefill can run the host-resident (cold) expert matmuls on the GPU by copying the
  // routed experts across (MOESTREAM_MIMO2_PREFILL_OP_OFFLOAD=1). The scheduler needs events, or
  // every weight copy synchronizes the destination stream.
  // Automatic: the expert copies cost about the same per chunk whatever its size (~52 GiB over
  // PCIe), so only large chunks beat the CPU path (measured crossover: 146 tokens CPU 37.7 tok/s vs
  // GPU 28.2; 516 tokens CPU 29.7 vs GPU 71.4). MOESTREAM_MIMO2_PREFILL_OP_OFFLOAD=0/1 forces it.
  const char *force_offload = moestream::graphcfg::env("MOESTREAM_MIMO2_PREFILL_OP_OFFLOAD");
  g->op_offload = tokens > 1 && g_model->ingraph_moe &&
                  (force_offload && force_offload[0]
                       ? force_offload[0] != '0'
                       : tokens >= moestream::graphcfg::integer("MOESTREAM_MIMO2_PREFILL_OFFLOAD_MIN", 256));
  if (g->op_offload) setenv("MOESTREAM_SCHED_EVENTS", "1", 1);
  g->sched = ggml_backend_sched_new(backends, nullptr, g->model->gpu1 ? 3 : 2, kNodes, false, g->op_offload);
  if (g->sched && g->op_offload && g->expert_copy.attach(g->sched))
    LOG_INFO("[MIMO2-GRAPH] S=%d prefill op-offload: cold experts compute on the GPU (selective copies)\n", tokens);
  if (!g->sched) return nullptr;
  g->input = host_input(*g, g->ctx, GGML_TYPE_F32, {c.hidden, tokens}, size_t(c.hidden) * tokens * 4, "mimo_input");
  g->positions = host_input(*g, g->ctx, GGML_TYPE_I32, {tokens}, size_t(tokens) * 4, "mimo_positions");
  const int mask_rows = ((tokens + 63) / 64) * 64;
  g->full_mask = host_input(*g, g->ctx, GGML_TYPE_F16, {g->kv_capacity, mask_rows},
                            size_t(g->kv_capacity) * mask_rows * 2, "mimo_full_mask");
  g->swa_mask = host_input(*g, g->ctx, GGML_TYPE_F16, {g->kv_capacity, mask_rows},
                           size_t(g->kv_capacity) * mask_rows * 2, "mimo_swa_mask");
  for (int64_t h : c.mimo2_head_count_kv) {
    if (g->indices.count(int(h))) continue;
    g->indices[int(h)] = host_input(*g, g->ctx, GGML_TYPE_I32, {h * tokens}, size_t(h) * tokens * 4, "mimo_kv_rows");
  }
  if (!g->input || !g->positions || !g->full_mask || !g->swa_mask) return nullptr;
  for (auto [h, t] : g->indices) if (!t) return nullptr;
  g->sites.reserve(c.n_layers);
  ggml_tensor *cur = g->input;
  for (int l = 0; l < c.n_layers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    cur = on_gpu(*g, ggml_add(g->ctx, cur, attention(*g, cur, l)));
    ggml_tensor *ffn;
    if (l < c.first_dense) {
      auto *x = norm(*g, cur, p + "ffn_norm.weight");
      auto *gate = ggml_mul_mat(g->ctx, weight(*g->model, p + "ffn_gate.weight"), x);
      auto *up = ggml_mul_mat(g->ctx, weight(*g->model, p + "ffn_up.weight"), x);
      ffn = on_gpu(*g, ggml_mul_mat(g->ctx, weight(*g->model, p + "ffn_down.weight"), ggml_swiglu_split(g->ctx, gate, up)));
    } else {
      if (g_model->ingraph_moe) {
        ffn = moe_ingraph(*g, cur, l);
      } else {
        g->sites.push_back({g.get(), l});
        ffn = ggml_map_custom1(g->ctx, cur, moe_node, 1, &g->sites.back());
        ggml_backend_sched_set_tensor_backend(g->sched, ffn, g->cpu);
      }
    }
    cur = on_gpu(*g, ggml_add(g->ctx, cur, ffn));
  }
  g->output = cur;
  ggml_set_output(g->output);
  ggml_build_forward_expand(g->graph, g->output);
  if (tokens == 1) {
    auto *last = norm(*g, cur, "output_norm.weight");
    g->logits = on_gpu(*g, ggml_mul_mat(g->ctx, weight(*g->model, "output.weight"), last));
    ggml_set_output(g->logits);
    ggml_build_forward_expand(g->graph, g->logits);
  }
  if (const long trace = moestream::graphcfg::integer("MOESTREAM_MIMO2_NODE_TRACE", 0)) {
    // Debug aid. 1: name each node before it runs (use with CUDA_LAUNCH_BLOCKING=1).
    // 2: after each F32 node, report the first few nodes holding non-finite / huge values.
    static long mode = 0;
    mode = trace;
    ggml_backend_sched_set_eval_callback(g->sched, [](ggml_tensor *t, bool ask, void *) {
      if (mode == 1 && ask) {
        std::fprintf(stderr, "[MIMO2-TRACE] %s op=%s shape=[%lld,%lld,%lld] src0=%s src1=%s\n", t->name,
                     ggml_op_name(t->op), (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
                     t->src[0] ? t->src[0]->name : "-", t->src[1] ? t->src[1]->name : "-");
        std::fflush(stderr);
      }
      if (mode == 4 && !ask && t->op == GGML_OP_MUL_MAT_ID && t->src[0] && t->src[2] && t->type == GGML_TYPE_F32) {
        static int shown4 = 0;
        if (shown4 < 2) {
          ++shown4;
          const ggml_tensor *ids = t->src[2];
          std::vector<int32_t> iv(size_t(ggml_nelements(ids)));
          ggml_backend_tensor_get(ids, iv.data(), 0, iv.size() * sizeof(int32_t));
          std::vector<float> v(size_t(ggml_nelements(t)));
          ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
          const int64_t d = t->ne[0], k = t->ne[1], S = t->ne[2];
          std::fprintf(stderr, "[MIMO2-GRID] %s (rows = route 0..%lld, cols = token 0..%lld; N=nonfinite . = ok, id in parens)\n",
                       t->name, (long long)k - 1, (long long)S - 1);
          for (int64_t j = 0; j < k; ++j) {
            std::fprintf(stderr, "[MIMO2-GRID] route %lld:", (long long)j);
            for (int64_t tok = 0; tok < S; ++tok) {
              bool bad = false; double mx = 0;
              for (int64_t i = 0; i < d; ++i) {
                const float x = v[size_t(tok * k * d + j * d + i)];
                if (!std::isfinite(x)) bad = true; else mx = std::max(mx, (double)std::fabs(x));
              }
              std::fprintf(stderr, " %s(%d,%.2g)", bad ? "N" : ".", iv[size_t(tok * k + j)], mx);
            }
            std::fprintf(stderr, "\n");
          }
          std::fflush(stderr);
        }
      }
      if (mode == 3 && !ask && t->op == GGML_OP_MUL_MAT_ID && t->src[0] && t->src[2]) {
        static int shown = 0;
        if (shown < 3) {
          ++shown;
          const ggml_tensor *w = t->src[0], *ids = t->src[2];
          std::vector<int32_t> v(size_t(ggml_nelements(ids)));
          ggml_backend_tensor_get(ids, v.data(), 0, v.size() * sizeof(int32_t));
          std::fprintf(stderr, "[MIMO2-IDS] %s w=%s type=%s ne=[%lld,%lld,%lld] nb1=%zu nb2=%zu backend_buf=%s ids_ne=[%lld,%lld] ids:",
                       t->name, w->name, ggml_type_name(w->type), (long long)w->ne[0], (long long)w->ne[1],
                       (long long)w->ne[2], w->nb[1], w->nb[2], w->buffer ? ggml_backend_buffer_name(w->buffer) : "none",
                       (long long)ids->ne[0], (long long)ids->ne[1]);
          for (size_t i = 0; i < std::min<size_t>(v.size(), 24); ++i) std::fprintf(stderr, " %d", v[i]);
          std::fprintf(stderr, "\n");
          std::fflush(stderr);
        }
      }
      if (mode == 2 && !ask && t->type == GGML_TYPE_F32 && ggml_nelements(t) <= (1 << 22)) {
        static int reported = 0;
        if (reported < 8) {
          std::vector<float> v(size_t(ggml_nelements(t)));
          ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
          float mx = 0.0f; bool bad = false;
          for (float x : v) { if (!std::isfinite(x)) { bad = true; break; } mx = std::max(mx, std::fabs(x)); }
          if (bad || mx > 1e6f) {
            ++reported;
            std::fprintf(stderr, "[MIMO2-NAN] %s op=%s shape=[%lld,%lld,%lld] %s max=%g src0=%s src1=%s\n", t->name,
                         ggml_op_name(t->op), (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
                         bad ? "NON-FINITE" : "HUGE", mx, t->src[0] ? t->src[0]->name : "-",
                         t->src[1] ? t->src[1]->name : "-");
            std::fflush(stderr);
          }
        }
      }
      return true;
    }, nullptr);
  }
  if (!ggml_backend_sched_alloc_graph(g->sched, g->graph)) return nullptr;
  {
    // Measured, not estimated: what this graph itself holds on each device.
    const double GiB = double(1ull << 30);
    const auto buf = [&](ggml_backend_t b) { return b ? double(ggml_backend_sched_get_buffer_size(g->sched, b)) / GiB : 0.0; };
    int cur_dev = 0; cudaGetDevice(&cur_dev);
    double free_gib[2] = {-1, -1}, total_gib[2] = {0, 0};
    int n_dev = 0; cudaGetDeviceCount(&n_dev);
    for (int d = 0; d < std::min(n_dev, 2); ++d) {
      size_t f = 0, t = 0; cudaSetDevice(d);
      if (cudaMemGetInfo(&f, &t) == cudaSuccess) { free_gib[d] = double(f) / GiB; total_gib[d] = double(t) / GiB; }
    }
    cudaSetDevice(cur_dev);
    LOG_INFO("[MIMO2-VRAM] S=%d graph compute buffers: GPU0=%.3f GiB GPU1=%.3f GiB CPU=%.3f GiB | KV=%.3f GiB | free now: GPU0 %.2f/%.2f GiB, GPU1 %.2f/%.2f GiB\n",
             tokens, buf(g->cuda), buf(g->model->gpu1), buf(g->cpu),
             double(ggml_backend_buffer_get_size(g->model->kv_buffer)) / GiB,
             free_gib[0], total_gib[0], free_gib[1], total_gib[1]);
  }
  LOG_INFO("[MIMO2-GRAPH] built S=%d layers=%d nodes=%d splits=%d; %s\n",
           tokens, c.n_layers, ggml_graph_n_nodes(g->graph), ggml_backend_sched_get_n_splits(g->sched),
           g_model->ingraph_moe ? "in-graph hot/cold MoE" : "shared native MoE cache");
  g->ready = true;
  return g;
}

void set_inputs(Graph &g, int position, const int *ids) {
  const auto &c = g.engine->model_file.architecture_config;
  g.position = position; g.token_ids.assign(ids, ids + g.S);
  auto *pos = static_cast<int32_t *>(g.positions->data);
  for (int s = 0; s < g.S; ++s) pos[s] = position + s;
  for (auto [h, t] : g.indices) {
    auto *rows = static_cast<int32_t *>(t->data);
    for (int s = 0; s < g.S; ++s)
      for (int head = 0; head < h; ++head)
        rows[s * h + head] = int32_t(mimo2_kv_row(head, g.kv_capacity, position + s));
  }
  auto *full = static_cast<ggml_fp16_t *>(g.full_mask->data);
  auto *swa = static_cast<ggml_fp16_t *>(g.swa_mask->data);
  const int rows = int(g.full_mask->ne[1]);
  for (int s = 0; s < rows; ++s)
    for (int k = 0; k < g.kv_capacity; ++k) {
      const bool visible = s < g.S && mimo2_attention_visible(position + s, k, 0, false);
      full[size_t(s) * g.kv_capacity + k] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
      swa[size_t(s) * g.kv_capacity + k] = ggml_fp32_to_fp16(
          visible && mimo2_attention_visible(position + s, k, c.sliding_window, true) ? 0.0f : -INFINITY);
    }
}
} // namespace
#endif

bool mimo2_ingraph_moe_enabled() {
  return mimo2_full_graph_enabled() &&
         moestream::graphcfg::flag_unless_off("MOESTREAM_MIMO2_INGRAPH_MOE");
}

bool mimo2_full_graph_enabled() {
  return moestream::graphcfg::flag_unless_off("MOESTREAM_MIMO2_FULL_GRAPH");
}

void mimo2_full_graph_set_vram_cache(GpuTopology *topology) {
#if defined(MOESTREAM_USE_GGML_CUDA)
  g_topology = topology;
#else
  (void)topology;
#endif
}

void mimo2_bind_runtime(StreamingEngine &e, RAMSafetyManager &r, LockFreePrefetcher &p, LayerScratch &s) {
#if defined(MOESTREAM_USE_GGML_CUDA)
  g_runtime.bind(e, r, p, s);
  g_ram = &r;
  if (g_model && g_model->engine == &e) g_model->runtime = g_runtime;
#else
  (void)e; (void)r; (void)p; (void)s;
#endif
}

bool mimo2_full_graph_prepare(StreamingEngine &e, int tokens, int position, const int *ids) {
#if defined(MOESTREAM_USE_GGML_CUDA)
  try {
    if (!mimo2_full_graph_enabled() || !ids || tokens <= 0 || position < 0 ||
        tokens > e.max_t || position > e.max_t - tokens)
      throw std::runtime_error("unsupported request shape or disabled graph");
    if (!g_model || g_model->engine != &e) {
      mimo2_full_graph_shutdown();
      if (!create_state(e)) throw std::runtime_error("weight/KV allocation failed");
    }
    auto &slot = tokens == 1 ? g_decode : g_prefill;
    if (!slot || slot->S != tokens) {
      slot.reset(); slot = create_graph(e, tokens);
      if (!slot) throw std::runtime_error("graph allocation failed");
    }
    set_inputs(*slot, position, ids);
    return true;
  } catch (const std::exception &ex) {
    e.qwen4exp_full_graph_failure = std::string("MiMo2: ") + ex.what();
    LOG_ERROR("[MIMO2-GRAPH] %s\n", ex.what());
    return false;
  }
#else
  (void)e; (void)tokens; (void)position; (void)ids;
  return false;
#endif
}

bool mimo2_full_graph_run(StreamingEngine &e, float *hidden, int tokens, int position,
                          const int *ids, float *logits, int vocab) {
#if defined(MOESTREAM_USE_GGML_CUDA)
  if (!hidden || !mimo2_full_graph_prepare(e, tokens, position, ids)) return false;
  Graph &g = *(tokens == 1 ? g_decode : g_prefill);
  if (position == 0) ggml_backend_buffer_clear(g_model->kv_buffer, 0);
  const size_t bytes = size_t(tokens) * g.D * sizeof(float);
  g.callback_failure.clear();
  std::memcpy(g.input->data, hidden, bytes);
  ModelState &ms = *g_model;
  if (ms.ingraph_moe) {
    if (!ms.placement.hold_all(*ms.topology)) {
      e.qwen4exp_full_graph_failure = "MiMo2: unable to hold aliased VRAM experts";
      return false;
    }
    for (auto &entry : g.early_ready) {
      entry.second->residency_ready = false;
      entry.second->completed = false;
    }
  }
  g.expert_copy.reset();
  const auto t_chunk = std::chrono::steady_clock::now();
  const auto status = ggml_backend_sched_graph_compute(g.sched, g.graph);
  ggml_backend_sched_synchronize(g.sched);
  if (ms.ingraph_moe) ms.placement.release_all(*ms.topology);
  if (ms.ingraph_moe && tokens > 1) {
    const double ms_chunk = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_chunk).count();
    const auto &ct = ms.offload.store().counters();
    LOG_INFO("[MIMO2-PREFILL] pos=%d S=%d %.1f ms (%.1f tok/s) | store misses=%llu transient_loads=%llu | expert copies=%llu (%.2f GiB)\n",
             position, tokens, ms_chunk, tokens * 1000.0 / ms_chunk, (unsigned long long)ct.misses.load(),
             (unsigned long long)ct.transient_loads.load(), (unsigned long long)g.expert_copy.plane_copies,
             double(g.expert_copy.bytes) / double(1ull << 30));
  }
  if (ms.ingraph_moe && tokens == 1) {
    static int steps = 0;
    static auto t_prev = std::chrono::steady_clock::now();
    ++steps;
    if (steps % 16 == 0) {
      uint64_t calls = 0, split = 0; int64_t wait = 0, compute = 0, stall = 0;
      for (auto &entry : g.early_ready) {
        calls += entry.second->calls; split += entry.second->split_runs;
        wait += entry.second->wait_us; compute += entry.second->compute_us;
        stall += entry.second->split_stall_us;
      }
      const auto &ct = ms.offload.store().counters();
      const auto now = std::chrono::steady_clock::now();
      const double ms_tok = std::chrono::duration<double, std::milli>(now - t_prev).count() / 16.0;
      t_prev = now;
      LOG_INFO("[MIMO2-LANE] steps=%d %.1f ms/token | lane calls=%llu split=%llu wait=%.1f compute=%.1f stall=%.1f ms/token | "
               "store misses=%llu loads=%llu read=%.1f ms/token cold_routes=%llu\n",
               steps, ms_tok, (unsigned long long)calls, (unsigned long long)split, wait / 1000.0 / steps,
               compute / 1000.0 / steps, stall / 1000.0 / steps, (unsigned long long)ct.misses.load(),
               (unsigned long long)ct.loads.load(), ct.load_us.load() / 1000.0 / steps,
               (unsigned long long)ct.cold_routes.load());
    }
  }
  if (status != GGML_STATUS_SUCCESS || !g.callback_failure.empty()) {
    e.qwen4exp_full_graph_failure = g.callback_failure.empty()
        ? "MiMo2 graph execution failed" : "MiMo2: " + g.callback_failure;
    return false;
  }
  ggml_backend_tensor_get(g.output, hidden, 0, bytes);
  if (tokens == 1 && logits && vocab == e.model_file.architecture_config.vocab) {
    ggml_backend_tensor_get(g.logits, logits, 0, size_t(vocab) * sizeof(float));
    e.qwen4exp_full_graph_logits_ready = true;
  }
  e.full_graph_output_ready = true;
  return true;
#else
  (void)e; (void)hidden; (void)tokens; (void)position; (void)ids; (void)logits; (void)vocab;
  return false;
#endif
}

void mimo2_full_graph_shutdown() {
#if defined(MOESTREAM_USE_GGML_CUDA)
  g_decode.reset(); g_prefill.reset(); g_model.reset();
#endif
}
