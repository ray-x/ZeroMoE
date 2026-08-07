// family_ops.cpp — resolve_family_ops() table + the shared hyper-connection
// no-op/plain-add hooks + the per-CacheShape cache allocators.
//
// Pulls in engine.hpp for StreamingEngine's real fields (Lc/Rc/Kc/Vc, the
// AVX-512 residual add) -- unlike family_ops.hpp, this hits the usual
// cuda_runtime.h wall on a machine with no CUDA Toolkit (CLAUDE.md's
// documented, pre-existing limitation for every file that reaches
// engine.hpp; not a new gap this file introduces). Compiled into both the
// moestream and moestream_tests targets, alongside models/model_factory.cpp
// (CMakeLists.txt).
#include "family_ops.hpp"
#include "engine.hpp"
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

// Real attention functions (attention_layer.cpp). No shared header declares
// these today -- execution_layer.cpp itself reaches them the same way, via
// its own extern declarations at file scope (see execution_layer.cpp:18-19)
// -- so this mirrors the codebase's own established pattern rather than
// introducing a new one.
extern void mla_attention_absorbed_gguf(StreamingEngine &engine,
                                         int layer_idx,
                                         const float *hidden_states, int S,
                                         int pos_base, float *out_context);
extern void dense_attention_gguf(StreamingEngine &engine, int layer_idx,
                                  const float *hidden_states, int S,
                                  int pos_base, float *out_context);
// DeepSeek-V4 (attention_layer.cpp, Project B) -- same signature discipline
// as the two above.
extern void dsv4_attention_gguf(StreamingEngine &engine, int layer_idx,
                                 const float *hidden_states, int S,
                                 int pos_base, float *out_context);
// DeepSeek-V4's hyper-connection bracket hooks (hyper_connections.cpp,
// Project B) -- alloc_dsv4_swa_cache itself is defined further down in
// THIS file, alongside its alloc_mla_absorbed_cache/alloc_dense_kv_cache
// siblings, not in hyper_connections.cpp -- cache allocation isn't
// hyper-connection-specific, so it stays grouped with the other
// CacheShape allocators for the same reason they're grouped together.
extern void hc_pre_collapse(StreamingEngine &engine, int layer_idx,
                             int sub_block, float *hidden_states, int S,
                             int D);
extern void hc_post_combine(StreamingEngine &engine, int layer_idx,
                             int sub_block, float *hidden_states,
                             const float *sub_block_out, int S, int D);

// ---------------------------------------------------------------------
// Shared hyper-connection hooks (hc_mult == 1 degenerate case)
// ---------------------------------------------------------------------

void hc_pre_noop(StreamingEngine & /*engine*/, int /*layer_idx*/,
                  int /*sub_block*/, float * /*hidden_states*/, int /*S*/,
                  int /*D*/) {
  // Intentionally empty: families with hc_mult==1 have nothing to collapse
  // -- hidden_states already IS the single stream the sub-block should read.
}

void hc_post_plain_add(StreamingEngine & /*engine*/, int /*layer_idx*/,
                        int /*sub_block*/, float *hidden_states,
                        const float *sub_block_out, int S, int D) {
  // Lifted verbatim from layers_forward_gguf's two (previously identical,
  // inlined-twice) residual-add loops -- bit-for-bit identical output to
  // the pre-refactor code for every family that reaches this function.
  int n = S * D;
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 h = _mm512_loadu_ps(hidden_states + i);
    __m512 a = _mm512_loadu_ps(sub_block_out + i);
    _mm512_storeu_ps(hidden_states + i, _mm512_add_ps(h, a));
  }
  for (; i < n; ++i)
    hidden_states[i] += sub_block_out[i];
}

// ---------------------------------------------------------------------
// Per-CacheShape allocators (lifted from graph_initializer.cpp's old
// is_mla-gated block, unchanged in substance -- just given a name so
// resolve_family_ops's table entries can point at them).
// ---------------------------------------------------------------------

static void alloc_mla_absorbed_cache(StreamingEngine &engine,
                                      const ArchitectureConfig &c,
                                      int max_ctx) {
  engine.Lc.resize(c.n_layers + 1, nullptr);
  engine.Rc.resize(c.n_layers + 1, nullptr);
  engine.d_Lc.resize(c.n_layers + 1, nullptr);
  engine.d_Rc.resize(c.n_layers + 1, nullptr);
  size_t total_kv_bytes = 0;
  for (int i = 0; i <= c.n_layers; ++i) {
    size_t lc_bytes = (size_t)max_ctx * c.kv_lora * sizeof(float);
    size_t rc_bytes = (size_t)max_ctx * c.qk_rope * sizeof(float);
    posix_memalign((void **)&engine.Lc[i], 4096, lc_bytes);
    posix_memalign((void **)&engine.Rc[i], 4096, rc_bytes);
    std::memset(engine.Lc[i], 0, lc_bytes);
    std::memset(engine.Rc[i], 0, rc_bytes);
    total_kv_bytes += lc_bytes + rc_bytes;
  }
  // NOTE: this used to also mirror the latent/rope KV cache into VRAM here
  // (d_Lc/d_Rc/d_Lc_q4/d_Rc_q4, "Q4 then FP32" attempt). Removed: the live
  // MLA attention path (mla_attention_absorbed_gguf, src/attention_layer.cpp)
  // reads and writes ONLY engine.Lc[layer_idx]/engine.Rc[layer_idx] -- plain
  // host pointers, via AVX-512 SIMD -- and never touches d_Lc/d_Rc/d_Lc_q4/
  // d_Rc_q4 at all. There IS a real GPU kernel for this
  // (cuda/mla_attention.cu's launch_mla_attention) but it is never called
  // from anywhere in the codebase, and the kv_on_device/kv_quantized flags
  // that would gate using it are set here but never read anywhere either.
  // If launch_mla_attention is ever wired up, this allocation (and the
  // fields, still present on StreamingEngine) can come back.
  LOG_STREAM_INFO() << "[GRAPH] KV cache: host-only (" << total_kv_bytes / 1e9
                     << " GB) -- GPU KV mirror unused by the live attention "
                        "path, not allocated\n";
}

static void alloc_dense_kv_cache(StreamingEngine &engine,
                                  const ArchitectureConfig &c, int max_ctx) {
  // Dense K/V cache (Hunyuan/Llama): [max_t, n_head_kv*head_dim] per layer.
  int kvd = std::max(1, c.n_head_kv) * std::max(1, c.head_dim);
  engine.Kc.resize(c.n_layers, nullptr);
  engine.Vc.resize(c.n_layers, nullptr);
  size_t kv_bytes = (size_t)max_ctx * kvd * sizeof(float);
  for (int i = 0; i < c.n_layers; ++i) {
    posix_memalign((void **)&engine.Kc[i], 4096, kv_bytes);
    posix_memalign((void **)&engine.Vc[i], 4096, kv_bytes);
    std::memset(engine.Kc[i], 0, kv_bytes);
    std::memset(engine.Vc[i], 0, kv_bytes);
  }
  size_t total_kv_bytes = 2 * kv_bytes * c.n_layers;
  LOG_STREAM_INFO() << "[GRAPH] Dense KV cache: " << total_kv_bytes / 1e9
                     << " GB (" << c.n_head_kv << " kv heads x " << c.head_dim
                     << ")\n";
}

static void alloc_dsv4_swa_cache(StreamingEngine &engine,
                                  const ArchitectureConfig &c, int max_ctx) {
  // DeepSeek-V4 raw/SWA MLA cache (engine.hpp's KVc): one per-head-sized
  // latent per layer, sized by sliding_window NOT max_ctx -- the whole
  // point of SWA is bounding per-layer KV memory by the window, not the
  // full context length. Falls back to max_ctx if the file didn't set a
  // sliding_window (shouldn't happen for a real DeepSeek-V4 checkpoint,
  // but a 0-or-negative window would make every ring-index modulo either
  // divide-by-zero or degenerate to 1 slot -- clamping to max_ctx here
  // means a missing sliding_window fails safe into "acts like the full,
  // unbounded absorbed-MLA cache" rather than corrupting every attention
  // read/write with a bad modulus).
  int window = c.sliding_window > 0 ? c.sliding_window : max_ctx;
  engine.sliding_window = window;
  engine.KVc.resize(c.n_layers, nullptr);
  // Per-position element count: DeepSeek-V4's wkv tensor (attn_kv.weight)
  // projects to n_embd_head, a single head's worth. Sized as qk_nope +
  // qk_rope (the SAME split Q already uses, c.qk_nope/c.qk_rope) -- NOT a
  // separate/independent dimension: this cached latent stands in for both
  // K and V (no wk_b/wv_b absorption in V4), and dsv4_attention_gguf scores
  // it against Q as score_nope + score_rope exactly like V3's absorbed
  // scoring does, which only decomposes correctly if the cached latent's
  // nope/rope split has the SAME widths Q's does. dsv4_attention_gguf
  // (attention_layer.cpp) MUST read/write exactly qk_nope+qk_rope floats
  // per cached position -- the two are required to agree.
  int qh = c.qk_nope + c.qk_rope;
  size_t kv_bytes = (size_t)window * qh * sizeof(float);
  for (int i = 0; i < c.n_layers; ++i) {
    posix_memalign((void **)&engine.KVc[i], 4096, kv_bytes);
    std::memset(engine.KVc[i], 0, kv_bytes);
  }
  size_t total_kv_bytes = kv_bytes * c.n_layers;
  LOG_STREAM_INFO() << "[GRAPH] DeepSeek-V4 SWA KV cache: "
                     << total_kv_bytes / 1e9 << " GB (window=" << window
                     << " tokens/layer, " << c.n_layers << " layers)\n";
}

// ---------------------------------------------------------------------
// resolve_family_ops -- the single ModelFamily -> FamilyOps table.
// ---------------------------------------------------------------------

FamilyOps resolve_family_ops(ModelFamily fam, const ArchitectureConfig &c) {
  FamilyOps ops;
  ops.family = fam;
  switch (fam) {
  case ModelFamily::GLM_DSA:
  case ModelFamily::DeepSeek_V3: // DeepSeek-V3 shares GLM-5.2's MLA core
    ops.attn_type = AttentionType::MLA;
    ops.cache_shape = CacheShape::MLAAbsorbed;
    ops.default_gating = GatingFunc::Sigmoid;
    ops.attention_fn = &mla_attention_absorbed_gguf;
    ops.alloc_cache_fn = &alloc_mla_absorbed_cache;
    break;

  case ModelFamily::DeepSeek_V4:
    // Phase 1 only (correctness within each layer's local SWA window) --
    // see CLAUDE.md's DeepSeek-V4 plan writeup. c.hc_mult is a GGUF
    // hyperparameter (hyper_connection.count), already populated by
    // graph_initializer.cpp before this function is called -- hence
    // resolve_family_ops taking ArchitectureConfig at all, unlike every
    // other case here which needs nothing from it.
    ops.attn_type = AttentionType::MLA;
    ops.cache_shape = CacheShape::MLARawSWA;
    ops.default_gating = GatingFunc::SqrtSoftplus; // hard-required, not just a default
    ops.uses_hyper_connections = (c.hc_mult > 1);
    ops.hc_mult = c.hc_mult;
    ops.attention_fn = &dsv4_attention_gguf;
    ops.alloc_cache_fn = &alloc_dsv4_swa_cache;
    // Only wire the real hyper-connection hooks when the file actually sets
    // hc_mult > 1 -- an hc_mult<=1 DeepSeek-V4 file (shouldn't happen for a
    // real checkpoint, but defensively) falls back to the same shared no-op/
    // plain-add every other family uses, rather than making hc_pre_collapse/
    // hc_post_combine themselves handle a degenerate hc_mult==1 case.
    if (c.hc_mult > 1) {
      ops.hc_pre_fn = &hc_pre_collapse;
      ops.hc_post_fn = &hc_post_combine;
    }
    break;

  case ModelFamily::Hunyuan:
  case ModelFamily::Llama:
  case ModelFamily::GLM4_MOE:
  case ModelFamily::Qwen:
  case ModelFamily::GemmaMoE:
  case ModelFamily::KimiMoE:
  case ModelFamily::Unknown:
  default:
    // Matches today's actual pre-refactor behavior: nothing in the live GPU
    // path branches on Qwen/GemmaMoE/KimiMoE/Unknown either -- they all fell
    // through is_mla's `else` branch (dense) exactly like Hunyuan/Llama do.
    // Note: graph_initializer.cpp's OLD use_sigmoid_gate fallback compared
    // arch0 == "hy_v3" (missing hyphen -- the real family string everywhere
    // else is "hy-v3"), so that fallback silently never fired; Hunyuan's
    // default_gating stays Softmax here to match that ACTUAL prior behavior,
    // not the apparently-intended-but-dead sigmoid fallback. Not fixed as
    // part of this refactor -- fixing it would silently change Hunyuan's
    // numerics, a separate decision from "make dispatch not scattered."
    ops.attn_type = AttentionType::DenseQKV;
    ops.cache_shape = CacheShape::DenseKV;
    ops.default_gating = GatingFunc::Softmax;
    ops.attention_fn = &dense_attention_gguf;
    ops.alloc_cache_fn = &alloc_dense_kv_cache;
    break;
  }
  return ops;
}
