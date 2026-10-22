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
#include "full_graph_model.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <stdexcept>

// Real attention functions (attention_layer.cpp). No shared header declares
// these today -- execution_layer.cpp itself reaches them the same way, via
// its own extern declarations at file scope (see execution_layer.cpp:18-19)
// -- so this mirrors the codebase's own established pattern rather than
// introducing a new one.
extern void mla_attention_absorbed_gguf(StreamingEngine &engine,
                                         int layer_idx,
                                         const float *hidden_states, int S,
                                         int pos_base, float *out_context);
extern void bailingmoe3_attention_gguf(StreamingEngine &engine, int layer_idx,
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
extern void dsv41_attention_gguf(StreamingEngine &engine, int layer_idx,
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
// GLM5Next (GLM 5.3 Flash, attention_layer.cpp) -- hybrid per-layer dispatch
// (recurrent KDA vs absorbed-MLA DSA), same signature discipline as above.
extern void glm5next_attention_gguf(StreamingEngine &engine, int layer_idx,
                                     const float *hidden_states, int S,
                                     int pos_base, float *out_context);
// Qwen4Exp (Qwen3.8-Flash-Next, attention_layer.cpp) -- hybrid per-layer
// dispatch (recurrent GDN vs dense full attention), same signature.
extern void qwen4exp_attention_gguf(StreamingEngine &engine, int layer_idx,
                                     const float *hidden_states, int S,
                                     int pos_base, float *out_context);
// Qwen4Exp's low-rank hyper-connection bracket hooks (hyper_connections.cpp)
// -- a DIFFERENT HC variant from DeepSeek-V4's Sinkhorn form (norm/down/up/
// inject tensors, gated mean-collapse, no comb matrix).
extern void qwen4exp_hc_pre_mix(StreamingEngine &engine, int layer_idx,
                                  int sub_block, float *hidden_states, int S,
                                  int D);
extern void qwen4exp_hc_post_combine(StreamingEngine &engine, int layer_idx,
                                      int sub_block, float *hidden_states,
                                      const float *sub_block_out, int S,
                                      int D);
// Qwen4Exp's PLE n-gram hash embedding (ple_ngram.cpp) -- optional per-layer
// input hook (FamilyOps::layer_input_fn), runs before hc_pre_fn at PLE
// layers.
extern void qwen4exp_ple_layer_input(StreamingEngine &engine, int layer_idx,
                                      int S, int pos_base,
                                      const int *token_ids);
extern void deepseek41_engram_layer_input(StreamingEngine &engine, int layer_idx,
                                           int S, int pos_base,
                                           const int *token_ids);

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

static bool alloc_zeroed_host(StreamingEngine &engine, float **out,
                              size_t bytes, const char *what) {
  *out = nullptr;
  if (bytes == 0) return true;
  if (!engine.reserve_host_memory(bytes)) {
    LOG_ERROR("[MEM-GUARD] host cache allocation refused: %s (%.2f MB)\n",
              what, bytes / 1e6);
    return false;
  }
  void *p = nullptr;
  const int err = posix_memalign(&p, 4096, bytes);
  if (err != 0 || !p) {
    engine.release_host_memory(bytes);
    LOG_ERROR("[CACHE] posix_memalign failed for %s (%.2f MB, errno=%d)\n",
              what, bytes / 1e6, err);
    return false;
  }
  *out = static_cast<float *>(p);
  std::memset(*out, 0, bytes);
  return true;
}

static void alloc_mla_absorbed_cache(StreamingEngine &engine,
                                      const ArchitectureConfig &c,
                                      int max_ctx) {
  engine.Lc.resize(c.n_layers + 1, nullptr);
  engine.Rc.resize(c.n_layers + 1, nullptr);
  engine.d_Lc.resize(c.n_layers + 1, nullptr);
  engine.d_Rc.resize(c.n_layers + 1, nullptr);
  size_t total_kv_bytes = 0;
  for (int i = 0; i <= c.n_layers; ++i) {
    // max(1, ...) on the ELEMENT count: GLM5Next is nope-only (qk_rope == 0),
    // which would otherwise make rc_bytes 0 -- posix_memalign(0) may return
    // NULL, and the live MLA path then forms `Rc[i] + pos * 0` and memcpys 0
    // bytes from it (technically UB on a null base, universally harmless in
    // practice). One float removes the question entirely for every family.
    size_t lc_elems = (size_t)max_ctx * c.kv_lora;
    size_t rc_elems = std::max<size_t>((size_t)max_ctx * c.qk_rope, 1);
    size_t lc_bytes = lc_elems * sizeof(float);
    size_t rc_bytes = rc_elems * sizeof(float);
    if (!alloc_zeroed_host(engine, &engine.Lc[i], lc_bytes, "MLA latent KV") ||
        !alloc_zeroed_host(engine, &engine.Rc[i], rc_bytes, "MLA rope KV"))
      return;
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
    if (!alloc_zeroed_host(engine, &engine.Kc[i], kv_bytes, "dense K cache") ||
        !alloc_zeroed_host(engine, &engine.Vc[i], kv_bytes, "dense V cache"))
      return;
  }
  size_t total_kv_bytes = 2 * kv_bytes * c.n_layers;
  LOG_STREAM_INFO() << "[GRAPH] Dense KV cache: " << total_kv_bytes / 1e9
                     << " GB (" << c.n_head_kv << " kv heads x " << c.head_dim
                     << ")\n";
}

static void alloc_dsv4_swa_cache(StreamingEngine &engine,
                                  const ArchitectureConfig &c, int max_ctx) {
  // Every V4 layer retains a raw sliding-window cache. CSA/HCA layers add
  // their own K-only compressed cache below; the raw cache remains the local
  // attention source and is intentionally not replaced.
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
    if (!alloc_zeroed_host(engine, &engine.KVc[i], kv_bytes, "DeepSeek-V4 SWA KV"))
      return;
  }
  size_t total_kv_bytes = kv_bytes * c.n_layers;
  LOG_STREAM_INFO() << "[GRAPH] DeepSeek-V4 SWA KV cache: "
                     << total_kv_bytes / 1e9 << " GB (window=" << window
                     << " tokens/layer, " << c.n_layers << " layers)\n";

  engine.dsv4_compressed.clear();
  engine.dsv4_compressed.resize(c.n_layers);
  size_t compressed_bytes = 0;
  int csa_layers = 0, hca_layers = 0;
  const auto compressed_size = [&](int ratio, int indexer_dim) -> size_t {
    const size_t capacity = (size_t)dsv4_compressed_cache_capacity(max_ctx, ratio);
    const size_t state_size = (size_t)(ratio == kDsv4CsaRatio ? 2 * ratio : ratio);
    const size_t state_width = (size_t)(ratio == kDsv4CsaRatio ? 2 * qh : qh);
    size_t elems = capacity * (size_t)qh +
                   2 * state_size * state_width;
    if (ratio == kDsv4CsaRatio)
      elems += capacity * (size_t)indexer_dim +
               2 * state_size * (size_t)indexer_dim;
    return elems * sizeof(float);
  };
  for (int l = 0; l < c.n_layers; ++l) {
    const int ratio = c.dsv4_compress_ratios.empty()
        ? 0 : static_cast<int>(c.dsv4_compress_ratios[(size_t)l]);
    if (ratio != 0)
      compressed_bytes += compressed_size(
          ratio, ratio == kDsv4CsaRatio ? c.dsv4_indexer_key_length : 0);
  }
  if (compressed_bytes && !engine.reserve_host_memory(compressed_bytes)) {
    LOG_ERROR("[MEM-GUARD] DeepSeek-V4 compressed state refused: %.2f MB\n",
              compressed_bytes / 1e6);
    return;
  }
  for (int l = 0; l < c.n_layers; ++l) {
    const int ratio = c.dsv4_compress_ratios.empty()
        ? 0 : static_cast<int>(c.dsv4_compress_ratios[(size_t)l]);
    if (ratio == 0) continue;
    Dsv4CompressedLayerState &state = engine.dsv4_compressed[(size_t)l];
    state.configure(ratio, qh, max_ctx,
                    ratio == kDsv4CsaRatio ? c.dsv4_indexer_key_length : 0);
    if (ratio == kDsv4CsaRatio) ++csa_layers;
    if (ratio == kDsv4HcaRatio) ++hca_layers;
  }
  if (csa_layers + hca_layers > 0) {
    LOG_STREAM_INFO() << "[GRAPH] DeepSeek-V4 compressed KV state: "
                      << compressed_bytes / 1e9 << " GB (" << csa_layers
                      << " CSA layer(s), " << hca_layers << " HCA layer(s), "
                      << "single contiguous sequence)\n";
  }
}

// DeepSeek-V4.1 uses the same raw SWA cache geometry as V4, but its
// compressed KV and indexer caches are shared between source layers and its
// compressor only supports ratio 1/2. Keep allocation separate from the V4
// ratio-4/128 state machine.
static void alloc_dsv41_swa_cache(StreamingEngine &engine,
                                  const ArchitectureConfig &c, int max_ctx) {
  const int window = c.sliding_window > 0 ? c.sliding_window : max_ctx;
  engine.sliding_window = window;
  engine.KVc.resize(c.n_layers, nullptr);
  const int qh = c.qk_nope + c.qk_rope;
  const size_t kv_bytes = (size_t)window * qh * sizeof(float);
  for (int l = 0; l < c.n_layers; ++l) {
    if (!alloc_zeroed_host(engine, &engine.KVc[l], kv_bytes,
                           "DeepSeek-V4.1 SWA KV"))
      return;
  }

  engine.dsv41_compressed.clear();
  engine.dsv41_compressed.resize(c.n_layers);
  engine.dsv41_kv_source_for_layer.assign(c.n_layers, -1);
  engine.dsv41_index_source_for_layer.assign(c.n_layers, -1);

  auto source_for = [](const std::vector<int64_t> &sources,
                       int layer, int ratio,
                       const ArchitectureConfig &cfg) -> int {
    int result = -1;
    for (int64_t source : sources) {
      if (source < 0 || source >= cfg.n_layers || source > layer)
        continue;
      const int sl = (int)source;
      const int sr = cfg.dsv4_compress_ratios.empty()
                         ? 0
                         : (int)cfg.dsv4_compress_ratios[(size_t)sl];
      if (sr != ratio) continue;
      result = sl;
    }
    return result;
  };

  size_t compressed_bytes = 0;
  for (int l = 0; l < c.n_layers; ++l) {
    const int ratio = c.dsv4_compress_ratios.empty()
                          ? 0
                          : (int)c.dsv4_compress_ratios[(size_t)l];
    if (ratio <= 0) continue;
    const int kv_source = source_for(c.dsv41_kv_source_layers, l, ratio, c);
    const int index_source = source_for(c.dsv41_index_source_layers, l, ratio, c);
    engine.dsv41_kv_source_for_layer[(size_t)l] = kv_source;
    engine.dsv41_index_source_for_layer[(size_t)l] = index_source;
    if (kv_source == l) {
      const size_t cap = (size_t)dsv4_compressed_cache_capacity(max_ctx, ratio);
      compressed_bytes += cap * (size_t)qh * sizeof(float);
      // A KV source owns the index key cache even when a later layer owns
      // the query-side indexer that scores it.
      compressed_bytes += cap * (size_t)c.dsv4_indexer_key_length * sizeof(float);
      compressed_bytes += 2 * (size_t)ratio * (size_t)qh * sizeof(float);
    }
  }
  if (compressed_bytes && !engine.reserve_host_memory(compressed_bytes)) {
    LOG_ERROR("[MEM-GUARD] DeepSeek-V4.1 compressed state refused: %.2f MB\n",
              compressed_bytes / 1e6);
    return;
  }
  for (int l = 0; l < c.n_layers; ++l) {
    if (engine.dsv41_kv_source_for_layer[(size_t)l] != l) continue;
    const int ratio = (int)c.dsv4_compress_ratios[(size_t)l];
    const int index_dim = c.dsv4_indexer_key_length;
    engine.dsv41_compressed[(size_t)l].configure(ratio, qh, max_ctx, index_dim);
  }
  LOG_STREAM_INFO() << "[GRAPH] DeepSeek-V4.1 raw SWA/compressed state: "
                    << compressed_bytes / 1e9 << " GB, shared source caches\n";
}

// GLM5Next (GLM 5.3 Flash): DSA layers reuse the absorbed-MLA latent cache
// (Lc/Rc -- allocated for every layer; KDA layers simply never touch theirs,
// cheaper than a sparse layer-indexed map), and KDA layers additionally get
// their recurrent state (conv history + gated-delta-rule state, engine.hpp's
// glm5next_conv_history/glm5next_recur_state).
static void alloc_glm5next_cache(StreamingEngine &engine,
                                  const ArchitectureConfig &c, int max_ctx) {
  alloc_mla_absorbed_cache(engine, c, max_ctx);

  const int H = c.n_heads;
  const int hd = c.kda_head_dim;
  const int d_inner = H * hd;
  const int hist = c.ssm_conv_kernel - 1;
  engine.glm5next_conv_history.resize(c.n_layers);
  engine.glm5next_recur_state.resize(c.n_layers);
  size_t total_bytes = 0;
  int n_kda = 0;
  const auto is_kda = [&](int l) {
    return !c.bailingmoe3_head_count_kv.empty()
               ? bailingmoe3_is_kda_layer(c, l)
               : glm5next_is_kda_layer(c, l);
  };
  for (int l = 0; l < c.n_layers; ++l) {
    if (!is_kda(l)) continue;
    ++n_kda;
    total_bytes += ((size_t)3 * d_inner * hist + (size_t)H * hd * hd) * sizeof(float);
  }
  if (total_bytes && !engine.reserve_host_memory(total_bytes)) {
    LOG_ERROR("[MEM-GUARD] GLM5Next recurrent state refused: %.2f MB\n",
              total_bytes / 1e6);
    return;
  }
  for (int l = 0; l < c.n_layers; ++l) {
    if (!is_kda(l)) continue;
    engine.glm5next_conv_history[l].assign((size_t)3 * d_inner * hist, 0.0f);
    engine.glm5next_recur_state[l].assign((size_t)H * hd * hd, 0.0f);
  }
  LOG_STREAM_INFO() << "[GRAPH] GLM5Next KDA recurrent state: "
                    << total_bytes / 1e9 << " GB across " << n_kda
                    << " KDA layer(s) (conv history " << 3 * d_inner << "x" << hist
                    << " + state " << H << "x" << hd << "x" << hd << " per layer)\n";
}

// Qwen4Exp (Qwen3.8-Flash-Next): full-attention layers use the dense KV
// cache (Kc/Vc -- allocated for every layer; GDN layers simply never touch
// theirs, same "cheaper than a sparse layer-indexed map" tradeoff as
// GLM5Next's Lc/Rc above), and GDN layers additionally get their recurrent
// state (conv history + gated-delta-rule state, engine.hpp's
// qwen4exp_conv_history/qwen4exp_recur_state). PLE layers get their dilated
// conv history + the n-gram token window.
static void alloc_qwen4exp_cache(StreamingEngine &engine,
                                   const ArchitectureConfig &c, int max_ctx) {
  alloc_dense_kv_cache(engine, c, max_ctx);

  // Qwen4Exp's shared MTP checkpoint (config.mtp_model_path) adds one
  // extra full-attention layer at index c.n_layers -- alloc_dense_kv_cache
  // above sizes Kc/Vc to EXACTLY c.n_layers (correct for the ordinary,
  // non-MTP case, and shared with other dense-KV families that never
  // carry an extra layer). GLM5Next's own MLA cache already reserves this
  // same "+1" slot unconditionally (alloc_mla_absorbed_cache, this file);
  // do the equivalent here, but only when a shared MTP file is actually
  // configured, since qwen4exp_full_attention_gguf(engine, c.n_layers,
  // ...) (called from qwen4exp_mtp_forward, execution_layer.cpp) indexes
  // engine.Kc[c.n_layers]/Vc[c.n_layers] directly -- an out-of-bounds
  // read/write on the plain std::vector<float*> otherwise. Harmless to
  // allocate even if later validation (graph_initializer.cpp) declines to
  // activate MTP (missing/invalid tensors) -- one unused layer's worth of
  // KV cache, not charged if reserve_host_memory below refuses it.
  if (engine.has_mtp_model) {
    const int mtp_layer = c.n_layers;
    engine.Kc.resize(mtp_layer + 1, nullptr);
    engine.Vc.resize(mtp_layer + 1, nullptr);
    const int kvd = std::max(1, c.n_head_kv) * std::max(1, c.head_dim);
    const size_t kv_bytes = (size_t)max_ctx * kvd * sizeof(float);
    if (!alloc_zeroed_host(engine, &engine.Kc[mtp_layer], kv_bytes,
                           "Qwen4Exp MTP dense K cache") ||
        !alloc_zeroed_host(engine, &engine.Vc[mtp_layer], kv_bytes,
                           "Qwen4Exp MTP dense V cache")) {
      LOG_ERROR("[MEM-GUARD] Qwen4Exp MTP KV cache (layer %d) refused; MTP "
                "validation will fail closed\n", mtp_layer);
    }
  }

  const int n_v_heads = c.ssm_time_step_rank;
  const int hd = c.ssm_state_size;
  const int conv_dim =
      2 * c.ssm_group_count * hd + c.ssm_inner_size; // q|k|v channels
  const int hist = c.ssm_conv_kernel - 1;
  engine.qwen4exp_conv_history.resize(c.n_layers);
  engine.qwen4exp_recur_state.resize(c.n_layers);
  size_t total_bytes = 0;
  int n_gdn = 0;
  const auto add_checked = [](size_t &total, size_t bytes) -> bool {
    if (bytes > std::numeric_limits<size_t>::max() - total) return false;
    total += bytes;
    return true;
  };
  for (int l = 0; l < c.n_layers; ++l) {
    if (!qwen4exp_is_gdn_layer(c, l)) continue;
    ++n_gdn;
    const size_t elems = (size_t)conv_dim * (size_t)hist +
                         (size_t)n_v_heads * (size_t)hd * (size_t)hd;
    if (elems > std::numeric_limits<size_t>::max() / sizeof(float) ||
        !add_checked(total_bytes, elems * sizeof(float))) {
      LOG_ERROR("[MEM-GUARD] Qwen4Exp recurrent-state size overflow\n");
      engine.host_memory_allocation_failed = true;
      return;
    }
  }
  engine.qwen4exp_ple_conv_history.resize(c.n_layers);
  size_t ple_bytes = 0;
  for (int64_t l64 : c.ple_layers) {
    const int l = (int)l64;
    if (l < 0 || l >= c.n_layers) {
      LOG_ERROR("[MEM-GUARD] Qwen4Exp PLE layer index out of range: %d\n", l);
      engine.host_memory_allocation_failed = true;
      return;
    }
    const size_t ple_hist = (size_t)std::max(0, c.ple_conv_kernel - 1) *
                            (size_t)std::max(0, c.ple_ngram_size);
    const size_t hc_dim = (size_t)std::max(0, c.hc_mult) *
                          (size_t)std::max(0, c.hidden);
    if (ple_hist != 0 &&
        hc_dim > std::numeric_limits<size_t>::max() / ple_hist) {
      LOG_ERROR("[MEM-GUARD] Qwen4Exp PLE-state size overflow\n");
      engine.host_memory_allocation_failed = true;
      return;
    }
    const size_t elems = hc_dim * ple_hist;
    if (elems > std::numeric_limits<size_t>::max() / sizeof(float) ||
        !add_checked(ple_bytes, elems * sizeof(float))) {
      LOG_ERROR("[MEM-GUARD] Qwen4Exp PLE-state size overflow\n");
      engine.host_memory_allocation_failed = true;
      return;
    }
  }
  const size_t ple_tok_count =
      (size_t)std::max(0, c.ple_ngram_size - 1);
  if (ple_tok_count > std::numeric_limits<size_t>::max() / sizeof(int32_t) ||
      !add_checked(ple_bytes, ple_tok_count * sizeof(int32_t))) {
    LOG_ERROR("[MEM-GUARD] Qwen4Exp PLE token-history size overflow\n");
    engine.host_memory_allocation_failed = true;
    return;
  }
  if (ple_bytes > std::numeric_limits<size_t>::max() - total_bytes) {
    LOG_ERROR("[MEM-GUARD] Qwen4Exp host-state size overflow\n");
    engine.host_memory_allocation_failed = true;
    return;
  }
  const size_t host_state_bytes = total_bytes + ple_bytes;
  if (host_state_bytes && !engine.reserve_host_memory(host_state_bytes)) {
    LOG_ERROR("[MEM-GUARD] Qwen4Exp recurrent state refused: %.2f MB\n",
              (double)host_state_bytes / 1e6);
    return;
  }
  for (int l = 0; l < c.n_layers; ++l) {
    if (!qwen4exp_is_gdn_layer(c, l)) continue;
    engine.qwen4exp_conv_history[l].assign((size_t)conv_dim * hist, 0.0f);
    engine.qwen4exp_recur_state[l].assign((size_t)n_v_heads * hd * hd, 0.0f);
  }
  LOG_STREAM_INFO() << "[GRAPH] Qwen4Exp GDN recurrent state: "
                    << total_bytes / 1e9 << " GB across " << n_gdn
                    << " GDN layer(s) (conv history " << conv_dim << "x" << hist
                    << " + state " << n_v_heads << "x" << hd << "x" << hd
                    << " per layer)\n";

  for (int64_t l64 : c.ple_layers) {
    const int l = (int)l64;
    const size_t ple_hist =
        (size_t)(c.ple_conv_kernel - 1) * (size_t)c.ple_ngram_size;
    const size_t hc_dim = (size_t)c.hc_mult * (size_t)c.hidden;
    engine.qwen4exp_ple_conv_history[(size_t)l].assign(hc_dim * ple_hist, 0.0f);
  }
  if (!c.ple_layers.empty()) {
    engine.qwen4exp_ple_tok_hist.assign((size_t)std::max(0, c.ple_ngram_size - 1),
                                        (int32_t)c.ple_eos_token_id);
    engine.qwen4exp_ple_next_pos = 0;
    LOG_STREAM_INFO() << "[GRAPH] Qwen4Exp PLE state: " << ple_bytes / 1e9
                      << " GB conv history across " << c.ple_layers.size()
                      << " PLE layer(s) + " << engine.qwen4exp_ple_tok_hist.size()
                      << "-token n-gram window\n";
  }
}

// Shared by every family that uses mHC hyper-connections (DeepSeek-V4.1,
// GLM5Next): the per-layer collapse/combine hooks are the same form in all of
// them, only the model-level collapse differs and lives in the family's own
// code. Wired only when the GGUF actually sets hc_mult > 1; otherwise the
// family keeps the plain residual add every other family uses.
static void use_hyper_connections(FamilyOps &ops, const ArchitectureConfig &c) {
  ops.uses_hyper_connections = (c.hc_mult > 1);
  ops.hc_mult = c.hc_mult;
  if (c.hc_mult > 1) {
    ops.hc_pre_fn = &hc_pre_collapse;
    ops.hc_post_fn = &hc_post_combine;
  }
}

static void graph_owned_cache(StreamingEngine &, const ArchitectureConfig &, int) {}
static void graph_only_attention(StreamingEngine &, int, const float *, int, int, float *) {
  throw std::runtime_error("this family requires its registered full graph");
}

// ---------------------------------------------------------------------
// resolve_family_ops -- the single ModelFamily -> FamilyOps table.
// ---------------------------------------------------------------------

FamilyOps resolve_family_ops(ModelFamily fam, const ArchitectureConfig &c) {
  FamilyOps ops;
  ops.family = fam;
  ops.graph_model = full_graph_model(fam);
  switch (fam) {
  case ModelFamily::MiMo2:
    ops.attn_type = AttentionType::DenseQKV;
    ops.cache_shape = CacheShape::GraphOwnedKV;
    ops.default_gating = GatingFunc::Sigmoid;
    ops.attention_fn = &graph_only_attention;
    ops.alloc_cache_fn = &graph_owned_cache;
    break;
  case ModelFamily::GLM_DSA:
  case ModelFamily::DeepSeek_V3: // DeepSeek-V3 shares GLM-5.2's MLA core
    ops.attn_type = AttentionType::MLA;
    ops.cache_shape = CacheShape::MLAAbsorbed;
    ops.default_gating = GatingFunc::Sigmoid;
    ops.attention_fn = &mla_attention_absorbed_gguf;
    ops.alloc_cache_fn = &alloc_mla_absorbed_cache;
    break;

  case ModelFamily::DeepSeek_V4:
    // Raw SWA plus CSA/HCA compressed attention. c.hc_mult is a GGUF
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

  case ModelFamily::DeepSeek_V41:
    ops.attn_type = AttentionType::MLA;
    ops.cache_shape = CacheShape::MLARawSWA;
    ops.default_gating = GatingFunc::SqrtSoftplus;
    use_hyper_connections(ops, c);
    ops.attention_fn = &dsv41_attention_gguf;
    ops.alloc_cache_fn = &alloc_dsv41_swa_cache;
    ops.layer_input_fn = &deepseek41_engram_layer_input;
    break;

  case ModelFamily::GLM5NEXT:
    // Hybrid KDA (recurrent) + DSA (nope-only absorbed MLA). The DSA side
    // reuses the MLAAbsorbed cache shape (Lc/Rc); the KDA side's recurrent
    // state is allocated alongside it by alloc_glm5next_cache. Per-layer
    // HC hooks are the SAME build_hc_pre/build_hc_post form as DeepSeek-V4
    // (glm5next.cpp inherits them unchanged) -- only the MODEL-LEVEL
    // collapse differs (hc_mean, handled inside hc_collapse_output).
    // default_gating: GLM-family MoE convention (sigmoid router + selection
    // bias + routed_scaling_factor, per the converter's
    // norm_topk_prob/routed_scaling_factor/exp_probs_b wiring); the GGUF's
    // own expert_gating_func key still overrides this when set.
    ops.attn_type = AttentionType::MLA;
    ops.cache_shape = CacheShape::MLAAbsorbed;
    ops.default_gating = GatingFunc::Sigmoid;
    use_hyper_connections(ops, c);
    ops.attention_fn = &glm5next_attention_gguf;
    ops.alloc_cache_fn = &alloc_glm5next_cache;
    break;

  case ModelFamily::BailingMoE3:
    ops.attn_type = AttentionType::MLA;
    ops.cache_shape = CacheShape::MLAAbsorbed;
    ops.default_gating = GatingFunc::Sigmoid;
    ops.attention_fn = &bailingmoe3_attention_gguf;
    ops.alloc_cache_fn = &alloc_glm5next_cache;
    break;

  case ModelFamily::Qwen4Exp:
    // Hybrid GDN (recurrent) + dense full attention, low-rank hyper-
    // connections, PLE n-gram embedding, softmax-gated MoE. CacheShape is
    // DenseKV (full-attention layers only use it; GDN layers' recurrent
    // state is allocated alongside by alloc_qwen4exp_cache).
    // default_gating: Softmax, the Qwen3Next-convention router
    // (qwen4exp.cpp's build_layer_ffn passes
    // LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX); the GGUF's own
    // expert_gating_func key still overrides.
    // QSA sparse attention on the full-attention layers is DEFERRED in v1
    // (they run dense) -- same staging as GLM5NEXT's own indexer.
    ops.attn_type = AttentionType::SSM;
    ops.cache_shape = CacheShape::DenseKV;
    ops.default_gating = GatingFunc::Softmax;
    ops.uses_hyper_connections = (c.hc_mult > 1);
    ops.hc_mult = c.hc_mult;
    ops.attention_fn = &qwen4exp_attention_gguf;
    ops.alloc_cache_fn = &alloc_qwen4exp_cache;
    if (c.hc_mult > 1) {
      ops.hc_pre_fn = &qwen4exp_hc_pre_mix;
      ops.hc_post_fn = &qwen4exp_hc_post_combine;
    }
    if (!c.ple_layers.empty()) {
      ops.layer_input_fn = &qwen4exp_ple_layer_input;
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
