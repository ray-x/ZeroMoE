#include "cuda_manager.hpp"
#include "graph_config.hpp"
#include "dense_dispatch.hpp"
#include "dsv41_attention.hpp"
#include "dsv41_trace.hpp"
#include "engine.hpp"
#include "ggml_gdn_proj_backend.hpp"
#include "ggml_glm5next_proj_backend.hpp"
#include "ggml_hc_backend.hpp"
#include "kernels_cpu.hpp"
#include "kquant_dequant.hpp"
#include "perf_timer.hpp"
#include "qwen4exp_gdn.hpp" // qwen4exp_gdn_delta_step (the real, unit-tested GDN step)
#include "rope.hpp" // rope_yarn_inplace / yarn_kq_scale (the real, unit-tested RoPE)
#include "tensor_trace.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <unordered_map>
#include <vector>

extern bool qwen4exp_hc_device_live();

void sync_glm5next_kda_gpu_states(StreamingEngine &engine) {
  if (!engine.gpu || !engine.glm5next_kda_gpu)
    return;
  const auto &c = engine.model_file.architecture_config;
  const int H = c.n_heads;
  const int hd = c.kda_head_dim;
  if (H <= 0 || hd <= 0)
    return;
  const size_t elements = (size_t)H * hd * hd;
  for (int layer = 0; layer < c.n_layers; ++layer) {
    if (!glm5next_is_kda_layer(c, layer) ||
        layer >= (int)engine.glm5next_recur_state.size())
      continue;
    if (!engine.gpu->glm5next_kda_sync_state_to_host(
            layer, engine.glm5next_recur_state[(size_t)layer].data(),
            elements)) {
      LOG_STREAM_WARN() << "[GLM5NEXT-KDA] failed to synchronize layer "
                        << layer << " device state to host checkpoint\n";
    }
  }
}

void invalidate_glm5next_kda_gpu_states(StreamingEngine &engine) {
  if (engine.gpu && engine.glm5next_kda_gpu)
    engine.gpu->glm5next_kda_invalidate_device_states();
}

// A/B control for the S>1 (prefill) score GPU path ONLY. Set
// ZEROMOE_SCORE_PREFILL_LEGACY=1 to force prefill back through the CPU
// AVX-512 score loop while leaving decode (S==1) on the GPU path exactly as
// before -- i.e. it reproduces the pre-2026-09-02 gate from the SAME binary,
// so an A/B can be run interleaved in one session instead of against a
// baseline recorded earlier (this project has already been burned once by
// trusting a cross-session reference number; see CLAUDE.md's "GLM-5.2 is much
// slower than a week-old reference build" entry). Same function-local-static
// idiom as kquant_matmul.cu's own ZEROMOE_KQUANT_LEGACY.
// ---------------------------------------------------------------------------
// ZEROMOE_WUV_INT8=1  --  int8-persistent WUV storage (OFF by default)
// ---------------------------------------------------------------------------
// Stores the MLA score down-projection weights (WUV) as int8 + one f32 scale
// per output row instead of BF16, roughly HALVING their per-layer VRAM cost:
//   BF16: H*kvl*vh*2 bytes          int8: H*kvl*vh*1 + H*vh*4 bytes
// (upload_wuv_int8 / wuv_int8_batched_host / launch_dequant_wuv_int8, see
// cuda_manager.hpp). The GEMM itself is unchanged -- a per-call kernel
// re-expands the weights to BF16 and feeds the SAME BF16-matched strided-
// batched cuBLAS call the BF16 path uses, so this is purely a storage-format
// change, not a different numerical path.
//
// WHEN THIS IS WORTH TURNING ON: only when GPU VRAM is the binding constraint.
// WUV competes with qabsorb and the score KV mirror for a small post-static-
// pool margin on the CoreEngine GPU (~0.3-0.4GB on this box), and it is the
// one that loses -- every layer that misses out falls back to the CPU WUV
// loop. Half the bytes per layer means either roughly twice the layers
// covered in the same budget, or the same coverage for half the VRAM.
//
// WHY IT IS NOT THE DEFAULT -- real hardware, GLM-5.3, n=3+ per arm,
// interleaved from one binary, cache hit 74% in EVERY arm (no cache confound):
//
//     WUV GPU coverage   18/78  ->  35/78      (nearly doubled, as predicted)
//     CPU fallbacks       2460  ->   1763
//     score=             4.39s  ->  4.03s      (-8.2%, every int8 run below
//                                               every BF16 run -- real, not
//                                               noise)
//     whole request     94.03s  ->  94.25s     (FLAT, overlapping bands)
//
// The bucket-level win is real and repeatable; it just does not reach wall
// time. score= is only ~4.7% of a request that is ~85% expert streaming, so a
// 0.36s saving sits below this hardware's own run-to-run noise floor. Paying
// int8's accuracy cost (~17% worst-case relative error vs exact F32 at real
// dims -- see Test 15's own informational figure) for a saving you cannot
// measure end-to-end is a bad trade unless the VRAM itself is what you need.
//
// A HYPOTHESIS THAT MEASUREMENT KILLED, recorded so it is not re-derived: the
// per-call dequant kernel re-expands the whole [H,kvl,vh] weight set on EVERY
// call (~16.8MB of GPU writes x 1435 calls/request at real dims), which looks
// like it must eat the win and make this a CPU-time-for-GPU-time trade. It
// does not -- a dedicated counter (see perf_timer.hpp's "int8-persistent WUV
// variant active" report line) measures it at 0.015s total, 0.011ms/call.
// Essentially free. The saving is not being consumed; it is just small.
//
// HISTORY: shelved 2026-08-29 after a 113,000-error NaN cascade, with a
// standing "do not re-enable without finding the root cause" warning. That
// warning is SUPERSEDED. The cascade's FIRST error was
//   ActiveLayerUploadWorkers: cudaMemcpyAsync launch failed ... invalid
//   argument
// -- the exact signature of the slab_capacity=0 bug root-caused and fixed on
// 2026-09-01 (a MOVE-mode promotion reported capacity 0 on a still-ALLOCATED
// buffer, so ensure_slab_capacity free+realloc'd it out from under queued
// upload work). int8 was a VICTIM of that bug, not its cause: re-tested 4x on
// real hardware after the fix, 0 [ERROR] lines and coherent output every run.
// Correctness is covered by Test 15 (S=1) and Test 22 (S>1, incl. an
// anti-aliasing check the S=1 test structurally cannot perform).
static bool wuv_int8_enabled() {
  static const bool on = [] {
    const char *e = zeromoe::graphcfg::env("ZEROMOE_WUV_INT8");
    return e && e[0] == '1';
  }();
  return on;
}

static bool score_prefill_legacy_forced() {
  static const bool forced = [] {
    const char *e = zeromoe::graphcfg::env("ZEROMOE_SCORE_PREFILL_LEGACY");
    return e && e[0] == '1';
  }();
  return forced;
}


static bool gdn_full_stats_enabled() {
  return Logger::instance().would_log(LogLevel::DEBUG);
}

static void log_gdn_full_stats(const char *label, int layer_idx,
                               const float *data, int S, int D) {
  if (!gdn_full_stats_enabled() || !data || S <= 0 || D <= 0)
    return;
  double max_abs = 0.0;
  int max_row = -1;
  int bad = 0;
  for (int s = 0; s < S; ++s) {
    double row_max = 0.0;
    for (int d = 0; d < D; ++d) {
      const float x = data[(size_t)s * D + d];
      if (!std::isfinite(x)) {
        ++bad;
        continue;
      }
      row_max = std::max(row_max, (double)std::fabs(x));
    }
    if (row_max > max_abs) {
      max_abs = row_max;
      max_row = s;
    }
  }
  LOG_DEBUG("[GDN-FULL-STATS] %s layer=%d bad=%d max_abs=%g max_row=%d\n", label,
           layer_idx, bad, max_abs, max_row);
}

extern void matmul(float *y, const float *x, const float *W, int S, int I,
                   int O);
extern void softmax_row(float *x, int size);

// mla_attention_absorbed_gguf has never had NaN instrumentation -- these
// bisect its internal stages the same way execution_layer.cpp's check_finite
// bisects expert_ffn/moe_forward_gguf. Local copy rather than sharing that
// one (file-local static there) to avoid cross-TU plumbing for a diagnostic.
//
// Also emits a VERBOSE-gated L2 norm of the LAST TOKEN's slice (row S-1 of
// an [S, row_width] buffer), added while bisecting a real DeepSeek-V4-Flash
// run's ~25x hidden-state norm jump across a single layer's attention block
// (generation_loop.hpp's [TRACE l=...] post-attn line) -- that trace only
// sees the combined pre/post-attention hidden norm, not which internal
// stage of dsv4_attention_gguf (Q/RoPE, KV/MQA blend, or the grouped
// attn_output_a/b projection -- the latter's on-disk layout is explicitly
// unverified per this file's own dsv4_attention_gguf header comment) is
// where the magnitude actually inflates. row_width=0 skips the per-row norm
// (used for buffers, like the whole KVc ring, that aren't one-row-per-token).
static bool check_finite_attn(const char *label, int layer_idx,
                              const float *data, size_t n, int row_width = 0) {
  int nan_count = 0, inf_count = 0;
  for (size_t i = 0; i < n; ++i) {
    if (std::isnan(data[i]))
      ++nan_count;
    else if (std::isinf(data[i]))
      ++inf_count;
  }
  if (nan_count > 0 || inf_count > 0) {
    LOG_ERROR("[MLA] %s layer=%d: %d/%zu NaN, %d/%zu Inf.\n", label, layer_idx,
              nan_count, n, inf_count, n);
    return false;
  }
  if (row_width > 0 && trace_gate("MLA")) {
    trace_row("MLA", layer_idx, label, data, n, row_width);
  }
  return true;
}

// dense_proj() now lives in include/dense_dispatch.hpp (shared with the router
// and lm_head projections in execution_layer.cpp / generation_loop.hpp).

// Interleaved RoPE (LLAMA_ROPE_TYPE_NORM) with YaRN long-context scaling, on a
// vector of `dim` elements at sequence position `pos`. Applied to the decoupled
// qk_rope portion of MLA (the nope portion is unrotated) and, in
// dense_attention_gguf, to the full head_dim of Q/K.
//
// The real implementation lives in include/rope.hpp (rope_yarn_inplace) so it
// is unit-tested in isolation by tests/test_components.cpp::test_rope. This
// wrapper just unpacks ArchitectureConfig into the primitives rope.hpp takes.
//
// CONVENTION: interleaved rotation -- pairs (2j, 2j+1) -- REQUIRED by every MLA
// target this project runs (glm-dsa / ChatGLM / DeepSeek-V2/V3); confirmed
// against llama.cpp (rope_type = LLAMA_ROPE_TYPE_NORM for GLM_DSA/CHATGLM) and
// rope_interleave. The previous implementation here used NeoX
// rotate-half (pairs (k, k+half)) which scrambled positional encoding on every
// MLA layer and produced "always random text" -- see test_rope's known-answer
// case for the exact failure mode.
// Reduces to plain interleaved RoPE when c.rope_ext_factor == 0 (the default
// for any model without rope.scaling.type=yarn, including GLM-5.2).
static inline void apply_rope_yarn(float *v, int dim, int pos, float theta_base,
                                   const ArchitectureConfig &c) {
  rope_yarn_inplace(v, dim, pos, theta_base, c.rope_ext_factor,
                    c.rope_freq_scale, c.rope_attn_factor, c.rope_yarn_log_mul,
                    c.rope_n_ctx_orig);
}

// Attention-score temperature scale for YaRN, matching deepseek2.cpp's
// kq_scale derivation. Real implementation in include/rope.hpp (yarn_kq_scale);
// reduces to plain 1/sqrt(qh) when c.rope_ext_factor==0 (same no-YaRN reduction
// as apply_rope_yarn above).
static inline float compute_yarn_kq_scale(const ArchitectureConfig &c, int qh) {
  return yarn_kq_scale(c.rope_ext_factor, c.rope_freq_scale, c.rope_attn_factor,
                       c.rope_yarn_log_mul, qh);
}

// Weightless per-head RMSNorm (no learned weight tensor) matching llama.cpp's
// `q = ggml_rms_norm(ctx0, q, norm_rms_eps)` in deepseek4.cpp's
// build_attention_impl -- applied to the FULL [qh] Q vector right after the
// attn_q_b projection, before the nope/rope split and before RoPE. Confirmed
// by direct read of deepseek4.cpp: this call has no corresponding weight
// tensor multiply (unlike attn_q_a_norm/attn_kv_a_norm elsewhere in this
// file, which ARE weighted). Was missing entirely from dsv4_attention_gguf.
static inline void rmsnorm_inplace_noweight(float *v, int dim, float eps) {
  double ss = 0.0;
  for (int i = 0; i < dim; ++i)
    ss += (double)v[i] * v[i];
  float r = 1.0f / std::sqrt((float)(ss / dim) + eps);
  for (int i = 0; i < dim; ++i)
    v[i] *= r;
}

void extract_gguf_row_q4_0(const QuantizedTensor &tensor, int row_idx,
                           float *__restrict__ out_f32) {
  int I = tensor.meta.shape[1];
  int blocks_per_row = I / 32;
  const block_q4_0 *row_ptr =
      reinterpret_cast<const block_q4_0 *>(tensor.host_ptr) +
      (row_idx * blocks_per_row);

  const __m128i m4 = _mm_set1_epi8(0x0F);

  for (int b = 0; b < blocks_per_row; ++b) {
    float scale = fp16_to_fp32(row_ptr[b].d);
    __m512 v_scale = _mm512_set1_ps(scale);

    // 16 bytes = 32 nibbles: low nibble of byte i -> element 2*i, high nibble
    // -> element 2*i+1
    __m128i raw_bytes =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(row_ptr[b].qs));
    __m128i lo_nib = _mm_and_si128(raw_bytes, m4); // low nibbles (even indices)
    __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(raw_bytes, 4),
                                   m4); // high nibbles (odd indices)

    // Interleave: [lo0,hi0, lo1,hi1, ...] to get elements in order
    __m128i interleaved_lo =
        _mm_unpacklo_epi8(lo_nib, hi_nib); // bytes 0-7 -> 16 values
    __m128i interleaved_hi =
        _mm_unpackhi_epi8(lo_nib, hi_nib); // bytes 8-15 -> 16 values

    // Convert to f32 with bias subtraction (Q4_0: stored as unsigned [0,15],
    // bias = 8)
    __m512i vals_0 = _mm512_cvtepu8_epi32(interleaved_lo);
    __m512i vals_1 = _mm512_cvtepu8_epi32(interleaved_hi);
    __m512 f_0 =
        _mm512_cvtepi32_ps(_mm512_sub_epi32(vals_0, _mm512_set1_epi32(8)));
    __m512 f_1 =
        _mm512_cvtepi32_ps(_mm512_sub_epi32(vals_1, _mm512_set1_epi32(8)));

    _mm512_storeu_ps(out_f32 + (b * 32) + 0, _mm512_mul_ps(f_0, v_scale));
    _mm512_storeu_ps(out_f32 + (b * 32) + 16, _mm512_mul_ps(f_1, v_scale));
  }
}

// wuk_cache/wuv_cache: moved from mla_attention_absorbed_gguf's own
// function-local statics to file scope (2026-08-12) so warm_mla_absorb_
// cache below can populate them too -- see that function's own comment.
// Same lifetime either way (both are effectively program-lifetime storage),
// this is purely about who else can reach them.
static std::vector<std::vector<float>> wuk_cache, wuv_cache;

// Dequantize attn_k_b/attn_v_b (W_UK/W_UV) for one layer into wuk_cache/
// wuv_cache, if not already cached. STATIC MODEL WEIGHTS ONLY -- identical
// on every call for a given layer_idx, completely independent of the
// prompt/tokens/S (confirmed: this function takes no token data at all).
// That's what makes it safe to call eagerly, once, right after model load
// finishes (warm_mla_absorb_cache) instead of lazily on first real
// attention call -- a real prompt's own size/content can never change what
// this computes. Returns false only on a genuine missing-weight error
// (logged); the two callers handle that differently (the real attention
// path zeros its output and bails, warmup just skips this layer and
// continues to the next). Idempotent -- a second call for an
// already-cached layer_idx is a cheap no-op (WUK.empty() check), so the
// lazy call site inside mla_attention_absorbed_gguf is left completely
// unchanged as a safety net for any call that happens before warmup runs
// (or if warmup is ever skipped/disabled).
static bool ensure_wuk_wuv_cached(StreamingEngine &engine, int layer_idx) {
  const auto &c = engine.model_file.architecture_config;
  int H = c.n_heads;
  int kvl = c.kv_lora;
  int nope = c.qk_nope;
  int vh = c.v_head;
  std::string p = "blk." + std::to_string(layer_idx) + ".";

  auto wkb_it = engine.resident_layers.find(p + "attn_k_b.weight");
  auto wvb_it = engine.resident_layers.find(p + "attn_v_b.weight");
  if (wkb_it == engine.resident_layers.end() ||
      wvb_it == engine.resident_layers.end() || !wkb_it->second.host_ptr ||
      !wvb_it->second.host_ptr) {
    return false;
  }

  if ((int)wuk_cache.size() <= layer_idx) {
    const int cache_layers =
        c.n_layers +
        ((engine.mtp_enabled && c.n_layer_nextn > 0) ? c.n_layer_nextn : 0);
    wuk_cache.resize(cache_layers);
    wuv_cache.resize(cache_layers);
  }
  std::vector<float> &WUK = wuk_cache[layer_idx];
  std::vector<float> &WUV = wuv_cache[layer_idx];
  if (!WUK.empty())
    return true; // already cached -- warmup already ran, or a prior real call
                 // did

  WUK.resize((size_t)nope * kvl * H);
  WUV.resize((size_t)kvl * vh * H);
  // Sanity check BEFORE dequanting: nope/kvl/H come from ArchitectureConfig
  // (GGUF metadata KEYS, e.g. qk_nope_head_dim), completely independent of
  // the tensor's own ne[] dims read at load time. If a wrong metadata key
  // or wrong default made any of those config values drift from the REAL
  // tensor, WUK.size() undercounts or overcounts the real element count,
  // and dequant_row() below either stops short or reads past the tensor's
  // real byte extent into whatever memory follows it -- which would
  // produce exactly a PARTIAL nan/garbage pattern (early elements, still
  // within the real tensor, decode fine; later ones don't), not total
  // corruption. That partial pattern is what's actually been observed.
  uint64_t real_numel_k = 1;
  for (auto d : wkb_it->second.meta.shape)
    real_numel_k *= d;
  uint64_t real_numel_v = 1;
  for (auto d : wvb_it->second.meta.shape)
    real_numel_v *= d;
  if (layer_idx == 0) {
    LOG_STREAM_INFO() << "[MLA] attn_k_b.weight type="
                      << gpu_type_name(wkb_it->second.meta.type)
                      << " numel=" << real_numel_k
                      << " (config nope*kvl*H=" << WUK.size() << ") | "
                      << "attn_v_b.weight type="
                      << gpu_type_name(wvb_it->second.meta.type)
                      << " numel=" << real_numel_v
                      << " (config kvl*vh*H=" << WUV.size() << ")\n";
  }
  if (real_numel_k != WUK.size() || real_numel_v != WUV.size()) {
    std::string shape_k, shape_v;
    for (auto d : wkb_it->second.meta.shape)
      shape_k += std::to_string(d) + ",";
    for (auto d : wvb_it->second.meta.shape)
      shape_v += std::to_string(d) + ",";
    LOG_ERROR(
        "[MLA] layer=%d tensor/config shape MISMATCH: attn_k_b real shape=[%s] "
        "(numel=%llu) vs config nope*kvl*H=%d*%d*%d=%zu; "
        "attn_v_b real shape=[%s] (numel=%llu) vs config kvl*vh*H=%d*%d*%d=%zu "
        "-- "
        "dequant_row is about to read the WRONG number of elements from "
        "host_ptr.\n",
        layer_idx, shape_k.c_str(), (unsigned long long)real_numel_k, nope, kvl,
        H, WUK.size(), shape_v.c_str(), (unsigned long long)real_numel_v, kvl,
        vh, H, WUV.size());
  }
  dequant_row(wkb_it->second.meta.type, wkb_it->second.host_ptr, WUK.data(),
              (int)WUK.size());
  dequant_row(wvb_it->second.meta.type, wvb_it->second.host_ptr, WUV.data(),
              (int)WUV.size());
  check_finite_attn("WUK (attn_k_b dequant)", layer_idx, WUK.data(),
                    WUK.size());
  check_finite_attn("WUV (attn_v_b dequant)", layer_idx, WUV.data(),
                    WUV.size());
  return true;
}

// Call ONCE, right after model load finishes and before any prompt is
// timed (main.cpp) -- moves attn_k_b/attn_v_b dequant (ensure_wuk_wuv_
// cached above) out of the first real prefill call's own timing window
// entirely. Real measurement (2026-08-12): this cost 1.59s inside the
// FIRST prefill call's "absorb" sub-phase (confirmed to be a one-time,
// model-only cost -- see ensure_wuk_wuv_cached's own comment -- so it was
// never really a "prefill" cost, just something that happened to run
// during prefill because prefill is whichever forward pass runs first).
// Safe to call unconditionally: ensure_wuk_wuv_cached is idempotent, a
// missing-weight layer is just skipped here (logged) rather than zeroing
// an output buffer that doesn't exist yet at this point in startup, and
// the lazy call site inside mla_attention_absorbed_gguf is completely
// unchanged as a fallback for any layer this didn't warm (e.g. a
// DeepSeek-V4 model, which uses the structurally separate dsv4_attention_
// gguf/woa_cache path this doesn't touch).
// Reserved so qabsorb/score/WUV's own cudaMemGetInfo-based budget checks
// below never claim the WHOLE currently-free margin on a dual-role GPU
// (e.g. CoreEngine/GPU1, ~1.8-1.9GB free after static pools on this
// project's own hardware). Without this, CudaManager::ensure_buffers'
// shared d_x/d_y scratch -- grown lazily by attn_output's dense_proj call
// (dense_matmul_host) and rmsnorm_gpu_host, both routed through it, and
// needing to grow largest during PREFILL's bigger S -- had nothing left to
// grow into: a real, deterministic (6/6 runs) production incident showed
// `ensure_buffers: d_x grow to 0.72 MB failed: out of memory` firing 78/78
// times (once per layer) every single run, forcing attn_output's GPU
// dense_proj onto its CPU fallback for part of every layer during prefill
// -- a real ~40% cost on that one step (outproj 6.48s -> 3.85s once fixed
// by disabling a DIFFERENT feature, ram_pinned_budget_gb, that happened to
// eat further into the same already-razor-thin margin; see CLAUDE.md).
// Lowering the dual-role slice size does NOT fix this on its own -- these
// budget checks size themselves off *however much is currently free*, so
// they simply claim more, reproducing the same near-zero remainder. This
// reserve is a direct, scoped fix: hold back a fixed slice BEFORE either
// budget check runs, so d_x/d_y's later prefill-time grow has somewhere to
// go regardless of how aggressively qabsorb/score/WUV spend the rest.
// Prefill grows d_x/d_y/d_score_combined with the batch (about 65 KB + 110 KB per token); 256MB covers decode and
// short prompts but a 426-token GLM-5.3 prefill still failed 446 cudaMallocs and ran attn_output on the CPU
// (outproj 131 s of 400 s). ZEROMOE_DENSE_SCRATCH_RESERVE_MB raises it.
static size_t dense_scratch_reserve_bytes() {
  static const size_t v = [] {
    const char *e = zeromoe::graphcfg::env("ZEROMOE_DENSE_SCRATCH_RESERVE_MB");
    const long mb = (e && e[0]) ? std::atol(e) : 256;
    return (size_t)(mb > 0 ? mb : 256) * 1024ULL * 1024ULL;
  }();
  return v;
}

void warm_mla_absorb_cache(StreamingEngine &engine) {
  const auto &c = engine.model_file.architecture_config;
  if (c.n_layers <= 0 || c.kv_lora <= 0)
    return; // not an MLA-absorbed-attention model
  // A GLM MTP tail is an ordinary absorbed-MLA/DSA block at index
  // c.n_layers. Warm its static WUK/WUV data and optional GPU mirrors too;
  // target KV state remains sized from c.n_layers, while the allocator's
  // existing +1 slot is reserved for this exact tail.
  const int cache_layers =
      c.n_layers +
      ((engine.mtp_enabled && c.n_layer_nextn > 0) ? c.n_layer_nextn : 0);
  PERF_START(mla_absorb_warmup);
  int warmed = 0, skipped = 0;
  for (int layer_idx = 0; layer_idx < cache_layers; ++layer_idx) {
    // GLM5Next: KDA (recurrent) layers have no attn_k_b/attn_v_b at all
    // -- skip them explicitly rather than logging them as "missing
    // weights" (which would suggest a load problem that doesn't exist).
    if (glm5next_is_kda_layer(c, layer_idx)) {
      ++skipped;
      continue;
    }
    if (ensure_wuk_wuv_cached(engine, layer_idx))
      ++warmed;
    else
      ++skipped;
  }
  PERF_END(mla_absorb_warmup, t_mla_absorb_warmup);
  LOG_STREAM_INFO()
      << "[MLA] Warmed attn_k_b/attn_v_b dequant cache for " << warmed
      << " layer(s) "
         "("
      << skipped << " skipped, missing weights) in "
      << perf().t_mla_absorb_warmup.load()
      << "s -- moved out of the first prefill call's own timing window.\n";

  // qabsorb BF16 batched GPU fast-path (EngineConfig::attn_qabsorb_gpu) --
  // one-time per-layer WUK upload, right here alongside the CPU dequant
  // above (same "static model weight, safe to do eagerly at startup"
  // reasoning ensure_wuk_wuv_cached's own comment already established).
  // Real, measured cudaMemGetInfo headroom at this point in startup
  // (model load + VRAM allocation fully done, same reason this whole
  // function is called here rather than lazily), capped by the config
  // ceiling -- never assumes free VRAM, and a layer that doesn't fit just
  // stays on the CPU qabsorb path (n_qabsorb_gpu_fallback_cpu), same
  // best-effort discipline as every other GPU fast-path in this codebase.
  if (engine.attn_qabsorb_gpu && engine.gpu) {
    int H = c.n_heads, nope = c.qk_nope, kvl = c.kv_lora;
    cudaSetDevice(engine.gpu->device_id);
    size_t free_b = 0, total_b = 0;
    size_t budget = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
      size_t cfg_cap = (size_t)(engine.attn_qabsorb_gpu_max_vram_gb * 1e9);
      size_t usable_free = (free_b > dense_scratch_reserve_bytes())
                               ? (free_b - dense_scratch_reserve_bytes())
                               : 0;
      budget = std::min(usable_free, cfg_cap);
    } else {
      LOG_STREAM_WARN()
          << "[MLA] qabsorb BF16 GPU fast-path: cudaMemGetInfo failed -- "
             "skipping, all layers stay on the CPU qabsorb path\n";
    }
    int gpu_warmed = 0, gpu_skipped = 0;
    size_t used = 0;
    size_t per_layer_bytes =
        (size_t)H * (size_t)nope * (size_t)kvl * sizeof(uint16_t);
    for (int layer_idx = 0; layer_idx < cache_layers; ++layer_idx) {
      if ((size_t)layer_idx >= wuk_cache.size() ||
          wuk_cache[layer_idx].empty()) {
        ++gpu_skipped;
        continue;
      }
      if (used + per_layer_bytes > budget) {
        ++gpu_skipped;
        continue;
      }
      if (engine.gpu->upload_wuk_bf16(layer_idx, wuk_cache[layer_idx], H, nope,
                                      kvl, budget - used)) {
        used += per_layer_bytes;
        ++gpu_warmed;
      } else {
        ++gpu_skipped;
      }
    }
    LOG_STREAM_INFO() << "[MLA] qabsorb BF16 GPU fast-path: " << gpu_warmed
                      << "/" << cache_layers << " layer(s) uploaded ("
                      << gpu_skipped
                      << " skipped -- VRAM budget exhausted or "
                         "missing weights), "
                      << (used / 1e9) << " GB of " << (budget / 1e9)
                      << " GB budget used\n";
  }

  // score GPU fast-path (EngineConfig::attn_score_gpu) -- allocate each
  // layer's persistent KV latent cache GPU mirror here, alongside qabsorb's
  // own startup warmup above (same "do it once at startup, real
  // cudaMemGetInfo headroom, never assume free VRAM" reasoning). Sized to
  // engine.max_t so it never needs a mid-generation realloc -- the mirror
  // is filled incrementally per-token afterward (mla_attention_absorbed_
  // gguf's own KV-write step, via upload_lc_rc_rows), never re-uploaded
  // wholesale. A layer whose mirror doesn't fit just stays on the CPU
  // score path (n_score_gpu_fallback_cpu), same best-effort discipline.
  if (engine.attn_score_gpu && engine.gpu) {
    int rope = c.qk_rope, kvl = c.kv_lora;
    // attn_score_gpu_max_context: caps the mirror's ALLOCATED size
    // independently of context_size, freeing VRAM for WUV below without
    // reducing the model's real context_size -- see that field's own
    // comment (engine_config.hpp) for the real VRAM math this exists
    // to solve. 0 (default) means no cap.
    int mirror_max_t = (engine.attn_score_gpu_max_context > 0 &&
                        engine.attn_score_gpu_max_context < engine.max_t)
                           ? engine.attn_score_gpu_max_context
                           : engine.max_t;
    cudaSetDevice(engine.gpu->device_id);
    size_t free_b = 0, total_b = 0;
    size_t budget = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
      size_t cfg_cap = (size_t)(engine.attn_score_gpu_max_vram_gb * 1e9);
      size_t usable_free = (free_b > dense_scratch_reserve_bytes())
                               ? (free_b - dense_scratch_reserve_bytes())
                               : 0;
      budget = std::min(usable_free, cfg_cap);
    } else {
      LOG_STREAM_WARN()
          << "[MLA] score GPU fast-path: cudaMemGetInfo failed -- "
             "skipping, all layers stay on the CPU score path\n";
    }
    int gpu_warmed = 0, gpu_skipped = 0;
    size_t used = 0;
    size_t per_layer_bytes =
        (size_t)mirror_max_t * ((size_t)kvl + (size_t)rope) * sizeof(float);
    // Tracks which layers actually got a mirror -- WUV upload below
    // only proceeds for these (a layer's WUV is pure wasted VRAM
    // without a mirror, since score_gpu_batched_host requires BOTH to
    // even attempt the GPU path; see wuv_bf16_batched_host's own call
    // site, attention_layer.cpp's score dispatch).
    std::vector<bool> mirror_ready(cache_layers, false);
    for (int layer_idx = 0; layer_idx < cache_layers; ++layer_idx) {
      if (used + per_layer_bytes > budget) {
        ++gpu_skipped;
        continue;
      }
      if (engine.gpu->ensure_lc_rc_gpu(layer_idx, mirror_max_t, kvl, rope,
                                       budget - used)) {
        used += per_layer_bytes;
        mirror_ready[layer_idx] = true;
        ++gpu_warmed;
      } else {
        ++gpu_skipped;
      }
    }
    LOG_STREAM_INFO() << "[MLA] score GPU fast-path: " << gpu_warmed << "/"
                      << cache_layers << " layer(s) KV mirror allocated ("
                      << gpu_skipped
                      << " skipped -- VRAM budget "
                         "exhausted), "
                      << (used / 1e9) << " GB of " << (budget / 1e9)
                      << " GB budget used (mirror_max_t=" << mirror_max_t
                      << ", context_size=" << engine.max_t
                      << (mirror_max_t < engine.max_t
                              ? ", CAPPED via attn_score_gpu_max_context"
                              : "")
                      << ")\n";

    // WUV int8 upload -- the score GPU path's own down-projection step
    // (wuv_int8_batched_host, cuda_manager.hpp), added after a real
    // production run showed the CPU per-head OMP-parallel WUV loop
    // costing 0.916ms/call (2.86s over one 40-token decode run) -- the
    // SAME "OMP team-spawn cost exceeds a tiny per-call workload"
    // pattern qabsorb's own gather fix already diagnosed and fixed at a
    // different call site. Shares the remaining attn_score_gpu_max_
    // vram_gb budget with the Lc/Rc mirror above (both are part of the
    // same feature).
    // 2026-08-29: switched from a flat BF16 upload to int8+per-row-scale
    // (upload_wuv_int8) -- BF16's real per-layer footprint (H*kvl*vh*2
    // bytes, e.g. 64*512*256*2=16.8MB/layer) only fit 21/78 layers in
    // this GPU's already-exhausted remaining budget; int8 (1 byte/
    // element + a small per-row scale) roughly halves that, buying real
    // coverage in the SAME existing budget with no VRAM displaced from
    // anywhere else -- see CLAUDE.md's WUV-coverage entries for the
    // real numbers that motivated this.
    int vh = c.v_head;
    int gpu_warmed_wuv = 0, gpu_skipped_wuv = 0;
    size_t used_wuv = 0;
    size_t remaining_budget = (used < budget) ? (budget - used) : 0;
    // 2026-08-29 -> 2026-09-02, the corrected history of this block.
    // An int8-persistent WUV format was tried here to roughly double GPU
    // coverage within this GPU's already-exhausted VRAM budget, and was
    // REVERTED the same day after real hardware showed a severe NaN
    // cascade (113K+ errors, garbled output), bisected to this change.
    // The conclusion drawn at the time -- "the real root cause is still
    // unknown, a future session should NOT re-enable this" -- was WRONG,
    // and is superseded. The cascade's FIRST error was
    //   ActiveLayerUploadWorkers: cudaMemcpyAsync launch failed ... invalid
    //   argument
    // i.e. the slab_capacity=0 bug root-caused and fixed on 2026-09-01,
    // in a completely different subsystem: a MOVE-mode promotion reported
    // capacity 0 on a still-ALLOCATED buffer, so ensure_slab_capacity
    // free+realloc'd it out from under queued upload work. The int8 path
    // was a VICTIM of that bug, not its cause -- re-tested 4x on real
    // hardware after the fix, 0 [ERROR] lines and coherent output every
    // run. (The one hypothesis actually tested back in 2026-08-29 -- the
    // unbudgeted dequant scratch buffer -- was correctly ruled out; the
    // mistake was concluding "cause unknown, therefore this path is
    // guilty" rather than "cause unknown, therefore keep looking".)
    //
    // It is still OFF by default, but for a completely different reason:
    // it simply is not a throughput win. See wuv_int8_enabled() at the
    // top of this file for the real A/B numbers -- in short, coverage
    // 18/78 -> 35/78 and score= -8.2%, both real and repeatable, but
    // whole-request time flat, because score= is only ~4.7% of a request
    // dominated by expert streaming. Worth enabling when VRAM is the
    // binding constraint; otherwise int8's accuracy cost buys nothing
    // measurable.
    //
    // GENERALIZABLE LESSON, worth restating alongside this file's others:
    // a correctness bug's FIRST symptom can name a subsystem that is only
    // a downstream victim. Bisection correctly proved "the cascade
    // disappears when this change is reverted" -- but that establishes
    // that the change EXPOSES the bug, not that it CONTAINS it. Those are
    // different claims, and the difference cost this feature four days.
    const bool wuv_int8 = wuv_int8_enabled();
    // int8 stores one byte per weight plus one f32 scale per output row,
    // vs BF16's two bytes per weight -- a little under half the footprint,
    // which is the entire point of the format (more layers covered in the
    // SAME already-exhausted budget).
    size_t per_layer_wuv_bytes =
        wuv_int8 ? ((size_t)c.n_heads * (size_t)c.kv_lora * (size_t)vh *
                        sizeof(int8_t) +
                    (size_t)c.n_heads * (size_t)vh * sizeof(float))
                 : ((size_t)c.n_heads * (size_t)c.kv_lora * (size_t)vh *
                    sizeof(uint16_t));
    // The int8 path needs a transient full-BF16 dequant scratch buffer
    // (d_wuv_dequant_bf16, H*kvl*vh*2 bytes) that is allocated lazily and
    // is NOT otherwise charged to this budget. Reserve it up front so the
    // per-layer loop cannot spend VRAM that buffer will later need. NOTE:
    // this reserve was already tried on 2026-08-29 and did NOT fix the
    // corruption -- it is kept because unbudgeted VRAM on an already-
    // starved GPU is a real hazard on its own, not because it is the fix.
    if (wuv_int8) {
      size_t dequant_scratch =
          (size_t)c.n_heads * (size_t)c.kv_lora * (size_t)vh * sizeof(uint16_t);
      remaining_budget = (remaining_budget > dequant_scratch)
                             ? (remaining_budget - dequant_scratch)
                             : 0;
    }
    for (int layer_idx = 0; layer_idx < cache_layers; ++layer_idx) {
      if (!mirror_ready[layer_idx]) {
        ++gpu_skipped_wuv;
        continue;
      } // WUV is wasted VRAM without a mirror
      if ((size_t)layer_idx >= wuv_cache.size() ||
          wuv_cache[layer_idx].empty()) {
        ++gpu_skipped_wuv;
        continue;
      }
      if (used_wuv + per_layer_wuv_bytes > remaining_budget) {
        ++gpu_skipped_wuv;
        continue;
      }
      if (wuv_int8
              ? engine.gpu->upload_wuv_int8(layer_idx, wuv_cache[layer_idx],
                                            c.n_heads, c.kv_lora, vh,
                                            remaining_budget - used_wuv)
              : engine.gpu->upload_wuv_bf16(layer_idx, wuv_cache[layer_idx],
                                            c.n_heads, c.kv_lora, vh,
                                            remaining_budget - used_wuv)) {
        used_wuv += per_layer_wuv_bytes;
        ++gpu_warmed_wuv;
      } else {
        ++gpu_skipped_wuv;
      }
    }
    LOG_STREAM_INFO() << "[MLA] WUV BF16 GPU fast-path: " << gpu_warmed_wuv
                      << "/" << cache_layers << " layer(s) uploaded ("
                      << gpu_skipped_wuv
                      << " skipped -- VRAM budget exhausted, "
                         "no mirror, or missing weights), "
                      << (used_wuv / 1e9) << " GB of "
                      << (remaining_budget / 1e9)
                      << " GB remaining budget used\n";
  }
}

void mla_attention_absorbed_gguf(StreamingEngine &engine, int layer_idx,
                                 const float *hidden_states, int S,
                                 int pos_base, float *out_context) {
  PERF_SCOPE(t_attn);
  PERF_START(attn_preamble);
  const auto &c = engine.model_file.architecture_config;
  int H = c.n_heads;
  int D = c.hidden;
  int kvl = c.kv_lora;
  int nope = c.qk_nope;
  int rope = c.qk_rope;
  int vh = c.v_head;
  int qh = nope + rope;
  std::string p = "blk." + std::to_string(layer_idx) + ".";
  float rope_theta =
      engine.model_file.get_config_float("rope.freq_base", 10000.0f);
  PERF_END(attn_preamble, t_attn_preamble);

  // ---- 1. attention input norm (F32 rmsnorm on GPU) ----
  PERF_START(attn_norm);
  std::vector<float> normed((size_t)S * D);
  rmsnorm_proj(engine, p + "attn_norm.weight", hidden_states, normed.data(), S,
               D, c.eps);
  check_finite_attn("normed (post attn_norm rmsnorm)", layer_idx, normed.data(),
                    (size_t)S * D, D);
  PERF_END(attn_norm, t_attn_norm);

  // ---- 2. project Q: q_a -> q_a_norm -> q_b  =>  Q [S, H*qh] ----
  // FIX: the live loop previously fed RAW hidden_states as the query and never
  // projected, so MLA attention was structurally wrong. This wires the real
  // low-rank query projection. RoPE is applied to q_rope in step 2b below.
  PERF_START(attn_qproj);
  int q_lora = engine.model_file.get_config_int("attention.q_lora_rank", 2048);
  std::vector<float> Q((size_t)S * H * qh, 0.0f);
  {
    std::vector<float> ql_buf((size_t)S * q_lora);
    {
      NamedTimerScope _t("attn_q_a");
      dense_proj(engine, p + "attn_q_a.weight", normed.data(), ql_buf.data(), S,
                 D, q_lora);
    }
    rmsnorm_proj(engine, p + "attn_q_a_norm.weight", ql_buf.data(),
                 ql_buf.data(), S, q_lora, c.eps);
    {
      NamedTimerScope _t("attn_q_b");
      dense_proj(engine, p + "attn_q_b.weight", ql_buf.data(), Q.data(), S,
                 q_lora, H * qh);
    }
  }
  check_finite_attn("Q (post attn_q_a/q_a_norm/q_b projection, pre-RoPE)",
                    layer_idx, Q.data(), Q.size());

  // ---- 2b. apply RoPE to the decoupled q_rope portion of each head ----
  for (int s = 0; s < S; ++s) {
    int pos = pos_base + s;
    for (int h = 0; h < H; ++h) {
      float *q_rope = Q.data() + ((size_t)s * H + h) * qh + nope;
      apply_rope_yarn(q_rope, rope, pos, rope_theta, c);
    }
  }
  check_finite_attn("Q (post RoPE)", layer_idx, Q.data(), Q.size(), H * qh);
  PERF_END(attn_qproj, t_attn_qproj);

  // ---- 3. project KV (kv_a_mqa) and WRITE Lc / Rc ----
  // FIX: Lc/Rc were allocated+zeroed but never written in the live path, so
  // the KV cache stayed empty. This populates the absorbed latent + rope caches
  // at [pos_base+s].
  PERF_START(attn_kvproj);
  {
    int kv_dim = kvl + rope;
    std::vector<float> kv((size_t)S * kv_dim);
    {
      NamedTimerScope _t("attn_kv_a_mqa");
      dense_proj(engine, p + "attn_kv_a_mqa.weight", normed.data(), kv.data(),
                 S, D, kv_dim);
    }
    check_finite_attn("kv (post attn_kv_a_mqa projection, pre-norm/RoPE)",
                      layer_idx, kv.data(), kv.size());
    auto kvn = engine.resident_layers.find(p + "attn_kv_a_norm.weight");
    const float *kvn_w =
        (kvn != engine.resident_layers.end() && kvn->second.host_ptr)
            ? reinterpret_cast<float *>(kvn->second.host_ptr)
            : nullptr;
    for (int s = 0; s < S; ++s) {
      int pos = pos_base + s;
      if (pos >= engine.max_t)
        continue;
      float *kv_row = kv.data() + (size_t)s * kv_dim;
      float *latent = kv_row;       // first kvl elements
      float *k_rope = kv_row + kvl; // last rope elements
      float *Lc_dst = engine.Lc[layer_idx] + (int64_t)pos * kvl;
      if (kvn_w)
        rmsnorm_row(Lc_dst, latent, kvn_w, kvl, c.eps);
      else
        std::memcpy(Lc_dst, latent, (size_t)kvl * sizeof(float));
      apply_rope_yarn(k_rope, rope, pos, rope_theta, c);
      std::memcpy(engine.Rc[layer_idx] + (int64_t)pos * rope, k_rope,
                  (size_t)rope * sizeof(float));
    }
    // score GPU fast-path (EngineConfig::attn_score_gpu): push the rows
    // just written above to this layer's persistent GPU mirror, same
    // positions/same call, so score_gpu_batched_host always sees a
    // mirror that's exactly as current as the CPU-side Lc/Rc it would
    // otherwise read. Best-effort -- upload_lc_rc_rows returning false
    // just means this layer's mirror was never allocated (VRAM budget
    // at startup); the score step below falls back to CPU either way.
    if (engine.attn_score_gpu && engine.gpu) {
      int valid_end = std::min(pos_base + S, engine.max_t);
      int valid_rows = valid_end - pos_base;
      if (valid_rows > 0) {
        engine.gpu->upload_lc_rc_rows(layer_idx, engine.Lc[layer_idx],
                                      engine.Rc[layer_idx], pos_base,
                                      valid_rows, kvl, rope);
      }
    }
  }
  PERF_END(attn_kvproj, t_attn_kvproj);

  // ---- 4. absorbed scoring + value aggregation, now over projected Q + real
  // KV ---- GLM-5.2 (glm-dsa) shares DeepSeek-V3's MLA core: same tensors &
  // absorption. GLM-5.2-only extras NOT handled here: DSA indexer (sparse top-k
  // attention — full causal is equivalent for ctx <= indexer.top_k), MTP/NextN
  // (unused). WUK/WUV (attn_k_b/attn_v_b dequant) moved to
  // ensure_wuk_wuv_cached (file scope, above) + warm_mla_absorb_cache
  // (2026-08-12) -- STATIC MODEL WEIGHTS, identical on every call for a given
  // layer_idx, so this is now normally already warm by the time we get here
  // (t_attn_absorb should read ~0, confirming warmup ran) with this call
  // surviving only as a safety-net fallback for whatever warmup didn't cover.
  PERF_START(attn_absorb);
  bool have_wuk_wuv = ensure_wuk_wuv_cached(engine, layer_idx);
  PERF_END(attn_absorb, t_attn_absorb);
  if (!have_wuk_wuv) {
    LOG_ERROR("[MLA] layer %d missing attn_k_b/attn_v_b — zeroing attention\n",
              layer_idx);
    std::memset(out_context, 0, (size_t)S * D * sizeof(float));
    return;
  }
  std::vector<float> &WUK = wuk_cache[layer_idx];
  std::vector<float> &WUV = wuv_cache[layer_idx];

  // Batch q_absorbed across ALL S tokens, once per head, BEFORE the
  // per-(s,h) score loop below -- WUK_h is the SAME weight matrix every
  // token in this batch reuses, so one real GEMM per head (S rows at
  // once) gives it genuine reuse instead of S separate GEMVs each
  // re-reading it fresh. Real prefill-specific win (S>1) -- at S=1
  // (decode) this degenerates to exactly the old per-token call
  // (matmul(...,1,nope,kvl)), byte-identical, no separate code path.
  // Bonus, not just arithmetic intensity: matmul (kernels_cpu.cpp) has
  // its OWN internal `#pragma omp parallel for` over its O=kvl output
  // dim -- but OMP nested parallelism is disabled by default in this
  // codebase (confirmed: no omp_set_nested/omp_set_max_active_levels
  // call anywhere), so calling it from INSIDE the per-(s,h) `collapse(2)`
  // parallel region below (the old code) silently ran matmul's own
  // parallel region on a single thread every time. Calling it here,
  // before that outer parallel region starts, lets matmul's own
  // parallelism actually engage.
  // Numerically identical to the old per-token calls, not just
  // "close": matmul's inner loop accumulates independently per (s,o)
  // pair (fresh accumulator each row, kernels_cpu.cpp) -- batching rows
  // into one call changes nothing about any single row's own
  // computation or accumulation order.
  PERF_START(attn_qabsorb_batch);
  std::vector<float> Q_absorbed((size_t)H * S * kvl);
  {
    // REAL, MEASURED REGRESSION caught and fixed same-session (2026-08-
    // 12): the first version of this pre-pass called matmul() once PER
    // HEAD in a plain serial for-loop, relying on matmul's OWN internal
    // `#pragma omp parallel for` (kernels_cpu.cpp) for parallelism.
    // That's a fresh OMP team spawn+join PER HEAD PER LAYER (H*n_layers
    // = 64*75 = 4800 team spawns for one 11-token prefill) -- measured
    // at 13.84 GFLOPS actually achieved vs. ~10.4 GFLOPs of real work,
    // i.e. spawn/join overhead dominating, not compute. Fixed by
    // parallelizing ACROSS HEADS ourselves in ONE team (matching the
    // old per-(s,h) loop's own one-spawn-per-layer pattern exactly),
    // and letting matmul's own internal parallel region stay inert
    // inside it -- OMP nested parallelism is disabled by default in
    // this codebase (confirmed, see this block's own header comment),
    // so calling matmul() from inside this outer parallel region never
    // spawns a second team; it just runs matmul's O=kvl loop on
    // whichever single thread is already executing this head's
    // iteration.
    //
    // Gather ALL heads' nope-portion into ONE [H,S,nope] buffer up
    // front (2026-08-14, EngineConfig::attn_qabsorb_gpu) -- shared by
    // both the GPU batched path below (needs one contiguous [H,S,nope]
    // host buffer to BF16-convert and upload in a single H2D transfer,
    // CudaManager::qabsorb_bf16_batched_host) and the CPU fallback
    // (same per-head slice it always used, just already gathered
    // instead of gathered inline per head -- numerically identical,
    // same bytes, just gathered once instead of per-branch).
    // 2026-08-14: gathering is small enough (H*S*nope floats, e.g. 48KB
    // at decode's real S=1/H=64/nope=192 dims) that spawning an OMP team
    // costs MORE than the memcpy work itself at S=1 -- confirmed via a
    // real-hardware isolated bench (tests/main_test.cpp's bench_qabsorb_
    // cpu_vs_gpu follow-up): S=1 omp_parallel ran 3.1-3.4x SLOWER than
    // plain serial (spawn/join overhead dominating a workload this
    // small), while S=12 (prefill, ~12x the bytes) still benefits from
    // parallelizing (0.36-0.37x, i.e. ~2.7x faster). Gate on S>1, which
    // cleanly matches this codebase's own decode(S=1)-vs-prefill(S>1)
    // split rather than an arbitrary threshold.
    std::vector<float> q_nope_gather_all((size_t)H * S * nope);
    if (S > 1) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int h = 0; h < H; ++h) {
        float *dst = q_nope_gather_all.data() + (size_t)h * S * nope;
        for (int s = 0; s < S; ++s) {
          const float *head_q = Q.data() + ((size_t)s * H + h) * qh;
          std::memcpy(dst + (size_t)s * nope, head_q, nope * sizeof(float));
        }
      }
    } else {
      for (int h = 0; h < H; ++h) {
        float *dst = q_nope_gather_all.data() + (size_t)h * S * nope;
        for (int s = 0; s < S; ++s) {
          const float *head_q = Q.data() + ((size_t)s * H + h) * qh;
          std::memcpy(dst + (size_t)s * nope, head_q, nope * sizeof(float));
        }
      }
    }

    // GPU batched fast-path (EngineConfig::attn_qabsorb_gpu, default
    // off -- see that config field's own comment for why this is
    // deliberately per-model/hardware configurable, not a blanket
    // default: DeepSeek-V4 doesn't even reach this code path at all
    // (dsv4_attention_gguf has no absorption/WUK), and its CPU
    // attention path is already much faster relative to GLM-5.2's
    // absorbed path on at least one real box, so a GPU win here should
    // never be assumed to generalize across models/hardware without a
    // real A/B). Best-effort: qabsorb_bf16_batched_host returns false
    // (this layer's WUK BF16 upload never succeeded, or the cublas
    // call itself failed) and the CPU loop below runs unchanged.
    bool gpu_ok = false;
    if (engine.attn_qabsorb_gpu && engine.gpu) {
      gpu_ok = engine.gpu->qabsorb_bf16_batched_host(
          layer_idx, q_nope_gather_all.data(), Q_absorbed.data(), H, S, nope,
          kvl);
      if (!gpu_ok)
        perf().n_qabsorb_gpu_fallback_cpu.fetch_add(1,
                                                    std::memory_order_relaxed);
    }
    if (!gpu_ok) {
// 2026-08-28: same nested-OMP-region fix as the two WUV call
// sites in this file (see their own comments for the full
// real-hardware perf-profile evidence) -- matmul()'s internal
// `#pragma omp parallel for` was being spawned once per head
// from inside this already-parallel outer loop. Lower real-
// world volume than the WUV sites (attn_qabsorb_gpu defaults
// to true in this project's own settled configs and CLAUDE.md
// documents 0 CPU fallbacks on a real, well-configured run --
// this branch is a genuine fallback, not the common case), but
// the fix is free and correct regardless of how often it
// fires. dot_f32_avx512 is matmul()'s own per-output-row
// kernel; S is always 1 here in practice for this fallback
// (decode -- prefill's S>1 case goes through the GPU batched
// path or, on ITS OWN GPU failure, would need S handled in the
// outer loop below too, which this preserves via the s-loop).
#pragma omp parallel for schedule(dynamic, 1)
      for (int h = 0; h < H; ++h) {
        float *out_h = Q_absorbed.data() + (size_t)h * S * kvl;
        const float *in_h = q_nope_gather_all.data() + (size_t)h * S * nope;
        const float *wuk_h = WUK.data() + (size_t)h * nope * kvl;
        for (int s = 0; s < S; ++s) {
          const float *in_s = in_h + (size_t)s * nope;
          float *out_s = out_h + (size_t)s * kvl;
          for (int o = 0; o < kvl; ++o) {
            out_s[o] = dot_f32_avx512(in_s, wuk_h + (size_t)o * nope, nope);
          }
        }
      }
    }
  }
  PERF_END(attn_qabsorb_batch, t_attn_qabsorb_batch);

  PERF_START(attn_score);
  std::vector<float> head_contexts((size_t)S * H * vh, 0.0f);
  float attn_scale = compute_yarn_kq_scale(c, qh);

  // score GPU fast-path (EngineConfig::attn_score_gpu), decode (S==1)
  // only -- prefill (S>1) has a DIFFERENT past_context_len per row within
  // the same batch, out of scope for this first cut (see that config
  // field's own comment). 3 plain GEMMs + a softmax kernel replace the
  // CPU AVX-512 per-head loop below; unlike qabsorb, MLA's latent K/V is
  // SHARED across every head so no per-head batching is needed -- see
  // tests/bench_score_cpu_vs_gpu.cu for the real-hardware feasibility
  // numbers (5x-51x speedup, growing with context length) this is based
  // on. Best-effort: score_gpu_batched_host returns false (this layer's
  // KV mirror never allocated, or a CUDA/cublas call failed) and the
  // existing CPU loop below runs unchanged.
  bool score_gpu_ok = false;
  if (engine.attn_score_gpu && engine.gpu &&
      !(S > 1 && score_prefill_legacy_forced())) {
    // 2026-09-01: the `S == 1` gate that used to be here is gone. Its
    // stated reason -- "prefill has a DIFFERENT past_context_len per row
    // within the same batch" -- is now handled where it belongs, inside
    // the softmax kernel's per-row causal limit (cuda/attn_score_softmax.
    // cu), so prefill no longer has to fall back to the CPU loop below.
    //
    // T is the total key count this batch attends over: the last query in
    // the batch sits at absolute position pos_base+S-1, so keys [0, pos_
    // base+S) must exist in the layer's Lc/Rc mirror. score_gpu_batched_
    // host re-checks this against the mirror's real extent and returns
    // false (CPU fallback, unchanged) if the mirror is too small -- which
    // is also what keeps the T>mirror_max_t case safe rather than silently
    // truncating context.
    // Clamped to engine.max_t for the same reason the mirror upload above
    // clamps valid_rows: if a batch ever ran past the context limit, the
    // rows beyond it were never written to the mirror. Clamping here makes
    // score_gpu_batched_host's own `pos_base + S > T` check fail, so the
    // call falls back to the CPU loop rather than reading mirror rows that
    // were never uploaded. Not reachable in normal operation (prefill is
    // bounded by max_t upstream) -- this just makes the failure mode a
    // fallback instead of silent garbage.
    int past_context_len = std::min(pos_base + S, engine.max_t);
    // Q_absorbed is already [H,S,kvl] (see the qabsorb loop above, which
    // writes h*S*kvl + s*kvl) -- exactly the [H*S, kvl] view the batched
    // GEMM needs, so still no gather for it at any S. Q_rope does need a
    // gather, because Q is [S,H,qh] (s-major) while the score path wants
    // [H,S,rope] (h-major) -- a transpose, not just a strided copy.
    std::vector<float> Q_rope_gather((size_t)H * S * rope);
    for (int h = 0; h < H; ++h) {
      for (int s = 0; s < S; ++s) {
        const float *head_q = Q.data() + ((size_t)s * H + h) * qh;
        std::memcpy(Q_rope_gather.data() + ((size_t)h * S + s) * rope,
                    head_q + nope, rope * sizeof(float));
      }
    }
    std::vector<float> context_buf((size_t)H * S * kvl);
    score_gpu_ok = engine.gpu->score_gpu_batched_host(
        layer_idx, Q_absorbed.data(), Q_rope_gather.data(), context_buf.data(),
        H, past_context_len, kvl, rope, attn_scale, S, pos_base);
    if (score_gpu_ok) {
      // WUV down-projection: GPU-batched first (wuv_bf16_batched_host,
      // same strided-batched cuBLAS approach qabsorb already proved),
      // falling back to the CPU per-head OMP-parallel loop on any
      // failure. The CPU loop was the ORIGINAL implementation here --
      // real production data showed it costing 0.916ms/call (2.86s
      // over one 40-token decode run), the SAME "OMP team-spawn cost
      // exceeds a tiny per-call workload" pattern qabsorb's own gather
      // fix already diagnosed and fixed at a different call site.
      // t_score_gpu_wuv (via PERF_START/END below) now measures only
      // the FALLBACK path's cost, not the common case.
      bool wuv_gpu_ok =
          wuv_int8_enabled()
              ? engine.gpu->wuv_int8_batched_host(layer_idx, context_buf.data(),
                                                  head_contexts.data(), H, kvl,
                                                  vh, S)
              : engine.gpu->wuv_bf16_batched_host(layer_idx, context_buf.data(),
                                                  head_contexts.data(), H, kvl,
                                                  vh, S);
      if (!wuv_gpu_ok) {
        perf().n_wuv_gpu_fallback_cpu.fetch_add(1, std::memory_order_relaxed);
        PERF_START(score_wuv);
// 2026-08-28: this call site's own comment above already
// diagnosed "the SAME OMP team-spawn cost exceeds a tiny
// per-call workload pattern qabsorb's own gather fix
// already diagnosed" but never actually applied the fix --
// matmul()'s internal `#pragma omp parallel for` was still
// being spawned once per head, nested inside this outer
// parallel region. A real `perf record -g` profile of a
// live decode run (with nested parallelism disabled
// process-wide, so the inner region never spawns real
// extra threads but still pays real per-call OMP-runtime-
// entry cost) confirmed this call site as a genuine,
// still-executing hotspot -- WUV's own real coverage gap
// (see CLAUDE.md's WUV-coverage entries: 38.5-61.5% of
// layers fall back to this exact CPU path depending on
// GPU1 VRAM headroom) means this fires often in practice,
// not just as a rare edge case. Fixed the same way the
// sibling !score_gpu_ok WUV loop was fixed just above in
// this file: dot_f32_avx512 is matmul()'s own per-output-
// row kernel (byte-identical unroll/reduce/tail structure,
// kernels_cpu.cpp) -- computes exactly what matmul() did,
// just without the redundant inner OMP entry, since the
// outer `#pragma omp parallel for` over h already provides
// all the parallelism this workload needs.
// 2026-08-29: num_threads(attn_score_threads()) added --
// this loop previously had no thread-count control at all
// (defaulted to omp_get_max_threads()=10). A real isolated
// microbenchmark at this exact shape (H=64,vh=256,kvl=512,
// GLM-5.2's real dims) found this is memory-bandwidth-bound,
// not compute-bound (~22-32 GFLOPS achieved, far below AVX-
// 512 FMA peak) -- and that this box's real bandwidth
// saturates around 6 threads (its own physical core count,
// 0.5387ms/call), with 10 threads (0.5664ms/call) and
// especially 12 (0.6707ms/call, full hyperthread
// oversubscription) both measurably WORSE, not better.
// Reuses the SAME dedicated-team knob the 3 sibling per-head
// AVX-512 loops in this file already use (dense_dispatch.
// hpp) rather than adding new config surface -- real-hardware
// verification pending; see CLAUDE.md for the isolated
// benchmark's full sweep and this call site's own real A/B.
// 2026-09-01: S added. Note the two buffers use DIFFERENT
// major orders -- context_buf is [H,S,kvl] (h-major, as
// score_gpu_batched_host produces it) while head_contexts is
// [S,H,vh] (s-major, the layout the rest of this function and
// its caller expect). Getting this backwards would not crash;
// it would silently scramble which query's context feeds which
// output row, so the indexing is spelled out rather than
// folded into pointer arithmetic. Parallelised over h only
// (not collapse(2)) to keep ONE team spawn per call -- the
// same nested-OMP lesson as this call site's own history
// above; at S>1 there is already ample work per h.
#pragma omp parallel for schedule(dynamic, 1) num_threads(attn_score_threads())
        for (int h = 0; h < H; ++h) {
          const float *wuv_h = WUV.data() + (size_t)h * kvl * vh;
          for (int s = 0; s < S; ++s) {
            const float *ctx_hs =
                context_buf.data() + ((size_t)h * S + s) * kvl;
            float *out_hs = head_contexts.data() + ((size_t)s * H + h) * vh;
            for (int o = 0; o < vh; ++o) {
              out_hs[o] = dot_f32_avx512(ctx_hs, wuv_h + (size_t)o * kvl, kvl);
            }
          }
        }
        PERF_END(score_wuv, t_score_gpu_wuv);
      }
    } else {
      perf().n_score_gpu_fallback_cpu.fetch_add(1, std::memory_order_relaxed);
    }
  }

  if (!score_gpu_ok) {
// num_threads(attn_score_threads()): dedicated, possibly-smaller team
// size for this causal score loop specifically -- see EngineConfig::
// attn_score_threads' own comment (dense_dispatch.hpp) for why this is
// separate from the process-wide omp_max_threads default. Defaults to
// omp_get_max_threads() (the same value this pragma would have used
// anyway) when unset, so this is a genuine no-op for any config that
// doesn't set attn_score_threads.
#pragma omp parallel for collapse(2) schedule(dynamic, 1)                      \
    num_threads(attn_score_threads())
    for (int s = 0; s < S; ++s) {
      for (int h = 0; h < H; ++h) {
        int current_pos = pos_base + s;
        int past_context_len = current_pos + 1;

        const float *head_q = Q.data() + ((size_t)s * H + h) * qh;
        const float *q_rope = head_q + nope;
        const float *q_absorbed = Q_absorbed.data() + ((size_t)h * S + s) * kvl;

        std::vector<float> attn_scores(past_context_len);
        const float *layer_lc_cache = engine.Lc[layer_idx];
        const float *layer_rc_cache = engine.Rc[layer_idx];

        for (int t = 0; t < past_context_len; ++t) {
          const float *L_t = layer_lc_cache + (t * kvl);
          const float *R_t = layer_rc_cache + (t * rope);

          float score_nope = 0.0f;
          __m512 v_score_nope = _mm512_setzero_ps();
          for (int i = 0; i < kvl; i += 16) {
            v_score_nope =
                _mm512_fmadd_ps(_mm512_loadu_ps(q_absorbed + i),
                                _mm512_loadu_ps(L_t + i), v_score_nope);
          }
          score_nope = _mm512_reduce_add_ps(v_score_nope);

          float score_rope = 0.0f;
          __m512 v_score_rope = _mm512_setzero_ps();
          for (int i = 0; i < rope; i += 16) {
            v_score_rope =
                _mm512_fmadd_ps(_mm512_loadu_ps(q_rope + i),
                                _mm512_loadu_ps(R_t + i), v_score_rope);
          }
          score_rope = _mm512_reduce_add_ps(v_score_rope);

          attn_scores[t] = (score_nope + score_rope) * attn_scale;
        }

        softmax_row(attn_scores.data(), past_context_len);

        std::vector<float> latent_accumulation(kvl, 0.0f);
        for (int t = 0; t < past_context_len; ++t) {
          float alpha_t = attn_scores[t];
          const float *L_t = layer_lc_cache + (t * kvl);

          for (int i = 0; i < kvl; i += 16) {
            __m512 v_lat = _mm512_loadu_ps(latent_accumulation.data() + i);
            __m512 v_cache = _mm512_loadu_ps(L_t + i);
            _mm512_storeu_ps(
                latent_accumulation.data() + i,
                _mm512_fmadd_ps(_mm512_set1_ps(alpha_t), v_cache, v_lat));
          }
        }

        // v[vh] = latent_agg[kvl] @ W_UV[h]   (W_UV[h] is [vh, kvl])
        //
        // 2026-08-28 real-hardware perf finding: this call site was
        // spawning its OWN OMP thread team (matmul()'s internal
        // `#pragma omp parallel for`, kernels_cpu.cpp) from INSIDE the
        // outer `#pragma omp parallel for collapse(2)` loop above --
        // once per (s,h) iteration, i.e. up to H times per layer per
        // token at decode (S=1). A real `perf record -g` profile of a
        // live decode run found this call site's own OMP-team-entry
        // overhead (nested parallelism is disabled process-wide in
        // this codebase, so the inner region never actually spawns
        // extra threads, but still pays real per-call runtime-entry
        // cost) was THE single largest hotspot in the entire process
        // -- matmul()'s own omp-region clone symbol alone accounted
        // for ~10.7% of all sampled CPU cycles, and >90% of that time
        // was gomp_team_barrier_wait's spin-then-futex loop (confirmed
        // via direct disassembly of the sampled addresses), not real
        // compute. Directly analogous to (and fixed via the exact same
        // proven pattern as) the S==1 outproj call site above in this
        // same file: "ONE combined OMP parallel region... instead of N
        // separate matmul() calls each spawning its own thread team" --
        // here the outer collapse(2) loop already IS that one combined
        // region, so the fix is simply to stop the inner call from
        // trying to parallelize again, not to add a new outer one.
        // dot_f32_avx512 is matmul()'s own per-output-row kernel
        // (byte-identical unroll/reduce/tail structure, kernels_cpu.cpp)
        // -- this loop computes EXACTLY what the removed matmul() call
        // computed, just without the redundant inner OMP entry.
        float *head_out_dest = head_contexts.data() + ((size_t)s * H + h) * vh;
        const float *wuv_h = WUV.data() + (size_t)h * kvl * vh;
        for (int o = 0; o < vh; ++o) {
          head_out_dest[o] = dot_f32_avx512(latent_accumulation.data(),
                                            wuv_h + (size_t)o * kvl, kvl);
        }
      }
    }
  } // !score_gpu_ok

  if (engine.family_ops.family == ModelFamily::BailingMoE3) {
    // BailingMoE3 gates each attention head before the output projection.
    std::vector<float> gates((size_t)S * H);
    dense_proj(engine, p + "attn_gate.weight", normed.data(), gates.data(),
               S, D, H);
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < H; ++h) {
        const float gate = 1.0f / (1.0f + std::exp(-gates[(size_t)s * H + h]));
        float *head = head_contexts.data() + ((size_t)s * H + h) * vh;
        for (int i = 0; i < vh; ++i) head[i] *= gate;
      }
  }
  check_finite_attn("head_contexts (post absorbed attention loop)", layer_idx,
                    head_contexts.data(), head_contexts.size(), H * vh);
  PERF_END(attn_score, t_attn_score);

  // ---- 5. output projection (attn_output) via dense GPU dispatch ----
  PERF_START(attn_outproj);
  dense_proj(engine, p + "attn_output.weight", head_contexts.data(),
             out_context, S, H * vh, D);
  check_finite_attn("out_context (post attn_output projection)", layer_idx,
                    out_context, (size_t)S * D, D);
  PERF_END(attn_outproj, t_attn_outproj);
}

// ---------------------------------------------------------------------------
// GLM5Next (GLM 5.3 Flash) recurrent KDA layer + hybrid attention dispatch.
// ---------------------------------------------------------------------------
// Reference: llama.cpp branch glm5next (src/models/glm5next.cpp's
// build_kda_layer + src/models/delta-net-base.cpp's
// build_delta_net_autoregressive). A KDA layer is DeltaNet-like but NOT the
// same as this repo's Qwen recurrent path (src/cpu_qwen_moe.cpp):
//   - separate attn_q/attn_k/attn_v projections + SEPARATE per-channel conv
//     weights (ssm_conv1d_{q,k,v}), concatenated into one conv over q|k|v
//   - the decay gate is PER-KEY-CHANNEL (a [head_dim] vector per head), not
//     Qwen's per-head scalar:
//       g = kda_gate_lower_bound * sigmoid(-( (f_b(f_a(x)) + dt_bias) * ssm_a
//       ))
//     with ssm_a = -exp(A_log) (kimi-k3 convention, per the converter) and
//     kda_gate_lower_bound < 0 -- so g is a NEGATIVE log-decay, exp(g) in
//     (exp(lb), 1). The lower_bound SCALES the sigmoid; it is not a clamp,
//     and it is not the softplus branch (that's kimi-k3-without-the-key, a
//     different function -- the converter hard-requires the key).
//   - beta = sigmoid(ssm_beta(x)) per head; output gate is a PLAIN sigmoid
//     (g_b(g_a(x))), not the SiLU that a FusedRMSNormGated default would
//     imply.
//   - Q/K are L2-normed per head with ggml_l2_norm's exact formula
//     (x / max(sqrt(sum(x^2)), 1e-6) -- NOT x / sqrt(sum + eps)).
//   - shared head count with the DSA layers (the converter requires
//     linear_attn_config.num_heads == num_attention_heads), and k_dim ==
//     v_dim == kda_head_dim -- no Qwen-style key/value head repeat.
// The recurrence itself is the per-token (autoregressive) delta rule, which
// is EXACT for any S (the reference's chunked path is purely an
// optimization) -- correctness-first, matching this session's CPU-first
// decision for KDA.
//
// UNVERIFIED ON REAL HARDWARE (new model-family bring-up): validate against
// a real GLM-5.3-Flash GGUF with a known prompt before trusting output.

// Per-layer dequant cache for the concatenated KDA conv weights
// ([3*d_inner, d_conv] = [ssm_conv1d_q ; ssm_conv1d_k ; ssm_conv1d_v],
// channel-major) -- static model weights, identical on every call, so
// dequantized once and reused (same pattern as wuk_cache/wuv_cache above).
static std::vector<std::vector<float>> g_glm5next_conv_w_cache;

// Build the narrow weight descriptors used by the optional GLM5NEXT KDA
// projection graph. The runner copies one layer's weights into its own
// reusable CUDA buffer; it deliberately does not borrow resident-layer device
// pointers because those pointers belong to the generic dense pool and may be
// reclaimed/reassigned independently of this experimental graph.
static bool glm5next_proj_ref(StreamingEngine &engine, const std::string &name,
                              GgmlGlm5nextProjWeightRef &out) {
  auto it = engine.resident_layers.find(name);
  if (it == engine.resident_layers.end() || !it->second.host_ptr)
    return false;
  const auto &w = it->second;
  if (w.meta.shape.size() < 2 || w.meta.shape[0] == 0 || w.meta.shape[1] == 0)
    return false;
  out.host_data = w.host_ptr;
  out.type = (uint32_t)w.meta.type;
  out.ne0 = (int64_t)w.meta.shape[0];
  out.ne1 = (int64_t)w.meta.shape[1];
  out.nbytes = w.meta.size_bytes;
  return true;
}

static bool
glm5next_kda_proj_fused(StreamingEngine &engine, int layer_idx,
                        const std::string &p, int D, int H, int hd, int d_inner,
                        int S, const float *hidden_states,
                        std::vector<float> &q, std::vector<float> &k,
                        std::vector<float> &v, std::vector<float> &f_b,
                        std::vector<float> &beta, std::vector<float> &g_b) {
  if (!engine.glm5next_dense_prefill_fusion || S <= 1 || !engine.gpu ||
      !GgmlGlm5nextKdaProjRunner::compiled_in())
    return false;
  GgmlGlm5nextProjWeightRef refs[8];
  const char *names[8] = {"attn_q.weight",  "attn_k.weight",  "attn_v.weight",
                          "ssm_f_a.weight", "ssm_f_b.weight", "ssm_beta.weight",
                          "ssm_g_a.weight", "ssm_g_b.weight"};
  for (int i = 0; i < 8; ++i)
    if (!glm5next_proj_ref(engine, p + names[i], refs[i]))
      return false;

  auto &runner = GgmlGlm5nextKdaProjRunner::instance();
  if (!runner.ensure(engine.gpu->device_id, D, d_inner, hd, H, S) ||
      !runner.ensure_layer(layer_idx, refs[0], refs[1], refs[2], refs[3],
                           refs[4], refs[5], refs[6], refs[7]))
    return false;
  q.resize((size_t)S * d_inner);
  k.resize((size_t)S * d_inner);
  v.resize((size_t)S * d_inner);
  f_b.resize((size_t)S * d_inner);
  beta.resize((size_t)S * H);
  g_b.resize((size_t)S * d_inner);
  if (!runner.run(layer_idx, hidden_states, S, q.data(), k.data(), v.data(),
                  f_b.data(), beta.data(), g_b.data()))
    return false;
  return true;
}

// Dequant a whole small tensor to F32. Returns false (logged) if missing.
// Family-neutral (renamed from glm5next_dequant_tensor 2026-09-03 -- now
// shared by the GLM5Next KDA and Qwen4Exp GDN paths).
static bool attn_dequant_tensor_f32(StreamingEngine &engine,
                                    const std::string &name,
                                    std::vector<float> &out,
                                    size_t expected_numel) {
  auto it = engine.resident_layers.find(name);
  if (it == engine.resident_layers.end() || !it->second.host_ptr) {
    LOG_ERROR("[ATTN-DEQUANT] missing tensor %s\n", name.c_str());
    return false;
  }
  uint64_t real = 1;
  for (auto d : it->second.meta.shape)
    real *= d;
  if (real != expected_numel) {
    LOG_ERROR("[ATTN-DEQUANT] %s: real numel=%llu != expected=%zu -- "
              "ArchitectureConfig/metadata mismatch, refusing to read past the "
              "tensor.\n",
              name.c_str(), (unsigned long long)real, expected_numel);
    return false;
  }
  out.resize(expected_numel);
  dequant_row(it->second.meta.type, it->second.host_ptr, out.data(),
              (int)expected_numel);
  return true;
}

// Causal short conv over the concatenated q|k|v channels, with SiLU on the
// output and a persistent per-layer history of the last (kernel-1) samples
// per channel. Lifted from cpu_qwen_moe.cpp's causal_conv1d (itself verified
// against ggml's ggml_ssm_conv: k=0 is the OLDEST tap, k=kernel-1 the current
// sample; history carries the receptive field across decode-time S=1 calls).
// x/out: [S, channels]; w: [channels, kernel] row-major (the GGUF conv
// tensors are {d_conv, 1, d_inner, 1} in ggml ne order = d_conv taps
// contiguous per channel = exactly this layout); history: [channels, kernel-1].
// (Renamed from glm5next_causal_conv1d 2026-09-03 -- the function was always
// generic over (channels, kernel); now shared by GLM5Next KDA and Qwen4Exp
// GDN, whose fused ssm_conv1d has the same [channels, kernel] row-major
// layout.)
// Not static: exposed for main_test.cpp's GPU-vs-CPU correctness test
// (see Test 28, "GDN causal conv1d+SiLU GPU path") to compare
// qwen4exp_conv1d_silu_gpu_host's new GPU kernel directly against this
// already-proven CPU reference, rather than an independent from-scratch
// implementation that could share a bug with the GPU port.
void attn_causal_conv1d_silu(const float *x, const float *w, float *out, int S,
                             int channels, int kernel, float *history) {
  const int hist = kernel - 1;

  // 2026-09-04: this was CHANNEL-OUTER, and x is [S, channels] row-major, so
  // the inner `x[(j-hist)*channels + c]` strided by channels*4 = 40 KB per
  // step at qwen4exp's conv_dim=10240 -- every load its own cache line, and
  // essentially every one a miss. Measured 61.97s of a 267s prefill (23%),
  // which is ~0.17 GFLOPS against ~10.4 GFLOPs of real work: three orders of
  // magnitude off, i.e. purely a memory-access-pattern problem, not compute.
  //
  // Now TOKEN-OUTER / channel-inner: for a fixed (s,k) the source row
  // x[(s+k-hist)] is contiguous in c, so both the activation and (after the
  // transpose below) the weight are unit-stride and the accumulate is a
  // straight AVX-512 FMA. Rows are independent, so s also parallelises.
  //
  // The weights arrive as [channels][kernel]; with c innermost that would be
  // a stride-`kernel` gather, so they are transposed once per call into
  // [kernel][channels]. That is channels*kernel = 40960 elements against the
  // S*channels*kernel = 144M element-loads the main loop does -- free.
  static thread_local std::vector<float> wt_tls;
  if ((int)wt_tls.size() < kernel * channels)
    wt_tls.resize((size_t)kernel * channels);
  float *wt = wt_tls.data();
  for (int c = 0; c < channels; ++c)
    for (int k = 0; k < kernel; ++k)
      wt[(size_t)k * channels + c] = w[(size_t)c * kernel + k];

#pragma omp parallel for schedule(static)
  for (int s = 0; s < S; ++s) {
    float *orow = out + (size_t)s * channels;
    bool first = true;
    for (int k = 0; k < kernel; ++k) {
      const int j = s + k; // position in [history ++ x]
      const float *wk = wt + (size_t)k * channels;
      if (j >= hist) {
        const float *xr = x + (size_t)(j - hist) * channels;
        int c = 0;
        if (first) {
          for (; c + 16 <= channels; c += 16)
            _mm512_storeu_ps(orow + c, _mm512_mul_ps(_mm512_loadu_ps(xr + c),
                                                     _mm512_loadu_ps(wk + c)));
          for (; c < channels; ++c)
            orow[c] = xr[c] * wk[c];
        } else {
          for (; c + 16 <= channels; c += 16)
            _mm512_storeu_ps(orow + c,
                             _mm512_fmadd_ps(_mm512_loadu_ps(xr + c),
                                             _mm512_loadu_ps(wk + c),
                                             _mm512_loadu_ps(orow + c)));
          for (; c < channels; ++c)
            orow[c] += xr[c] * wk[c];
        }
      } else {
        // Only reachable for s < hist (3 rows), so the strided history
        // read stays scalar rather than justifying a second code path.
        if (first)
          for (int c = 0; c < channels; ++c)
            orow[c] = history[(size_t)c * hist + j] * wk[c];
        else
          for (int c = 0; c < channels; ++c)
            orow[c] += history[(size_t)c * hist + j] * wk[c];
      }
      first = false;
    }
    // SiLU on the conv output (not on the projections) -- glm5next.cpp.
    for (int c = 0; c < channels; ++c)
      orow[c] = orow[c] / (1.0f + std::exp(-orow[c]));
  }

  if (hist > 0) {
    std::vector<float> new_hist((size_t)channels * hist);
    for (int c = 0; c < channels; ++c) {
      const float *h = history + (size_t)c * hist;
      for (int j = 0; j < hist; ++j) {
        int pos = S + j;
        new_hist[(size_t)c * hist + j] =
            (pos < hist) ? h[pos] : x[(size_t)(pos - hist) * channels + c];
      }
    }
    std::memcpy(history, new_hist.data(),
                (size_t)channels * hist * sizeof(float));
  }
}

// Per-token gated delta rule with PER-KEY-CHANNEL decay (KDA), matching
// delta-net-base.cpp's build_delta_net_autoregressive op-for-op:
//   state[k][v] *= exp(g[k])                    (decay, per key channel k)
//   sk[v]        = sum_k state[k][v] * k[k]
//   state[k][v] += k[k] * (v[v] - sk[v]) * beta
//   out[v]       = sum_k state[k][v] * q[k] / sqrt(head_dim)
// state: [H, hd, hd] row-major ([key, value] per head), persistent across
// calls. Q/K L2-normed by the caller. g_log is the NEGATIVE log-decay
// [S, H, hd]; beta is sigmoided [S, H].
static void glm5next_kda_delta_rule(const float *Q, const float *K,
                                    const float *V, const float *beta,
                                    const float *g_log, float *state,
                                    float *out, int S, int H, int hd) {
  const float q_scale = 1.0f / std::sqrt((float)hd);
  std::vector<float> sk(hd);
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < H; ++h) {
      float *st = state + (size_t)h * hd * hd;
      const float *q_h = Q + ((size_t)s * H + h) * hd;
      const float *k_h = K + ((size_t)s * H + h) * hd;
      const float *v_h = V + ((size_t)s * H + h) * hd;
      const float *g_h = g_log + ((size_t)s * H + h) * hd;
      const float b = beta[(size_t)s * H + h];
      float *o_h = out + ((size_t)s * H + h) * hd;

      for (int k = 0; k < hd; ++k) {
        const float decay = std::exp(g_h[k]);
        float *row = st + (size_t)k * hd;
        for (int v = 0; v < hd; ++v)
          row[v] *= decay;
      }
      std::memset(sk.data(), 0, (size_t)hd * sizeof(float));
      for (int k = 0; k < hd; ++k) {
        const float kk = k_h[k];
        const float *row = st + (size_t)k * hd;
        for (int v = 0; v < hd; ++v)
          sk[v] += row[v] * kk;
      }
      for (int k = 0; k < hd; ++k) {
        const float kk = k_h[k];
        float *row = st + (size_t)k * hd;
        for (int v = 0; v < hd; ++v)
          row[v] += kk * (v_h[v] - sk[v]) * b;
      }
      std::memset(o_h, 0, (size_t)hd * sizeof(float));
      for (int k = 0; k < hd; ++k) {
        const float qq = q_h[k] * q_scale;
        const float *row = st + (size_t)k * hd;
        for (int v = 0; v < hd; ++v)
          o_h[v] += row[v] * qq;
      }
    }
  }
}

static void glm5next_kda_gguf(StreamingEngine &engine, int layer_idx,
                              const float *hidden_states, int S, int pos_base,
                              float *out_context) {
  PERF_SCOPE(t_attn);
  const auto &c = engine.model_file.architecture_config;
  const int D = c.hidden;
  const int H = c.n_heads;
  const int hd = c.kda_head_dim;
  const int d_inner = H * hd;
  const int d_conv = c.ssm_conv_kernel;
  const float lb = c.kda_gate_lower_bound;
  const bool bailing = engine.family_ops.family == ModelFamily::BailingMoE3;
  const std::string p = "blk." + std::to_string(layer_idx) + ".";
  (void)pos_base; // KDA is recurrent -- no positions, no RoPE, no KV cache

  if (D <= 0 || H <= 0 || hd <= 0 || d_conv <= 1 || !(lb < 0.0f)) {
    LOG_ERROR(
        "[GLM5NEXT-KDA] layer=%d invalid dims: D=%d H=%d kda_head_dim=%d "
        "ssm_conv_kernel=%d gate_lower_bound=%f -- zeroing attention output\n",
        layer_idx, D, H, hd, d_conv, lb);
    std::memset(out_context, 0, (size_t)S * D * sizeof(float));
    return;
  }
  if (layer_idx >= (int)engine.glm5next_conv_history.size() ||
      layer_idx >= (int)engine.glm5next_recur_state.size()) {
    LOG_ERROR(
        "[GLM5NEXT-KDA] layer=%d: recurrent state not allocated "
        "(conv_history=%zu recur_state=%zu entries) -- alloc_glm5next_cache "
        "did not run? Zeroing attention output.\n",
        layer_idx, engine.glm5next_conv_history.size(),
        engine.glm5next_recur_state.size());
    std::memset(out_context, 0, (size_t)S * D * sizeof(float));
    return;
  }

  // New-sequence reset. generate_text (generation_loop.hpp) prefills every
  // request with pos_base=0 and rewrites the KV cache from position 0, so
  // each prompt is an INDEPENDENT sequence -- but a recurrent layer's state
  // IS its compressed history, and this state is allocated once by
  // alloc_glm5next_cache and otherwise never cleared. Without this, turn 2+ of
  // a REPL session starts from the previous turn's accumulated state and
  // degrades (observed 2026-09-04 on qwen4exp: a prompt that answers correctly
  // and in full as turn 1 emitted only <|im_end|> as turn 3 of a multi-turn
  // session). pos_base==0 is already this codebase's established "new
  // sequence" signal -- see dsv4_attention_gguf's compressed_state->reset()
  // and ple_ngram.cpp's own position-discontinuity EOS-pad, both of which
  // reset their cross-call state exactly this way. The GDN/KDA recurrent
  // state was the one form that never did.
  if (pos_base == 0) {
    std::fill(engine.glm5next_conv_history[layer_idx].begin(),
              engine.glm5next_conv_history[layer_idx].end(), 0.0f);
    std::fill(engine.glm5next_recur_state[layer_idx].begin(),
              engine.glm5next_recur_state[layer_idx].end(), 0.0f);
  }

  // ---- 1. attention input norm ----
  std::vector<float> normed((size_t)S * D);
  rmsnorm_proj(engine, p + "attn_norm.weight", hidden_states, normed.data(), S,
               D, c.eps);
  check_finite_attn("KDA normed (post attn_norm rmsnorm)", layer_idx,
                    normed.data(), normed.size());

  // ---- 2. Q/K/V projections, interleaved into one [S, 3*d_inner] buffer
  // (q block, then k, then v -- matching the reference's ggml_concat order
  // and the concatenated conv weight built in step 3) ----
  std::vector<float> qkv((size_t)S * 3 * d_inner);
  std::vector<float> fused_q, fused_k, fused_v, fused_f_b, fused_beta,
      fused_g_b;
  const bool fused_kda_proj = !bailing && glm5next_kda_proj_fused(
      engine, layer_idx, p, D, H, hd, d_inner, S, normed.data(), fused_q,
      fused_k, fused_v, fused_f_b, fused_beta, fused_g_b);
  {
    if (fused_kda_proj) {
      for (int s = 0; s < S; ++s) {
        std::memcpy(qkv.data() + (size_t)s * 3 * d_inner,
                    fused_q.data() + (size_t)s * d_inner,
                    (size_t)d_inner * sizeof(float));
        std::memcpy(qkv.data() + (size_t)s * 3 * d_inner + d_inner,
                    fused_k.data() + (size_t)s * d_inner,
                    (size_t)d_inner * sizeof(float));
        std::memcpy(qkv.data() + (size_t)s * 3 * d_inner + 2 * d_inner,
                    fused_v.data() + (size_t)s * d_inner,
                    (size_t)d_inner * sizeof(float));
      }
    } else {
      std::vector<float> tmp((size_t)S * d_inner);
      dense_proj(engine, p + "attn_q.weight", normed.data(), tmp.data(), S, D,
                 d_inner);
      for (int s = 0; s < S; ++s)
        std::memcpy(qkv.data() + (size_t)s * 3 * d_inner,
                    tmp.data() + (size_t)s * d_inner,
                    (size_t)d_inner * sizeof(float));
      dense_proj(engine, p + "attn_k.weight", normed.data(), tmp.data(), S, D,
                 d_inner);
      for (int s = 0; s < S; ++s)
        std::memcpy(qkv.data() + (size_t)s * 3 * d_inner + d_inner,
                    tmp.data() + (size_t)s * d_inner,
                    (size_t)d_inner * sizeof(float));
      dense_proj(engine, p + "attn_v.weight", normed.data(), tmp.data(), S, D,
                 d_inner);
      for (int s = 0; s < S; ++s)
        std::memcpy(qkv.data() + (size_t)s * 3 * d_inner + 2 * d_inner,
                    tmp.data() + (size_t)s * d_inner,
                    (size_t)d_inner * sizeof(float));
    }
  }
  check_finite_attn("KDA qkv (post q/k/v projections, pre-conv)", layer_idx,
                    qkv.data(), qkv.size());

  // ---- 3. causal short conv over q|k|v (concatenated weights, cached) ----
  if ((int)g_glm5next_conv_w_cache.size() < c.n_layers)
    g_glm5next_conv_w_cache.resize(c.n_layers);
  std::vector<float> &conv_w = g_glm5next_conv_w_cache[layer_idx];
  if (conv_w.empty()) {
    conv_w.resize((size_t)3 * d_inner * d_conv);
    std::vector<float> one;
    bool ok = attn_dequant_tensor_f32(engine, p + "ssm_conv1d_q.weight", one,
                                      (size_t)d_inner * d_conv);
    if (ok)
      std::memcpy(conv_w.data(), one.data(), one.size() * sizeof(float));
    if (ok)
      ok = attn_dequant_tensor_f32(engine, p + "ssm_conv1d_k.weight", one,
                                   (size_t)d_inner * d_conv);
    if (ok)
      std::memcpy(conv_w.data() + (size_t)d_inner * d_conv, one.data(),
                  one.size() * sizeof(float));
    if (ok)
      ok = attn_dequant_tensor_f32(engine, p + "ssm_conv1d_v.weight", one,
                                   (size_t)d_inner * d_conv);
    if (ok)
      std::memcpy(conv_w.data() + (size_t)2 * d_inner * d_conv, one.data(),
                  one.size() * sizeof(float));
    if (!ok) {
      conv_w.clear();
      std::memset(out_context, 0, (size_t)S * D * sizeof(float));
      return;
    }
    check_finite_attn("KDA conv_w (q|k|v concatenated, dequant)", layer_idx,
                      conv_w.data(), conv_w.size());
  }

  std::vector<float> &conv_hist = engine.glm5next_conv_history[layer_idx];
  if ((int)conv_hist.size() != 3 * d_inner * (d_conv - 1))
    conv_hist.assign((size_t)3 * d_inner * (d_conv - 1), 0.0f);
  std::vector<float> conv_out((size_t)S * 3 * d_inner);
  // The same row-major causal-conv primitive is used by Qwen4Exp. Reuse
  // its already-validated CUDA bridge here so GLM KDA does not spend its
  // entire short-convolution phase in the CPU DRAM loop. Failure is
  // per-call and correctness-safe: the reference implementation below is
  // unchanged.
  bool conv_gpu_ok = engine.glm5next_kda_gpu && engine.gpu &&
                     engine.gpu->qwen4exp_conv1d_silu_gpu_host(
                         qkv.data(), conv_w.data(), conv_out.data(), S,
                         3 * d_inner, d_conv, conv_hist.data());
  if (!conv_gpu_ok) {
    attn_causal_conv1d_silu(qkv.data(), conv_w.data(), conv_out.data(), S,
                            3 * d_inner, d_conv, conv_hist.data());
  }
  check_finite_attn("KDA conv_out (post conv + SiLU)", layer_idx,
                    conv_out.data(), conv_out.size());

  // ---- 4. split into Q/K/V [S, H, hd] and L2-normalize Q/K per head ----
  // ggml_l2_norm's exact formula: x / max(sqrt(sum(x^2)), 1e-6) -- NOT
  // x / sqrt(sum + eps) (confirmed against ggml_compute_forward_l2_norm_f32).
  std::vector<float> Q((size_t)S * d_inner), K((size_t)S * d_inner),
      V((size_t)S * d_inner);
  for (int s = 0; s < S; ++s) {
    const float *src = conv_out.data() + (size_t)s * 3 * d_inner;
    std::memcpy(Q.data() + (size_t)s * d_inner, src,
                (size_t)d_inner * sizeof(float));
    std::memcpy(K.data() + (size_t)s * d_inner, src + d_inner,
                (size_t)d_inner * sizeof(float));
    std::memcpy(V.data() + (size_t)s * d_inner, src + 2 * d_inner,
                (size_t)d_inner * sizeof(float));
  }
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < H; ++h) {
      float *q_h = Q.data() + ((size_t)s * H + h) * hd;
      float *k_h = K.data() + ((size_t)s * H + h) * hd;
      double sq = 0.0, sk2 = 0.0;
      for (int i = 0; i < hd; ++i) {
        sq += (double)q_h[i] * q_h[i];
        sk2 += (double)k_h[i] * k_h[i];
      }
      const float rq = 1.0f / std::fmax(std::sqrt((float)sq), 1e-6f);
      const float rk = 1.0f / std::fmax(std::sqrt((float)sk2), 1e-6f);
      for (int i = 0; i < hd; ++i) {
        q_h[i] *= rq;
        k_h[i] *= rk;
      }
    }
  }

  // ---- 5. decay gate g and beta (both read the LAYER INPUT, not the
  // convolved q/k/v -- glm5next.cpp's `inp`) ----
  // GLM uses f_b(f_a(normed)); Bailing projects directly to d_inner.
  std::vector<float> g_log((size_t)S * d_inner);
  {
    if (fused_kda_proj)
      g_log = fused_f_b;
    else if (bailing)
      dense_proj(engine, p + "ssm_f_a.weight", normed.data(), g_log.data(),
                 S, D, d_inner);
    else {
      std::vector<float> fa((size_t)S * hd);
      dense_proj(engine, p + "ssm_f_a.weight", normed.data(), fa.data(), S, D,
                 hd);
      dense_proj(engine, p + "ssm_f_b.weight", fa.data(), g_log.data(), S, hd,
                 d_inner);
    }
    std::vector<float> dt_bias, ssm_a;
    if (!attn_dequant_tensor_f32(engine, p + "ssm_dt.bias", dt_bias,
                                 (size_t)d_inner) ||
        !attn_dequant_tensor_f32(engine, p + "ssm_a", ssm_a, (size_t)H)) {
      std::memset(out_context, 0, (size_t)S * D * sizeof(float));
      return;
    }
    for (int s = 0; s < S; ++s) {
      float *g_s = g_log.data() + (size_t)s * d_inner;
      for (int h = 0; h < H; ++h) {
        const float a_h = ssm_a[h]; // = -exp(A_log) < 0
        for (int i = 0; i < hd; ++i) {
          const int idx = h * hd + i;
          const float f = (g_s[idx] + dt_bias[idx]) * a_h;
          g_s[idx] = bailing ? lb / (1.0f + std::exp(-f))
                               : lb / (1.0f + std::exp(f));
        }
      }
    }
  }
  check_finite_attn("KDA g (log-decay, pre-exp)", layer_idx, g_log.data(),
                    g_log.size());

  // beta = sigmoid(ssm_beta(normed))  [S, H]
  std::vector<float> beta((size_t)S * H);
  if (fused_kda_proj)
    beta = fused_beta;
  else
    dense_proj(engine, p + "ssm_beta.weight", normed.data(), beta.data(), S, D,
               H);
  for (auto &b : beta)
    b = 1.0f / (1.0f + std::exp(-b));

  // ---- 6. recurrent gated delta rule (persistent per-layer state) ----
  std::vector<float> &state = engine.glm5next_recur_state[layer_idx];
  if ((int)state.size() != H * hd * hd)
    state.assign((size_t)H * hd * hd, 0.0f);
  std::vector<float> core_out((size_t)S * d_inner);
  bool kda_gpu_ok =
      engine.glm5next_kda_gpu && engine.gpu &&
      engine.gpu->glm5next_kda_delta_rule_gpu_device(
          Q.data(), K.data(), V.data(), beta.data(), g_log.data(), state.data(),
          core_out.data(), layer_idx, S, H, hd, pos_base == 0);
  if (!kda_gpu_ok) {
    if (engine.glm5next_kda_gpu && engine.gpu)
      perf().n_glm5next_kda_gpu_fallback_cpu.fetch_add(
          1, std::memory_order_relaxed);
    glm5next_kda_delta_rule(Q.data(), K.data(), V.data(), beta.data(),
                            g_log.data(), state.data(), core_out.data(), S, H,
                            hd);
  }
  check_finite_attn("KDA core_out (post delta rule, pre gate/norm)", layer_idx,
                    core_out.data(), core_out.size());

  // ---- 7. output gate: per-head RMSNorm (ssm_norm) * PLAIN sigmoid gate
  // (g_b(g_a(normed))) -- NOT the SiLU a FusedRMSNormGated default implies ----
  std::vector<float> gate((size_t)S * d_inner);
  if (fused_kda_proj) {
    gate = fused_g_b;
  } else if (bailing) {
    dense_proj(engine, p + "ssm_g_a.weight", normed.data(), gate.data(),
               S, D, d_inner);
  } else {
    std::vector<float> ga((size_t)S * hd);
    dense_proj(engine, p + "ssm_g_a.weight", normed.data(), ga.data(), S, D,
               hd);
    dense_proj(engine, p + "ssm_g_b.weight", ga.data(), gate.data(), S, hd,
               d_inner);
  }
  std::vector<float> norm_w;
  if (!attn_dequant_tensor_f32(engine, p + "ssm_norm.weight", norm_w,
                               (size_t)hd)) {
    std::memset(out_context, 0, (size_t)S * D * sizeof(float));
    return;
  }
  std::vector<float> gated((size_t)S * d_inner);
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < H; ++h) {
      const float *x = core_out.data() + ((size_t)s * H + h) * hd;
      const float *g_h = gate.data() + ((size_t)s * H + h) * hd;
      float *g_out = gated.data() + ((size_t)s * H + h) * hd;
      double msq = 0.0;
      for (int i = 0; i < hd; ++i)
        msq += (double)x[i] * x[i];
      const float inv = 1.0f / std::sqrt((float)(msq / hd) + c.eps);
      for (int i = 0; i < hd; ++i)
        g_out[i] = x[i] * inv * norm_w[i] / (1.0f + std::exp(-g_h[i]));
    }
  }
  check_finite_attn("KDA gated (post norm*sigmoid-gate)", layer_idx,
                    gated.data(), gated.size());

  // ---- 8. output projection (attn_output: [d_inner, D]) ----
  dense_proj(engine, p + "attn_output.weight", gated.data(), out_context, S,
             d_inner, D);
  check_finite_attn("KDA out_context (post attn_output projection)", layer_idx,
                    out_context, (size_t)S * D);
}

// GLM5Next hybrid per-layer dispatch: the head_count_kv array (GGUF metadata)
// is the layer-type switch -- 0 = recurrent KDA, nonzero = full-attention DSA.
// DSA layers reuse the absorbed-MLA path verbatim: glm5next's DSA is nope-only
// (qk_rope == 0, validated at load time), which mla_attention_absorbed_gguf
// handles correctly (RoPE loops no-op, the rope score term vanishes, and the
// GPU score fast-path supports rope==0 by running only the nope GEMM and
// feeding an explicit zero rope-score buffer to the shared softmax kernel.
// v1: the Lightning indexer (sparse top-k attention on DSA layers) is
// DEFERRED -- DSA layers attend densely over the full KV cache.
void glm5next_attention_gguf(StreamingEngine &engine, int layer_idx,
                             const float *hidden_states, int S, int pos_base,
                             float *out_context) {
  if (glm5next_is_kda_layer(engine.model_file.architecture_config, layer_idx)) {
    glm5next_kda_gguf(engine, layer_idx, hidden_states, S, pos_base,
                      out_context);
  } else {
    mla_attention_absorbed_gguf(engine, layer_idx, hidden_states, S, pos_base,
                                out_context);
  }
}

void bailingmoe3_attention_gguf(StreamingEngine &engine, int layer_idx,
                                const float *hidden_states, int S, int pos_base,
                                float *out_context) {
  if (bailingmoe3_is_kda_layer(engine.model_file.architecture_config, layer_idx))
    glm5next_kda_gguf(engine, layer_idx, hidden_states, S, pos_base,
                      out_context);
  else
    mla_attention_absorbed_gguf(engine, layer_idx, hidden_states, S, pos_base,
                                out_context);
}

// ---------------------------------------------------------------------------
// Qwen4Exp (Qwen3.8-Flash-Next): recurrent GDN layer + dense full-attention
// layer + hybrid dispatch.
// ---------------------------------------------------------------------------
// Reference: llama.cpp branch qwen4exp/qwen3.8-flash-next
// (src/models/qwen4exp.cpp's build_layer_attn_linear / build_layer_attn +
// src/models/delta-net-base.cpp's build_delta_net_autoregressive). GDN is
// DeltaNet-like but NOT the same as this repo's GLM5Next KDA path (above):
//   - ONE fused attn_qkv projection ([q|k|v], key_dim*2+value_dim channels)
//     and ONE fused per-channel conv weight (ssm_conv1d), not separate
//     q/k/v projections and conv weights
//   - the decay gate is a PER-VALUE-HEAD SCALAR:
//       g = softplus(ssm_alpha(x) + ssm_dt) * ssm_a     (then exp() applied
//     at use; ssm_a = -exp(A_log), the "noscan" convention), not KDA's
//     per-key-channel lower-bound-scaled sigmoid
//   - key/query head count (ssm.group_count) can DIFFER from the value head
//     count (ssm.time_step_rank): v-head h reads q/k head (h % group_count),
//     matching the converter's TILED V-head reorder (conversion/qwen.py's
//     _LinearAttentionVReorderBase) and ggml's tiled repeat broadcast
//   - Q/K are L2-normed per head with ggml_l2_norm's exact formula
//     (x / max(sqrt(sum(x^2)), eps) -- verified in ggml-cpu/ops.cpp's
//     ggml_compute_forward_l2_norm_f32)
//   - the output gate z (attn_gate projection) is applied as a PLAIN
//     sigmoid inside the gated RMS norm (build_norm_gated: "the one
//     numerical difference from Qwen3.5's GDN: sigmoid output gate, not
//     silu")
//   - there is NO attn_norm: the low-rank hyper-connection mix
//     (qwen4exp_hc_pre_mix, hyper_connections.cpp) already normalized the
//     sub-block input
// The recurrence itself is the per-token (autoregressive) delta rule, which
// is EXACT for any S (the reference's chunked path is purely an
// optimization) -- correctness-first, matching the CPU-first decision for
// GLM5Next's KDA.
//
// UNVERIFIED ON REAL HARDWARE (new model-family bring-up): validate against
// a real Qwen3.8-Flash-Next GGUF with a known prompt before trusting output.

// Per-layer dequant cache for the fused GDN conv weight ([conv_dim, d_conv]
// row-major) -- static model weight, identical on every call, dequantized
// once (same pattern as g_glm5next_conv_w_cache / wuk_cache).
static std::vector<std::vector<float>> g_qwen4exp_conv_w_cache;

// Per-token gated delta rule with PER-HEAD SCALAR decay (GDN), matching
// delta-net-base.cpp's build_delta_net_autoregressive op-for-op:
//   state[k][v] *= exp(g)                      (decay, scalar per v-head)
//   sk[v]        = sum_k state[k][v] * k[k]
//   state[k][v] += k[k] * (v[v] - sk[v]) * beta
//   out[v]       = sum_k state[k][v] * q[k] / sqrt(head_dim)
// state: [H_v, hd, hd] row-major ([key, value] per head), persistent across
// calls. Q/K L2-normed by the caller. g_log is the NEGATIVE log-decay
// [S, H_v]; beta is sigmoided [S, H_v]. Q/K carry only H_k heads; v-head h
// reads q/k head h % H_k (tiled broadcast -- see the file-header comment).
// state: [H_v, hd, hd] row-major ([key, value] per head), persistent across
// calls. Q/K L2-normed by the caller. g_log is the NEGATIVE log-decay
// [S, H_v]; beta is sigmoided [S, H_v]. Q/K carry only H_k heads; v-head h
// reads q/k head h % H_k (tiled broadcast -- see the file-header comment).
// The per-(token, head) step itself is qwen4exp_gdn_delta_step
// (include/qwen4exp_gdn.hpp) -- pure and unit-tested, same discipline as
// rope.hpp/hyper_connections.hpp.
static void qwen4exp_gdn_delta_rule(const float *Q, const float *K,
                                    const float *V, const float *beta,
                                    const float *g_log, float *state,
                                    float *out, int S, int H_k, int H_v, int hd) {
  // Heads are fully independent (own state region) -- one OMP team per
  // call, real work per head (hd*hd state), matching this project's "one
  // spawn per real workload" discipline.
  // Prefill uses the CHUNKED closed form; decode (S==1) stays on the
  // sequential recurrence. This mirrors llama.cpp exactly --
  // delta-net-base.cpp:483 dispatches build_delta_net_autoregressive only
  // when n_tokens==1 and build_delta_net_chunking (CS=64) otherwise. Both
  // compute the same rule; qwen4exp_gdn_delta_chunk is verified against
  // qwen4exp_gdn_delta_step itself in tests/test_components.cpp
  // ("qwen4exp GDN chunked vs sequential", max diff ~9e-8, i.e. float
  // rounding). Opt-in for now via ZEROMOE_GDN_CHUNKED=1 so the two can be
  // A/B'd on real hardware before it becomes the default; same env-flag
  // precedent as ZEROMOE_KQUANT_LEGACY / ZEROMOE_WUV_INT8.
  static const bool gdn_chunked = [] {
    const char *e = zeromoe::graphcfg::env("ZEROMOE_GDN_CHUNKED");
    return e && e[0] == '1';
  }();
  // 64 matches llama.cpp's own CS for the non-KDA (GDN) branch.
  constexpr int kGdnChunk = 64;

  // GPU chunkwise path (EngineConfig::qwen4exp_gdn_gpu). Same chunk size and
  // same closed form as the CPU chunk function below -- chunks stay
  // sequential (the state carries) but all H_v heads and hd columns run in
  // parallel, which the CPU form cannot do. Prefill only, matching
  // llama.cpp's own n_tokens==1 dispatch rule for the sequential variant.

#pragma omp parallel for schedule(static)
  for (int h = 0; h < H_v; ++h) {
    const int hk = h % H_k;
    float *st = state + (size_t)h * hd * hd;
    if (gdn_chunked && S > 1) {
      // Gather this head's rows contiguously -- Q/K/V are interleaved
      // by head in the [S, H, hd] layout, and the chunk kernel wants
      // [C, hd] blocks.
      static thread_local std::vector<float> qc, kc, vc, bc, gc, oc;
      for (int s0 = 0; s0 < S; s0 += kGdnChunk) {
        const int C = std::min(kGdnChunk, S - s0);
        const size_t n = (size_t)C * hd;
        if (qc.size() < n) {
          qc.resize(n);
          kc.resize(n);
          vc.resize(n);
          oc.resize(n);
        }
        if ((int)bc.size() < C) {
          bc.resize((size_t)C);
          gc.resize((size_t)C);
        }
        for (int i = 0; i < C; ++i) {
          const int s = s0 + i;
          std::memcpy(qc.data() + (size_t)i * hd,
                      Q + ((size_t)s * H_k + hk) * hd,
                      (size_t)hd * sizeof(float));
          std::memcpy(kc.data() + (size_t)i * hd,
                      K + ((size_t)s * H_k + hk) * hd,
                      (size_t)hd * sizeof(float));
          std::memcpy(vc.data() + (size_t)i * hd,
                      V + ((size_t)s * H_v + h) * hd,
                      (size_t)hd * sizeof(float));
          bc[(size_t)i] = beta[(size_t)s * H_v + h];
          gc[(size_t)i] = g_log[(size_t)s * H_v + h];
        }
        qwen4exp_gdn_delta_chunk(qc.data(), kc.data(), vc.data(), bc.data(),
                                 gc.data(), st, oc.data(), C, hd);
        for (int i = 0; i < C; ++i)
          std::memcpy(out + ((size_t)(s0 + i) * H_v + h) * hd,
                      oc.data() + (size_t)i * hd, (size_t)hd * sizeof(float));
      }
    } else {
      for (int s = 0; s < S; ++s) {
        qwen4exp_gdn_delta_step(
            Q + ((size_t)s * H_k + hk) * hd, K + ((size_t)s * H_k + hk) * hd,
            V + ((size_t)s * H_v + h) * hd, beta[(size_t)s * H_v + h],
            g_log[(size_t)s * H_v + h], st, out + ((size_t)s * H_v + h) * hd,
            hd);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// ggml GPU path for GDN's four shared-input projections (opt-in, gdn_proj_
// ggml). See include/ggml_gdn_proj_backend.hpp for the measurement.
//
// UNLIKE hc_ggml, there is no "all-or-nothing per forward pass" state to
// protect here -- nothing persists across layers or calls except the
// read-only weight brackets, so a failure on THIS call can fall back to the
// existing per-call dense_proj path for THIS call alone with no divergent-
// state risk (contrast hc_stream_buf, which genuinely IS live state shared
// across the whole pass).
// ---------------------------------------------------------------------------
static bool gdn_proj_ggml_usable(StreamingEngine &engine, int D, int conv_dim,
                                 int value_dim, int H_v) {
  if (!gdn_proj_ggml() || !GgmlGdnProjRunner::compiled_in())
    return false;
  const int dev = moe_ggml_device() >= 0 ? moe_ggml_device() : 0;
  return GgmlGdnProjRunner::instance().ensure(dev, D, conv_dim, value_dim, H_v);
}

// Pulls one layer's four weights (plus the small dt_bias/ssm_a decay-gate
// constants, dequantized via the SAME attn_dequant_tensor_f32 helper the
// existing CPU path already uses for them) out of the resident dense pool
// and hands them to the runner. No-op if already registered.
static bool gdn_proj_ggml_register(StreamingEngine &engine, int layer_idx,
                                   const std::string &p, int H_v) {
  GgmlGdnProjRunner &r = GgmlGdnProjRunner::instance();
  if (r.weights_ready(layer_idx))
    return true;

  auto ref = [&](const std::string &name, GgmlGdnProjWeightRef &out) -> bool {
    auto it = engine.resident_layers.find(name);
    if (it == engine.resident_layers.end() || !it->second.host_ptr)
      return false;
    const auto &W = it->second;
    if (W.meta.shape.size() < 2)
      return false;
    out.host_data = W.host_ptr;
    out.type = (uint32_t)W.meta.type;
    out.ne0 = (int64_t)W.meta.shape[0];
    out.ne1 = (int64_t)W.meta.shape[1];
    out.nbytes = W.meta.size_bytes;
    out.device_data = W.device_ptr;
    out.device_bytes = W.device_ptr ? W.meta.size_bytes : 0;
    return true;
  };
  GgmlGdnProjWeightRef qkv, gate, alpha, beta;
  if (!ref(p + "attn_qkv.weight", qkv) || !ref(p + "attn_gate.weight", gate) ||
      !ref(p + "ssm_alpha.weight", alpha) || !ref(p + "ssm_beta.weight", beta))
    return false;

  std::vector<float> dt_bias, ssm_a;
  if (!attn_dequant_tensor_f32(engine, p + "ssm_dt.bias", dt_bias,
                               (size_t)H_v) ||
      !attn_dequant_tensor_f32(engine, p + "ssm_a", ssm_a, (size_t)H_v))
    return false;
  return r.ensure_weights(layer_idx, qkv, gate, alpha, beta, dt_bias.data(),
                          ssm_a.data());
}

// Registers layer_idx's causal-conv weight with the SAME ggml runner used
// for the 4-projection fusion (opt-in ON TOP of gdn_proj_ggml via
// EngineConfig::qwen4exp_gdn_conv_gpu) -- see run_with_conv's own comment
// for why chaining the two removes a real, Llama.cpp-profile-confirmed
// ~65-70x-oversized round trip (2026-09-05: their own ssm_conv kernel costs
// 2.80ms for all 36 layers at S=512; our separate-round-trip GPU conv1d path
// costs ~180-190ms for the same 36 layers, almost entirely H2D+D2H of the
// [S,channels] activation this fusion avoids re-uploading in the first
// place). Reuses g_qwen4exp_conv_w_cache exactly as the CPU path's own lazy
// dequant does, so whichever runs first populates it for the other.
static bool gdn_proj_ggml_register_conv(StreamingEngine &engine, int layer_idx,
                                        const std::string &p, int conv_dim,
                                        int d_conv) {
  GgmlGdnProjRunner &r = GgmlGdnProjRunner::instance();
  if (r.conv_weight_ready(layer_idx))
    return true;
  const auto &c = engine.model_file.architecture_config;
  if ((int)g_qwen4exp_conv_w_cache.size() < c.n_layers)
    g_qwen4exp_conv_w_cache.resize(c.n_layers);
  std::vector<float> &conv_w = g_qwen4exp_conv_w_cache[layer_idx];
  if (conv_w.empty()) {
    if (!attn_dequant_tensor_f32(engine, p + "ssm_conv1d.weight", conv_w,
                                 (size_t)conv_dim * d_conv))
      return false;
    check_finite_attn("GDN conv_w (dequant)", layer_idx, conv_w.data(),
                      conv_w.size());
  }
  GgmlGdnProjWeightRef ref;
  ref.host_data = conv_w.data();
  ref.type = 0; // GGML_TYPE_F32 == 0 -- attn_dequant_tensor_f32 always
                // produces F32 host floats; avoids pulling ggml.h into
                // this file just for one enum value (ensure_conv_weight's
                // own implementation, which DOES include ggml.h, validates
                // this numerically).
  ref.ne0 = d_conv;
  ref.ne1 = conv_dim;
  ref.nbytes = conv_w.size() * sizeof(float);
  return r.ensure_conv_weight(layer_idx, ref);
}

// Register the two weights used only by the complete device-resident GDN
// graph. Norm weights are one-dimensional in GGUF while ssm_out is a normal
// [value_dim, hidden] projection, so this intentionally does not reuse the
// four-projection helper's two-dimensional shape requirement.
static bool gdn_full_ggml_register(StreamingEngine &engine, int layer_idx,
                                   const std::string &p, int value_dim, int D,
                                   int H_v) {
  GgmlGdnProjRunner &r = GgmlGdnProjRunner::instance();
  if (r.full_weights_ready(layer_idx))
    return true;
  auto ref = [&](const std::string &name, GgmlGdnProjWeightRef &out,
                 bool one_dim) -> bool {
    auto it = engine.resident_layers.find(name);
    if (it == engine.resident_layers.end() || !it->second.host_ptr)
      return false;
    const auto &W = it->second;
    if (one_dim) {
      if (W.meta.shape.empty())
        return false;
      out.ne0 = W.meta.shape[0];
      out.ne1 = W.meta.shape.size() > 1 ? W.meta.shape[1] : 1;
    } else {
      if (W.meta.shape.size() < 2)
        return false;
      out.ne0 = W.meta.shape[0];
      out.ne1 = W.meta.shape[1];
    }
    out.host_data = W.host_ptr;
    out.type = (uint32_t)W.meta.type;
    out.nbytes = W.meta.size_bytes;
    out.device_data = W.device_ptr;
    out.device_bytes = W.device_ptr ? W.meta.size_bytes : 0;
    return true;
  };
  GgmlGdnProjWeightRef norm, out;
  if (!ref(p + "ssm_norm.weight", norm, true) ||
      !ref(p + "ssm_out.weight", out, false))
    return false;
  if (norm.ne0 != value_dim / H_v || norm.ne1 != 1 || out.ne0 != value_dim ||
      out.ne1 != D)
    return false;
  return r.ensure_full_weights(layer_idx, norm, out);
}

static void qwen4exp_gdn_gguf(StreamingEngine &engine, int layer_idx,
                              const float *hidden_states, int S, int pos_base,
                              float *out_context) {
  PERF_SCOPE(t_attn);
  const auto &c = engine.model_file.architecture_config;
  const int D = c.hidden;
  const int H_k = c.ssm_group_count;    // key/query head count
  const int H_v = c.ssm_time_step_rank; // value head count
  const int hd = c.ssm_state_size;      // per-head key AND value dim
  const int key_dim = H_k * hd;
  const int value_dim = c.ssm_inner_size; // == H_v * hd (validated at load)
  const int conv_dim = 2 * key_dim + value_dim;
  const int d_conv = c.ssm_conv_kernel;
  const std::string p = "blk." + std::to_string(layer_idx) + ".";
  (void)pos_base; // GDN is recurrent -- no positions, no RoPE, no KV cache

  if (D <= 0 || H_k <= 0 || H_v <= 0 || hd <= 0 || d_conv <= 1 ||
      H_v % H_k != 0 || value_dim != H_v * hd) {
    LOG_ERROR("[QWEN4EXP-GDN] layer=%d invalid dims: D=%d H_k=%d H_v=%d hd=%d "
              "conv_kernel=%d inner=%d -- zeroing attention output\n",
              layer_idx, D, H_k, H_v, hd, d_conv, c.ssm_inner_size);
    std::memset(out_context, 0, (size_t)S * D * sizeof(float));
    return;
  }
  if (layer_idx >= (int)engine.qwen4exp_conv_history.size() ||
      layer_idx >= (int)engine.qwen4exp_recur_state.size()) {
    LOG_ERROR(
        "[QWEN4EXP-GDN] layer=%d: recurrent state not allocated "
        "(conv_history=%zu recur_state=%zu entries) -- alloc_qwen4exp_cache "
        "did not run? Zeroing attention output.\n",
        layer_idx, engine.qwen4exp_conv_history.size(),
        engine.qwen4exp_recur_state.size());
    std::memset(out_context, 0, (size_t)S * D * sizeof(float));
    return;
  }

  // New-sequence reset. generate_text (generation_loop.hpp) prefills every
  // request with pos_base=0 and rewrites the KV cache from position 0, so
  // each prompt is an INDEPENDENT sequence -- but a recurrent layer's state
  // IS its compressed history, and this state is allocated once by
  // alloc_qwen4exp_cache and otherwise never cleared. Without this, turn 2+ of
  // a REPL session starts from the previous turn's accumulated state and
  // degrades (observed 2026-09-04 on qwen4exp: a prompt that answers correctly
  // and in full as turn 1 emitted only <|im_end|> as turn 3 of a multi-turn
  // session). pos_base==0 is already this codebase's established "new
  // sequence" signal -- see dsv4_attention_gguf's compressed_state->reset()
  // and ple_ngram.cpp's own position-discontinuity EOS-pad, both of which
  // reset their cross-call state exactly this way. The GDN/KDA recurrent
  // state was the one form that never did.
  if (pos_base == 0) {
    std::fill(engine.qwen4exp_conv_history[layer_idx].begin(),
              engine.qwen4exp_conv_history[layer_idx].end(), 0.0f);
    std::fill(engine.qwen4exp_recur_state[layer_idx].begin(),
              engine.qwen4exp_recur_state[layer_idx].end(), 0.0f);
  }

  // NOTE: no attn_norm -- the HC mix already normalized the input
  // (qwen4exp has no per-layer norm tensors at all).

  // ---- 1. fused QKV + z-gate projections ----
  // Grow-only thread_local scratch instead of ten fresh, value-initialised
  // std::vectors per call. At qwen4exp's real prefill shape these total
  // ~60 MB PER LAYER (qkv/conv_out alone are ~14.5 MB each at S=441), so
  // the old code malloc'd AND zeroed ~2.2 GB across a 36-layer GDN pass
  // before doing a single useful flop. Exactly the pattern already fixed in
  // hyper_connections.cpp's qwen4exp_hc_mix_impl; this function had it too.
  // Every buffer below is fully overwritten before it is read (dense_proj,
  // the conv, the split memcpy, the delta rule and the norm loop each write
  // their whole output), so dropping the zero-init is safe -- verified by
  // an end-to-end output comparison, not by inspection alone.
  static thread_local std::vector<float> gdn_qkv_b, gdn_z_b, gdn_conv_b,
      gdn_Q_b, gdn_K_b, gdn_V_b, gdn_glog_b, gdn_beta_b, gdn_core_b,
      gdn_gated_b;
  auto gdn_grow = [](std::vector<float> &v, size_t n) -> float * {
    if (v.size() < n)
      v.resize(n);
    return v.data();
  };
  float *const qkv = gdn_grow(gdn_qkv_b, (size_t)S * conv_dim);
  float *const z = gdn_grow(gdn_z_b, (size_t)S * value_dim);
  // g_log/beta are step 4's outputs, but the ggml fused-projection path
  // (below) computes all four of attn_qkv/attn_gate/ssm_alpha/ssm_beta --
  // plus their decay-log/sigmoid postprocessing -- from the SAME
  // hidden_states in one graph, so they're allocated here rather than at
  // step 4's original location.
  float *const g_log = gdn_grow(gdn_glog_b, (size_t)S * H_v);
  float *const beta = gdn_grow(gdn_beta_b, (size_t)S * H_v);
  // conv_out/conv_hist are step 2's outputs, hoisted here too: the fully-
  // fused path below (gdn_conv_fused_ok) writes conv_out DIRECTLY from
  // this same graph, skipping step 2 (and the raw qkv it would otherwise
  // need) entirely.
  float *const conv_out = gdn_grow(gdn_conv_b, (size_t)S * conv_dim);
  std::vector<float> &conv_hist = engine.qwen4exp_conv_history[layer_idx];
  if ((int)conv_hist.size() != conv_dim * (d_conv - 1))
    conv_hist.assign((size_t)conv_dim * (d_conv - 1), 0.0f);
  std::vector<float> &recur_state = engine.qwen4exp_recur_state[layer_idx];
  if ((int)recur_state.size() != H_v * hd * hd)
    recur_state.assign((size_t)H_v * hd * hd, 0.0f);

  // Fast path: keep the whole GDN sub-block on the GGML CUDA backend.
  // This is deliberately attempted before the older split path and falls
  // back without changing its state if any registration, allocation or
  // graph execution precondition is unavailable.
  if (S > 0 && gdn_full_ggml() &&
      gdn_proj_ggml_usable(engine, D, conv_dim, value_dim, H_v) &&
      gdn_proj_ggml_register(engine, layer_idx, p, H_v) &&
      gdn_proj_ggml_register_conv(engine, layer_idx, p, conv_dim, d_conv) &&
      gdn_full_ggml_register(engine, layer_idx, p, value_dim, D, H_v)) {
    PERF_START(gdnfull);
    const void *device_output = nullptr;
    bool full_ok = false;
    bool device_full_ok = false;
    bool host_input_ready = true;
    bool device_state_attempted = false;
    if (engine.qwen4exp_device_hc_input) {
      device_state_attempted = true;
      device_full_ok = GgmlGdnProjRunner::instance().run_full_device_tensor(
          layer_idx, engine.qwen4exp_device_hc_input, S, &device_output,
          conv_hist.data(), d_conv, recur_state.data(), c.eps,
          /*device_state_resident=*/S == 1);
      full_ok = device_full_ok;
      if (full_ok) {
        engine.qwen4exp_device_attn_output = device_output;
        engine.qwen4exp_device_attn_output_ready = true;
      } else {
        // The host path below is a real fallback. Recover the input
        // that pre_mix intentionally left on the device before
        // trying it; hidden_states is not authoritative in this
        // device-input form.  The same recovery is deliberately
        // fallback changes only the GDN execution side.
        if (!GgmlHcRunner::instance().copy_pre_mix_to_host(
                engine.qwen4exp_device_hc_input,
                const_cast<float *>(hidden_states), S)) {
          host_input_ready = false;
        }
      }
    }
    if (!full_ok && host_input_ready) {
      if (device_state_attempted)
        GgmlGdnProjRunner::instance().sync_full_device_states_to_host();
      log_gdn_full_stats("input", layer_idx, hidden_states, S, D);
      full_ok = GgmlGdnProjRunner::instance().run_full(
          layer_idx, hidden_states, S, out_context, conv_hist.data(), d_conv,
          recur_state.data(), c.eps);
    }
    if (full_ok) {
      if (device_full_ok) {
        perf().n_gdn_full_device_success.fetch_add(1,
                                                   std::memory_order_relaxed);
      } else {
        perf().n_gdn_full_host_success.fetch_add(1, std::memory_order_relaxed);
      }
      PERF_END(gdnfull, t_gdn_full);
      log_gdn_full_stats("output", layer_idx, out_context, S, D);
      if (!engine.qwen4exp_device_attn_output_ready)
        check_finite_attn("GDN full graph out_context", layer_idx, out_context,
                          (size_t)S * D, D);
      return;
    }
    if (device_state_attempted)
      perf().n_gdn_full_device_failure.fetch_add(1, std::memory_order_relaxed);
    perf().n_gdn_full_split_fallback.fetch_add(1, std::memory_order_relaxed);
  }

  bool gdn_proj_ggml_ok =
      false; // 4 projections fused; qkv still needs step 2's own conv
  bool gdn_conv_fused_ok =
      false; // conv1d chained into the SAME graph; conv_out is already final
  PERF_START(gdnqkv);
  if (gdn_proj_ggml_usable(engine, D, conv_dim, value_dim, H_v) &&
      gdn_proj_ggml_register(engine, layer_idx, p, H_v)) {
    // Prefer the fully-fused path (proj + conv1d + SiLU in one graph,
    // zero host round trip for qkv -- see run_with_conv's own comment)
    // whenever it's available. S>1 only, same "decode's single token
    // isn't worth it" convention as the separate GPU conv1d path below
    // and every other prefill-scoped GPU path in this function. Falls
    // back to the proj-only fusion on ANY failure, per-call, no
    // divergent state either way -- same discipline as gdn_proj_ggml's
    // own fallback to per-call dense_proj.
    if (S > 1 && engine.qwen4exp_gdn_conv_gpu &&
        gdn_proj_ggml_register_conv(engine, layer_idx, p, conv_dim, d_conv)) {
      gdn_conv_fused_ok = GgmlGdnProjRunner::instance().run_with_conv(
          layer_idx, hidden_states, S, conv_out, z, g_log, beta,
          conv_hist.data(), d_conv);
    }
    if (!gdn_conv_fused_ok) {
      gdn_proj_ggml_ok = GgmlGdnProjRunner::instance().run(
          layer_idx, hidden_states, S, qkv, z, g_log, beta);
    }
  }
  if (!gdn_conv_fused_ok && !gdn_proj_ggml_ok) {
    dense_proj(engine, p + "attn_qkv.weight", hidden_states, qkv, S, D,
               conv_dim);
    dense_proj(engine, p + "attn_gate.weight", hidden_states, z, S, D,
               value_dim);
  }
  PERF_END(gdnqkv, t_gdn_qkv_proj);
  // Nothing to trace when gdn_conv_fused_ok: qkv was never materialised
  // host-side in that path (the whole point of fusing it away).
  if (!gdn_conv_fused_ok)
    check_finite_attn("GDN qkv (post attn_qkv, pre-conv)", layer_idx, qkv,
                      (size_t)S * conv_dim);

  // ---- 2. causal short conv over q|k|v (fused weight, cached) ----
  // Skipped entirely when gdn_conv_fused_ok: conv_out is already final,
  // computed device-resident inside the SAME graph step 1 just built.
  if (!gdn_conv_fused_ok) {
    if ((int)g_qwen4exp_conv_w_cache.size() < c.n_layers)
      g_qwen4exp_conv_w_cache.resize(c.n_layers);
    std::vector<float> &conv_w = g_qwen4exp_conv_w_cache[layer_idx];
    if (conv_w.empty()) {
      if (!attn_dequant_tensor_f32(engine, p + "ssm_conv1d.weight", conv_w,
                                   (size_t)conv_dim * d_conv)) {
        std::memset(out_context, 0, (size_t)S * D * sizeof(float));
        return;
      }
      check_finite_attn("GDN conv_w (dequant)", layer_idx, conv_w.data(),
                        conv_w.size());
    }

    PERF_START(gdnconv);
    // GPU path (EngineConfig::qwen4exp_gdn_conv_gpu): prefill only
    // (S>1), same convention as the fully-fused path above -- decode's
    // single-token transfer isn't worth the H2D/D2H round trip. Falls
    // back to the byte-identical CPU path (below) on any failure,
    // including the fallback's own history update, so `conv_hist`
    // always ends the call in a consistent state regardless of which
    // path ran.
    bool conv_gpu_ok = false;
    if (engine.qwen4exp_gdn_conv_gpu && engine.gpu && S > 1) {
      conv_gpu_ok = engine.gpu->qwen4exp_conv1d_silu_gpu_host(
          qkv, conv_w.data(), conv_out, S, conv_dim, d_conv, conv_hist.data());
      if (!conv_gpu_ok)
        perf().n_gdn_conv1d_gpu_fallback_cpu.fetch_add(
            1, std::memory_order_relaxed);
    }
    if (!conv_gpu_ok) {
      attn_causal_conv1d_silu(qkv, conv_w.data(), conv_out, S, conv_dim, d_conv,
                              conv_hist.data());
    }
    PERF_END(gdnconv, t_gdn_conv);
  }
  check_finite_attn("GDN conv_out (post conv + SiLU)", layer_idx, conv_out,
                    (size_t)S * conv_dim);

  // ---- 3. split into Q/K/V and L2-normalize Q/K per head ----
  // ggml_l2_norm's exact formula: x / max(sqrt(sum(x^2)), eps) -- NOT
  // x / sqrt(sum + eps) (verified in ggml_compute_forward_l2_norm_f32).
  float *const Q = gdn_grow(gdn_Q_b, (size_t)S * key_dim);
  float *const K = gdn_grow(gdn_K_b, (size_t)S * key_dim);
  float *const V = gdn_grow(gdn_V_b, (size_t)S * value_dim);
  PERF_START(gdnl2);
  // Both loops below are per-token independent (each s writes only its own
  // rows of Q/K/V), so parallelising leaves them bit-identical.
#pragma omp parallel for if (S > 1) schedule(static)
  for (int s = 0; s < S; ++s) {
    const float *src = conv_out + (size_t)s * conv_dim;
    std::memcpy(Q + (size_t)s * key_dim, src, (size_t)key_dim * sizeof(float));
    std::memcpy(K + (size_t)s * key_dim, src + key_dim,
                (size_t)key_dim * sizeof(float));
    std::memcpy(V + (size_t)s * value_dim, src + 2 * key_dim,
                (size_t)value_dim * sizeof(float));
  }
#pragma omp parallel for if (S > 1) schedule(static)
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < H_k; ++h) {
      qwen4exp_l2_norm_inplace(Q + ((size_t)s * H_k + h) * hd, hd, c.eps);
      qwen4exp_l2_norm_inplace(K + ((size_t)s * H_k + h) * hd, hd, c.eps);
    }
  }

  PERF_END(gdnl2, t_gdn_l2);
  // ---- 4. decay gate g and beta (both read the SUB-BLOCK INPUT, not the
  // convolved q/k/v -- qwen4exp.cpp's build_layer_attn_linear) ----
  // g = softplus(ssm_alpha(x) + ssm_dt) * ssm_a   [S, H_v] (negative log-decay)
  // beta = sigmoid(ssm_beta(x))  [S, H_v]
  // Both already computed by the ggml fused-projection path above (step 1)
  // when gdn_proj_ggml_ok OR gdn_conv_fused_ok -- either graph applies the
  // identical decay-log/sigmoid formulas as graph ops, so this whole block
  // is skipped rather than redone. (Real bug caught via a non-zero
  // "gates=" bucket during the conv-fusion A/B, 2026-09-05: this condition
  // originally checked gdn_proj_ggml_ok only, so the fully-fused path's
  // already-correct g_log/beta were being silently, redundantly
  // recomputed here every call -- not a correctness bug (dense_proj
  // reproduces the same math from the same weights) but a real
  // performance one, defeating part of the point of fusing conv in.)
  PERF_START(gdngt);
  if (!gdn_proj_ggml_ok && !gdn_conv_fused_ok) {
    dense_proj(engine, p + "ssm_alpha.weight", hidden_states, g_log, S, D, H_v);
    std::vector<float> dt_bias, ssm_a;
    if (!attn_dequant_tensor_f32(engine, p + "ssm_dt.bias", dt_bias,
                                 (size_t)H_v) ||
        !attn_dequant_tensor_f32(engine, p + "ssm_a", ssm_a, (size_t)H_v)) {
      std::memset(out_context, 0, (size_t)S * D * sizeof(float));
      return;
    }
    for (int s = 0; s < S; ++s) {
      float *g_s = g_log + (size_t)s * H_v;
      for (int h = 0; h < H_v; ++h)
        g_s[h] = qwen4exp_gdn_decay_log(g_s[h], dt_bias[h], ssm_a[h]);
    }

    dense_proj(engine, p + "ssm_beta.weight", hidden_states, beta, S, D, H_v);
    for (size_t bi = 0; bi < (size_t)S * H_v; ++bi)
      beta[bi] = 1.0f / (1.0f + std::exp(-beta[bi]));
  }
  check_finite_attn("GDN g (log-decay, pre-exp)", layer_idx, g_log,
                    (size_t)S * H_v);
  PERF_END(gdngt, t_gdn_gates);
  // ---- 5. recurrent gated delta rule (persistent per-layer state) ----
  std::vector<float> &state = recur_state;
  float *const core_out = gdn_grow(gdn_core_b, (size_t)S * value_dim);
  PERF_START(gdndelta);
  qwen4exp_gdn_delta_rule(Q, K, V, beta, g_log, state.data(), core_out, S, H_k,
                          H_v, hd);
  PERF_END(gdndelta, t_gdn_delta);
  check_finite_attn("GDN core_out (post delta rule, pre gate/norm)", layer_idx,
                    core_out, (size_t)S * value_dim);

  // ---- 6. gated output norm: per-head weighted RMSNorm (ssm_norm) *
  // PLAIN sigmoid gate z (attn_gate projection) -- build_norm_gated,
  // "the one numerical difference from Qwen3.5's GDN" (sigmoid, not silu) ----
  std::vector<float> norm_w;
  if (!attn_dequant_tensor_f32(engine, p + "ssm_norm.weight", norm_w,
                               (size_t)hd)) {
    std::memset(out_context, 0, (size_t)S * D * sizeof(float));
    return;
  }
  float *const gated = gdn_grow(gdn_gated_b, (size_t)S * value_dim);
  PERF_START(gdnon);
  // Per-token independent (each (s,h) touches only its own hd-slice of
  // `gated`), so this is bit-identical to the serial form. MEASURED
  // 2026-09-05: 0.63 s of a 2.68 s GDN bucket, the single largest phase,
  // and it was running on one thread.
#pragma omp parallel for schedule(static)
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < H_v; ++h) {
      const float *x = core_out + ((size_t)s * H_v + h) * hd;
      const float *z_h = z + ((size_t)s * H_v + h) * hd;
      float *g_out = gated + ((size_t)s * H_v + h) * hd;
      double msq = 0.0;
      for (int i = 0; i < hd; ++i)
        msq += (double)x[i] * x[i];
      const float inv = 1.0f / std::sqrt((float)(msq / hd) + c.eps);
      for (int i = 0; i < hd; ++i)
        g_out[i] = x[i] * inv * norm_w[i] / (1.0f + std::exp(-z_h[i]));
    }
  }
  check_finite_attn("GDN gated (post norm*sigmoid-gate)", layer_idx, gated,
                    (size_t)S * value_dim);

  PERF_END(gdnon, t_gdn_outnorm);
  // ---- 7. output projection (ssm_out: [value_dim, D]) ----
  PERF_START(gdnout);
  dense_proj(engine, p + "ssm_out.weight", gated, out_context, S, value_dim, D);
  PERF_END(gdnout, t_gdn_outproj);
  check_finite_attn("GDN out_context (post ssm_out projection)", layer_idx,
                    out_context, (size_t)S * D);
}

// Qwen4Exp full-attention layer (qwen4exp.cpp's build_layer_attn, QSA
// top-k DEFERRED -- dense over the whole KV cache in v1). Differences from
// this file's generic dense_attention_gguf (Hunyuan/Llama):
//   - NO attn_norm (the HC mix already normalized the input)
//   - ONE fused attn_q projection holding [q|gate] INTERLEAVED PER HEAD
//     (head h's row pair: q at offset 0, gate at offset head_dim)
//   - per-head QK RMS norms with learned weights (attn_q_norm/attn_k_norm)
//   - partial INTERLEAVED RoPE over the first rope_n_rot dims only
//     (mrope sections degenerate to plain interleaved RoPE for text-only
//     positions -- every section's position is the same token position)
//   - sigmoid output gate on the attention result before attn_output
// Exposed (not static) so qwen4exp_mtp_forward (execution_layer.cpp) can
// reuse this exact math for the shared MTP checkpoint's own extra layer
// (blk.<n_layers>.*, architecturally an ordinary full-attention layer --
// verified against a real tensor dump: identical attn_q/k/v/output shapes
// to a normal full-attention trunk layer). See attention_layer.hpp (or the
// extern declaration at the qwen4exp_mtp_forward call site) for the
// signature.
void qwen4exp_full_attention_gguf(StreamingEngine &engine, int layer_idx,
                                  const float *hidden_states, int S,
                                  int pos_base, float *out_context) {
  PERF_SCOPE(t_attn);
  const auto &c = engine.model_file.architecture_config;
  const int D = c.hidden;
  const int H = c.n_heads;
  const int Hkv = std::max(1, c.n_head_kv);
  const int hd = c.head_dim;
  const int n_rot = c.rope_n_rot > 0 ? c.rope_n_rot : hd;
  const int group = std::max(1, H / Hkv); // query heads sharing one kv head
  const std::string p = "blk." + std::to_string(layer_idx) + ".";
  float rope_theta =
      engine.model_file.get_config_float("rope.freq_base", 10000.0f);

  if (D <= 0 || H <= 0 || hd <= 0) {
    LOG_ERROR("[QWEN4EXP-ATTN] layer=%d invalid dims: D=%d H=%d head_dim=%d "
              "-- zeroing attention output\n",
              layer_idx, D, H, hd);
    std::memset(out_context, 0, (size_t)S * D * sizeof(float));
    return;
  }

  // ---- 1. fused Q|gate + K/V projections (NO attn_norm -- HC already
  // normalized the sub-block input) ----
  std::vector<float> q_full((size_t)S * H * 2 * hd);
  std::vector<float> K((size_t)S * Hkv * hd), V((size_t)S * Hkv * hd);
  std::vector<float> qn_w, kn_w;
  bool norms_ok = attn_dequant_tensor_f32(engine, p + "attn_q_norm.weight",
                                          qn_w, (size_t)hd) &&
                  attn_dequant_tensor_f32(engine, p + "attn_k_norm.weight",
                                          kn_w, (size_t)hd);
  if (!norms_ok) {
    std::memset(out_context, 0, (size_t)S * D * sizeof(float));
    return;
  }
  const void *device_hidden = nullptr;
  if (engine.qwen4exp_device_hc_input)
    device_hidden =
        GgmlHcRunner::instance().device_data(engine.qwen4exp_device_hc_input);
  bool direct_qkv = false;
  if (device_hidden && engine.gpu && S > 1) {
    auto device_projection = [&](const std::string &name, float *out,
                                 int O) -> bool {
      auto it = engine.resident_layers.find(name);
      return it != engine.resident_layers.end() && it->second.device_ptr &&
             engine.gpu->dense_matmul_device_host(
                 it->second.meta.type, it->second.device_ptr,
                 (const float *)device_hidden, out, S, D, O);
    };
    direct_qkv =
        device_projection(p + "attn_q.weight", q_full.data(), H * 2 * hd) &&
        device_projection(p + "attn_k.weight", K.data(), Hkv * hd) &&
        device_projection(p + "attn_v.weight", V.data(), Hkv * hd);
    if (direct_qkv) {
      static bool logged_direct_qkv = false;
      if (!logged_direct_qkv) {
        logged_direct_qkv = true;
        LOG_INFO("[QWEN4EXP-ATTN] full-attention Q/K/V projections "
                 "consume HC's device activation\n");
      }
    }
  }
  if (!direct_qkv) {
    // A partial device projection attempt may have failed after HC kept
    // its result off-host. Recover the authoritative activation before
    // using the established host dense_proj fallback.
    if (device_hidden && !GgmlHcRunner::instance().copy_pre_mix_to_host(
                             engine.qwen4exp_device_hc_input,
                             const_cast<float *>(hidden_states), S)) {
      std::memset(out_context, 0, (size_t)S * D * sizeof(float));
      return;
    }
    dense_proj(engine, p + "attn_q.weight", hidden_states, q_full.data(), S, D,
               H * 2 * hd);
    dense_proj(engine, p + "attn_k.weight", hidden_states, K.data(), S, D,
               Hkv * hd);
    dense_proj(engine, p + "attn_v.weight", hidden_states, V.data(), S, D,
               Hkv * hd);
  }

  // De-interleave per head: q at offset 0, gate at offset head_dim
  // (qwen4exp.cpp's ggml_view_3d calls on Qcur_full).  The CUDA QKV path
  // already performed this split and the following normalization/RoPE.
  std::vector<float> Q((size_t)S * H * hd), gate((size_t)S * H * hd);
  {
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < H; ++h) {
        const float *row = q_full.data() + ((size_t)s * H + h) * 2 * hd;
        std::memcpy(Q.data() + ((size_t)s * H + h) * hd, row,
                    (size_t)hd * sizeof(float));
        std::memcpy(gate.data() + ((size_t)s * H + h) * hd, row + hd,
                    (size_t)hd * sizeof(float));
      }

    // ---- 2. per-head QK RMS norms (learned weights), then partial
    // interleaved RoPE over the first n_rot dims ----
    for (int s = 0; s < S; ++s) {
      int pos = pos_base + s;
      for (int h = 0; h < H; ++h) {
        float *q_h = Q.data() + ((size_t)s * H + h) * hd;
        rmsnorm_row(q_h, q_h, qn_w.data(), hd, c.eps);
        if (n_rot > 0)
          rope_yarn_inplace(q_h, n_rot, pos, rope_theta,
                            /*rope_ext_factor=*/0.0f, /*freq_scale=*/1.0f,
                            /*attn_factor=*/1.0f, /*yarn_log_mul=*/0.0f,
                            /*n_ctx_orig=*/0);
      }
      for (int h = 0; h < Hkv; ++h) {
        float *k_h = K.data() + ((size_t)s * Hkv + h) * hd;
        rmsnorm_row(k_h, k_h, kn_w.data(), hd, c.eps);
        if (n_rot > 0)
          rope_yarn_inplace(k_h, n_rot, pos, rope_theta, 0.0f, 1.0f, 1.0f, 0.0f,
                            0);
      }
    }
  }

  // ---- 3. write K/V into the dense cache ----
  const int kvd = Hkv * hd;
  {
    for (int s = 0; s < S; ++s) {
      int pos = pos_base + s;
      if (pos >= engine.max_t)
        continue; // bounds check, mirrors dense path
      std::memcpy(engine.Kc[layer_idx] + (size_t)pos * kvd,
                  K.data() + (size_t)s * kvd, (size_t)kvd * sizeof(float));
      std::memcpy(engine.Vc[layer_idx] + (size_t)pos * kvd,
                  V.data() + (size_t)s * kvd, (size_t)kvd * sizeof(float));
    }
  }

  // ---- 4. causal GQA attention (dense; QSA top-k deferred) ----
  // kq_scale: qwen4exp.cpp uses 1/sqrt(head_dim) when f_attention_scale
  // is unset (it is, for this model).
  const float kq_scale = 1.0f / std::sqrt((float)hd);
  std::vector<float> head_contexts((size_t)S * H * hd, 0.0f);

  {
#pragma omp parallel for collapse(2) schedule(dynamic, 1)                      \
    num_threads(attn_score_threads())
    for (int s = 0; s < S; ++s) {
      for (int h = 0; h < H; ++h) {
        int current_pos = pos_base + s;
        int context_len = current_pos + 1;
        if (context_len > engine.max_t)
          context_len = engine.max_t;
        int kv_head = h / group;
        const float *q_h = Q.data() + ((size_t)s * H + h) * hd;

        // Per-thread grow-only score buffer. This was a
        // `std::vector<float> scores(context_len)` constructed HERE, i.e.
        // once per (token, head): 3189 tokens x 24 heads x 12 dense
        // layers = ~918,000 heap allocations for ONE 3,189-token prefill.
        // thread_local is correct inside this OMP region (one buffer per
        // worker) and grow-only, so it settles at the longest context
        // this layer sees and never reallocates again. The explicit `<`
        // guard rather than a bare resize() keeps a shorter context from
        // re-zero-initialising the buffer every call.
        static thread_local std::vector<float> scores_tls;
        if ((int)scores_tls.size() < context_len)
          scores_tls.resize((size_t)context_len);
        float *scores = scores_tls.data();
        for (int t = 0; t < context_len; ++t) {
          const float *k_t =
              engine.Kc[layer_idx] + (size_t)t * kvd + (size_t)kv_head * hd;
          // Was a scalar `for (i<hd) dot += q_h[i]*k_t[i]`.
          // dot_f32_avx512 is this codebase's own AVX-512 row dot
          // (kernels_cpu.cpp), already used at three other sites in
          // THIS file. hd=256 here, so 16 vector FMAs per score.
          scores[t] = dot_f32_avx512(q_h, k_t, hd) * kq_scale;
        }
        softmax_row(scores, context_len);

        float *out_h = head_contexts.data() + ((size_t)s * H + h) * hd;
        for (int t = 0; t < context_len; ++t) {
          const float *v_t =
              engine.Vc[layer_idx] + (size_t)t * kvd + (size_t)kv_head * hd;
          const float alpha = scores[t];
          // Was a scalar `for (i<hd) out_h[i] += alpha*v_t[i]`. Same
          // axpy shape (and same vectorisation) as the weighted
          // accumulate in execution_layer.cpp. Scalar tail kept for a
          // non-multiple-of-16 head_dim.
          const __m512 va = _mm512_set1_ps(alpha);
          int i2 = 0;
          for (; i2 + 16 <= hd; i2 += 16)
            _mm512_storeu_ps(out_h + i2,
                             _mm512_fmadd_ps(va, _mm512_loadu_ps(v_t + i2),
                                             _mm512_loadu_ps(out_h + i2)));
          for (; i2 < hd; ++i2)
            out_h[i2] += alpha * v_t[i2];
        }
      }
    }
  }
  check_finite_attn("ATTN head_contexts (post attention, pre gate)",
                      layer_idx, head_contexts.data(), head_contexts.size());

  // ---- 5. sigmoid output gate (from the fused Q projection), then wo ----
  {
    for (size_t i = 0; i < head_contexts.size(); ++i)
      head_contexts[i] /= (1.0f + std::exp(-gate[i]));
    check_finite_attn("ATTN gated (sigmoid output gate)", layer_idx,
                      head_contexts.data(), head_contexts.size());
    dense_proj(engine, p + "attn_output.weight", head_contexts.data(),
               out_context, S, H * hd, D);
    check_finite_attn("ATTN out_context (post attn_output projection)",
                      layer_idx, out_context, (size_t)S * D);

    // Host HC post-combine is only a fallback now that full attention can
    // consume and produce device data. Make its input authoritative if a
    // projection, score kernel, or output handoff failed mid-layer.
    if (device_hidden && !GgmlHcRunner::instance().copy_pre_mix_to_host(
                             engine.qwen4exp_device_hc_input,
                             const_cast<float *>(hidden_states), S)) {
      LOG_ERROR("[QWEN4EXP-ATTN] failed to recover HC input for host "
                "fallback at layer=%d\n",
                layer_idx);
    }
  }
}

// Fallback bridge for the rare case where a device GDN graph completed but
// the following HC post graph declined its input. Normally the result goes
// directly from GDN's persistent output into HC and this is never called.
bool qwen4exp_copy_gdn_device_output_to_host(const void *device_output,
                                             float *host, int S) {
  return GgmlGdnProjRunner::instance().copy_full_output_to_host(device_output,
                                                                host, S);
}

// Qwen4Exp hybrid per-layer dispatch: the recurrent-layer map (GGUF
// attention.recurrent_layers, or the full_attention_interval fallback) is
// the layer-type switch -- GDN (recurrent) vs dense full attention.
void qwen4exp_attention_gguf(StreamingEngine &engine, int layer_idx,
                             const float *hidden_states, int S, int pos_base,
                             float *out_context) {
  if (qwen4exp_is_gdn_layer(engine.model_file.architecture_config, layer_idx)) {
    PERF_SCOPE(t_attn_qwen_gdn);
    qwen4exp_gdn_gguf(engine, layer_idx, hidden_states, S, pos_base,
                      out_context);
  } else {
    PERF_SCOPE(t_attn_qwen_dense);
    qwen4exp_full_attention_gguf(engine, layer_idx, hidden_states, S, pos_base,
                                 out_context);
  }
}

struct Dsv4CompressorWeights {
  const float *ape = nullptr;
  const float *norm = nullptr;
};

static const QuantizedTensor *
dsv4_resident_tensor(const StreamingEngine &engine, const std::string &name) {
  const auto it = engine.resident_layers.find(name);
  return it != engine.resident_layers.end() && it->second.host_ptr ? &it->second
                                                                   : nullptr;
}

static bool dsv4_load_compressor_weights(const StreamingEngine &engine,
                                         const std::string &prefix,
                                         const char *stem,
                                         Dsv4CompressorWeights &weights) {
  const QuantizedTensor *ape =
      dsv4_resident_tensor(engine, prefix + stem + "_ape.weight");
  const QuantizedTensor *norm =
      dsv4_resident_tensor(engine, prefix + stem + "_norm.weight");
  if (!ape || !norm || ape->meta.type != GGUFType::F32 ||
      norm->meta.type != GGUFType::F32) {
    LOG_ERROR("[DSV4] Missing or non-F32 %s_{ape,norm} tensors\n", stem);
    return false;
  }
  weights.ape = reinterpret_cast<const float *>(ape->host_ptr);
  weights.norm = reinterpret_cast<const float *>(norm->host_ptr);
  return true;
}

static bool dsv4_has_projection_pair(const StreamingEngine &engine,
                                     const std::string &prefix,
                                     const char *stem) {
  return dsv4_resident_tensor(engine, prefix + stem + "_kv.weight") &&
         dsv4_resident_tensor(engine, prefix + stem + "_gate.weight");
}

static void dsv4_store_compressor_state(Dsv4CompressedLayerState &state,
                                        int pos, const float *kv,
                                        const float *score, const float *ape,
                                        int width, std::vector<float> &kv_state,
                                        std::vector<float> &score_state) {
  const int slot = dsv4_state_slot(pos, state.state_size);
  float *kv_dst = kv_state.data() + (size_t)slot * width;
  float *score_dst = score_state.data() + (size_t)slot * width;
  const float *ape_row = ape + (size_t)(pos % state.ratio) * width;
  for (int i = 0; i < width; ++i) {
    kv_dst[i] = kv[i];
    score_dst[i] = score[i] + ape_row[i];
  }
}

static void dsv4_compress_state_block(const Dsv4CompressedLayerState &state,
                                      const std::vector<float> &kv_state,
                                      const std::vector<float> &score_state,
                                      int block_end_pos, int dim, bool overlap,
                                      float *out) {
  const float neg_inf = -std::numeric_limits<float>::infinity();
  const int ratio = state.ratio;
  const int current_start = block_end_pos + 1 - ratio;

  for (int d = 0; d < dim; ++d) {
    float max_score = neg_inf;
    for (int t = 0; t < (overlap ? 2 * ratio : ratio); ++t) {
      const bool previous = overlap && t < ratio;
      const int pos =
          current_start + (previous ? -ratio + t : t - (overlap ? ratio : 0));
      const int offset = previous ? 0 : (overlap ? dim : 0);
      const float score =
          pos < 0 ? neg_inf
                  : score_state[(size_t)dsv4_state_slot(pos, state.state_size) *
                                    (overlap ? 2 * dim : dim) +
                                offset + d];
      max_score = std::max(max_score, score);
    }

    float denominator = 0.0f;
    float weighted_value = 0.0f;
    for (int t = 0; t < (overlap ? 2 * ratio : ratio); ++t) {
      const bool previous = overlap && t < ratio;
      const int pos =
          current_start + (previous ? -ratio + t : t - (overlap ? ratio : 0));
      const int offset = previous ? 0 : (overlap ? dim : 0);
      if (pos < 0)
        continue; // synthetic zero value / -inf score
      const size_t row = (size_t)dsv4_state_slot(pos, state.state_size) *
                             (overlap ? 2 * dim : dim) +
                         offset + d;
      const float weight = std::exp(score_state[row] - max_score);
      denominator += weight;
      weighted_value += weight * kv_state[row];
    }
    out[d] = denominator > 0.0f ? weighted_value / denominator : 0.0f;
  }
}

static bool dsv4_commit_compressed_key(
    Dsv4CompressedLayerState &state, int block_end_pos, int nope, int rope,
    float rope_theta, float rope_ext_factor, float rope_freq_scale,
    float rope_attn_factor, float rope_yarn_log_mul, int rope_n_ctx_orig,
    float norm_eps, const Dsv4CompressorWeights &weights,
    const std::vector<float> &kv_state, const std::vector<float> &score_state,
    std::vector<float> &keys, int key_dim) {
  const int block = block_end_pos / state.ratio;
  if (block < 0 || block >= state.cache_capacity) {
    LOG_ERROR("[DSV4] Compressed-cache block %d is outside capacity %d\n",
              block, state.cache_capacity);
    return false;
  }
  std::vector<float> key(key_dim);
  dsv4_compress_state_block(state, kv_state, score_state, block_end_pos,
                            key_dim, state.is_csa(), key.data());
  rmsnorm_row(key.data(), key.data(), weights.norm, key_dim, norm_eps);
  rope_yarn_inplace(key.data() + (key_dim - rope), rope,
                    block_end_pos + 1 - state.ratio, rope_theta,
                    rope_ext_factor, rope_freq_scale, rope_attn_factor,
                    rope_yarn_log_mul, rope_n_ctx_orig, true);
  std::memcpy(keys.data() + (size_t)block * key_dim, key.data(),
              (size_t)key_dim * sizeof(float));
  return true;
}

// ============================================================
// DeepSeek-V4 MLA attention. Ratio-zero layers use raw sliding-window
// attention; ratio-4 CSA layers add overlap-compressed keys selected by the
// Lightning Indexer; ratio-128 HCA layers add completed compressed blocks.
// Structured like
// mla_attention_absorbed_gguf above (same attn_norm -> q_a/q_a_norm/q_b ->
// RoPE-on-rope-half Q projection, reused verbatim), but the KV side has NO
// absorption: attn_kv.weight produces one per-head-sized latent directly
// (no wk_b/wv_b, no separate kv_lora_rank), used as BOTH K and V (MQA-
// style attention over the raw cached latent. New pieces V3 doesn't have:
// compressed key state, attention sinks, a de-rope step
// on the attention output, and a grouped low-rank output projection
// (attn_output_a/attn_output_b) replacing the single attn_output matmul.
//
// UNVERIFIED ON REAL HARDWARE end-to-end, but two real bugs found and fixed
// via a direct line-by-line comparison against llama.cpp's real
// deepseek4.cpp (not just paraphrase/research notes) after real-hardware
// generation produced coherent-looking-but-garbage output even after a
// separate, unrelated tid2eid on-disk corruption bug was fixed:
//   1. This function was feeding the model-global c.rope_ext_factor/
//      c.rope_freq_scale (correct for V2/V3/GLM's mla_attention_absorbed_gguf,
//      which applies YaRN uniformly per the model's rope.scaling.type
//      metadata) into RoPE and the attention kq_scale here too. Confirmed
//      against deepseek4.cpp's build_attention_impl: V4 only ever enables
//      ext_factor/freq_scale for CSA/HCA "compress-rope" layers
//      (dsv4_compress_ratios[il] != 0); for every
//      ordinary layer it forces ext_factor=0/freq_scale=1/attn_factor=1
//      UNCONDITIONALLY, and the softmax scale is hardcoded plain
//      1/sqrt(n_embd_head) at every build_attn_mha call site -- no mscale
//      term at all, regardless of the checkpoint's global YaRN metadata.
//      Fixed by hardcoding plain values at all 4 RoPE/scale call sites in
//      this function instead of threading through c.rope_ext_factor.
//   2. Missing a weightless (no learned weight tensor) per-head RMSNorm on
//      the full [qh] Q vector immediately after the attn_q_b projection,
//      before the nope/rope split and before RoPE -- confirmed present in
//      deepseek4.cpp as `q = ggml_rms_norm(ctx0, q, norm_rms_eps)`, entirely
//      separate from the WEIGHTED attn_q_a_norm applied earlier in the
//      q_a->q_a_norm->q_b decompose. Added as rmsnorm_inplace_noweight.
// (a) attn_kv_norm applying to the WHOLE qh-wide latent before splitting
// nope/rope (not nope-only) is CONFIRMED correct as implemented here,
// matching deepseek4.cpp's own "wkv -> kv_norm -> split -> rope" ordering
// exactly. (b) the exact on-disk byte layout of the grouped attn_output_a
// tensor's {H*qh/o_groups, o_lora_rank, o_groups} shape is still unverified
// on real hardware.
// ============================================================
void dsv4_attention_gguf(StreamingEngine &engine, int layer_idx,
                         const float *hidden_states, int S, int pos_base,
                         float *out_context) {
  PERF_SCOPE(t_attn);
  // Sub-phase timers added to match mla_attention_absorbed_gguf's own
  // breakdown (this file, above) -- dsv4_attention_gguf is the function
  // ACTUALLY called for DeepSeek-V4 (mla_attention_absorbed_gguf is V2/
  // V3/GLM only), but had none of these until now, which is why every
  // [REQUEST] line's "Attention (MLA)" sub-phase breakdown always showed
  // preamble/norm/qproj/kvproj/absorb/score/outproj all at exactly 0.00
  // despite t_attn itself (PERF_SCOPE above) being real and substantial --
  // the counters existed and were wired into every report format string,
  // just never written to by whichever function actually ran for this
  // architecture. No attn_absorb phase here: V4 does K==V MQA with no
  // absorption step (see this function's own header comment), so
  // t_attn_absorb correctly stays 0 for this architecture -- that is not
  // a gap, it is a real "this stage doesn't exist" difference from V2/V3.
  PERF_START(dsv4_preamble);
  const auto &c = engine.model_file.architecture_config;
  int H = c.n_heads;
  int D = c.hidden;
  int nope = c.qk_nope;
  int rope = c.qk_rope;
  int qh = nope + rope; // per-head latent width, K==V, no separate v_head
  std::string p = "blk." + std::to_string(layer_idx) + ".";
  float rope_theta =
      engine.model_file.get_config_float("rope.freq_base", 10000.0f);
  int window = engine.sliding_window > 0 ? engine.sliding_window : engine.max_t;
  const int ratio =
      (!c.dsv4_compress_ratios.empty() &&
       layer_idx < (int)c.dsv4_compress_ratios.size())
          ? static_cast<int>(c.dsv4_compress_ratios[(size_t)layer_idx])
          : 0;
  const bool compressed = ratio != 0;
  Dsv4CompressedLayerState *compressed_state =
      compressed && layer_idx < (int)engine.dsv4_compressed.size()
          ? &engine.dsv4_compressed[(size_t)layer_idx]
          : nullptr;
  if (compressed && (!compressed_state || !compressed_state->enabled() ||
                     compressed_state->ratio != ratio)) {
    LOG_ERROR("[DSV4] Layer %d compressed attention state is unavailable\n",
              layer_idx);
    std::memset(out_context, 0, (size_t)S * D * sizeof(float));
    return;
  }
  // ZeroMoE starts each request at position zero. Its cache model is a
  // single contiguous sequence, so reset this layer's compressor state at
  // that boundary rather than retaining llama.cpp-style sequence snapshots.
  if (compressed && pos_base == 0)
    compressed_state->reset();

  const float dsv4_rope_theta =
      compressed ? c.dsv4_compress_rope_freq_base : rope_theta;
  const float dsv4_rope_ext_factor = compressed ? c.rope_ext_factor : 0.0f;
  const float dsv4_rope_freq_scale = compressed ? c.rope_freq_scale : 1.0f;
  const float dsv4_rope_attn_factor = compressed ? c.rope_attn_factor : 1.0f;
  const int dsv4_rope_n_ctx_orig = compressed ? c.rope_n_ctx_orig : 0;

  // Confirms this function is actually reached for this checkpoint and
  // surfaces whether YaRN (rope_ext_factor != 0) is active, plus the
  // grouped-outproj shape (o_group_count/o_lora_rank/group_in) needed to
  // reason about whether GPU dispatch could help there -- was a raw
  // fprintf(stderr, ...) (bypasses log_path entirely, easy to miss and
  // not captured in ds4_debug.log), converted to LOG_STREAM_INFO so it's
  // reliably in the log like everything else.
  {
    static bool dsv4_debug_once = false;
    if (!dsv4_debug_once) {
      dsv4_debug_once = true;
      int o_groups_dbg = std::max(1, c.o_group_count);
      LOG_STREAM_INFO()
          << "[DSV4-DEBUG] dsv4_attention_gguf reached: layer=" << layer_idx
          << " H=" << H << " nope=" << nope << " rope=" << rope << " qh=" << qh
          << " rope_ext_factor=" << c.rope_ext_factor
          << " rope_freq_scale=" << c.rope_freq_scale
          << " rope_theta=" << rope_theta << " window=" << window
          << " compress_ratio=" << ratio << " q_lora="
          << engine.model_file.get_config_int("attention.q_lora_rank", 2048)
          << " o_group_count=" << c.o_group_count
          << " o_lora_rank=" << c.o_lora_rank
          << " group_in=(H*qh)/o_groups=" << ((H * qh) / o_groups_dbg) << "\n";
    }
  }
  PERF_END(dsv4_preamble, t_attn_preamble);

  // ---- 1. attention input norm (same as V3) ----
  PERF_START(dsv4_norm);
  std::vector<float> normed((size_t)S * D);
  rmsnorm_proj(engine, p + "attn_norm.weight", hidden_states, normed.data(), S,
               D, c.eps);
  if (engine.family_ops.family == ModelFamily::DeepSeek_V41) {
    dsv41_trace::save("attn-input", layer_idx, pos_base,
                      hidden_states + (size_t)(S - 1) * D, D);
    if (S > 1)
      dsv41_trace::save("attn-input-seq", layer_idx, pos_base, hidden_states,
                        (size_t)S * D);
    dsv41_trace::save("attn-norm", layer_idx, pos_base,
                      normed.data() + (size_t)(S - 1) * D, D);
    if (S > 1)
      dsv41_trace::save("attn-norm-seq", layer_idx, pos_base, normed.data(),
                        (size_t)S * D);
  }
  check_finite_attn("normed (post attn_norm rmsnorm)", layer_idx, normed.data(),
                    (size_t)S * D);
  PERF_END(dsv4_norm, t_attn_norm);

  // ---- 2. project Q: q_a -> q_a_norm -> q_b => Q[S, H*qh], RoPE on the
  // rope-half only. Identical code shape to mla_attention_absorbed_gguf's
  // own step 2/2b -- V4 keeps the same low-rank Q decompose pattern. ----
  PERF_START(dsv4_qproj);
  int q_lora = engine.model_file.get_config_int("attention.q_lora_rank", 2048);
  std::vector<float> Q((size_t)S * H * qh, 0.0f);
  std::vector<float> ql_buf((size_t)S * q_lora);
  dense_proj(engine, p + "attn_q_a.weight", normed.data(), ql_buf.data(), S, D,
             q_lora);
  rmsnorm_proj(engine, p + "attn_q_a_norm.weight", ql_buf.data(), ql_buf.data(),
               S, q_lora, c.eps);
  if (engine.family_ops.family == ModelFamily::DeepSeek_V41)
    dsv41_trace::save("q-lora", layer_idx, pos_base,
                      ql_buf.data() + (size_t)(S - 1) * q_lora, q_lora);
  if (engine.family_ops.family == ModelFamily::DeepSeek_V41 && S > 1)
    dsv41_trace::save("q-lora-seq", layer_idx, pos_base, ql_buf.data(),
                      ql_buf.size());
  dense_proj(engine, p + "attn_q_b.weight", ql_buf.data(), Q.data(), S, q_lora,
             H * qh);
  // FIX: this was the one stage in this function without a check (every
  // other stage already has one -- confirmed via a real audit) -- the
  // matching V3 path (mla_attention_absorbed_gguf, this file, line 146)
  // already checks its equivalent Q projection output pre-RoPE; this one
  // didn't, so a NaN introduced by attn_q_a/attn_q_a_norm/attn_q_b here
  // would only surface (if at all) at the post-RoPE check below, one
  // stage later than necessary.
  check_finite_attn("Q (post attn_q_a/q_a_norm/q_b projection, pre-RoPE)",
                    layer_idx, Q.data(), Q.size(), H * qh);
  // The original V4 path has a weightless per-head RMSNorm here. V4.1's
  // reference Attention applies only q_a_norm before q_b, so skip that
  // legacy normalization when this function is used for V4.1 ratio-zero
  // layers.
  if (engine.family_ops.family != ModelFamily::DeepSeek_V41) {
    for (int s = 0; s < S; ++s) {
      for (int h = 0; h < H; ++h) {
        rmsnorm_inplace_noweight(Q.data() + ((size_t)s * H + h) * qh, qh,
                                 c.eps);
      }
    }
    check_finite_attn("Q (post per-head RMSNorm, pre-RoPE)", layer_idx,
                      Q.data(), Q.size(), H * qh);
  }
  // CSA/HCA use the checkpoint's dedicated compressed-RoPE base plus
  // YaRN scaling; raw-SWA layers stay on the plain V4 RoPE path.
  for (int s = 0; s < S; ++s) {
    int pos = pos_base + s;
    for (int h = 0; h < H; ++h) {
      float *q_rope = Q.data() + ((size_t)s * H + h) * qh + nope;
      rope_yarn_inplace(q_rope, rope, pos, dsv4_rope_theta,
                        dsv4_rope_ext_factor, dsv4_rope_freq_scale,
                        dsv4_rope_attn_factor, c.rope_yarn_log_mul,
                        dsv4_rope_n_ctx_orig, /*dsv4_mscale=*/compressed);
    }
  }
  if (engine.family_ops.family == ModelFamily::DeepSeek_V41)
    dsv41_trace::save("query", layer_idx, pos_base,
                      Q.data() + (size_t)(S - 1) * H * qh,
                      (size_t)H * qh);
  if (engine.family_ops.family == ModelFamily::DeepSeek_V41 && S > 1)
    dsv41_trace::save("query-seq", layer_idx, pos_base, Q.data(), Q.size());
  check_finite_attn("Q (post RoPE)", layer_idx, Q.data(), Q.size(), H * qh);
  PERF_END(dsv4_qproj, t_attn_qproj);

  // ---- 3. project raw KV and update any compressed-state machine. ----
  PERF_START(dsv4_kvproj);
  std::vector<float> raw_kv_prefill((size_t)S * qh);
  {
    std::vector<float> &kv = raw_kv_prefill;
    dense_proj(engine, p + "attn_kv.weight", normed.data(), kv.data(), S, D,
               qh);
    check_finite_attn("kv (post attn_kv projection, pre-norm/RoPE)", layer_idx,
                      kv.data(), kv.size(), qh);
    rmsnorm_proj(engine, p + "attn_kv_a_norm.weight", kv.data(), kv.data(), S,
                 qh, c.eps);

    Dsv4CompressorWeights compressor_weights;
    Dsv4CompressorWeights indexer_compressor_weights;
    std::vector<float> compressor_kv;
    std::vector<float> compressor_gate;
    std::vector<float> indexer_compressor_kv;
    std::vector<float> indexer_compressor_gate;
    if (compressed) {
      if (!dsv4_has_projection_pair(engine, p, "attn_compressor") ||
          !dsv4_load_compressor_weights(engine, p, "attn_compressor",
                                        compressor_weights)) {
        std::memset(out_context, 0, (size_t)S * D * sizeof(float));
        return;
      }
      const int compressor_width = compressed_state->state_width;
      compressor_kv.resize((size_t)S * compressor_width);
      compressor_gate.resize((size_t)S * compressor_width);
      dense_proj(engine, p + "attn_compressor_kv.weight", normed.data(),
                 compressor_kv.data(), S, D, compressor_width);
      dense_proj(engine, p + "attn_compressor_gate.weight", normed.data(),
                 compressor_gate.data(), S, D, compressor_width);

      if (compressed_state->is_csa()) {
        if (!dsv4_has_projection_pair(engine, p, "indexer_compressor") ||
            !dsv4_load_compressor_weights(engine, p, "indexer_compressor",
                                          indexer_compressor_weights)) {
          std::memset(out_context, 0, (size_t)S * D * sizeof(float));
          return;
        }
        const int indexer_width = 2 * c.dsv4_indexer_key_length;
        indexer_compressor_kv.resize((size_t)S * indexer_width);
        indexer_compressor_gate.resize((size_t)S * indexer_width);
        dense_proj(engine, p + "indexer_compressor_kv.weight", normed.data(),
                   indexer_compressor_kv.data(), S, D, indexer_width);
        dense_proj(engine, p + "indexer_compressor_gate.weight", normed.data(),
                   indexer_compressor_gate.data(), S, D, indexer_width);
      }
    }

    for (int s = 0; s < S; ++s) {
      int pos = pos_base + s;
      int slot = pos % window;
      float *kv_row = kv.data() + (size_t)s * qh;
      float *k_rope = kv_row + nope;
      rope_yarn_inplace(k_rope, rope, pos, dsv4_rope_theta,
                        dsv4_rope_ext_factor, dsv4_rope_freq_scale,
                        dsv4_rope_attn_factor, c.rope_yarn_log_mul,
                        dsv4_rope_n_ctx_orig, /*dsv4_mscale=*/compressed);
      if (s == S - 1 &&
          engine.family_ops.family == ModelFamily::DeepSeek_V41)
        dsv41_trace::save("raw-kv", layer_idx, pos_base, kv_row, qh);
      if (s == S - 1 && Logger::instance().would_log(LogLevel::VERBOSE)) {
        double sq = 0.0;
        for (int i = 0; i < qh; ++i)
          sq += (double)kv_row[i] * kv_row[i];
        LOG_STREAM_VERBOSE()
            << "[MLA-TRACE l=" << layer_idx
            << "] kv_row (post norm+RoPE, pre-cache-write) |last_row|="
            << std::sqrt(sq) << "\n";
      }
      std::memcpy(engine.KVc[layer_idx] + (int64_t)slot * qh, kv_row,
                  (size_t)qh * sizeof(float));

      // dsv4_score_gpu (EngineConfig): keep this layer's raw-window
      // ring-buffer GPU mirror in sync, one row at a time -- ratio==4
      // (CSA) layers are excluded, they stay entirely on CPU (see
      // dsv4_score_gpu's own comment, engine_config.hpp).
      // ensure_dsv4_kvc_gpu is idempotent (a cheap no-op once this
      // layer's mirror is already allocated), so no extra static
      // per-layer cache is needed at this call site.
      if (ratio != kDsv4CsaRatio && dsv4_score_gpu_enabled() && engine.gpu) {
        if (engine.gpu->ensure_dsv4_kvc_gpu(
                layer_idx, window, qh,
                (size_t)(engine.dsv4_score_gpu_max_vram_gb * 1e9))) {
          engine.gpu->upload_dsv4_kvc_row(layer_idx, kv_row, slot, qh);
        }
      }

      if (!compressed)
        continue;
      dsv4_store_compressor_state(
          *compressed_state, pos,
          compressor_kv.data() + (size_t)s * compressed_state->state_width,
          compressor_gate.data() + (size_t)s * compressed_state->state_width,
          compressor_weights.ape, compressed_state->state_width,
          compressed_state->state_kv, compressed_state->state_score);
      if (compressed_state->is_csa()) {
        const int indexer_width = 2 * c.dsv4_indexer_key_length;
        dsv4_store_compressor_state(
            *compressed_state, pos,
            indexer_compressor_kv.data() + (size_t)s * indexer_width,
            indexer_compressor_gate.data() + (size_t)s * indexer_width,
            indexer_compressor_weights.ape, indexer_width,
            compressed_state->indexer_state_kv,
            compressed_state->indexer_state_score);
      }
      if ((pos + 1) % ratio != 0)
        continue;

      if (!dsv4_commit_compressed_key(
              *compressed_state, pos, nope, rope, dsv4_rope_theta,
              dsv4_rope_ext_factor, dsv4_rope_freq_scale, dsv4_rope_attn_factor,
              c.rope_yarn_log_mul, dsv4_rope_n_ctx_orig, c.eps,
              compressor_weights, compressed_state->state_kv,
              compressed_state->state_score, compressed_state->keys, qh)) {
        std::memset(out_context, 0, (size_t)S * D * sizeof(float));
        return;
      }
      // dsv4_score_gpu: mirror the newly-committed HCA block to this
      // layer's GPU cache. HCA only (ratio==128) -- CSA's own
      // compressed cache stays CPU-only, see dsv4_score_gpu's comment.
      if (compressed_state->is_hca() && dsv4_score_gpu_enabled() &&
          engine.gpu) {
        const int block = pos / compressed_state->ratio;
        if (engine.gpu->ensure_dsv4_hca_gpu(
                layer_idx, compressed_state->cache_capacity, qh,
                (size_t)(engine.dsv4_score_gpu_max_vram_gb * 1e9))) {
          engine.gpu->upload_dsv4_hca_row(
              layer_idx, compressed_state->keys.data() + (size_t)block * qh,
              block, qh);
        }
      }
      if (compressed_state->is_csa() &&
          !dsv4_commit_compressed_key(
              *compressed_state, pos, c.dsv4_indexer_key_length - rope, rope,
              dsv4_rope_theta, dsv4_rope_ext_factor, dsv4_rope_freq_scale,
              dsv4_rope_attn_factor, c.rope_yarn_log_mul, dsv4_rope_n_ctx_orig,
              c.eps, indexer_compressor_weights,
              compressed_state->indexer_state_kv,
              compressed_state->indexer_state_score,
              compressed_state->indexer_keys, c.dsv4_indexer_key_length)) {
        std::memset(out_context, 0, (size_t)S * D * sizeof(float));
        return;
      }
    }
  }
  if (engine.family_ops.family == ModelFamily::DeepSeek_V41 && S > 1)
    dsv41_trace::save("raw-kv-sequence", layer_idx, pos_base,
                      raw_kv_prefill.data(), raw_kv_prefill.size());
  check_finite_attn("KVc (post write)", layer_idx, engine.KVc[layer_idx],
                    (size_t)window * qh);
  PERF_END(dsv4_kvproj, t_attn_kvproj);
  // Max per-row norm across every cache row actually in-window for the
  // newest token (s==S-1) -- the real quantity the softmax convex-
  // combination bound (head_contexts <= max over blended rows) must be
  // checked against, since head_contexts blends up to `window` distinct
  // historical rows, not just the just-written one traced above. A single
  // large-magnitude row anywhere in that window is enough for softmax to
  // legitimately land close to it and still respect the bound.
  if (Logger::instance().would_log(LogLevel::VERBOSE)) {
    int current_pos = pos_base + (S - 1);
    int past_context_len = std::min(current_pos + 1, window);
    double max_sq = 0.0;
    int max_t = -1;
    for (int t = 0; t < past_context_len; ++t) {
      int abs_pos = current_pos - past_context_len + 1 + t;
      int slot = abs_pos % window;
      const float *cached = engine.KVc[layer_idx] + (int64_t)slot * qh;
      double sq = 0.0;
      for (int i = 0; i < qh; ++i)
        sq += (double)cached[i] * cached[i];
      if (sq > max_sq) {
        max_sq = sq;
        max_t = t;
      }
    }
    LOG_STREAM_VERBOSE() << "[MLA-TRACE l=" << layer_idx
                         << "] KVc window max |row|=" << std::sqrt(max_sq)
                         << " (t=" << max_t << "/" << past_context_len << ")\n";
  }

  // ---- 4. MQA attention over the raw cached latent (K==V, no
  // absorption), windowed to the last `window` positions, plus a
  // per-head attention-sink logit. ----
  PERF_START(dsv4_score);
  auto sinks_it = engine.resident_layers.find(p + "attn_sinks.weight");
  const float *sinks =
      (sinks_it != engine.resident_layers.end() && sinks_it->second.host_ptr)
          ? reinterpret_cast<float *>(sinks_it->second.host_ptr)
          : nullptr;

  // CSA selects compressed cache rows with the Lightning Indexer. The
  // indexer query comes from the q_lora latent before attn_q_b, whereas
  // its learned per-head projection comes from this layer's normalized
  // hidden state. Both are exactly the tensors used by deepseek4.cpp.
  //
  // GAP vs deepseek4.cpp (audited 2026-08-31, not fixed -- believed a
  // no-op at F32): llama.cpp additionally rotates indexer_q/indexer_k
  // through a fixed random Hadamard matrix (llama_mul_mat_hadamard,
  // llama-impl.h) before this dot product, "always" (independent of K
  // quantization) per that file's own comment. A rotation applied
  // identically to both operands of a dot product doesn't change an exact
  // dot product (R^T R = I for an orthonormal Hadamard matrix, so
  // (Rq)*(Rk) = q*k), and ReLU is applied to the dot AFTER rotation either
  // way -- so in this function's all-F32 CPU path this rotation should be
  // a pure no-op on the actual score values. It matters upstream only for
  // reducing quantization error in a quantized K/indexer cache, which this
  // CPU path doesn't have yet. Revisit if/when a quantized indexer cache
  // is added here.
  std::vector<std::vector<int>> csa_selected((size_t)S);
  if (compressed && compressed_state->is_csa()) {
    if (!dsv4_resident_tensor(engine, p + "indexer.proj.weight") ||
        !dsv4_resident_tensor(engine, p + "indexer.attn_q_b.weight")) {
      LOG_ERROR("[DSV4] CSA layer %d is missing Lightning Indexer weights\n",
                layer_idx);
      std::memset(out_context, 0, (size_t)S * D * sizeof(float));
      return;
    }
    const int indexer_heads = c.dsv4_indexer_head_count;
    const int indexer_dim = c.dsv4_indexer_key_length;
    std::vector<float> indexer_q((size_t)S * indexer_heads * indexer_dim);
    std::vector<float> indexer_proj((size_t)S * indexer_heads);
    dense_proj(engine, p + "indexer.attn_q_b.weight", ql_buf.data(),
               indexer_q.data(), S, q_lora, indexer_heads * indexer_dim);
    dense_proj(engine, p + "indexer.proj.weight", normed.data(),
               indexer_proj.data(), S, D, indexer_heads);
    for (int s = 0; s < S; ++s) {
      const int pos = pos_base + s;
      for (int h = 0; h < indexer_heads; ++h) {
        float *q_rope = indexer_q.data() +
                        ((size_t)s * indexer_heads + h) * indexer_dim +
                        indexer_dim - rope;
        rope_yarn_inplace(q_rope, rope, pos, dsv4_rope_theta,
                          dsv4_rope_ext_factor, dsv4_rope_freq_scale,
                          dsv4_rope_attn_factor, c.rope_yarn_log_mul,
                          dsv4_rope_n_ctx_orig, true);
      }

      const int visible =
          std::min(dsv4_completed_compressed_blocks(pos, kDsv4CsaRatio),
                   compressed_state->cache_capacity);
      std::vector<float> lid_scores((size_t)visible, 0.0f);
      const float scale =
          1.0f / std::sqrt((float)(indexer_dim * indexer_heads));
      for (int block = 0; block < visible; ++block) {
        const float *key =
            compressed_state->indexer_keys.data() + (size_t)block * indexer_dim;
        float score = 0.0f;
        for (int h = 0; h < indexer_heads; ++h) {
          const float *q =
              indexer_q.data() + ((size_t)s * indexer_heads + h) * indexer_dim;
          float dot = 0.0f;
          for (int d = 0; d < indexer_dim; ++d)
            dot += q[d] * key[d];
          score += std::max(0.0f, dot) *
                   indexer_proj[(size_t)s * indexer_heads + h] * scale;
        }
        lid_scores[(size_t)block] = score;
      }
      csa_selected[(size_t)s] =
          dsv4_top_k_indices(lid_scores, c.dsv4_indexer_top_k);
    }
  }

  std::vector<float> head_contexts((size_t)S * H * qh, 0.0f);
  // DeepSeek-V4's attention scale is always plain 1/sqrt(head_size), even
  // when CSA/HCA use compressed RoPE/YaRN for Q and K.
  const float attn_scale = 1.0f / std::sqrt((float)qh);
  auto raw_key = [&](int abs_pos) -> const float * {
    if (pos_base == 0 && abs_pos >= 0 && abs_pos < S)
      return raw_kv_prefill.data() + (size_t)abs_pos * qh;
    return engine.KVc[layer_idx] + (int64_t)(abs_pos % window) * qh;
  };

  // dsv4_score_gpu: try ONE joint GPU softmax over [raw window | HCA
  // compressed rows | sink] for the whole [H,qh] token at once, decode
  // (S==1) only -- ratio==4 (CSA) layers never take this path (see
  // dsv4_score_gpu's own comment, engine_config.hpp). On success this
  // fills head_contexts directly and the per-head loop below skips its
  // own CPU scoring (de-rope still runs unconditionally). On any failure
  // (mirror not ready, VRAM budget, cublas error) gpu_score_ok stays
  // false and every head falls through to the existing CPU path,
  // unchanged -- never a hard requirement, same discipline as every other
  // GPU fast-path in this file.
  bool gpu_score_ok = false;
  if (S == 1 && ratio != kDsv4CsaRatio && dsv4_score_gpu_enabled() &&
      engine.gpu) {
    const int current_pos0 = pos_base;
    const int past_context_len0 = std::min(current_pos0 + 1, window);
    const int oldest_pos0 = current_pos0 - past_context_len0 + 1;
    const int oldest_slot0 = oldest_pos0 % window;
    const int hca_visible0 = (compressed && compressed_state->is_hca())
                                 ? std::min(dsv4_completed_compressed_blocks(
                                                current_pos0, kDsv4HcaRatio),
                                            compressed_state->cache_capacity)
                                 : 0;
    gpu_score_ok = engine.gpu->dsv4_score_gpu_batched_host(
        layer_idx, Q.data(), H, qh, window, past_context_len0, oldest_slot0,
        hca_visible0, sinks, attn_scale, head_contexts.data());
    if (!gpu_score_ok) {
      perf().n_dsv4_score_gpu_fallback_cpu.fetch_add(1,
                                                     std::memory_order_relaxed);
    }
  }

// num_threads(attn_score_threads()): dedicated, possibly-smaller team
// size for this causal score loop specifically -- see EngineConfig::
// attn_score_threads' own comment (dense_dispatch.hpp) for why this is
// separate from the process-wide omp_max_threads default. Defaults to
// omp_get_max_threads() (the same value this pragma would have used
// anyway) when unset, so this is a genuine no-op for any config that
// doesn't set attn_score_threads.
#pragma omp parallel for collapse(2) schedule(dynamic, 1)                      \
    num_threads(attn_score_threads())
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < H; ++h) {
      int current_pos = pos_base + s;
      float *head_out_dest = head_contexts.data() + ((size_t)s * H + h) * qh;

      // gpu_score_ok (dsv4_score_gpu): head_out_dest is already filled
      // by dsv4_score_gpu_batched_host above -- skip the CPU scoring
      // below entirely, but still run the de-rope step, same as the
      // CPU path would.
      if (!gpu_score_ok) {
        // SWA window: the last `window` positions ending at current_pos
        // (inclusive), never reaching further back than position 0.
        int past_context_len = std::min(current_pos + 1, window);

        const float *head_q = Q.data() + ((size_t)s * H + h) * qh;
        const std::vector<int> *selected_csa =
            (compressed && compressed_state->is_csa())
                ? &csa_selected[(size_t)s]
                : nullptr;
        const int hca_visible = (compressed && compressed_state->is_hca())
                                    ? std::min(dsv4_completed_compressed_blocks(
                                                   current_pos, kDsv4HcaRatio),
                                               compressed_state->cache_capacity)
                                    : 0;
        const int compressed_count =
            selected_csa ? static_cast<int>(selected_csa->size()) : hca_visible;
        std::vector<float> attn_scores((size_t)past_context_len +
                                       compressed_count + (sinks ? 1 : 0));

        auto score_key = [head_q, qh, attn_scale](const float *key) {
          float score = 0.0f;
          __m512 v_score = _mm512_setzero_ps();
          for (int i = 0; i + 16 <= qh; i += 16) {
            v_score = _mm512_fmadd_ps(_mm512_loadu_ps(head_q + i),
                                      _mm512_loadu_ps(key + i), v_score);
          }
          score = _mm512_reduce_add_ps(v_score);
          for (int i = (qh / 16) * 16; i < qh; ++i)
            score += head_q[i] * key[i];
          return score * attn_scale;
        };

        for (int t = 0; t < past_context_len; ++t) {
          // t=0 is the OLDEST position still in-window
          // (current_pos - past_context_len + 1); ring slot for the
          // absolute position current_pos-past_context_len+1+t.
          int abs_pos = current_pos - past_context_len + 1 + t;
          const float *cached = raw_key(abs_pos);
          attn_scores[(size_t)t] = score_key(cached);
        }
        for (int i = 0; i < compressed_count; ++i) {
          const int block = selected_csa ? (*selected_csa)[(size_t)i] : i;
          const float *key = compressed_state->keys.data() + (size_t)block * qh;
          attn_scores[(size_t)past_context_len + i] = score_key(key);
        }
        // Attention sink: an extra always-present logit, no
        // corresponding value contribution -- contributes to the
        // softmax denominator only, damping every real weight.
        if (sinks)
          attn_scores[(size_t)past_context_len + compressed_count] = sinks[h];

        softmax_row(attn_scores.data(), (int)attn_scores.size());

        for (int t = 0; t < past_context_len; ++t) {
          int abs_pos = current_pos - past_context_len + 1 + t;
          const float *cached = raw_key(abs_pos);
          float w = attn_scores[t];
          for (int i = 0; i < qh; ++i)
            head_out_dest[i] += w * cached[i];
        }
        for (int i = 0; i < compressed_count; ++i) {
          const int block = selected_csa ? (*selected_csa)[(size_t)i] : i;
          const float *key = compressed_state->keys.data() + (size_t)block * qh;
          const float w = attn_scores[(size_t)past_context_len + i];
          for (int d = 0; d < qh; ++d)
            head_out_dest[d] += w * key[d];
        }
        // sink's own softmax mass has no value vector to blend in --
        // it just shrinks the real weights above, already applied.
      } // !gpu_score_ok

      // Match the Q/K RoPE configuration on the output's rope half.
      rope_yarn_inplace_inverse(head_out_dest + nope, rope, current_pos,
                                dsv4_rope_theta, dsv4_rope_ext_factor,
                                dsv4_rope_freq_scale, dsv4_rope_attn_factor,
                                c.rope_yarn_log_mul, dsv4_rope_n_ctx_orig,
                                /*dsv4_mscale=*/compressed);
    }
  }
  check_finite_attn("head_contexts (post MQA + de-rope)", layer_idx,
                    head_contexts.data(), head_contexts.size(), H * qh);
  if (engine.family_ops.family == ModelFamily::DeepSeek_V41)
    dsv41_trace::save("attn-ctx", layer_idx, pos_base,
                      head_contexts.data() + (size_t)(S - 1) * H * qh,
                      (size_t)H * qh);
  if (engine.family_ops.family == ModelFamily::DeepSeek_V41 && S > 1)
    dsv41_trace::save("attn-ctx-seq", layer_idx, pos_base,
                      head_contexts.data(), head_contexts.size());
  PERF_END(dsv4_score, t_attn_score);
  // The row_width=H*qh check above measures the norm of ALL H heads
  // concatenated for the last token -- NOT comparable to "KVc window max
  // |row|" above, which is a single qh-sized row. Concatenating H
  // independently-bounded heads can legitimately scale the combined norm
  // up to sqrt(H) times a single head's, even when every individual head
  // respects the softmax convex-combination bound (each head's own
  // weights sum to 1, so each head's own qh-sized output is bounded by
  // the max qh-sized cached row it blended) -- the same category of
  // mismatch already found in the Q per-head-RMSNorm trace above. This is
  // the genuine apples-to-apples check: max norm of any SINGLE head's
  // qh-sized segment for the last token, to test the per-head bound
  // directly against "KVc window max |row|".
  if (Logger::instance().would_log(LogLevel::VERBOSE)) {
    double max_sq = 0.0;
    int max_h = -1;
    const float *last_tok = head_contexts.data() + (size_t)(S - 1) * H * qh;
    for (int h = 0; h < H; ++h) {
      const float *seg = last_tok + (size_t)h * qh;
      double sq = 0.0;
      for (int i = 0; i < qh; ++i)
        sq += (double)seg[i] * seg[i];
      if (sq > max_sq) {
        max_sq = sq;
        max_h = h;
      }
    }
    LOG_STREAM_VERBOSE() << "[MLA-TRACE l=" << layer_idx
                         << "] head_contexts per-head max |head|="
                         << std::sqrt(max_sq) << " (h=" << max_h << "/" << H
                         << ")\n";
  }

  // ---- 5. grouped low-rank output projection: attn_output_a (per
  // group) -> concat -> attn_output_b (single dense matmul). ----
  PERF_START(dsv4_outproj);
  int o_groups = std::max(1, c.o_group_count);
  int o_rank = c.o_lora_rank;
  int group_in = (H * qh) / o_groups;

  auto woa_it = engine.resident_layers.find(p + "attn_output_a.weight");
  if (woa_it == engine.resident_layers.end() || !woa_it->second.host_ptr) {
    LOG_ERROR("[DSV4] layer %d missing attn_output_a -- zeroing attention\n",
              layer_idx);
    std::memset(out_context, 0, (size_t)S * D * sizeof(float));
    return;
  }
  // Dequantize once per layer, cached -- same pattern as V3's WUK/WUV
  // (attn_output_a is a static model weight, identical on every call for
  // this layer_idx). (b) layout interpretation -- see this function's
  // own header comment: treated as o_groups contiguous blocks, each
  // [o_rank, group_in] row-major (matmul()'s W[O,I] convention), block g
  // starting at g*o_rank*group_in.
  static std::vector<std::vector<float>> woa_cache;
  if ((int)woa_cache.size() <= layer_idx)
    woa_cache.resize(c.n_layers);
  std::vector<float> &WOA = woa_cache[layer_idx];
  if (WOA.empty()) {
    PERF_START(dsv4_outproj_dequant);
    WOA.resize((size_t)o_groups * o_rank * group_in);
    dequant_row(woa_it->second.meta.type, woa_it->second.host_ptr, WOA.data(),
                (int)WOA.size());
    check_finite_attn("WOA (attn_output_a dequant)", layer_idx, WOA.data(),
                      WOA.size());
    PERF_END(dsv4_outproj_dequant, t_attn_outproj_dequant);
  }
  // dsv4_outproj_gpu (config, opt-in -- see EngineConfig's own comment):
  // one-time persistent VRAM upload of this SAME dequantized WOA, keyed
  // by layer_idx like woa_cache above. nullptr entries are the common
  // case unless the config flag is on; upload is attempted at most once
  // per layer (guarded by the cache slot being nullptr, not by the flag
  // alone) and a failed upload just leaves the slot nullptr forever --
  // the S==1 dispatch below always checks for a real pointer before
  // trying the GPU path, so a partial/failed upload can never crash,
  // only fall back to the existing CPU path for that layer.
  static std::vector<float *> woa_dev_cache;
  if (dsv4_outproj_gpu_enabled() && engine.gpu) {
    if ((int)woa_dev_cache.size() <= layer_idx)
      woa_dev_cache.resize(c.n_layers, nullptr);
    if (!woa_dev_cache[layer_idx]) {
      woa_dev_cache[layer_idx] =
          engine.gpu->upload_persistent_f32(WOA.data(), WOA.size());
      if (woa_dev_cache[layer_idx]) {
        LOG_STREAM_INFO() << "[DSV4] layer=" << layer_idx << " attn_output_a: "
                          << (WOA.size() * sizeof(float)) / 1e6
                          << " MB uploaded to VRAM (dsv4_outproj_gpu)\n";
      } else {
        LOG_STREAM_WARN()
            << "[DSV4] layer=" << layer_idx
            << " attn_output_a: VRAM upload "
               "failed -- falling back to CPU outproj for this layer\n";
      }
    }
  }

  // S==1 (decode -- the overwhelmingly common case, ~40 of every 41 calls
  // per layer across a real generation): ONE combined OMP parallel region
  // over every (group, output) pair, instead of o_groups SEPARATE
  // matmul() calls each spawning its own thread team. A first attempt
  // here went fully serial (dot_f32_avx512, no OMP at all) on the
  // assumption that per-call work was too small to be worth threading --
  // that was WRONG, confirmed on real hardware: outproj went from 11.76s
  // to 24.44s (more than DOUBLED), meaning the real per-call compute
  // (group_in x o_rank FMAs, times o_groups) genuinely benefits from this
  // machine's 6 cores/12 threads, and losing that parallelism cost far
  // more than the thread-spawn overhead ever did. The actual fix is
  // narrower: keep the parallelism, cut the NUMBER of times a thread team
  // gets spun up per token from o_groups down to 1, by flattening the
  // (group, output) index space into a single "#pragma omp parallel for"
  // instead of nesting o_groups separate ones. Each iteration computes
  // the exact same dot product matmul() would (dot_f32_avx512 is
  // matmul()'s own per-output-row kernel), so results are identical to
  // both the original matmul()-per-group version and the (reverted)
  // serial one -- only the parallelization GRANULARITY changed.
  std::vector<float> grouped_out((size_t)S * o_groups * o_rank);
  if (S == 1) {
    const float *in_row = head_contexts.data();
    float *out_flat = grouped_out.data();
    // GPU path (dsv4_outproj_gpu, opt-in): only taken when this layer's
    // weight actually made it to VRAM (woa_dev_cache entry non-null) --
    // see cuda_manager.hpp's grouped_output_proj_host for the full
    // reasoning (memory-bandwidth-bound on CPU, ~12x faster on GPU once
    // resident, verified numerically matching in a standalone
    // benchmark). Falls straight through to the CPU path below on
    // ANY failure (grouped_output_proj_host returns false) -- never a
    // hard failure, matching this codebase's standing GPU-dispatch
    // discipline (dense_dispatch.hpp's dense_proj does the same).
    bool gpu_ok = false;
    // V4.1 graph callbacks share the original quantized dense pool. The
    // grouped Q2_K kernel follows dot_f32_avx512's exact FMA/reduction order,
    // so raw-SWA layers can use it too without a second dequantized weight
    // allocation or a routing change downstream.
    if (engine.family_ops.family == ModelFamily::DeepSeek_V41 &&
        engine.gpu && woa_it->second.device_ptr &&
        woa_it->second.meta.type == GGUFType::Q2_K) {
      const size_t group_stride =
          (size_t)row_bytes(GGUFType::Q2_K, group_in) * o_rank;
      gpu_ok = engine.gpu->grouped_output_proj_q2k_host(
          woa_it->second.device_ptr, group_stride, o_groups, o_rank,
          group_in, in_row, out_flat);
    }
    if (!gpu_ok && dsv4_outproj_gpu_enabled() && engine.gpu &&
        (int)woa_dev_cache.size() > layer_idx && woa_dev_cache[layer_idx]) {
      gpu_ok = engine.gpu->grouped_output_proj_host(woa_dev_cache[layer_idx],
                                                    o_groups, o_rank, group_in,
                                                    in_row, out_flat);
    }
    if (!gpu_ok) {
      const float *woa_data = WOA.data();
#pragma omp parallel for schedule(static)
      for (int go = 0; go < o_groups * o_rank; ++go) {
        int g = go / o_rank;
        int o = go % o_rank;
        const float *in_group = in_row + (size_t)g * group_in;
        const float *w_row =
            woa_data + (size_t)g * o_rank * group_in + (size_t)o * group_in;
        out_flat[go] = dot_f32_avx512(in_group, w_row, group_in);
      }
    }
  } else {
    bool gpu_ok = false;
    if (engine.family_ops.family == ModelFamily::DeepSeek_V41 && engine.gpu &&
        woa_it->second.device_ptr && woa_it->second.meta.type == GGUFType::Q2_K) {
      gpu_ok = engine.gpu->grouped_output_proj_q2k_host(
          woa_it->second.device_ptr,
          (size_t)row_bytes(GGUFType::Q2_K, group_in) * o_rank,
          o_groups, o_rank, group_in, head_contexts.data(), grouped_out.data(), S);
    }
    for (int s = 0; !gpu_ok && s < S; ++s) {
      const float *in_row = head_contexts.data() + (size_t)s * H * qh;
      for (int g = 0; g < o_groups; ++g) {
        const float *in_group = in_row + (size_t)g * group_in;
        float *out_group =
            grouped_out.data() + ((size_t)s * o_groups + g) * o_rank;
        matmul(out_group, in_group, WOA.data() + (size_t)g * o_rank * group_in,
               1, group_in, o_rank);
      }
    }
  }
  check_finite_attn("grouped_out (post attn_output_a)", layer_idx,
                    grouped_out.data(), grouped_out.size(), o_groups * o_rank);

  dense_proj(engine, p + "attn_output_b.weight", grouped_out.data(),
             out_context, S, o_groups * o_rank, D);
  check_finite_attn("out_context (post attn_output_b projection)", layer_idx,
                    out_context, (size_t)S * D, D);
  PERF_END(dsv4_outproj, t_attn_outproj);
}

namespace {

bool dsv41_load_f32(const StreamingEngine &engine, const std::string &name,
                    size_t expected, std::vector<float> &out) {
  auto it = engine.resident_layers.find(name);
  if (it == engine.resident_layers.end() || !it->second.host_ptr)
    return false;
  uint64_t n = 1;
  for (uint64_t dim : it->second.meta.shape)
    n *= dim;
  if (n != expected)
    return false;
  out.resize(expected);
  if (it->second.meta.type == GGUFType::F32)
    std::memcpy(out.data(), it->second.host_ptr, expected * sizeof(float));
  else
    dequant_row(it->second.meta.type, it->second.host_ptr, out.data(),
                (int)expected);
  return true;
}

} // namespace

// TEMPORARY DIAGNOSTIC (2026-09-12) -- phase-by-phase bisection of
// dsv41_attention_gguf's ratio!=0 branch, following on from the
// PERF_SCOPE(t_attn)/g_diag_attn_call_ms accounting fix above (that fix
// proved the whole function costs ~20-21s/request; these accumulators find
// WHICH part of it does). Same style as g_diag_attn_call_ms/
// g_diag_layer_loop_total_ms (execution_layer.cpp): plain static double
// wall-clock accumulators, single-writer (attention is called from the one
// generation thread; the only parallel region in this function, the
// per-head score loop, is deliberately measured as one bracket around the
// whole `#pragma omp parallel for`, not per-thread), reset once per request
// from generate_text (generation_loop.hpp), read via the diag_get_dsv41_*
// getters right before print_summary("REQUEST", ...). *_calls counts are
// tracked alongside the two conditional phases (owns_kv/owns_index) so a
// low per-request total isn't misread as "cheap per call" when it's
// actually "rare, but expensive when it happens" (see caller comment).
// Left in place after this investigation for a later cleanup/productionize
// pass -- do not remove.
static double g_diag_dsv41_calls = 0.0;
static double g_diag_dsv41_qrope_ms = 0.0;        // phase (a)
static double g_diag_dsv41_kvrope_ms = 0.0;       // phase (b)
static double g_diag_dsv41_ownskv_ms = 0.0;       // phase (c), owns_kv only
static double g_diag_dsv41_ownskv_calls = 0.0;
static double g_diag_dsv41_indexer_qk_ms = 0.0;   // phase (d), owns_index only
static double g_diag_dsv41_indexer_score_ms = 0.0; // phase (e), owns_index only
static double g_diag_dsv41_ownsindex_calls = 0.0;
static double g_diag_dsv41_perhead_ms = 0.0;      // phase (f)
static double g_diag_dsv41_outproj_ms = 0.0;      // phase (g)
static double g_diag_dsv41_woa_dequant_ms = 0.0;  // subset of (g): one-time
static double g_diag_dsv41_woa_dequant_calls = 0.0;
void diag_reset_dsv41_attn_phases() {
  g_diag_dsv41_calls = 0.0;
  g_diag_dsv41_qrope_ms = 0.0;
  g_diag_dsv41_kvrope_ms = 0.0;
  g_diag_dsv41_ownskv_ms = 0.0;
  g_diag_dsv41_ownskv_calls = 0.0;
  g_diag_dsv41_indexer_qk_ms = 0.0;
  g_diag_dsv41_indexer_score_ms = 0.0;
  g_diag_dsv41_ownsindex_calls = 0.0;
  g_diag_dsv41_perhead_ms = 0.0;
  g_diag_dsv41_outproj_ms = 0.0;
  g_diag_dsv41_woa_dequant_ms = 0.0;
  g_diag_dsv41_woa_dequant_calls = 0.0;
}
double diag_get_dsv41_calls() { return g_diag_dsv41_calls; }
double diag_get_dsv41_qrope_ms() { return g_diag_dsv41_qrope_ms; }
double diag_get_dsv41_kvrope_ms() { return g_diag_dsv41_kvrope_ms; }
double diag_get_dsv41_ownskv_ms() { return g_diag_dsv41_ownskv_ms; }
double diag_get_dsv41_ownskv_calls() { return g_diag_dsv41_ownskv_calls; }
double diag_get_dsv41_indexer_qk_ms() { return g_diag_dsv41_indexer_qk_ms; }
double diag_get_dsv41_indexer_score_ms() { return g_diag_dsv41_indexer_score_ms; }
double diag_get_dsv41_ownsindex_calls() { return g_diag_dsv41_ownsindex_calls; }
double diag_get_dsv41_perhead_ms() { return g_diag_dsv41_perhead_ms; }
double diag_get_dsv41_outproj_ms() { return g_diag_dsv41_outproj_ms; }
double diag_get_dsv41_woa_dequant_ms() { return g_diag_dsv41_woa_dequant_ms; }
double diag_get_dsv41_woa_dequant_calls() { return g_diag_dsv41_woa_dequant_calls; }

// out[0..qh) += sum_i w[i] * keys[i][0..qh). Accumulators stay in registers
// across all keys (the scalar `out[d] += w*key[d]` loop re-reads and re-writes
// `out` per key and measured ~10x slower than the dot-product pass over the
// same keys). Sum order per dimension is unchanged (keys in order).
static void dsv41_weighted_sum_avx512(float *out, const float *w,
                                      const float *const *keys, size_t n,
                                      int qh) {
  int d = 0;
  for (; d + 64 <= qh; d += 64) {
    __m512 a0 = _mm512_loadu_ps(out + d), a1 = _mm512_loadu_ps(out + d + 16),
           a2 = _mm512_loadu_ps(out + d + 32), a3 = _mm512_loadu_ps(out + d + 48);
    for (size_t i = 0; i < n; ++i) {
      const __m512 wv = _mm512_set1_ps(w[i]);
      const float *k = keys[i] + d;
      a0 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(k), a0);
      a1 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(k + 16), a1);
      a2 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(k + 32), a2);
      a3 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(k + 48), a3);
    }
    _mm512_storeu_ps(out + d, a0);
    _mm512_storeu_ps(out + d + 16, a1);
    _mm512_storeu_ps(out + d + 32, a2);
    _mm512_storeu_ps(out + d + 48, a3);
  }
  for (; d < qh; ++d)
    for (size_t i = 0; i < n; ++i) out[d] += w[i] * keys[i][d];
}

// DeepSeek-V4.1 attention. Ratio-zero layers are delegated to the already
// verified V4 raw-SWA implementation. Compressed layers use V4.1's own
// ratio-1/2 pooling and shared source-layer caches; its indexer is the direct
// wk/k_norm path from the pooled latent (V4's overlap compressor is absent).
static void dsv41_attention_impl(StreamingEngine &engine, int layer_idx,
                          const float *hidden_states, int S, int pos_base,
                          float *out_context, const float *projected_normed,
                          const float *projected_ql, const float *projected_q,
                          const float *projected_kv,
                          bool context_only) {
  const auto &c = engine.model_file.architecture_config;
  const int D = c.hidden;
  const int ratio =
      (layer_idx >= 0 && layer_idx < (int)c.dsv4_compress_ratios.size())
          ? (int)c.dsv4_compress_ratios[(size_t)layer_idx]
          : 0;
  if (layer_idx == 0 && pos_base == 0) {
    for (auto &state : engine.dsv41_compressed)
      state.reset();
    engine.dsv41_candidate_positions.clear();
  }
  if (ratio == 0) {
    dsv4_attention_gguf(engine, layer_idx, hidden_states, S, pos_base,
                        out_context);
    dsv41_trace::save("attn-out-projected", layer_idx, pos_base,
                      out_context + (size_t)(S - 1) * D, D);
    return;
  }

  // FOUND THIS SESSION (deepseek-v41-decode-speed-session): this whole
  // ratio!=0 branch -- the DSA/compressed-KV sliding-window path, the one
  // every real DeepSeek-V4.1 layer transition actually takes -- had ZERO
  // PERF_SCOPE/PERF_START coverage. Only the ratio==0 fallback above
  // (dsv4_attention_gguf) was timed. Measured via a direct wall-clock
  // wrapper around this exact call vs. what t_attn reported for the same
  // call: attn_call_wall=20.03s vs t_attn=0.53s for one 37-token request --
  // a ~19.5s/request gap that alone explained essentially the ENTIRE
  // previously-mysterious "[REQUEST] wall vs named buckets" gap (~19-21s)
  // this whole investigation was chasing. This was a pure accounting bug:
  // the work was always happening, "attn=" in every [REQUEST] line to date
  // was simply not counting it. Fixed here; see the per-head loop below
  // for the REAL performance fix (OMP parallelization) found alongside it.
  PERF_SCOPE(t_attn);
  g_diag_dsv41_calls += 1.0;

  const int H = c.n_heads;
  const int nope = c.qk_nope;
  const int rope = c.qk_rope;
  const int qh = nope + rope;
  const int q_lora =
      engine.model_file.get_config_int("attention.q_lora_rank", 2048);
  const int window =
      engine.sliding_window > 0 ? engine.sliding_window : engine.max_t;
  const std::string p = "blk." + std::to_string(layer_idx) + ".";
  if (S == 1)
    dsv41_trace::save("attn-input", layer_idx, pos_base, hidden_states, D);
  const float rope_theta =
      engine.model_file.get_config_float("rope.freq_base", 10000.0f);
  const float rope_theta_v41 = c.dsv4_compress_rope_freq_base;

  if (layer_idx >= (int)engine.dsv41_kv_source_for_layer.size() ||
      layer_idx >= (int)engine.dsv41_index_source_for_layer.size()) {
    LOG_ERROR("[DSV41] layer %d has no source map\n", layer_idx);
    std::memset(out_context, 0, (size_t)S * D * sizeof(float));
    return;
  }
  const int kv_source = engine.dsv41_kv_source_for_layer[(size_t)layer_idx];
  const int index_source =
      engine.dsv41_index_source_for_layer[(size_t)layer_idx];
  if (kv_source < 0 || kv_source >= (int)engine.dsv41_compressed.size() ||
      !engine.dsv41_compressed[(size_t)kv_source].enabled()) {
    LOG_ERROR("[DSV41] layer %d has no usable compressed KV source\n",
              layer_idx);
    std::memset(out_context, 0, (size_t)S * D * sizeof(float));
    return;
  }
  Dsv41CompressedLayerState &state = engine.dsv41_compressed[(size_t)kv_source];
  const bool owns_kv = kv_source == layer_idx;
  const bool owns_index = index_source == layer_idx;

  // TEMPORARY DIAGNOSTIC (2026-09-12): phase (a) -- rmsnorm + Q-lora
  // projection/norm + Q projection + per-head RoPE.
  const auto _dsv41_t0 = std::chrono::high_resolution_clock::now();
  std::vector<float> normed((size_t)S * D);
  std::vector<float> ql((size_t)S * q_lora);
  std::vector<float> Q((size_t)S * H * qh);
  if (projected_normed) {
    std::memcpy(normed.data(), projected_normed, normed.size() * sizeof(float));
    std::memcpy(ql.data(), projected_ql, ql.size() * sizeof(float));
    std::memcpy(Q.data(), projected_q, Q.size() * sizeof(float));
  } else {
    rmsnorm_proj(engine, p + "attn_norm.weight", hidden_states, normed.data(), S,
                 D, c.eps);
    dense_proj(engine, p + "attn_q_a.weight", normed.data(), ql.data(), S, D,
               q_lora);
    rmsnorm_proj(engine, p + "attn_q_a_norm.weight", ql.data(), ql.data(), S,
                 q_lora, c.eps);
    dense_proj(engine, p + "attn_q_b.weight", ql.data(), Q.data(), S, q_lora,
               H * qh);
  }
  dsv41_trace::save("attn-norm", layer_idx, pos_base,
                    normed.data() + (size_t)(S - 1) * D, D);
  dsv41_trace::save("q-lora", layer_idx, pos_base,
                    ql.data() + (size_t)(S - 1) * q_lora, q_lora);
  dsv41_trace::save("query", layer_idx, pos_base,
                    Q.data() + (size_t)(S - 1) * H * qh, (size_t)H * qh);
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < H; ++h) {
      float *qrow = Q.data() + ((size_t)s * H + h) * qh;
      rope_yarn_inplace(qrow + nope, rope, pos_base + s, rope_theta_v41,
                        c.rope_ext_factor, c.rope_freq_scale,
                        c.rope_attn_factor, c.rope_yarn_log_mul,
                        c.rope_n_ctx_orig, true);
    }
  }
  g_diag_dsv41_qrope_ms += std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - _dsv41_t0).count();

  // TEMPORARY DIAGNOSTIC (2026-09-12): phase (b) -- raw sliding-window KV
  // projection/norm + RoPE + KV-cache memcpy. Always belongs to the current
  // layer (unlike compressed KV, which is only published by owns_kv).
  const auto _dsv41_t1 = std::chrono::high_resolution_clock::now();
  std::vector<float> kv((size_t)S * qh);
  if (projected_normed) {
    std::memcpy(kv.data(), projected_kv, kv.size() * sizeof(float));
  } else {
    dense_proj(engine, p + "attn_kv.weight", normed.data(), kv.data(), S, D, qh);
    rmsnorm_proj(engine, p + "attn_kv_a_norm.weight", kv.data(), kv.data(), S, qh,
                 c.eps);
  }
  dsv41_trace::save("raw-kv", layer_idx, pos_base,
                    kv.data() + (size_t)(S - 1) * qh, qh);
  for (int s = 0; s < S; ++s) {
    const int pos = pos_base + s;
    rope_yarn_inplace(kv.data() + (size_t)s * qh + nope, rope, pos,
                      rope_theta_v41, c.rope_ext_factor, c.rope_freq_scale,
                      c.rope_attn_factor, c.rope_yarn_log_mul,
                      c.rope_n_ctx_orig, true);
    const int slot = pos % window;
    std::memcpy(engine.KVc[(size_t)layer_idx] + (size_t)slot * qh,
                kv.data() + (size_t)s * qh, (size_t)qh * sizeof(float));
  }
  g_diag_dsv41_kvrope_ms += std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - _dsv41_t1).count();

  // The source layer publishes pooled compressed KV.  The source's indexer
  // consumes the same latent before compressed RoPE is applied.
  // TEMPORARY DIAGNOSTIC (2026-09-12): phase (c), owns_kv only -- see
  // g_diag_dsv41_ownskv_ms's own comment above the function. Wraps the
  // WHOLE owns_kv block (comp_kv/comp_gate projections, pooling math when
  // ratio>1, indexer-key publish, compressed-KV cache write) as one bracket
  // rather than sub-phases, since the task only calls out (c) as a single
  // conditional phase.
  const auto _dsv41_t2 = std::chrono::high_resolution_clock::now();
  if (owns_kv) {
    g_diag_dsv41_ownskv_calls += 1.0;
    std::vector<float> comp_kv((size_t)S * qh);
    dense_proj(engine, p + "attn_compressor_kv.weight", normed.data(),
               comp_kv.data(), S, D, qh);
    std::vector<float> comp_gate;
    if (ratio > 1) {
      comp_gate.resize((size_t)S * qh);
      dense_proj(engine, p + "attn_compressor_gate.weight", normed.data(),
                 comp_gate.data(), S, D, qh);
    }
    std::vector<float> comp_norm;
    if (!dsv41_load_f32(engine, p + "attn_compressor_norm.weight", qh,
                        comp_norm)) {
      LOG_ERROR("[DSV41] layer %d missing attn_compressor_norm\n", layer_idx);
      std::memset(out_context, 0, (size_t)S * D * sizeof(float));
      return;
    }
    std::vector<float> latent(qh);
    for (int s = 0; s < S; ++s) {
      const int pos = pos_base + s;
      const float *src_kv = comp_kv.data() + (size_t)s * qh;
      if (ratio == 1) {
        std::memcpy(latent.data(), src_kv, (size_t)qh * sizeof(float));
        rmsnorm_row(latent.data(), latent.data(), comp_norm.data(), qh, c.eps);
      } else {
        const int slot = pos % ratio;
        std::memcpy(state.pending_kv.data() + (size_t)slot * qh, src_kv,
                    (size_t)qh * sizeof(float));
        std::memcpy(state.pending_score.data() + (size_t)slot * qh,
                    comp_gate.data() + (size_t)s * qh,
                    (size_t)qh * sizeof(float));
        state.pending_count = std::max(state.pending_count, slot + 1);
        if ((pos + 1) % ratio != 0)
          continue;
        for (int d = 0; d < qh; ++d) {
          float mx = -std::numeric_limits<float>::infinity();
          for (int t = 0; t < ratio; ++t)
            mx = std::max(mx, state.pending_score[(size_t)t * qh + d]);
          float denom = 0.0f, value = 0.0f;
          for (int t = 0; t < ratio; ++t) {
            const float w =
                std::exp(state.pending_score[(size_t)t * qh + d] - mx);
            denom += w;
            value += w * state.pending_kv[(size_t)t * qh + d];
          }
          latent[(size_t)d] = denom > 0.0f ? value / denom : 0.0f;
        }
        rmsnorm_row(latent.data(), latent.data(), comp_norm.data(), qh, c.eps);
      }
      const int block = pos / ratio;
      if (block < 0 || block >= state.cache_capacity)
        continue;
      // The indexer's wk/k_norm consumes the RoPE-free latent. Only
      // after publishing that side cache may the attention copy be
      // rotated and stored in the compressed KV cache.
      if (state.indexer_key_dim > 0) {
        std::vector<float> ik(c.dsv4_indexer_key_length);
        dense_proj(engine, p + "indexer.attn_k.weight", latent.data(),
                   ik.data(), 1, qh, c.dsv4_indexer_key_length);
        rmsnorm_proj(engine, p + "indexer.k_norm.weight", ik.data(), ik.data(),
                     1, c.dsv4_indexer_key_length, c.eps);
        // Only the indexer's RoPE tail is rotated; the key cache is
        // otherwise the normalized projection of the unrotated latent.
        rope_yarn_inplace(ik.data() + c.dsv4_indexer_key_length - rope, rope,
                          pos + 1 - ratio, rope_theta_v41, c.rope_ext_factor,
                          c.rope_freq_scale, c.rope_attn_factor,
                          c.rope_yarn_log_mul, c.rope_n_ctx_orig, true);
        std::memcpy(state.indexer_keys.data() +
                        (size_t)block * c.dsv4_indexer_key_length,
                    ik.data(),
                    (size_t)c.dsv4_indexer_key_length * sizeof(float));
      }
      rope_yarn_inplace(latent.data() + nope, rope, pos + 1 - ratio,
                        rope_theta_v41, c.rope_ext_factor, c.rope_freq_scale,
                        c.rope_attn_factor, c.rope_yarn_log_mul,
                        c.rope_n_ctx_orig, true);
      std::memcpy(state.keys.data() + (size_t)block * qh, latent.data(),
                  (size_t)qh * sizeof(float));
    }
  }
  g_diag_dsv41_ownskv_ms += std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - _dsv41_t2).count();

  // Index source layers publish the compressed positions used by all
  // following consumers until the next source. The compact CPU top-k is
  // deterministic and mirrors llama.cpp's sort-by-position result.
  // TEMPORARY DIAGNOSTIC (2026-09-12): phases (d)+(e), owns_index only --
  // see g_diag_dsv41_indexer_qk_ms/g_diag_dsv41_indexer_score_ms's own
  // comment above the function. (d) covers the iq/iw dense_proj calls plus
  // each token's per-head RoPE application on iq; (e) covers the per-block
  // dot-product scoring loop plus dsv4_top_k_indices. Both phases share one
  // outer serial `for (int s...)` loop (no OMP here, unlike the per-head
  // loop below), so they are timed with brackets INSIDE that loop rather
  // than around the whole block.
  if (owns_index) {
    g_diag_dsv41_ownsindex_calls += 1.0;
    if (state.indexer_keys.empty()) {
      LOG_ERROR("[DSV41] layer %d has no indexer key cache\n", layer_idx);
      std::memset(out_context, 0, (size_t)S * D * sizeof(float));
      return;
    }
    std::vector<float> iq((size_t)S * c.dsv4_indexer_head_count *
                          c.dsv4_indexer_key_length);
    std::vector<float> iw((size_t)S * c.dsv4_indexer_head_count);
    const auto _dsv41_t3 = std::chrono::high_resolution_clock::now();
    dense_proj(engine, p + "indexer.attn_q_b.weight", ql.data(), iq.data(), S,
               q_lora, c.dsv4_indexer_head_count * c.dsv4_indexer_key_length);
    dense_proj(engine, p + "indexer.proj.weight", normed.data(), iw.data(), S,
               D, c.dsv4_indexer_head_count);
    g_diag_dsv41_indexer_qk_ms += std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - _dsv41_t3).count();
    for (int s = 0; s < S; ++s) {
      const int pos = pos_base + s;
      const int visible = std::min(dsv4_completed_compressed_blocks(pos, ratio),
                                   state.cache_capacity);
      const auto _dsv41_t4 = std::chrono::high_resolution_clock::now();
      for (int h = 0; h < c.dsv4_indexer_head_count; ++h) {
        rope_yarn_inplace(iq.data() +
                              ((size_t)s * c.dsv4_indexer_head_count + h) *
                                  c.dsv4_indexer_key_length +
                              c.dsv4_indexer_key_length - rope,
                          rope, pos, rope_theta_v41, c.rope_ext_factor,
                          c.rope_freq_scale, c.rope_attn_factor,
                          c.rope_yarn_log_mul, c.rope_n_ctx_orig, true);
      }
      g_diag_dsv41_indexer_qk_ms += std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - _dsv41_t4).count();
      const auto _dsv41_t5 = std::chrono::high_resolution_clock::now();
      std::vector<float> scores((size_t)visible, 0.0f);
      const float scale = 1.0f / std::sqrt((float)(c.dsv4_indexer_key_length *
                                                   c.dsv4_indexer_head_count));
      // Blocks are independent and each keeps its own serial accumulation
      // order, so the result is bit-identical to the serial loop (matters
      // for near-tie top-k). Only worth a team for long prompts.
      #pragma omp parallel for schedule(static) num_threads(attn_score_threads()) if (visible >= 64)
      for (int block = 0; block < visible; ++block) {
        const float *key = state.indexer_keys.data() +
                           (size_t)block * c.dsv4_indexer_key_length;
        float score = 0.0f;
        for (int h = 0; h < c.dsv4_indexer_head_count; ++h) {
          const float *q =
              iq.data() + ((size_t)s * c.dsv4_indexer_head_count + h) *
                              c.dsv4_indexer_key_length;
          float dot = 0.0f;
          for (int d = 0; d < c.dsv4_indexer_key_length; ++d)
            dot += q[d] * key[d];
          score += std::max(0.0f, dot) *
                   iw[(size_t)s * c.dsv4_indexer_head_count + h] * scale;
        }
        scores[(size_t)block] = score;
      }
      if (engine.dsv41_candidate_positions.size() != (size_t)S)
        engine.dsv41_candidate_positions.resize((size_t)S);
      engine.dsv41_candidate_positions[(size_t)s] =
          dsv4_top_k_indices(scores, c.dsv4_indexer_top_k);
      g_diag_dsv41_indexer_score_ms += std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - _dsv41_t5).count();
    }
  }

  if (engine.dsv41_candidate_positions.size() != (size_t)S) {
    LOG_ERROR("[DSV41] layer %d has no shared indexer top-k result\n",
              layer_idx);
    std::memset(out_context, 0, (size_t)S * D * sizeof(float));
    return;
  }
  if (dsv41_trace::enabled(layer_idx, pos_base + S - 1)) {
    const auto &selected = engine.dsv41_candidate_positions[(size_t)(S - 1)];
    std::vector<float> ids(selected.size());
    std::vector<float> keys(selected.size() * (size_t)qh);
    for (size_t i = 0; i < selected.size(); ++i) {
      ids[i] = static_cast<float>(selected[i]);
      if (selected[i] >= 0 && selected[i] < state.cache_capacity)
        std::memcpy(keys.data() + i * (size_t)qh,
                    state.keys.data() + (size_t)selected[i] * qh,
                    (size_t)qh * sizeof(float));
    }
    dsv41_trace::save("candidate-blocks", layer_idx, pos_base + S - 1,
                      ids.data(), ids.size());
    dsv41_trace::save("candidate-keys", layer_idx, pos_base + S - 1,
                      keys.data(), keys.size());
  }

  const auto sinks_it = engine.resident_layers.find(p + "attn_sinks.weight");
  const float *sinks =
      sinks_it != engine.resident_layers.end() && sinks_it->second.host_ptr
          ? reinterpret_cast<const float *>(sinks_it->second.host_ptr)
          : nullptr;
  std::vector<float> contexts((size_t)S * H * qh, 0.0f);
  const float attn_scale = 1.0f / std::sqrt((float)qh);
  auto raw_key = [&](int abs_pos) -> const float * {
    if (pos_base == 0 && abs_pos >= 0 && abs_pos < S)
      return kv.data() + (size_t)abs_pos * qh;
    return engine.KVc[(size_t)layer_idx] + (int64_t)(abs_pos % window) * qh;
  };
  // TEMPORARY DIAGNOSTIC (2026-09-12): phase (f) -- the already-OMP-
  // parallelized per-head score/softmax/weighted-sum loop. One bracket
  // around the whole `for (s)` (which itself contains the `#pragma omp
  // parallel for h`) rather than per-thread, since this is a single-
  // writer wall-clock accumulator.
  const auto _dsv41_t6 = std::chrono::high_resolution_clock::now();
  for (int s = 0; s < S; ++s) {
    const int pos = pos_base + s;
    const int past = std::min(pos + 1, window);
    const std::vector<int> &selected =
        engine.dsv41_candidate_positions[(size_t)s];
    // Parallelized across heads (found + fixed same session as the
    // PERF_SCOPE(t_attn) fix above): this loop was fully serial CPU,
    // walking `past + selected.size()` (up to dsv4_indexer_top_k=512)
    // AVX512 dot products per head with no OMP -- the single largest
    // component of the ~20s/request cost this whole investigation
    // chased. Every iteration is independent: `head_q`/`raw_key`/
    // `compressed_keys`/`sinks` are read-only shared data, `scores` is a
    // fresh per-iteration local (each thread gets its own, no sharing),
    // and `out` writes to a disjoint `contexts` slice per (s,h) -- no
    // restructuring needed, matches CLAUDE.md's own documented idiom
    // ("Parallelize across heads/experts in single outer #pragma omp
    // parallel for loops"). Verified byte-identical single-token output
    // before/after at temperature=0 across the standard 4-prompt test.
    // Dedicated team (see attn_score_threads) with one scores buffer per
    // thread: at decode context lengths the per-head work is microseconds,
    // so spawn/barrier cost against the prefetch/IO pools dominated.
    #pragma omp parallel num_threads(attn_score_threads())
    {
    std::vector<float> scores;
    std::vector<const float *> key_ptrs;
    #pragma omp for schedule(static)
    for (int h = 0; h < H; ++h) {
      const float *head_q = Q.data() + ((size_t)s * H + h) * qh;
      scores.assign((size_t)past + selected.size() + 1, 0.0f);
      for (int t = 0; t < past; ++t) {
        const int abs_pos = pos - past + 1 + t;
        const float *key = raw_key(abs_pos);
        scores[(size_t)t] = dot_f32_avx512(head_q, key, qh) * attn_scale;
      }
      const float *compressed_keys = state.keys.data();
      for (size_t i = 0; i < selected.size(); ++i) {
        const int block = selected[i];
        scores[(size_t)past + i] =
            dot_f32_avx512(head_q, compressed_keys + (size_t)block * qh, qh) *
            attn_scale;
      }
      scores.back() = sinks ? sinks[h] : 0.0f;
      softmax_row(scores.data(), (int)scores.size());
      float *out = contexts.data() + ((size_t)s * H + h) * qh;
      key_ptrs.resize((size_t)past + selected.size());
      for (int t = 0; t < past; ++t)
        key_ptrs[(size_t)t] = raw_key(pos - past + 1 + t);
      for (size_t i = 0; i < selected.size(); ++i)
        key_ptrs[(size_t)past + i] = compressed_keys + (size_t)selected[i] * qh;
      dsv41_weighted_sum_avx512(out, scores.data(), key_ptrs.data(),
                                key_ptrs.size(), qh);
      rope_yarn_inplace_inverse(out + nope, rope, pos, rope_theta_v41,
                                c.rope_ext_factor, c.rope_freq_scale,
                                c.rope_attn_factor, c.rope_yarn_log_mul,
                                c.rope_n_ctx_orig, true);
    }
    }
  }
  g_diag_dsv41_perhead_ms += std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - _dsv41_t6).count();
  dsv41_trace::save("attn-ctx", layer_idx, pos_base,
                    contexts.data() + (size_t)(S - 1) * H * qh,
                    (size_t)H * qh);
  if (context_only) {
    std::memcpy(out_context, contexts.data(), contexts.size() * sizeof(float));
    return;
  }

  // TEMPORARY DIAGNOSTIC (2026-09-12): phase (g) -- output projection
  // (woa_cache dequant on first use + per-group dot_f32_avx512 loop +
  // attn_output_b dense_proj), plus a separate sub-counter isolating just
  // the one-time woa_cache dequant_row cost (the cache is a function-local
  // static keyed by file path + tensor name, so it persists for the whole
  // process lifetime -- expect near-zero cost here during decode if
  // prefill already warmed every layer's entry, and investigate this
  // sub-counter specifically if it is NOT near-zero).
  const auto _dsv41_t7 = std::chrono::high_resolution_clock::now();
  const int groups = std::max(1, c.o_group_count);
  const int rank = c.o_lora_rank;
  const int group_in = (H * qh) / groups;
  std::vector<float> grouped((size_t)S * groups * rank);
  bool grouped_gpu = false;
  const auto output_a = engine.resident_layers.find(p + "attn_output_a.weight");
  if (output_a == engine.resident_layers.end() || !output_a->second.host_ptr) {
    LOG_ERROR("[DSV41] layer %d missing attn_output_a\n", layer_idx);
    std::memset(out_context, 0, (size_t)S * D * sizeof(float));
    return;
  }
  if (engine.gpu && output_a->second.device_ptr &&
      output_a->second.meta.type == GGUFType::Q2_K) {
    grouped_gpu = engine.gpu->grouped_output_proj_q2k_host(
        output_a->second.device_ptr,
        (size_t)row_bytes(GGUFType::Q2_K, group_in) * rank,
        groups, rank, group_in, contexts.data(), grouped.data(), S);
  }
  // Only the CPU fallback needs a dequantized weight mirror. The CUDA path
  // reads the original persistent Q2_K tensor and matches this dot exactly.
  if (!grouped_gpu) {
    const size_t woa_n = (size_t)groups * rank * group_in;
    static std::unordered_map<std::string, std::vector<float>> woa_cache;
    auto &WOA = woa_cache[engine.model_file.resolved_file_path + "#" + p +
                          "attn_output_a.weight"];
    if (WOA.empty()) {
      const auto dequant_start = std::chrono::high_resolution_clock::now();
      WOA.resize(woa_n);
      dequant_row(output_a->second.meta.type, output_a->second.host_ptr,
                  WOA.data(), (int)woa_n);
      g_diag_dsv41_woa_dequant_calls += 1.0;
      g_diag_dsv41_woa_dequant_ms += std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - dequant_start).count();
    }
    for (int token = 0; token < S; ++token) {
      const float *in = contexts.data() + (size_t)token * H * qh;
#pragma omp parallel for collapse(2)
      for (int group = 0; group < groups; ++group)
        for (int row = 0; row < rank; ++row)
          grouped[((size_t)token * groups + group) * rank + row] = dot_f32_avx512(
              in + (size_t)group * group_in,
              WOA.data() + ((size_t)group * rank + row) * group_in, group_in);
    }
  }
  dense_proj(engine, p + "attn_output_b.weight", grouped.data(), out_context, S,
             groups * rank, D);
  // 2026-09-13: also feed the STANDARD [REQUEST] line's own t_attn_outproj
  // counter, not just the DIAG3-only ms accumulator above -- t_attn_outproj
  // was scoped to mla_attention_absorbed_gguf's own (ratio==0, rarely-used
  // fallback) output projection only, so the standard line's outproj= field
  // stayed near-zero/misleading for THIS model's real (ratio!=0) DSA path,
  // even though [REQUEST-DIAG3]'s own outproj= already measured it
  // correctly. Diagnostic-only change (adds a timer read, no behavior
  // change) -- makes the standard line accurate without needing DIAG3
  // enabled to see where attn= time actually goes on this model.
  const double _dsv41_outproj_elapsed_s = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - _dsv41_t7).count();
  g_diag_dsv41_outproj_ms += _dsv41_outproj_elapsed_s * 1000.0;
  {
    double _old = perf().t_attn_outproj.load(std::memory_order_relaxed);
    while (!perf().t_attn_outproj.compare_exchange_weak(
        _old, _old + _dsv41_outproj_elapsed_s, std::memory_order_relaxed,
        std::memory_order_relaxed)) {}
  }
}

void dsv41_attention_gguf(StreamingEngine &engine, int layer_idx,
                         const float *hidden_states, int S, int pos_base,
                         float *out_context) {
  dsv41_attention_impl(engine, layer_idx, hidden_states, S, pos_base,
                       out_context, nullptr, nullptr, nullptr, nullptr,
                       false);
}

void deepseek41_attention_projected(StreamingEngine &engine, int layer,
                                    int pos, const float *normed,
                                    const float *q_lora, const float *query,
                                    const float *raw_kv, float *output) {
  dsv41_attention_impl(engine, layer, nullptr, 1, pos, output,
                       normed, q_lora, query, raw_kv, false);
}

void deepseek41_attention_projected_context(
    StreamingEngine &engine, int layer, int pos, const float *normed,
    const float *q_lora, const float *query, const float *raw_kv,
    float *contexts) {
  dsv41_attention_impl(engine, layer, nullptr, 1, pos, contexts,
                       normed, q_lora, query, raw_kv, true);
}

// ============================================================
// Standard GQA causal attention with a real dense KV cache (Kc/Vc), for
// non-MLA architectures (Hunyuan/Hy-V3/Llama-style). Structured like
// mla_attention_absorbed_gguf: norm -> QKV projection -> QK-norm -> RoPE ->
// write K/V into the cache -> full causal softmax attention over every
// cached position (not just the current token, unlike the previous dormant
// HYV3Backend::dense_attention stub) -> output projection.
// ============================================================
void dense_attention_gguf(StreamingEngine &engine, int layer_idx,
                          const float *hidden_states, int S, int pos_base,
                          float *out_context) {
  PERF_SCOPE(t_attn);
  const auto &c = engine.model_file.architecture_config;
  int D = c.hidden;
  int H = c.n_heads;
  int Hkv = std::max(1, c.n_head_kv);
  int hd = c.head_dim;
  int group = std::max(1, H / Hkv); // query heads sharing one kv head (GQA)
  std::string p = "blk." + std::to_string(layer_idx) + ".";
  float rope_theta =
      engine.model_file.get_config_float("rope.freq_base", 10000.0f);

  // ---- 1. attention input norm ----
  std::vector<float> normed((size_t)S * D);
  rmsnorm_proj(engine, p + "attn_norm.weight", hidden_states, normed.data(), S,
               D, c.eps);

  // ---- 2. Q/K/V projection ----
  std::vector<float> Q((size_t)S * H * hd), K((size_t)S * Hkv * hd),
      V((size_t)S * Hkv * hd);
  dense_proj(engine, p + "attn_q.weight", normed.data(), Q.data(), S, D,
             H * hd);
  dense_proj(engine, p + "attn_k.weight", normed.data(), K.data(), S, D,
             Hkv * hd);
  dense_proj(engine, p + "attn_v.weight", normed.data(), V.data(), S, D,
             Hkv * hd);

  // ---- 3/4. QK-norm + RoPE. Ordering differs by arch: original HunyuanMoE
  // ropes before QK-norm; Hy-V3 (and every other supported arch) norms
  // first — see ArchitectureConfig::rope_before_norm. ----
  auto qn_it = engine.resident_layers.find(p + "attn_q_norm.weight");
  auto kn_it = engine.resident_layers.find(p + "attn_k_norm.weight");
  const float *qn_w =
      (qn_it != engine.resident_layers.end() && qn_it->second.host_ptr)
          ? reinterpret_cast<float *>(qn_it->second.host_ptr)
          : nullptr;
  const float *kn_w =
      (kn_it != engine.resident_layers.end() && kn_it->second.host_ptr)
          ? reinterpret_cast<float *>(kn_it->second.host_ptr)
          : nullptr;

  auto apply_qk_norms = [&]() {
    if (qn_w)
      for (int s = 0; s < S; ++s)
        for (int h = 0; h < H; ++h)
          rmsnorm_row(Q.data() + ((size_t)s * H + h) * hd,
                      Q.data() + ((size_t)s * H + h) * hd, qn_w, hd, c.eps);
    if (kn_w)
      for (int s = 0; s < S; ++s)
        for (int h = 0; h < Hkv; ++h)
          rmsnorm_row(K.data() + ((size_t)s * Hkv + h) * hd,
                      K.data() + ((size_t)s * Hkv + h) * hd, kn_w, hd, c.eps);
  };
  auto apply_qk_rope = [&]() {
    for (int s = 0; s < S; ++s) {
      int pos = pos_base + s;
      for (int h = 0; h < H; ++h)
        apply_rope_yarn(Q.data() + ((size_t)s * H + h) * hd, hd, pos,
                        rope_theta, c);
      for (int h = 0; h < Hkv; ++h)
        apply_rope_yarn(K.data() + ((size_t)s * Hkv + h) * hd, hd, pos,
                        rope_theta, c);
    }
  };
  if (c.rope_before_norm) {
    apply_qk_rope();
    apply_qk_norms();
  } else {
    apply_qk_norms();
    apply_qk_rope();
  }

  // ---- 5. write K/V into the dense cache ----
  int kvd = Hkv * hd;
  for (int s = 0; s < S; ++s) {
    int pos = pos_base + s;
    if (pos >= engine.max_t)
      continue; // bounds check, mirrors MLA's Lc/Rc write
    std::memcpy(engine.Kc[layer_idx] + (size_t)pos * kvd,
                K.data() + (size_t)s * kvd, (size_t)kvd * sizeof(float));
    std::memcpy(engine.Vc[layer_idx] + (size_t)pos * kvd,
                V.data() + (size_t)s * kvd, (size_t)kvd * sizeof(float));
  }

  // ---- 6. causal GQA attention: full masked softmax over every cached
  // position up to and including the current one (not just a single
  // self-dot-product — replaces the previous dormant stub). Hunyuan/Hy-V3
  // use a plain 1/sqrt(head_dim) score scale, unlike DeepSeek/GLM-5.2's
  // YaRN mscale^2 adjustment (confirmed against hunyuan-moe.cpp/hy-v3.cpp:
  // both use plain kq_scale). ----
  float kq_scale = 1.0f / std::sqrt((float)hd);
  std::vector<float> head_contexts((size_t)S * H * hd, 0.0f);

// num_threads(attn_score_threads()): dedicated, possibly-smaller team
// size for this causal score loop specifically -- see EngineConfig::
// attn_score_threads' own comment (dense_dispatch.hpp) for why this is
// separate from the process-wide omp_max_threads default. Defaults to
// omp_get_max_threads() (the same value this pragma would have used
// anyway) when unset, so this is a genuine no-op for any config that
// doesn't set attn_score_threads.
#pragma omp parallel for collapse(2) schedule(dynamic, 1)                      \
    num_threads(attn_score_threads())
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < H; ++h) {
      int current_pos = pos_base + s;
      int context_len = current_pos + 1;
      int kv_head = h / group;
      const float *q_h = Q.data() + ((size_t)s * H + h) * hd;

      std::vector<float> scores(context_len);
      for (int t = 0; t < context_len; ++t) {
        const float *k_t =
            engine.Kc[layer_idx] + (size_t)t * kvd + (size_t)kv_head * hd;
        float dot = 0.0f;
        for (int i = 0; i < hd; ++i)
          dot += q_h[i] * k_t[i];
        scores[t] = dot * kq_scale;
      }
      softmax_row(scores.data(), context_len);

      float *out_h = head_contexts.data() + ((size_t)s * H + h) * hd;
      for (int t = 0; t < context_len; ++t) {
        const float *v_t =
            engine.Vc[layer_idx] + (size_t)t * kvd + (size_t)kv_head * hd;
        float alpha = scores[t];
        for (int i = 0; i < hd; ++i)
          out_h[i] += alpha * v_t[i];
      }
    }
  }

  // ---- 7. output projection ----
  dense_proj(engine, p + "attn_output.weight", head_contexts.data(),
             out_context, S, H * hd, D);
}
