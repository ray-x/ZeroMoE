#include "engine.hpp"
#include "kernels_cpu.hpp"
#include "cuda_manager.hpp"
#include "dense_dispatch.hpp"
#include "kquant_dequant.hpp"
#include "rope.hpp"        // rope_yarn_inplace / yarn_kq_scale (the real, unit-tested RoPE)
#include "perf_timer.hpp"
#include "tensor_trace.hpp"
#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>

extern void matmul(float* y, const float* x, const float* W, int S, int I, int O);
extern void softmax_row(float* x, int size);

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
static bool check_finite_attn(const char* label, int layer_idx, const float* data, size_t n,
                              int row_width = 0) {
    int nan_count = 0, inf_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i])) ++nan_count;
        else if (std::isinf(data[i])) ++inf_count;
    }
    if (nan_count > 0 || inf_count > 0) {
        LOG_ERROR("[MLA] %s layer=%d: %d/%zu NaN, %d/%zu Inf.\n", label, layer_idx, nan_count, n, inf_count, n);
        trace_abort_if_nan("MLA", true);
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
// The real implementation lives in include/rope.hpp (rope_yarn_inplace) so it is
// unit-tested in isolation by tests/test_components.cpp::test_rope. This wrapper
// just unpacks ArchitectureConfig into the primitives rope.hpp takes.
//
// CONVENTION: interleaved rotation -- pairs (2j, 2j+1) -- REQUIRED by every MLA
// target this project runs (glm-dsa / ChatGLM / DeepSeek-V2/V3); confirmed
// against llama.cpp (rope_type = LLAMA_ROPE_TYPE_NORM for GLM_DSA/CHATGLM) and
// rope_interleave. The previous implementation here used NeoX
// rotate-half (pairs (k, k+half)) which scrambled positional encoding on every
// MLA layer and produced "always random text" -- see test_rope's known-answer
// case for the exact failure mode.
// Reduces to plain interleaved RoPE when c.rope_ext_factor == 0 (the default for
// any model without rope.scaling.type=yarn, including GLM-5.2).
static inline void apply_rope_yarn(float* v, int dim, int pos, float theta_base,
                                    const ArchitectureConfig& c) {
    rope_yarn_inplace(v, dim, pos, theta_base, c.rope_ext_factor, c.rope_freq_scale,
                      c.rope_attn_factor, c.rope_yarn_log_mul, c.rope_n_ctx_orig);
}

// Attention-score temperature scale for YaRN, matching deepseek2.cpp's
// kq_scale derivation. Real implementation in include/rope.hpp (yarn_kq_scale);
// reduces to plain 1/sqrt(qh) when c.rope_ext_factor==0 (same no-YaRN reduction
// as apply_rope_yarn above).
static inline float compute_yarn_kq_scale(const ArchitectureConfig& c, int qh) {
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
static inline void rmsnorm_inplace_noweight(float* v, int dim, float eps) {
    double ss = 0.0;
    for (int i = 0; i < dim; ++i) ss += (double)v[i] * v[i];
    float r = 1.0f / std::sqrt((float)(ss / dim) + eps);
    for (int i = 0; i < dim; ++i) v[i] *= r;
}

void extract_gguf_row_q4_0(const QuantizedTensor& tensor, int row_idx, float* __restrict__ out_f32) {
    int I = tensor.meta.shape[1]; 
    int blocks_per_row = I / 32;
    const block_q4_0* row_ptr = reinterpret_cast<const block_q4_0*>(tensor.host_ptr) + (row_idx * blocks_per_row);

    const __m128i m4 = _mm_set1_epi8(0x0F);

    for (int b = 0; b < blocks_per_row; ++b) {
        float scale = fp16_to_fp32(row_ptr[b].d);
        __m512 v_scale = _mm512_set1_ps(scale);

        // 16 bytes = 32 nibbles: low nibble of byte i -> element 2*i, high nibble -> element 2*i+1
        __m128i raw_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_ptr[b].qs));
        __m128i lo_nib = _mm_and_si128(raw_bytes, m4);             // low nibbles (even indices)
        __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(raw_bytes, 4), m4); // high nibbles (odd indices)

        // Interleave: [lo0,hi0, lo1,hi1, ...] to get elements in order
        __m128i interleaved_lo = _mm_unpacklo_epi8(lo_nib, hi_nib); // bytes 0-7 -> 16 values
        __m128i interleaved_hi = _mm_unpackhi_epi8(lo_nib, hi_nib); // bytes 8-15 -> 16 values

        // Convert to f32 with bias subtraction (Q4_0: stored as unsigned [0,15], bias = 8)
        __m512i vals_0 = _mm512_cvtepu8_epi32(interleaved_lo);
        __m512i vals_1 = _mm512_cvtepu8_epi32(interleaved_hi);
        __m512 f_0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(vals_0, _mm512_set1_epi32(8)));
        __m512 f_1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(vals_1, _mm512_set1_epi32(8)));

        _mm512_storeu_ps(out_f32 + (b * 32) + 0,  _mm512_mul_ps(f_0, v_scale));
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
static bool ensure_wuk_wuv_cached(StreamingEngine& engine, int layer_idx) {
    const auto& c = engine.model_file.architecture_config;
    int H = c.n_heads;
    int kvl = c.kv_lora;
    int nope = c.qk_nope;
    int vh = c.v_head;
    std::string p = "blk." + std::to_string(layer_idx) + ".";

    auto wkb_it = engine.resident_layers.find(p + "attn_k_b.weight");
    auto wvb_it = engine.resident_layers.find(p + "attn_v_b.weight");
    if (wkb_it == engine.resident_layers.end() || wvb_it == engine.resident_layers.end() ||
        !wkb_it->second.host_ptr || !wvb_it->second.host_ptr) {
        return false;
    }

    if ((int)wuk_cache.size() <= layer_idx) {
        wuk_cache.resize(c.n_layers);
        wuv_cache.resize(c.n_layers);
    }
    std::vector<float>& WUK = wuk_cache[layer_idx];
    std::vector<float>& WUV = wuv_cache[layer_idx];
    if (!WUK.empty()) return true;  // already cached -- warmup already ran, or a prior real call did

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
    uint64_t real_numel_k = 1; for (auto d : wkb_it->second.meta.shape) real_numel_k *= d;
    uint64_t real_numel_v = 1; for (auto d : wvb_it->second.meta.shape) real_numel_v *= d;
    if (layer_idx == 0) {
        LOG_STREAM_INFO() << "[MLA] attn_k_b.weight type=" << gpu_type_name(wkb_it->second.meta.type)
                           << " numel=" << real_numel_k << " (config nope*kvl*H=" << WUK.size() << ") | "
                           << "attn_v_b.weight type=" << gpu_type_name(wvb_it->second.meta.type)
                           << " numel=" << real_numel_v << " (config kvl*vh*H=" << WUV.size() << ")\n";
    }
    if (real_numel_k != WUK.size() || real_numel_v != WUV.size()) {
        std::string shape_k, shape_v;
        for (auto d : wkb_it->second.meta.shape) shape_k += std::to_string(d) + ",";
        for (auto d : wvb_it->second.meta.shape) shape_v += std::to_string(d) + ",";
        LOG_ERROR("[MLA] layer=%d tensor/config shape MISMATCH: attn_k_b real shape=[%s] "
                  "(numel=%llu) vs config nope*kvl*H=%d*%d*%d=%zu; "
                  "attn_v_b real shape=[%s] (numel=%llu) vs config kvl*vh*H=%d*%d*%d=%zu -- "
                  "dequant_row is about to read the WRONG number of elements from host_ptr.\n",
                  layer_idx, shape_k.c_str(), (unsigned long long)real_numel_k, nope, kvl, H, WUK.size(),
                  shape_v.c_str(), (unsigned long long)real_numel_v, kvl, vh, H, WUV.size());
    }
    dequant_row(wkb_it->second.meta.type, wkb_it->second.host_ptr, WUK.data(), (int)WUK.size());
    dequant_row(wvb_it->second.meta.type, wvb_it->second.host_ptr, WUV.data(), (int)WUV.size());
    check_finite_attn("WUK (attn_k_b dequant)", layer_idx, WUK.data(), WUK.size());
    check_finite_attn("WUV (attn_v_b dequant)", layer_idx, WUV.data(), WUV.size());
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
void warm_mla_absorb_cache(StreamingEngine& engine) {
    const auto& c = engine.model_file.architecture_config;
    if (c.n_layers <= 0 || c.kv_lora <= 0) return;  // not an MLA-absorbed-attention model
    PERF_START(mla_absorb_warmup);
    int warmed = 0, skipped = 0;
    for (int layer_idx = 0; layer_idx < c.n_layers; ++layer_idx) {
        if (ensure_wuk_wuv_cached(engine, layer_idx)) ++warmed;
        else ++skipped;
    }
    PERF_END(mla_absorb_warmup, t_mla_absorb_warmup);
    LOG_STREAM_INFO() << "[MLA] Warmed attn_k_b/attn_v_b dequant cache for " << warmed << " layer(s) "
                          "(" << skipped << " skipped, missing weights) in " << perf().t_mla_absorb_warmup.load()
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
            budget = std::min(free_b, cfg_cap);
        } else {
            LOG_STREAM_WARN() << "[MLA] qabsorb BF16 GPU fast-path: cudaMemGetInfo failed -- "
                                  "skipping, all layers stay on the CPU qabsorb path\n";
        }
        int gpu_warmed = 0, gpu_skipped = 0;
        size_t used = 0;
        size_t per_layer_bytes = (size_t)H * (size_t)nope * (size_t)kvl * sizeof(uint16_t);
        for (int layer_idx = 0; layer_idx < c.n_layers; ++layer_idx) {
            if ((size_t)layer_idx >= wuk_cache.size() || wuk_cache[layer_idx].empty()) { ++gpu_skipped; continue; }
            if (used + per_layer_bytes > budget) { ++gpu_skipped; continue; }
            if (engine.gpu->upload_wuk_bf16(layer_idx, wuk_cache[layer_idx], H, nope, kvl, budget - used)) {
                used += per_layer_bytes;
                ++gpu_warmed;
            } else {
                ++gpu_skipped;
            }
        }
        LOG_STREAM_INFO() << "[MLA] qabsorb BF16 GPU fast-path: " << gpu_warmed << "/" << c.n_layers
                           << " layer(s) uploaded (" << gpu_skipped << " skipped -- VRAM budget exhausted or "
                              "missing weights), " << (used / 1e9) << " GB of " << (budget / 1e9)
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
        int mirror_max_t = (engine.attn_score_gpu_max_context > 0 && engine.attn_score_gpu_max_context < engine.max_t)
                            ? engine.attn_score_gpu_max_context : engine.max_t;
        cudaSetDevice(engine.gpu->device_id);
        size_t free_b = 0, total_b = 0;
        size_t budget = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            size_t cfg_cap = (size_t)(engine.attn_score_gpu_max_vram_gb * 1e9);
            budget = std::min(free_b, cfg_cap);
        } else {
            LOG_STREAM_WARN() << "[MLA] score GPU fast-path: cudaMemGetInfo failed -- "
                                  "skipping, all layers stay on the CPU score path\n";
        }
        int gpu_warmed = 0, gpu_skipped = 0;
        size_t used = 0;
        size_t per_layer_bytes = (size_t)mirror_max_t * ((size_t)kvl + (size_t)rope) * sizeof(float);
        // Tracks which layers actually got a mirror -- WUV upload below
        // only proceeds for these (a layer's WUV is pure wasted VRAM
        // without a mirror, since score_gpu_batched_host requires BOTH to
        // even attempt the GPU path; see wuv_bf16_batched_host's own call
        // site, attention_layer.cpp's score dispatch).
        std::vector<bool> mirror_ready(c.n_layers, false);
        for (int layer_idx = 0; layer_idx < c.n_layers; ++layer_idx) {
            if (used + per_layer_bytes > budget) { ++gpu_skipped; continue; }
            if (engine.gpu->ensure_lc_rc_gpu(layer_idx, mirror_max_t, kvl, rope, budget - used)) {
                used += per_layer_bytes;
                mirror_ready[layer_idx] = true;
                ++gpu_warmed;
            } else {
                ++gpu_skipped;
            }
        }
        LOG_STREAM_INFO() << "[MLA] score GPU fast-path: " << gpu_warmed << "/" << c.n_layers
                           << " layer(s) KV mirror allocated (" << gpu_skipped << " skipped -- VRAM budget "
                              "exhausted), " << (used / 1e9) << " GB of " << (budget / 1e9)
                           << " GB budget used (mirror_max_t=" << mirror_max_t << ", context_size=" << engine.max_t
                           << (mirror_max_t < engine.max_t ? ", CAPPED via attn_score_gpu_max_context" : "") << ")\n";

        // WUV BF16 upload -- the score GPU path's own down-projection step
        // (wuv_bf16_batched_host, cuda_manager.hpp), added after a real
        // production run showed the CPU per-head OMP-parallel WUV loop
        // costing 0.916ms/call (2.86s over one 40-token decode run) -- the
        // SAME "OMP team-spawn cost exceeds a tiny per-call workload"
        // pattern qabsorb's own gather fix already diagnosed and fixed at a
        // different call site. Shares the remaining attn_score_gpu_max_
        // vram_gb budget with the Lc/Rc mirror above (both are part of the
        // same feature). NOTE: WUV's own real footprint (H*kvl*vh*2 bytes,
        // e.g. 64*512*256*2=16.8MB/layer) is actually LARGER than the
        // mirror's per-layer cost at typical context sizes, not smaller as
        // originally assumed here -- corrected after a real production run
        // showed only 6/78 layers getting WUV coverage; see CLAUDE.md.
        int vh = c.v_head;
        int gpu_warmed_wuv = 0, gpu_skipped_wuv = 0;
        size_t used_wuv = 0;
        size_t remaining_budget = (used < budget) ? (budget - used) : 0;
        size_t per_layer_wuv_bytes = (size_t)c.n_heads * (size_t)c.kv_lora * (size_t)vh * sizeof(uint16_t);
        for (int layer_idx = 0; layer_idx < c.n_layers; ++layer_idx) {
            if (!mirror_ready[layer_idx]) { ++gpu_skipped_wuv; continue; }  // WUV is wasted VRAM without a mirror
            if ((size_t)layer_idx >= wuv_cache.size() || wuv_cache[layer_idx].empty()) { ++gpu_skipped_wuv; continue; }
            if (used_wuv + per_layer_wuv_bytes > remaining_budget) { ++gpu_skipped_wuv; continue; }
            if (engine.gpu->upload_wuv_bf16(layer_idx, wuv_cache[layer_idx], c.n_heads, c.kv_lora, vh,
                                             remaining_budget - used_wuv)) {
                used_wuv += per_layer_wuv_bytes;
                ++gpu_warmed_wuv;
            } else {
                ++gpu_skipped_wuv;
            }
        }
        LOG_STREAM_INFO() << "[MLA] WUV BF16 GPU fast-path: " << gpu_warmed_wuv << "/" << c.n_layers
                           << " layer(s) uploaded (" << gpu_skipped_wuv << " skipped -- VRAM budget exhausted, "
                              "no mirror, or missing weights), " << (used_wuv / 1e9) << " GB of " << (remaining_budget / 1e9)
                           << " GB remaining budget used\n";
    }
}

void mla_attention_absorbed_gguf(
    StreamingEngine& engine,
    int layer_idx,
    const float* hidden_states,
    int S,
    int pos_base,
    float* out_context
) {
    PERF_SCOPE(t_attn);
    PERF_START(attn_preamble);
    const auto& c = engine.model_file.architecture_config;
    int H = c.n_heads;
    int D = c.hidden;
    int kvl = c.kv_lora;
    int nope = c.qk_nope;
    int rope = c.qk_rope;
    int vh = c.v_head;
    int qh = nope + rope;
    std::string p = "blk." + std::to_string(layer_idx) + ".";
    float rope_theta = engine.model_file.get_config_float("rope.freq_base", 10000.0f);
    PERF_END(attn_preamble, t_attn_preamble);

    // ---- 1. attention input norm (F32 rmsnorm on GPU) ----
    PERF_START(attn_norm);
    std::vector<float> normed((size_t)S * D);
    rmsnorm_proj(engine, p + "attn_norm.weight", hidden_states, normed.data(), S, D, c.eps);
    check_finite_attn("normed (post attn_norm rmsnorm)", layer_idx, normed.data(), (size_t)S * D);
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
        { NamedTimerScope _t("attn_q_a");
          dense_proj(engine, p + "attn_q_a.weight", normed.data(), ql_buf.data(), S, D, q_lora); }
        rmsnorm_proj(engine, p + "attn_q_a_norm.weight", ql_buf.data(), ql_buf.data(), S, q_lora, c.eps);
        { NamedTimerScope _t("attn_q_b");
          dense_proj(engine, p + "attn_q_b.weight", ql_buf.data(), Q.data(), S, q_lora, H * qh); }
    }
    check_finite_attn("Q (post attn_q_a/q_a_norm/q_b projection, pre-RoPE)", layer_idx, Q.data(), Q.size());

    // ---- 2b. apply RoPE to the decoupled q_rope portion of each head ----
    for (int s = 0; s < S; ++s) {
        int pos = pos_base + s;
        for (int h = 0; h < H; ++h) {
            float* q_rope = Q.data() + ((size_t)s * H + h) * qh + nope;
            apply_rope_yarn(q_rope, rope, pos, rope_theta, c);
        }
    }
    check_finite_attn("Q (post RoPE)", layer_idx, Q.data(), Q.size());
    PERF_END(attn_qproj, t_attn_qproj);

    // ---- 3. project KV (kv_a_mqa) and WRITE Lc / Rc ----
    // FIX: Lc/Rc were allocated+zeroed but never written in the live path, so
    // the KV cache stayed empty. This populates the absorbed latent + rope caches
    // at [pos_base+s].
    PERF_START(attn_kvproj);
    {
        int kv_dim = kvl + rope;
        std::vector<float> kv((size_t)S * kv_dim);
        { NamedTimerScope _t("attn_kv_a_mqa");
          dense_proj(engine, p + "attn_kv_a_mqa.weight", normed.data(), kv.data(), S, D, kv_dim); }
        check_finite_attn("kv (post attn_kv_a_mqa projection, pre-norm/RoPE)", layer_idx, kv.data(), kv.size());
        auto kvn = engine.resident_layers.find(p + "attn_kv_a_norm.weight");
        const float* kvn_w = (kvn != engine.resident_layers.end() && kvn->second.host_ptr)
                             ? reinterpret_cast<float*>(kvn->second.host_ptr) : nullptr;
        for (int s = 0; s < S; ++s) {
            int pos = pos_base + s;
            if (pos >= engine.max_t) continue;
            float* kv_row = kv.data() + (size_t)s * kv_dim;
            float* latent = kv_row;          // first kvl elements
            float* k_rope = kv_row + kvl;    // last rope elements
            float* Lc_dst = engine.Lc[layer_idx] + (int64_t)pos * kvl;
            if (kvn_w) rmsnorm_row(Lc_dst, latent, kvn_w, kvl, c.eps);
            else       std::memcpy(Lc_dst, latent, (size_t)kvl * sizeof(float));
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
                engine.gpu->upload_lc_rc_rows(layer_idx, engine.Lc[layer_idx], engine.Rc[layer_idx],
                                               pos_base, valid_rows, kvl, rope);
            }
        }
    }
    PERF_END(attn_kvproj, t_attn_kvproj);

    // ---- 4. absorbed scoring + value aggregation, now over projected Q + real KV ----
    // GLM-5.2 (glm-dsa) shares DeepSeek-V3's MLA core: same tensors & absorption.
    // GLM-5.2-only extras NOT handled here: DSA indexer (sparse top-k attention —
    // full causal is equivalent for ctx <= indexer.top_k), MTP/NextN (unused).
    // WUK/WUV (attn_k_b/attn_v_b dequant) moved to ensure_wuk_wuv_cached
    // (file scope, above) + warm_mla_absorb_cache (2026-08-12) -- STATIC
    // MODEL WEIGHTS, identical on every call for a given layer_idx, so this
    // is now normally already warm by the time we get here (t_attn_absorb
    // should read ~0, confirming warmup ran) with this call surviving only
    // as a safety-net fallback for whatever warmup didn't cover.
    PERF_START(attn_absorb);
    bool have_wuk_wuv = ensure_wuk_wuv_cached(engine, layer_idx);
    PERF_END(attn_absorb, t_attn_absorb);
    if (!have_wuk_wuv) {
        LOG_ERROR("[MLA] layer %d missing attn_k_b/attn_v_b — zeroing attention\n", layer_idx);
        std::memset(out_context, 0, (size_t)S * D * sizeof(float));
        return;
    }
    std::vector<float>& WUK = wuk_cache[layer_idx];
    std::vector<float>& WUV = wuv_cache[layer_idx];

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
                float* dst = q_nope_gather_all.data() + (size_t)h * S * nope;
                for (int s = 0; s < S; ++s) {
                    const float* head_q = Q.data() + ((size_t)s * H + h) * qh;
                    std::memcpy(dst + (size_t)s * nope, head_q, nope * sizeof(float));
                }
            }
        } else {
            for (int h = 0; h < H; ++h) {
                float* dst = q_nope_gather_all.data() + (size_t)h * S * nope;
                for (int s = 0; s < S; ++s) {
                    const float* head_q = Q.data() + ((size_t)s * H + h) * qh;
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
                layer_idx, q_nope_gather_all.data(), Q_absorbed.data(), H, S, nope, kvl);
            if (!gpu_ok) perf().n_qabsorb_gpu_fallback_cpu.fetch_add(1, std::memory_order_relaxed);
        }
        if (!gpu_ok) {
            #pragma omp parallel for schedule(dynamic, 1)
            for (int h = 0; h < H; ++h) {
                matmul(Q_absorbed.data() + (size_t)h * S * kvl,
                       q_nope_gather_all.data() + (size_t)h * S * nope,
                       WUK.data() + (size_t)h * nope * kvl, S, nope, kvl);
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
    if (S == 1 && engine.attn_score_gpu && engine.gpu) {
        int past_context_len = pos_base + 1;
        // Q_absorbed is already [H,S,kvl] -- at S==1 this collapses to
        // exactly the [H,kvl] layout score_gpu_batched_host needs, no
        // extra gather. Q_rope needs a small [H,rope] gather (same shape
        // of work as qabsorb's own q_nope_gather_all, tiny at these dims).
        std::vector<float> Q_rope_gather((size_t)H * rope);
        for (int h = 0; h < H; ++h) {
            const float* head_q = Q.data() + (size_t)h * qh;  // S==1, so s=0 always
            std::memcpy(Q_rope_gather.data() + (size_t)h * rope, head_q + nope, rope * sizeof(float));
        }
        std::vector<float> context_buf((size_t)H * kvl);
        score_gpu_ok = engine.gpu->score_gpu_batched_host(
            layer_idx, Q_absorbed.data(), Q_rope_gather.data(), context_buf.data(),
            H, past_context_len, kvl, rope, attn_scale);
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
            bool wuv_gpu_ok = engine.gpu->wuv_bf16_batched_host(
                layer_idx, context_buf.data(), head_contexts.data(), H, kvl, vh);
            if (!wuv_gpu_ok) {
                perf().n_wuv_gpu_fallback_cpu.fetch_add(1, std::memory_order_relaxed);
                PERF_START(score_wuv);
                #pragma omp parallel for schedule(dynamic, 1)
                for (int h = 0; h < H; ++h) {
                    float* head_out_dest = head_contexts.data() + (size_t)h * vh;
                    matmul(head_out_dest, context_buf.data() + (size_t)h * kvl,
                           WUV.data() + (size_t)h * kvl * vh, 1, kvl, vh);
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
    #pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(attn_score_threads())
    for (int s = 0; s < S; ++s) {
        for (int h = 0; h < H; ++h) {
            int current_pos = pos_base + s;
            int past_context_len = current_pos + 1;

            const float* head_q = Q.data() + ((size_t)s * H + h) * qh;
            const float* q_rope = head_q + nope;
            const float* q_absorbed = Q_absorbed.data() + ((size_t)h * S + s) * kvl;

            std::vector<float> attn_scores(past_context_len);
            const float* layer_lc_cache = engine.Lc[layer_idx];
            const float* layer_rc_cache = engine.Rc[layer_idx];

            for (int t = 0; t < past_context_len; ++t) {
                const float* L_t = layer_lc_cache + (t * kvl);
                const float* R_t = layer_rc_cache + (t * rope);

                float score_nope = 0.0f;
                __m512 v_score_nope = _mm512_setzero_ps();
                for (int i = 0; i < kvl; i += 16) {
                    v_score_nope = _mm512_fmadd_ps(_mm512_loadu_ps(q_absorbed + i), _mm512_loadu_ps(L_t + i), v_score_nope);
                }
                score_nope = _mm512_reduce_add_ps(v_score_nope);

                float score_rope = 0.0f;
                __m512 v_score_rope = _mm512_setzero_ps();
                for (int i = 0; i < rope; i += 16) {
                    v_score_rope = _mm512_fmadd_ps(_mm512_loadu_ps(q_rope + i), _mm512_loadu_ps(R_t + i), v_score_rope);
                }
                score_rope = _mm512_reduce_add_ps(v_score_rope);

                attn_scores[t] = (score_nope + score_rope) * attn_scale;
            }

            softmax_row(attn_scores.data(), past_context_len);

            std::vector<float> latent_accumulation(kvl, 0.0f);
            for (int t = 0; t < past_context_len; ++t) {
                float alpha_t = attn_scores[t];
                const float* L_t = layer_lc_cache + (t * kvl);

                for (int i = 0; i < kvl; i += 16) {
                    __m512 v_lat = _mm512_loadu_ps(latent_accumulation.data() + i);
                    __m512 v_cache = _mm512_loadu_ps(L_t + i);
                    _mm512_storeu_ps(latent_accumulation.data() + i, _mm512_fmadd_ps(_mm512_set1_ps(alpha_t), v_cache, v_lat));
                }
            }

            // v[vh] = latent_agg[kvl] @ W_UV[h]   (W_UV[h] is [vh, kvl])
            float* head_out_dest = head_contexts.data() + ((size_t)s * H + h) * vh;
            matmul(head_out_dest, latent_accumulation.data(),
                   WUV.data() + (size_t)h * kvl * vh, 1, kvl, vh);
        }
    }
    }  // !score_gpu_ok

    check_finite_attn("head_contexts (post absorbed attention loop)", layer_idx,
                       head_contexts.data(), head_contexts.size());
    PERF_END(attn_score, t_attn_score);

    // ---- 5. output projection (attn_output) via dense GPU dispatch ----
    PERF_START(attn_outproj);
    dense_proj(engine, p + "attn_output.weight", head_contexts.data(), out_context, S, H * vh, D);
    check_finite_attn("out_context (post attn_output projection)", layer_idx, out_context, (size_t)S * D);
    PERF_END(attn_outproj, t_attn_outproj);
}

// ============================================================
// DeepSeek-V4 raw/SWA MLA attention (Phase 1: correctness within each
// layer's local sliding window; CSA/HCA compressed-attention tiers are
// Project C, not implemented here -- every layer is treated as if its
// compress_ratio were 0, i.e. plain SWA). Structured like
// mla_attention_absorbed_gguf above (same attn_norm -> q_a/q_a_norm/q_b ->
// RoPE-on-rope-half Q projection, reused verbatim), but the KV side has NO
// absorption: attn_kv.weight produces one per-head-sized latent directly
// (no wk_b/wv_b, no separate kv_lora_rank), used as BOTH K and V (MQA-
// style attention over the raw cached latent, not a compress/decompress
// round trip). New pieces V3 doesn't have: attention sinks, a de-rope step
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
//      ext_factor/freq_scale for HCA "compress-rope" layers
//      (dsv4_compress_ratios[il] != 0, a feature not implemented here --
//      every layer is treated as compress_ratio==0, see below); for every
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
void dsv4_attention_gguf(
    StreamingEngine& engine,
    int layer_idx,
    const float* hidden_states,
    int S,
    int pos_base,
    float* out_context
) {
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
    const auto& c = engine.model_file.architecture_config;
    int H = c.n_heads;
    int D = c.hidden;
    int nope = c.qk_nope;
    int rope = c.qk_rope;
    int qh = nope + rope; // per-head latent width, K==V, no separate v_head
    std::string p = "blk." + std::to_string(layer_idx) + ".";
    float rope_theta = engine.model_file.get_config_float("rope.freq_base", 10000.0f);
    int window = engine.sliding_window > 0 ? engine.sliding_window : engine.max_t;

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
            LOG_STREAM_INFO() << "[DSV4-DEBUG] dsv4_attention_gguf reached: layer=" << layer_idx
                << " H=" << H << " nope=" << nope << " rope=" << rope << " qh=" << qh
                << " rope_ext_factor=" << c.rope_ext_factor << " rope_freq_scale=" << c.rope_freq_scale
                << " rope_theta=" << rope_theta << " window=" << window
                << " q_lora=" << engine.model_file.get_config_int("attention.q_lora_rank", 2048)
                << " o_group_count=" << c.o_group_count << " o_lora_rank=" << c.o_lora_rank
                << " group_in=(H*qh)/o_groups=" << ((H * qh) / o_groups_dbg) << "\n";
        }
    }
    PERF_END(dsv4_preamble, t_attn_preamble);

    // ---- 1. attention input norm (same as V3) ----
    PERF_START(dsv4_norm);
    std::vector<float> normed((size_t)S * D);
    rmsnorm_proj(engine, p + "attn_norm.weight", hidden_states, normed.data(), S, D, c.eps);
    check_finite_attn("normed (post attn_norm rmsnorm)", layer_idx, normed.data(), (size_t)S * D);
    PERF_END(dsv4_norm, t_attn_norm);

    // ---- 2. project Q: q_a -> q_a_norm -> q_b => Q[S, H*qh], RoPE on the
    // rope-half only. Identical code shape to mla_attention_absorbed_gguf's
    // own step 2/2b -- V4 keeps the same low-rank Q decompose pattern. ----
    PERF_START(dsv4_qproj);
    int q_lora = engine.model_file.get_config_int("attention.q_lora_rank", 2048);
    std::vector<float> Q((size_t)S * H * qh, 0.0f);
    {
        std::vector<float> ql_buf((size_t)S * q_lora);
        dense_proj(engine, p + "attn_q_a.weight", normed.data(), ql_buf.data(), S, D, q_lora);
        rmsnorm_proj(engine, p + "attn_q_a_norm.weight", ql_buf.data(), ql_buf.data(), S, q_lora, c.eps);
        dense_proj(engine, p + "attn_q_b.weight", ql_buf.data(), Q.data(), S, q_lora, H * qh);
    }
    // FIX: this was the one stage in this function without a check (every
    // other stage already has one -- confirmed via a real audit) -- the
    // matching V3 path (mla_attention_absorbed_gguf, this file, line 146)
    // already checks its equivalent Q projection output pre-RoPE; this one
    // didn't, so a NaN introduced by attn_q_a/attn_q_a_norm/attn_q_b here
    // would only surface (if at all) at the post-RoPE check below, one
    // stage later than necessary.
    check_finite_attn("Q (post attn_q_a/q_a_norm/q_b projection, pre-RoPE)", layer_idx, Q.data(), Q.size(), H * qh);
    // Weightless per-head RMSNorm on the full [qh] Q vector -- see
    // rmsnorm_inplace_noweight's own comment above. Must run BEFORE the
    // RoPE loop below, matching deepseek4.cpp's ordering (norm, then split
    // nope/pe, then RoPE on the pe half only).
    for (int s = 0; s < S; ++s) {
        for (int h = 0; h < H; ++h) {
            rmsnorm_inplace_noweight(Q.data() + ((size_t)s * H + h) * qh, qh, c.eps);
        }
    }
    check_finite_attn("Q (post per-head RMSNorm, pre-RoPE)", layer_idx, Q.data(), Q.size(), H * qh);
    // V4 doesn't implement HCA/compress-rope layers (see this function's
    // own header comment) -- confirmed against llama.cpp's real
    // build_attention_impl (deepseek4.cpp): use_compress_rope is false for
    // every layer this function handles, which forces
    // ext_factor_l=0/freq_scale_l=1/attn_factor_l=1 there UNCONDITIONALLY,
    // regardless of the model's global rope.scaling.type GGUF metadata.
    // The model-global c.rope_ext_factor (used correctly by
    // mla_attention_absorbed_gguf for V2/V3/GLM, which DOES apply YaRN
    // uniformly) does NOT apply to V4's ordinary layers -- they are always
    // plain RoPE, even on a long-context checkpoint that sets
    // rope.scaling.type=yarn globally. Previously this used
    // c.rope_ext_factor/c.rope_freq_scale directly with dsv4_mscale=true,
    // which incorrectly engaged YaRN's ramp+mscale math whenever the
    // checkpoint declared YaRN scaling.
    for (int s = 0; s < S; ++s) {
        int pos = pos_base + s;
        for (int h = 0; h < H; ++h) {
            float* q_rope = Q.data() + ((size_t)s * H + h) * qh + nope;
            rope_yarn_inplace(q_rope, rope, pos, rope_theta, /*rope_ext_factor=*/0.0f,
                              /*rope_freq_scale=*/1.0f, /*rope_attn_factor=*/1.0f,
                              c.rope_yarn_log_mul, c.rope_n_ctx_orig, /*dsv4_mscale=*/false);
        }
    }
    check_finite_attn("Q (post RoPE)", layer_idx, Q.data(), Q.size(), H * qh);
    PERF_END(dsv4_qproj, t_attn_qproj);

    // ---- 3. project KV (single wkv, no absorption) and write the SWA ring
    // cache at pos % window. ----
    PERF_START(dsv4_kvproj);
    {
        std::vector<float> kv((size_t)S * qh);
        dense_proj(engine, p + "attn_kv.weight", normed.data(), kv.data(), S, D, qh);
        check_finite_attn("kv (post attn_kv projection, pre-norm/RoPE)", layer_idx, kv.data(), kv.size(), qh);
        // (a) whole-latent norm -- see this function's own header comment.
        // Real tensor name is attn_kv_a_norm (reused from V3's kv_a_norm
        // naming convention, confirmed against llama-arch.cpp's
        // LLM_TENSOR_ATTN_KV_NORM pattern) -- NOT attn_kv_norm. Found via a
        // real "missing weight blk.N.attn_kv_norm.weight -- passing input
        // through" warning spam on every layer of a real DeepSeek-V4-Flash
        // run; confirmed against this file's own earlier research notes,
        // which had the name right all along -- this was a transcription
        // slip introducing the bug during implementation, not a genuine
        // naming ambiguity.
        rmsnorm_proj(engine, p + "attn_kv_a_norm.weight", kv.data(), kv.data(), S, qh, c.eps);
        for (int s = 0; s < S; ++s) {
            int pos = pos_base + s;
            int slot = pos % window;
            float* kv_row = kv.data() + (size_t)s * qh;
            float* k_rope = kv_row + nope;
            // Plain RoPE for V4's ordinary layers -- see the matching
            // comment on the Q-side RoPE call above.
            rope_yarn_inplace(k_rope, rope, pos, rope_theta, /*rope_ext_factor=*/0.0f,
                              /*rope_freq_scale=*/1.0f, /*rope_attn_factor=*/1.0f,
                              c.rope_yarn_log_mul, c.rope_n_ctx_orig, /*dsv4_mscale=*/false);
            // Per-row norm of THIS token's cache row, right after norm+RoPE
            // and right before it becomes one of the vectors the softmax in
            // step 4 actually blends -- the earlier "kv (post attn_kv
            // projection, pre-norm/RoPE)" trace above measures a DIFFERENT,
            // pre-norm/pre-RoPE buffer, so it is not a valid stand-in for
            // "the norm of what gets blended" when checking the softmax
            // convex-combination bound against head_contexts below. Only
            // the newest row (s==S-1) is logged, matching every other
            // VERBOSE trace in this function's "last token" convention.
            if (s == S - 1 && Logger::instance().would_log(LogLevel::VERBOSE)) {
                double sq = 0.0;
                for (int i = 0; i < qh; ++i) sq += (double)kv_row[i] * kv_row[i];
                LOG_STREAM_VERBOSE() << "[MLA-TRACE l=" << layer_idx
                                     << "] kv_row (post norm+RoPE, pre-cache-write) |last_row|="
                                     << std::sqrt(sq) << "\n";
            }
            std::memcpy(engine.KVc[layer_idx] + (int64_t)slot * qh, kv_row,
                        (size_t)qh * sizeof(float));
        }
    }
    check_finite_attn("KVc (post write)", layer_idx, engine.KVc[layer_idx], (size_t)window * qh);
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
            const float* cached = engine.KVc[layer_idx] + (int64_t)slot * qh;
            double sq = 0.0;
            for (int i = 0; i < qh; ++i) sq += (double)cached[i] * cached[i];
            if (sq > max_sq) { max_sq = sq; max_t = t; }
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
    const float* sinks = (sinks_it != engine.resident_layers.end() && sinks_it->second.host_ptr)
                         ? reinterpret_cast<float*>(sinks_it->second.host_ptr) : nullptr;

    std::vector<float> head_contexts((size_t)S * H * qh, 0.0f);
    // CONFIRMED against llama.cpp's build_attention_impl (deepseek4.cpp):
    // every build_attn_mha call site there hardcodes plain
    // 1.0f/sqrtf(n_embd_head) -- no mscale multiplier at all, for any
    // layer this function handles (HCA/compress-rope layers aren't
    // implemented here, see this function's header comment). The earlier
    // "reuse V3's kq_scale formula" approach was wrong whenever the
    // checkpoint's global rope.scaling.type=yarn -- V4 never uses that
    // term outside HCA layers. Passing ext_factor=0 makes this reduce to
    // the correct plain 1/sqrt(qh) unconditionally.
    float attn_scale = yarn_kq_scale(/*rope_ext_factor=*/0.0f, /*rope_freq_scale=*/1.0f,
                                     /*rope_attn_factor=*/1.0f, c.rope_yarn_log_mul, qh);

    // num_threads(attn_score_threads()): dedicated, possibly-smaller team
    // size for this causal score loop specifically -- see EngineConfig::
    // attn_score_threads' own comment (dense_dispatch.hpp) for why this is
    // separate from the process-wide omp_max_threads default. Defaults to
    // omp_get_max_threads() (the same value this pragma would have used
    // anyway) when unset, so this is a genuine no-op for any config that
    // doesn't set attn_score_threads.
    #pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(attn_score_threads())
    for (int s = 0; s < S; ++s) {
        for (int h = 0; h < H; ++h) {
            int current_pos = pos_base + s;
            // SWA window: the last `window` positions ending at current_pos
            // (inclusive), never reaching further back than position 0.
            int past_context_len = std::min(current_pos + 1, window);

            const float* head_q = Q.data() + ((size_t)s * H + h) * qh;

            std::vector<float> attn_scores(past_context_len + (sinks ? 1 : 0));
            const float* layer_kv_cache = engine.KVc[layer_idx];

            for (int t = 0; t < past_context_len; ++t) {
                // t=0 is the OLDEST position still in-window
                // (current_pos - past_context_len + 1); ring slot for the
                // absolute position current_pos-past_context_len+1+t.
                int abs_pos = current_pos - past_context_len + 1 + t;
                int slot = abs_pos % window;
                const float* cached = layer_kv_cache + (int64_t)slot * qh;

                float score = 0.0f;
                __m512 v_score = _mm512_setzero_ps();
                for (int i = 0; i + 16 <= qh; i += 16)
                    v_score = _mm512_fmadd_ps(_mm512_loadu_ps(head_q + i), _mm512_loadu_ps(cached + i), v_score);
                score = _mm512_reduce_add_ps(v_score);
                for (int i = (qh / 16) * 16; i < qh; ++i) score += head_q[i] * cached[i];

                attn_scores[t] = score * attn_scale;
            }
            // Attention sink: an extra always-present logit, no
            // corresponding value contribution -- contributes to the
            // softmax denominator only, damping every real weight.
            if (sinks) attn_scores[past_context_len] = sinks[h];

            softmax_row(attn_scores.data(), (int)attn_scores.size());

            float* head_out_dest = head_contexts.data() + ((size_t)s * H + h) * qh;
            for (int t = 0; t < past_context_len; ++t) {
                int abs_pos = current_pos - past_context_len + 1 + t;
                int slot = abs_pos % window;
                const float* cached = layer_kv_cache + (int64_t)slot * qh;
                float w = attn_scores[t];
                for (int i = 0; i < qh; ++i) head_out_dest[i] += w * cached[i];
            }
            // sink's own softmax mass has no value vector to blend in --
            // it just shrinks the real weights above, already applied.

            // De-rope: undo the rope-half's positional rotation on the
            // AGGREGATED (multi-position-blended) output -- see this
            // function's own header comment on why this is needed here but
            // not in V3 (no absorption to keep rope-space and output-space
            // separate). Same theta/position/(plain, mscale=1) params as
            // the forward rotation applied to K above -- see the matching
            // comment on the Q-side RoPE call earlier in this function for
            // why ext_factor/freq_scale/attn_factor are forced plain here
            // rather than using the model-global c.rope_ext_factor.
            rope_yarn_inplace_inverse(head_out_dest + nope, rope, current_pos, rope_theta,
                                      /*rope_ext_factor=*/0.0f, /*rope_freq_scale=*/1.0f,
                                      /*rope_attn_factor=*/1.0f, c.rope_yarn_log_mul,
                                      c.rope_n_ctx_orig, /*dsv4_mscale=*/false);
        }
    }
    check_finite_attn("head_contexts (post MQA + de-rope)", layer_idx,
                       head_contexts.data(), head_contexts.size(), H * qh);
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
        const float* last_tok = head_contexts.data() + (size_t)(S - 1) * H * qh;
        for (int h = 0; h < H; ++h) {
            const float* seg = last_tok + (size_t)h * qh;
            double sq = 0.0;
            for (int i = 0; i < qh; ++i) sq += (double)seg[i] * seg[i];
            if (sq > max_sq) { max_sq = sq; max_h = h; }
        }
        LOG_STREAM_VERBOSE() << "[MLA-TRACE l=" << layer_idx
                             << "] head_contexts per-head max |head|=" << std::sqrt(max_sq)
                             << " (h=" << max_h << "/" << H << ")\n";
    }

    // ---- 5. grouped low-rank output projection: attn_output_a (per
    // group) -> concat -> attn_output_b (single dense matmul). ----
    PERF_START(dsv4_outproj);
    int o_groups = std::max(1, c.o_group_count);
    int o_rank = c.o_lora_rank;
    int group_in = (H * qh) / o_groups;

    auto woa_it = engine.resident_layers.find(p + "attn_output_a.weight");
    if (woa_it == engine.resident_layers.end() || !woa_it->second.host_ptr) {
        LOG_ERROR("[DSV4] layer %d missing attn_output_a -- zeroing attention\n", layer_idx);
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
    if ((int)woa_cache.size() <= layer_idx) woa_cache.resize(c.n_layers);
    std::vector<float>& WOA = woa_cache[layer_idx];
    if (WOA.empty()) {
        PERF_START(dsv4_outproj_dequant);
        WOA.resize((size_t)o_groups * o_rank * group_in);
        dequant_row(woa_it->second.meta.type, woa_it->second.host_ptr, WOA.data(), (int)WOA.size());
        check_finite_attn("WOA (attn_output_a dequant)", layer_idx, WOA.data(), WOA.size());
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
    static std::vector<float*> woa_dev_cache;
    if (dsv4_outproj_gpu_enabled() && engine.gpu) {
        if ((int)woa_dev_cache.size() <= layer_idx) woa_dev_cache.resize(c.n_layers, nullptr);
        if (!woa_dev_cache[layer_idx]) {
            woa_dev_cache[layer_idx] = engine.gpu->upload_persistent_f32(WOA.data(), WOA.size());
            if (woa_dev_cache[layer_idx]) {
                LOG_STREAM_INFO() << "[DSV4] layer=" << layer_idx << " attn_output_a: "
                                  << (WOA.size() * sizeof(float)) / 1e6
                                  << " MB uploaded to VRAM (dsv4_outproj_gpu)\n";
            } else {
                LOG_STREAM_WARN() << "[DSV4] layer=" << layer_idx << " attn_output_a: VRAM upload "
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
        const float* in_row = head_contexts.data();
        float* out_flat = grouped_out.data();
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
        if (dsv4_outproj_gpu_enabled() && engine.gpu &&
            (int)woa_dev_cache.size() > layer_idx && woa_dev_cache[layer_idx]) {
            gpu_ok = engine.gpu->grouped_output_proj_host(
                woa_dev_cache[layer_idx], o_groups, o_rank, group_in, in_row, out_flat);
        }
        if (!gpu_ok) {
            const float* woa_data = WOA.data();
            #pragma omp parallel for schedule(static)
            for (int go = 0; go < o_groups * o_rank; ++go) {
                int g = go / o_rank;
                int o = go % o_rank;
                const float* in_group = in_row + (size_t)g * group_in;
                const float* w_row = woa_data + (size_t)g * o_rank * group_in + (size_t)o * group_in;
                out_flat[go] = dot_f32_avx512(in_group, w_row, group_in);
            }
        }
    } else {
        for (int s = 0; s < S; ++s) {
            const float* in_row = head_contexts.data() + (size_t)s * H * qh;
            for (int g = 0; g < o_groups; ++g) {
                const float* in_group = in_row + (size_t)g * group_in;
                float* out_group = grouped_out.data() + ((size_t)s * o_groups + g) * o_rank;
                matmul(out_group, in_group, WOA.data() + (size_t)g * o_rank * group_in, 1, group_in, o_rank);
            }
        }
    }
    check_finite_attn("grouped_out (post attn_output_a)", layer_idx, grouped_out.data(), grouped_out.size(), o_groups * o_rank);

    dense_proj(engine, p + "attn_output_b.weight", grouped_out.data(), out_context, S, o_groups * o_rank, D);
    check_finite_attn("out_context (post attn_output_b projection)", layer_idx, out_context, (size_t)S * D, D);
    PERF_END(dsv4_outproj, t_attn_outproj);
}

// ============================================================
// Standard GQA causal attention with a real dense KV cache (Kc/Vc), for
// non-MLA architectures (Hunyuan/Hy-V3/Llama-style). Structured like
// mla_attention_absorbed_gguf: norm -> QKV projection -> QK-norm -> RoPE ->
// write K/V into the cache -> full causal softmax attention over every
// cached position (not just the current token, unlike the previous dormant
// HYV3Backend::dense_attention stub) -> output projection.
// ============================================================
void dense_attention_gguf(
    StreamingEngine& engine,
    int layer_idx,
    const float* hidden_states,
    int S,
    int pos_base,
    float* out_context
) {
    PERF_SCOPE(t_attn);
    const auto& c = engine.model_file.architecture_config;
    int D = c.hidden;
    int H = c.n_heads;
    int Hkv = std::max(1, c.n_head_kv);
    int hd = c.head_dim;
    int group = std::max(1, H / Hkv);  // query heads sharing one kv head (GQA)
    std::string p = "blk." + std::to_string(layer_idx) + ".";
    float rope_theta = engine.model_file.get_config_float("rope.freq_base", 10000.0f);

    // ---- 1. attention input norm ----
    std::vector<float> normed((size_t)S * D);
    rmsnorm_proj(engine, p + "attn_norm.weight", hidden_states, normed.data(), S, D, c.eps);

    // ---- 2. Q/K/V projection ----
    std::vector<float> Q((size_t)S * H * hd), K((size_t)S * Hkv * hd), V((size_t)S * Hkv * hd);
    dense_proj(engine, p + "attn_q.weight", normed.data(), Q.data(), S, D, H * hd);
    dense_proj(engine, p + "attn_k.weight", normed.data(), K.data(), S, D, Hkv * hd);
    dense_proj(engine, p + "attn_v.weight", normed.data(), V.data(), S, D, Hkv * hd);

    // ---- 3/4. QK-norm + RoPE. Ordering differs by arch: original HunyuanMoE
    // ropes before QK-norm; Hy-V3 (and every other supported arch) norms
    // first — see ArchitectureConfig::rope_before_norm. ----
    auto qn_it = engine.resident_layers.find(p + "attn_q_norm.weight");
    auto kn_it = engine.resident_layers.find(p + "attn_k_norm.weight");
    const float* qn_w = (qn_it != engine.resident_layers.end() && qn_it->second.host_ptr)
                        ? reinterpret_cast<float*>(qn_it->second.host_ptr) : nullptr;
    const float* kn_w = (kn_it != engine.resident_layers.end() && kn_it->second.host_ptr)
                        ? reinterpret_cast<float*>(kn_it->second.host_ptr) : nullptr;

    auto apply_qk_norms = [&]() {
        if (qn_w) for (int s = 0; s < S; ++s) for (int h = 0; h < H; ++h)
            rmsnorm_row(Q.data() + ((size_t)s * H + h) * hd, Q.data() + ((size_t)s * H + h) * hd, qn_w, hd, c.eps);
        if (kn_w) for (int s = 0; s < S; ++s) for (int h = 0; h < Hkv; ++h)
            rmsnorm_row(K.data() + ((size_t)s * Hkv + h) * hd, K.data() + ((size_t)s * Hkv + h) * hd, kn_w, hd, c.eps);
    };
    auto apply_qk_rope = [&]() {
        for (int s = 0; s < S; ++s) {
            int pos = pos_base + s;
            for (int h = 0; h < H; ++h)
                apply_rope_yarn(Q.data() + ((size_t)s * H + h) * hd, hd, pos, rope_theta, c);
            for (int h = 0; h < Hkv; ++h)
                apply_rope_yarn(K.data() + ((size_t)s * Hkv + h) * hd, hd, pos, rope_theta, c);
        }
    };
    if (c.rope_before_norm) { apply_qk_rope(); apply_qk_norms(); }
    else                    { apply_qk_norms(); apply_qk_rope(); }

    // ---- 5. write K/V into the dense cache ----
    int kvd = Hkv * hd;
    for (int s = 0; s < S; ++s) {
        int pos = pos_base + s;
        if (pos >= engine.max_t) continue;  // bounds check, mirrors MLA's Lc/Rc write
        std::memcpy(engine.Kc[layer_idx] + (size_t)pos * kvd, K.data() + (size_t)s * kvd, (size_t)kvd * sizeof(float));
        std::memcpy(engine.Vc[layer_idx] + (size_t)pos * kvd, V.data() + (size_t)s * kvd, (size_t)kvd * sizeof(float));
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
    #pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(attn_score_threads())
    for (int s = 0; s < S; ++s) {
        for (int h = 0; h < H; ++h) {
            int current_pos = pos_base + s;
            int context_len = current_pos + 1;
            int kv_head = h / group;
            const float* q_h = Q.data() + ((size_t)s * H + h) * hd;

            std::vector<float> scores(context_len);
            for (int t = 0; t < context_len; ++t) {
                const float* k_t = engine.Kc[layer_idx] + (size_t)t * kvd + (size_t)kv_head * hd;
                float dot = 0.0f;
                for (int i = 0; i < hd; ++i) dot += q_h[i] * k_t[i];
                scores[t] = dot * kq_scale;
            }
            softmax_row(scores.data(), context_len);

            float* out_h = head_contexts.data() + ((size_t)s * H + h) * hd;
            for (int t = 0; t < context_len; ++t) {
                const float* v_t = engine.Vc[layer_idx] + (size_t)t * kvd + (size_t)kv_head * hd;
                float alpha = scores[t];
                for (int i = 0; i < hd; ++i) out_h[i] += alpha * v_t[i];
            }
        }
    }

    // ---- 7. output projection ----
    dense_proj(engine, p + "attn_output.weight", head_contexts.data(), out_context, S, H * hd, D);
}