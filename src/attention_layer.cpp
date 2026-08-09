#include "engine.hpp"
#include "kernels_cpu.hpp"
#include "cuda_manager.hpp"
#include "dense_dispatch.hpp"
#include "kquant_dequant.hpp"
#include "rope.hpp"        // rope_yarn_inplace / yarn_kq_scale (the real, unit-tested RoPE)
#include "perf_timer.hpp"
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
        return false;
    }
    if (row_width > 0 && Logger::instance().would_log(LogLevel::VERBOSE)) {
        const float* last_row = data + (n - (size_t)row_width);
        double sq = 0.0;
        for (int i = 0; i < row_width; ++i) sq += (double)last_row[i] * last_row[i];
        LOG_STREAM_VERBOSE() << "[MLA-TRACE l=" << layer_idx << "] " << label
                             << " |last_row|=" << std::sqrt(sq) << "\n";
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
    }
    PERF_END(attn_kvproj, t_attn_kvproj);

    // ---- 4. absorbed scoring + value aggregation, now over projected Q + real KV ----
    // NOTE: attn_v_b is dequantized row-by-row via extract_gguf_row_q4_0 below,
    // which assumes Q4_0. For Q5_K/Q6_K/Q8_0 v_b this is numerically wrong — the
    // quant-correct v_b absorption (GPU v_b matmul) is a Step 2/3 follow-up.
    // Dequant the two MLA down-projections to f32 once per layer:
    //   attn_k_b (W_UK [qk_nope,kv_lora,n_head]) — folded into the query
    //   attn_v_b (W_UV [kv_lora,v_head,n_head])  — applied to the aggregated latent
    // GLM-5.2 (glm-dsa) shares DeepSeek-V3's MLA core: same tensors & absorption.
    // GLM-5.2-only extras NOT handled here: DSA indexer (sparse top-k attention —
    // full causal is equivalent for ctx <= indexer.top_k), MTP/NextN (unused).
    PERF_START(attn_absorb);
    auto wkb_it = engine.resident_layers.find(p + "attn_k_b.weight");
    auto wvb_it = engine.resident_layers.find(p + "attn_v_b.weight");
    if (wkb_it == engine.resident_layers.end() || wvb_it == engine.resident_layers.end() ||
        !wkb_it->second.host_ptr || !wvb_it->second.host_ptr) {
        LOG_ERROR("[MLA] layer %d missing attn_k_b/attn_v_b — zeroing attention\n", layer_idx);
        std::memset(out_context, 0, (size_t)S * D * sizeof(float));
        return;
    }
    // WUK/WUV dequantize attn_k_b/attn_v_b -- STATIC MODEL WEIGHTS, identical
    // on every call for a given layer_idx (never change across tokens). This
    // used to re-dequantize millions of elements per layer from scratch on
    // EVERY SINGLE attention call (every layer, every decode step) -- found
    // to be the dominant remaining per-token cost once the MoE expert
    // dequant path was fixed (attn=98.75s vs moe_expert=76.7s over a
    // 4-token run, i.e. attention alone exceeded the ENTIRE MoE pipeline).
    // Cached here: dequantized once per layer on first use, reused forever
    // after -- ne0 is the first shape dim, so each per-head block is
    // already the [O,I] row-major matrix the f32 matmul expects.
    static std::vector<std::vector<float>> wuk_cache, wuv_cache;
    if ((int)wuk_cache.size() <= layer_idx) {
        wuk_cache.resize(c.n_layers);
        wuv_cache.resize(c.n_layers);
    }
    std::vector<float>& WUK = wuk_cache[layer_idx];
    std::vector<float>& WUV = wuv_cache[layer_idx];
    if (WUK.empty()) {
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
    }
    PERF_END(attn_absorb, t_attn_absorb);

    PERF_START(attn_score);
    std::vector<float> head_contexts((size_t)S * H * vh, 0.0f);
    float attn_scale = compute_yarn_kq_scale(c, qh);

    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int s = 0; s < S; ++s) {
        for (int h = 0; h < H; ++h) {
            int current_pos = pos_base + s;
            int past_context_len = current_pos + 1;

            const float* head_q = Q.data() + ((size_t)s * H + h) * qh;
            const float* q_nope = head_q;
            const float* q_rope = head_q + nope;

            // q_absorbed[kvl] = q_nope[nope] @ W_UK[h]   (W_UK[h] is [kvl, nope])
            std::vector<float> q_absorbed(kvl);
            matmul(q_absorbed.data(), q_nope, WUK.data() + (size_t)h * nope * kvl, 1, nope, kvl);

            std::vector<float> attn_scores(past_context_len);
            const float* layer_lc_cache = engine.Lc[layer_idx]; 
            const float* layer_rc_cache = engine.Rc[layer_idx]; 

            for (int t = 0; t < past_context_len; ++t) {
                const float* L_t = layer_lc_cache + (t * kvl); 
                const float* R_t = layer_rc_cache + (t * rope); 

                float score_nope = 0.0f;
                __m512 v_score_nope = _mm512_setzero_ps();
                for (int i = 0; i < kvl; i += 16) {
                    v_score_nope = _mm512_fmadd_ps(_mm512_loadu_ps(q_absorbed.data() + i), _mm512_loadu_ps(L_t + i), v_score_nope);
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
    const auto& c = engine.model_file.architecture_config;
    int H = c.n_heads;
    int D = c.hidden;
    int nope = c.qk_nope;
    int rope = c.qk_rope;
    int qh = nope + rope; // per-head latent width, K==V, no separate v_head
    std::string p = "blk." + std::to_string(layer_idx) + ".";
    float rope_theta = engine.model_file.get_config_float("rope.freq_base", 10000.0f);
    int window = engine.sliding_window > 0 ? engine.sliding_window : engine.max_t;

    // TEMP DEBUG: unconditional (not gated by log level), fires once, to
    // confirm this function is actually reached for this checkpoint and
    // to surface whether YaRN (rope_ext_factor != 0) is active -- remove
    // once the garbage-output investigation concludes.
    {
        static bool dsv4_debug_once = false;
        if (!dsv4_debug_once) {
            dsv4_debug_once = true;
            fprintf(stderr,
                "[DSV4-DEBUG] dsv4_attention_gguf reached: layer=%d H=%d nope=%d rope=%d "
                "qh=%d rope_ext_factor=%.6f rope_freq_scale=%.6f rope_theta=%.1f window=%d "
                "q_lora=%d o_group_count=%d o_lora_rank=%d\n",
                layer_idx, H, nope, rope, qh, c.rope_ext_factor, c.rope_freq_scale,
                rope_theta, window, engine.model_file.get_config_int("attention.q_lora_rank", 2048),
                c.o_group_count, c.o_lora_rank);
        }
    }

    // ---- 1. attention input norm (same as V3) ----
    std::vector<float> normed((size_t)S * D);
    rmsnorm_proj(engine, p + "attn_norm.weight", hidden_states, normed.data(), S, D, c.eps);
    check_finite_attn("normed (post attn_norm rmsnorm)", layer_idx, normed.data(), (size_t)S * D);

    // ---- 2. project Q: q_a -> q_a_norm -> q_b => Q[S, H*qh], RoPE on the
    // rope-half only. Identical code shape to mla_attention_absorbed_gguf's
    // own step 2/2b -- V4 keeps the same low-rank Q decompose pattern. ----
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

    // ---- 3. project KV (single wkv, no absorption) and write the SWA ring
    // cache at pos % window. ----
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

    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
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
        WOA.resize((size_t)o_groups * o_rank * group_in);
        dequant_row(woa_it->second.meta.type, woa_it->second.host_ptr, WOA.data(), (int)WOA.size());
        check_finite_attn("WOA (attn_output_a dequant)", layer_idx, WOA.data(), WOA.size());
    }

    std::vector<float> grouped_out((size_t)S * o_groups * o_rank);
    for (int s = 0; s < S; ++s) {
        const float* in_row = head_contexts.data() + (size_t)s * H * qh;
        for (int g = 0; g < o_groups; ++g) {
            const float* in_group = in_row + (size_t)g * group_in;
            float* out_group = grouped_out.data() + ((size_t)s * o_groups + g) * o_rank;
            matmul(out_group, in_group, WOA.data() + (size_t)g * o_rank * group_in, 1, group_in, o_rank);
        }
    }
    check_finite_attn("grouped_out (post attn_output_a)", layer_idx, grouped_out.data(), grouped_out.size(), o_groups * o_rank);

    dense_proj(engine, p + "attn_output_b.weight", grouped_out.data(), out_context, S, o_groups * o_rank, D);
    check_finite_attn("out_context (post attn_output_b projection)", layer_idx, out_context, (size_t)S * D, D);
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

    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
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