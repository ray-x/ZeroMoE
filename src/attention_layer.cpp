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
static bool check_finite_attn(const char* label, int layer_idx, const float* data, size_t n) {
    int nan_count = 0, inf_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i])) ++nan_count;
        else if (std::isinf(data[i])) ++inf_count;
    }
    if (nan_count > 0 || inf_count > 0) {
        LOG_ERROR("[MLA] %s layer=%d: %d/%zu NaN, %d/%zu Inf.\n", label, layer_idx, nan_count, n, inf_count, n);
        return false;
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
// colibri (rope_interleave). The previous implementation here used NeoX
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
        dense_proj(engine, p + "attn_q_a.weight", normed.data(), ql_buf.data(), S, D, q_lora);
        rmsnorm_proj(engine, p + "attn_q_a_norm.weight", ql_buf.data(), ql_buf.data(), S, q_lora, c.eps);
        dense_proj(engine, p + "attn_q_b.weight", ql_buf.data(), Q.data(), S, q_lora, H * qh);
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
        dense_proj(engine, p + "attn_kv_a_mqa.weight", normed.data(), kv.data(), S, D, kv_dim);
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