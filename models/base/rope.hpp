// rope.hpp — RoPE (Rotary Position Embedding) with YaRN long-context scaling.
//
// Extracted verbatim from src/attention_layer.cpp's former static apply_rope_yarn /
// compute_yarn_kq_scale so the math is unit-testable in isolation (tests/
// test_components.cpp). This header is deliberately dependency-free beyond <cmath>:
// it takes PRIMITIVE parameters, not ArchitectureConfig, so it pulls in no CUDA /
// no AVX-512 / no engine headers and compiles in the plain CPU test harness.
// attention_layer.cpp keeps one-line wrappers that unpack ArchitectureConfig into
// these primitives.
//
// CONVENTION: interleaved rotate (pairs (2j, 2j+1)) -- LLAMA_ROPE_TYPE_NORM in
// llama.cpp, used by every MLA target this project runs (glm-dsa / ChatGLM /
// DeepSeek-V2/V3). Confirmed two ways:
//   - llama.cpp src/llama-model.cpp: rope_type for GLM_DSA and CHATGLM is
//     LLAMA_ROPE_TYPE_NORM; rope_norm (ggml) is the interleaved variant.
//   - rope_interleave pairs (2j, 2j+1).
// The previous implementation here used NeoX rotate-half (pairs (k, k+half) =
// LLAMA_ROPE_TYPE_NEOX), which scrambled the decoupled-rope positional encoding
// on every MLA layer and was the root cause of "always random text" output.
//
// YaRN: when rope_ext_factor == 0 (the default for any model without
// rope.scaling.type=yarn -- true for GLM-5.2) every formula below reduces to
// plain interleaved RoPE with no mscale / no correction ramp. The YaRN branch is
// preserved for forward-compatibility but is dead code on every current target.
#pragma once
#include <cmath>
#include <algorithm>   // std::min / std::max (YaRN corr ramp)

// NeoX-style was WRONG for these arches; this is the interleaved (NORM) rotation.
// Pairs (2j, 2j+1) for j in [0, dim/2). Frequency for pair j: theta_base^(-2j/dim).
//
// dsv4_mscale (default false): DeepSeek-V4 uses a genuinely DIFFERENT attn-
// factor formula from V3's here, not just "V3's formula with
// rope_yarn_log_mul forced to 0" -- confirmed by actually deriving both:
// forcing rope_yarn_log_mul=0 in the branch below (with rope_attn_factor's
// default of 1.0) collapses attn_factor_org and the (1+0.1*log(factor))
// denominator to the SAME expression, so rope_cache_mscale reduces to a
// constant 1.0, not to V4's documented mscale = 1/(1+0.1*log(1/freq_scale))
// (llama.cpp deepseek4.cpp's dsv4_rope_attn_factor). Graph_initializer.cpp
// still forces rope_yarn_log_mul=0 for DeepSeek-V4 (V3's formula has no use
// for that coefficient there either way), but the actual V4 mscale needs
// this separate branch, not just a zeroed coefficient -- caught by deriving
// the formula by hand before implementing, per this project's own standing
// "verify before assuming a coefficient reduces the formula" discipline.
inline void rope_yarn_inplace(float* v, int dim, int pos, float theta_base,
                              float rope_ext_factor, float rope_freq_scale,
                              float rope_attn_factor, float rope_yarn_log_mul,
                              int rope_n_ctx_orig, bool dsv4_mscale = false) {
    int half = dim / 2;
    float theta_scale = std::pow(theta_base, -2.0f / (float)dim);

    float rope_cache_mscale = rope_attn_factor;
    float corr_start = 0.0f, corr_end = (float)(dim - 1);
    if (rope_ext_factor != 0.0f) {
        float factor = 1.0f / rope_freq_scale;
        if (dsv4_mscale) {
            rope_cache_mscale = rope_attn_factor / (1.0f + 0.1f * std::log(factor));
        } else {
            auto get_mscale = [](float s, float m) {
                return s <= 1.0f ? 1.0f : (0.1f * m * std::log(s) + 1.0f);
            };
            float yarn_base = (rope_yarn_log_mul != 0.0f)
                ? get_mscale(factor, 1.0f) / get_mscale(factor, rope_yarn_log_mul)
                : get_mscale(factor, 1.0f);
            float attn_factor_org = rope_attn_factor * yarn_base;
            rope_cache_mscale = attn_factor_org / (1.0f + 0.1f * std::log(factor));
        }

        auto corr_dim = [&](float n_rot) {
            return (float)dim * std::log((float)rope_n_ctx_orig / (n_rot * 2.0f * (float)M_PI))
                 / (2.0f * std::log(theta_base));
        };
        corr_start = std::max(0.0f, std::floor(corr_dim(32.0f))); // beta_fast
        corr_end   = std::min((float)(dim - 1), std::ceil(corr_dim(1.0f))); // beta_slow
    }

    float theta_extrap = (float)pos;
    for (int j = 0; j < half; ++j) {
        float theta_interp = rope_freq_scale * theta_extrap;
        float t = theta_interp;
        float mscale = rope_cache_mscale;
        if (rope_ext_factor != 0.0f) {
            float y = ((float)j - corr_start) / std::max(0.001f, corr_end - corr_start);
            float ramp = 1.0f - std::min(1.0f, std::max(0.0f, y));
            float ramp_mix = ramp * rope_ext_factor;
            t = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            // V3's formula pre-divides rope_cache_mscale by this SAME term
            // above (attn_factor_org / (1+0.1*log(factor))) specifically so
            // this multiplication cancels it back out -- net effect
            // attn_factor_org, by design (matches deepseek2.cpp). V4's
            // dsv4_mscale branch computes its own COMPLETE final mscale
            // above already (rope_attn_factor / (1+0.1*log(factor)) IS the
            // documented formula, no further adjustment) -- applying this
            // multiplication there too would cancel that division right
            // back out, silently discarding the whole point of the
            // dsv4_mscale branch. Skip it for V4.
            if (!dsv4_mscale)
                mscale *= 1.0f + 0.1f * std::log(1.0f / rope_freq_scale);
        }
        float cos_t = std::cos(t) * mscale;
        float sin_t = std::sin(t) * mscale;

        // INTERLEAVED pairing (2j, 2j+1) -- NOT (k, k+half).
        float a = v[2*j], b = v[2*j + 1];
        v[2*j]       = a * cos_t - b * sin_t;
        v[2*j + 1]   = a * sin_t + b * cos_t;

        theta_extrap *= theta_scale;
    }
}

// TRANSPOSE of rope_yarn_inplace's rotation, matching ggml's own
// ggml_rope_ext_back semantics (a backward/gradient op, NOT a true
// mathematical inverse) -- rotates by -theta using the SAME mscale as the
// forward pass, not its reciprocal. Since forward is v' = mscale*R(t)*v,
// this computes mscale*R(-t)*v', which is a true round-trip to the ORIGINAL
// v only when mscale == 1 (rope_ext_factor == 0, plain RoPE -- true for
// every current target's main attention path). With YaRN/dsv4_mscale
// active (mscale != 1), round-tripping through both functions instead
// yields mscale^2 * v, not v -- this is the mathematically correct
// transpose behavior (matching what ggml_rope_ext_back actually computes),
// not a bug; test accordingly (assert plain-RoPE round-trip == identity,
// and separately assert the YaRN case's round-trip scales by mscale^2, in
// tests/test_components.cpp -- do NOT assert identity unconditionally).
// Added for DeepSeek-V4's "de-rope" step (dsv4_attention_gguf,
// attention_layer.cpp): V4 applies this to the attention OUTPUT, after MHA
// and before the grouped output projection -- see deepseek4.cpp's
// ggml_rope_ext_back call. Phase 1's raw/SWA layers use the model's MAIN
// RoPE config (same c.rope_ext_factor/rope_freq_scale/etc. the Q/K rotation
// itself uses, with dsv4_mscale=true for the V4-specific attn-factor
// formula above) -- if that model sets YaRN (ext_factor != 0), mscale != 1
// here too, so the mscale^2-not-identity round-trip behavior above is a
// real, reachable case for a long-context DeepSeek-V4 checkpoint, not just
// a theoretical corner. (Project C's separate, always-plain-RoPE compressed-
// layer config is a DIFFERENT, unrelated RoPE application, out of Phase-1
// scope.)
inline void rope_yarn_inplace_inverse(float* v, int dim, int pos, float theta_base,
                                      float rope_ext_factor, float rope_freq_scale,
                                      float rope_attn_factor, float rope_yarn_log_mul,
                                      int rope_n_ctx_orig, bool dsv4_mscale = false) {
    int half = dim / 2;
    float theta_scale = std::pow(theta_base, -2.0f / (float)dim);

    float rope_cache_mscale = rope_attn_factor;
    float corr_start = 0.0f, corr_end = (float)(dim - 1);
    if (rope_ext_factor != 0.0f) {
        float factor = 1.0f / rope_freq_scale;
        if (dsv4_mscale) {
            rope_cache_mscale = rope_attn_factor / (1.0f + 0.1f * std::log(factor));
        } else {
            auto get_mscale = [](float s, float m) {
                return s <= 1.0f ? 1.0f : (0.1f * m * std::log(s) + 1.0f);
            };
            float yarn_base = (rope_yarn_log_mul != 0.0f)
                ? get_mscale(factor, 1.0f) / get_mscale(factor, rope_yarn_log_mul)
                : get_mscale(factor, 1.0f);
            float attn_factor_org = rope_attn_factor * yarn_base;
            rope_cache_mscale = attn_factor_org / (1.0f + 0.1f * std::log(factor));
        }

        auto corr_dim = [&](float n_rot) {
            return (float)dim * std::log((float)rope_n_ctx_orig / (n_rot * 2.0f * (float)M_PI))
                 / (2.0f * std::log(theta_base));
        };
        corr_start = std::max(0.0f, std::floor(corr_dim(32.0f)));
        corr_end   = std::min((float)(dim - 1), std::ceil(corr_dim(1.0f)));
    }

    float theta_extrap = (float)pos;
    for (int j = 0; j < half; ++j) {
        float theta_interp = rope_freq_scale * theta_extrap;
        float t = theta_interp;
        float mscale = rope_cache_mscale;
        if (rope_ext_factor != 0.0f) {
            float y = ((float)j - corr_start) / std::max(0.001f, corr_end - corr_start);
            float ramp = 1.0f - std::min(1.0f, std::max(0.0f, y));
            float ramp_mix = ramp * rope_ext_factor;
            t = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            // V3's formula pre-divides rope_cache_mscale by this SAME term
            // above (attn_factor_org / (1+0.1*log(factor))) specifically so
            // this multiplication cancels it back out -- net effect
            // attn_factor_org, by design (matches deepseek2.cpp). V4's
            // dsv4_mscale branch computes its own COMPLETE final mscale
            // above already (rope_attn_factor / (1+0.1*log(factor)) IS the
            // documented formula, no further adjustment) -- applying this
            // multiplication there too would cancel that division right
            // back out, silently discarding the whole point of the
            // dsv4_mscale branch. Skip it for V4.
            if (!dsv4_mscale)
                mscale *= 1.0f + 0.1f * std::log(1.0f / rope_freq_scale);
        }
        float cos_t = std::cos(t) * mscale;
        float sin_t = std::sin(t) * mscale;

        // Inverse rotation: same cos_t/sin_t, sign pattern flipped relative
        // to the forward function above (rotate by -t instead of +t).
        float a = v[2*j], b = v[2*j + 1];
        v[2*j]       =  a * cos_t + b * sin_t;
        v[2*j + 1]   = -a * sin_t + b * cos_t;

        theta_extrap *= theta_scale;
    }
}

// Attention-score temperature scale for YaRN, matching deepseek2.cpp's kq_scale
// derivation. Reduces to plain 1/sqrt(qh) when rope_ext_factor==0 (no YaRN).
inline float yarn_kq_scale(float rope_ext_factor, float rope_freq_scale,
                           float rope_attn_factor, float rope_yarn_log_mul, int qh) {
    if (rope_ext_factor == 0.0f) return 1.0f / std::sqrt((float)qh);

    float factor = 1.0f / rope_freq_scale;
    auto get_mscale = [](float s, float m) {
        return s <= 1.0f ? 1.0f : (0.1f * m * std::log(s) + 1.0f);
    };
    float yarn_base = (rope_yarn_log_mul != 0.0f)
        ? get_mscale(factor, 1.0f) / get_mscale(factor, rope_yarn_log_mul)
        : get_mscale(factor, 1.0f);
    float attn_factor_org = rope_attn_factor * yarn_base;
    float mscale_final = attn_factor_org * (1.0f + 0.1f * rope_yarn_log_mul * std::log(factor));
    return (mscale_final * mscale_final) / std::sqrt((float)qh);
}
