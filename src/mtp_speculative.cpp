#include "mtp_speculative.hpp"

#include "async_prefetcher.hpp"
#include "dense_dispatch.hpp"
#include "engine.hpp"
#include "gguf_tokenizer.hpp"
#include "kquant_dequant.hpp"
#include "logger.hpp"
#include "ram_manager.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

extern void layers_forward_gguf(StreamingEngine &engine,
                                RAMSafetyManager &ram_mgr,
                                LockFreePrefetcher &prefetcher,
                                float *hidden_states, int S, int pos_base,
                                const int *token_ids);
// Same two hooks generation_loop.hpp brackets every layers_forward_gguf()
// call with for a multi-stream Hyper-Connections model (hc_mult > 1) --
// hc_expand_embedding seeds the transient, per-call engine.hc_stream_buf
// from this call's own token embeddings (it does NOT persist meaningful
// state across calls, unlike glm5next_recur_state/glm5next_conv_history,
// which MtpSpeculator's take_snapshot/restore_snapshot already handle
// separately), and hc_collapse_output folds it back into hidden_states.
// target_forward() below is a second, independent caller of
// layers_forward_gguf() (the target trunk, for verification) and must
// bracket it exactly the same way or its per-layer HC hooks read stale
// hc_stream_buf content left over from an unrelated earlier call.
extern void hc_expand_embedding(StreamingEngine &engine,
                                const float *hidden_states, int S, int D);
extern void hc_collapse_output(StreamingEngine &engine, float *hidden_states,
                               int S, int D);

MtpSpeculator::MtpSpeculator(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                             LockFreePrefetcher &prefetcher)
    : engine_(engine), ram_mgr_(ram_mgr), prefetcher_(prefetcher) {}

namespace {

size_t mtp_host_peak_estimate(const StreamingEngine &engine,
                              size_t prompt_rows, int draft_tokens) {
    const auto &c = engine.model_file.architecture_config;
    const size_t D = c.hidden > 0 ? (size_t)c.hidden : 0;
    const size_t V = c.vocab > 0 ? (size_t)c.vocab : 0;
    const size_t K = draft_tokens > 0 ? (size_t)draft_tokens : 1;
    const size_t max_size = std::numeric_limits<size_t>::max();
    size_t float_elems = 0;
    const auto add_mul = [&](size_t n, size_t m) {
        if (n > 0 && m > max_size / n) {
            float_elems = max_size;
            return;
        }
        const size_t add = n * m;
        if (add > max_size - float_elems) float_elems = max_size;
        else float_elems += add;
    };

    // Prompt bridge plus the maximum simultaneous target/MTP rows in one
    // speculative verification cycle.  The factors intentionally cover
    // vector copies made by target_logits_batch and the acceptance buffers;
    // prompt_hidden itself is owned by the generation loop and is not charged
    // a second time here.
    add_mul(2 * std::min(prompt_rows, max_size / 2), D);
    add_mul(8 * std::min(K, max_size / 8) + 32, D);
    add_mul(3 * std::min(K, max_size / 3) + 8, V);

    // GLM KDA rollback snapshots are host vectors and must be included even
    // though their normal persistent state is charged by the engine cache
    // allocator.  This is one additional copy per live state during MTP.
    for (const auto &state : engine.glm5next_recur_state)
        add_mul(state.size(), 1);
    for (const auto &state : engine.glm5next_conv_history)
        add_mul(state.size(), 1);

    if (float_elems == max_size) return max_size;
    const size_t bytes = float_elems > max_size / sizeof(float)
                             ? max_size
                             : float_elems * sizeof(float);
    constexpr size_t kScratchMargin = 256ull * 1024ull * 1024ull;
    return bytes > max_size - kScratchMargin ? max_size
                                             : bytes + kScratchMargin;
}

struct HostReservation {
    size_t bytes = 0;
    ~HostReservation() { MemoryGuard::release_host_bytes(bytes); }
};

} // namespace

bool MtpSpeculator::supported() const {
    const auto &c = engine_.model_file.architecture_config;
    const auto fam = engine_.family_ops.family;
    // hc_mult > 1 (multi-stream Hyper-Connections) used to be an outright
    // block here. Checked against the real glm5next.cpp reference source:
    // the NextN/MTP block is built with `GGML_ASSERT(!layer.hc_attn_fn &&
    // "the NextN block has no mHC mixer")` -- it is an ordinary
    // plain-residual DSA layer regardless of the trunk's own hc_mult, fed
    // from `build_hc_mean(...)` (unweighted, matching this engine's own
    // hc_collapse_output for GLM5NEXT) followed by output_norm -- i.e. the
    // SAME single-stream [S,D] hidden state GLM_DSA's non-HC trunk already
    // produces. generation_loop.hpp's hc_active branch runs
    // hc_expand_embedding/hc_collapse_output around layers_forward_gguf
    // unconditionally (not gated on mtp_enabled), and skips the GLM5NEXT
    // full-graph fast path whenever mtp_enabled is set, so `prompt_hidden`
    // handed to run() below is already collapsed to single-stream by the
    // time MTP ever sees it. glm_mtp_forward() itself makes no HC calls --
    // correct, since the NextN block never enters HC-stream space at all.
    // GLM_DSA has no HC tensors (hc_mult<=1 there always), so this only
    // changes behavior for GLM5NEXT.
    return engine_.mtp_enabled && c.n_layer_nextn == 1 && c.hidden > 0 &&
           c.vocab > 0 &&
           (fam == ModelFamily::GLM5NEXT || fam == ModelFamily::GLM_DSA) &&
           engine_.layers_experts.size() > (size_t)c.n_layers;
}

bool MtpSpeculator::embed(int token, float *out) const {
    const auto &c = engine_.model_file.architecture_config;
    auto it = engine_.resident_layers.find("token_embd.weight");
    if (it == engine_.resident_layers.end() || !it->second.host_ptr ||
        token < 0 || token >= c.vocab)
        return false;
    const int rb = row_bytes(it->second.meta.type, c.hidden);
    return rb > 0 &&
           dequant_row(it->second.meta.type,
                       it->second.host_ptr + (int64_t)token * rb, out,
                       c.hidden);
}

bool MtpSpeculator::normalized_target_hidden(const float *raw, float *out) const {
    const auto &c = engine_.model_file.architecture_config;
    auto it = engine_.resident_layers.find("output_norm.weight");
    if (it == engine_.resident_layers.end() || !it->second.host_ptr)
        return false;
    rmsnorm_row(out, raw, reinterpret_cast<const float *>(it->second.host_ptr),
                c.hidden, c.eps);
    return std::all_of(out, out + c.hidden,
                       [](float v) { return std::isfinite(v); });
}

bool MtpSpeculator::target_forward(const std::vector<int> &tokens, int pos_base,
                                   std::vector<float> &raw_hidden) {
    if (tokens.empty()) return false;
    const int D = engine_.model_file.architecture_config.hidden;
    std::vector<float> hidden((size_t)tokens.size() * D, 0.0f);
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (!embed(tokens[i], hidden.data() + i * D)) return false;
    }
    // Mirror generation_loop.hpp's own hc_expand_embedding/layers_forward_
    // gguf/hc_collapse_output bracket exactly -- this is a second, separate
    // caller of the target trunk (verification), not just the normal
    // prefill/decode path, and needs the identical wrapping for a
    // multi-stream HC model (see the extern declarations' own comment
    // above for why this is required, not optional).
    const bool hc_active = engine_.model_file.architecture_config.hc_mult > 1;
    if (hc_active)
        hc_expand_embedding(engine_, hidden.data(), (int)tokens.size(), D);
    layers_forward_gguf(engine_, ram_mgr_, prefetcher_, hidden.data(),
                        (int)tokens.size(), pos_base, tokens.data());
    // Mirrors generation_loop.hpp's own guard: skip the manual collapse if a
    // full-graph fast path already produced final output directly. This is
    // structurally unreachable today (every full-graph path is gated
    // !engine.mtp_enabled, and MTP is only ever active with mtp_enabled
    // true), but checked explicitly rather than relied upon, exactly like
    // the primary call site does.
    if (hc_active && !engine_.qwen4exp_full_graph_output_ready &&
        !engine_.glm5next_full_graph_output_ready &&
        !engine_.deepseek_v4_full_graph_output_ready)
        hc_collapse_output(engine_, hidden.data(), (int)tokens.size(), D);
    if (!std::all_of(hidden.begin(), hidden.end(),
                     [](float v) { return std::isfinite(v); })) {
        LOG_STREAM_ERROR() << "[MTP] target verification produced non-finite "
                              "hidden state\n";
        return false;
    }
    raw_hidden = std::move(hidden);
    return true;
}

bool MtpSpeculator::target_logits_batch(const std::vector<float> &raw_hidden,
                                        int rows,
                                        std::vector<float> &logits) {
    const auto &c = engine_.model_file.architecture_config;
    if (rows <= 0 || raw_hidden.size() != (size_t)rows * c.hidden)
        return false;
    auto out_it = engine_.resident_layers.find("output.weight");
    if (out_it == engine_.resident_layers.end() || !out_it->second.host_ptr)
        return false;

    std::vector<float> normalized = raw_hidden;
    auto norm_it = engine_.resident_layers.find("output_norm.weight");
    if (norm_it == engine_.resident_layers.end() || !norm_it->second.host_ptr)
        return false;
    rmsnorm_proj(engine_, "output_norm.weight", raw_hidden.data(),
                 normalized.data(), rows, c.hidden, c.eps);
    logits.resize((size_t)rows * c.vocab);
    dense_proj(engine_, "output.weight", normalized.data(), logits.data(), rows,
               c.hidden, c.vocab);
    return std::all_of(logits.begin(), logits.end(),
                       [](float v) { return std::isfinite(v); });
}

bool MtpSpeculator::make_distribution(const float *logits, int vocab,
                                      const std::vector<int> &context,
                                      float temperature, float top_p,
                                      std::vector<float> &probs) const {
    if (!logits || vocab <= 0) return false;
    std::vector<float> work(logits, logits + vocab);
    constexpr float repetition_penalty = 1.3f;
    constexpr int repetition_window = 64;
    const int begin = std::max(0, (int)context.size() - repetition_window);
    for (int i = begin; i < (int)context.size(); ++i) {
        const int token = context[(size_t)i];
        if (token < 0 || token >= vocab) continue;
        float &v = work[(size_t)token];
        v = (v > 0.0f) ? v / repetition_penalty : v * repetition_penalty;
    }
    if (!std::all_of(work.begin(), work.end(),
                     [](float v) { return std::isfinite(v); }))
        return false;

    probs.assign((size_t)vocab, 0.0f);
    if (temperature <= 0.0f) {
        const int best = (int)std::distance(
            work.begin(), std::max_element(work.begin(), work.end()));
        probs[(size_t)best] = 1.0f;
        return true;
    }

    const float max_logit = *std::max_element(work.begin(), work.end());
    const float inv_temp = 1.0f / std::max(temperature, 1e-4f);
    double sum = 0.0;
    for (int i = 0; i < vocab; ++i) {
        const float p = std::exp((work[(size_t)i] - max_logit) * inv_temp);
        if (!std::isfinite(p)) return false;
        probs[(size_t)i] = p;
        sum += p;
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) return false;
    for (float &p : probs) p = (float)(p / sum);

    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<int> indices((size_t)vocab);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&](int a, int b) { return probs[(size_t)a] > probs[(size_t)b]; });
        float cumulative = 0.0f;
        int keep = vocab;
        for (int i = 0; i < vocab; ++i) {
            cumulative += probs[(size_t)indices[(size_t)i]];
            if (cumulative >= top_p) {
                keep = i + 1;
                break;
            }
        }
        for (int i = keep; i < vocab; ++i)
            probs[(size_t)indices[(size_t)i]] = 0.0f;
        double kept_sum = 0.0;
        for (float p : probs) kept_sum += p;
        if (!(kept_sum > 0.0)) return false;
        for (float &p : probs) p = (float)(p / kept_sum);
    }
    return true;
}

int MtpSpeculator::sample_distribution(const std::vector<float> &probs,
                                       std::mt19937 &rng) const {
    if (probs.empty()) return -1;
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    const float target = uniform(rng);
    float cumulative = 0.0f;
    int last = -1;
    for (int i = 0; i < (int)probs.size(); ++i) {
        if (probs[(size_t)i] <= 0.0f) continue;
        last = i;
        cumulative += probs[(size_t)i];
        if (cumulative >= target) return i;
    }
    return last;
}

bool MtpSpeculator::emit_token(std::vector<int> &sequence, int token, int eos_id,
                               const MtpEmitFn &emit) const {
    if (token < 0) return false;
    sequence.push_back(token);
    if (emit && !emit(token)) return false;
    return token != eos_id;
}

bool MtpSpeculator::run(const std::vector<int> &prompt_ids,
                        const std::vector<float> &prompt_hidden,
                        int max_new_tokens, int eos_id, float temperature,
                        float top_p, std::vector<int> &sequence,
                        std::mt19937 &rng, const MtpEmitFn &emit) {
    const auto &c = engine_.model_file.architecture_config;
    const int D = c.hidden;
    const int V = c.vocab;
    if (!supported() || prompt_ids.empty() || max_new_tokens < 2 ||
        prompt_hidden.size() != (size_t)prompt_ids.size() * D ||
        sequence.size() != prompt_ids.size())
        return false;

    const int max_k = std::min(std::max(1, engine_.mtp_draft_tokens),
                               max_new_tokens - 1);
    const size_t host_peak = mtp_host_peak_estimate(
        engine_, prompt_ids.size(), max_k);
    constexpr size_t kMtpWatermark = 512ull * 1024ull * 1024ull;
    if (host_peak == std::numeric_limits<size_t>::max() ||
        !MemoryGuard::try_reserve_host_bytes(host_peak, kMtpWatermark)) {
        LOG_STREAM_WARN() << "[MTP] host-memory admission denied: need ~"
                          << (host_peak == std::numeric_limits<size_t>::max()
                                  ? -1.0
                                  : (double)host_peak / 1.0e9)
                          << " GB transient scratch, reserved="
                          << (double)MemoryGuard::reserved_host_bytes() / 1.0e9
                          << " GB, budget="
                          << (double)MemoryGuard::host_budget_bytes() / 1.0e9
                          << " GB; using ordinary target decoding\n";
        return false;
    }
    HostReservation host_reservation{host_peak};
    LOG_STREAM_INFO() << "[MTP] host-memory admission: ~"
                      << (double)host_peak / 1.0e9
                      << " GB transient peak reserved (budget="
                      << (double)MemoryGuard::host_budget_bytes() / 1.0e9
                      << " GB)\n";

    // The reference MTP driver leaves the final prompt token separate.  It
    // first processes prompt[0 .. N-2] with target hidden rows shifted by one,
    // then feeds prompt[N-1] with the MTP hidden row produced for prompt[N-2].
    // Feeding the whole prompt in one call would use target h[N-2] for the
    // final token and skips the trained MTP-to-MTP handoff at the boundary.
    const int prompt_prefix_rows = std::max(0, (int)prompt_ids.size() - 1);
    std::vector<float> mtp_input((size_t)prompt_prefix_rows * D, 0.0f);
    for (int s = 1; s < prompt_prefix_rows; ++s) {
        if (!normalized_target_hidden(prompt_hidden.data() + (s - 1) * D,
                                      mtp_input.data() + (size_t)s * D))
            return false;
    }
    std::vector<float> mtp_hidden((size_t)D, 0.0f);
    if (prompt_prefix_rows > 0) {
        std::vector<float> prefix_hidden((size_t)prompt_prefix_rows * D);
        if (!glm_mtp_forward(engine_, ram_mgr_, prefetcher_, prompt_ids.data(),
                             mtp_input.data(), prompt_prefix_rows, 0,
                             prefix_hidden.data(), nullptr))
            return false;
        std::memcpy(mtp_hidden.data(),
                    prefix_hidden.data() + (size_t)(prompt_prefix_rows - 1) * D,
                    (size_t)D * sizeof(float));
    }

    // `mtp_input_h` is the MTP hidden row paired with the current token.  The
    // block returns `mtp_hidden` for that token, which supplies the next
    // proposal step.  Keeping both rows is important: the target verification
    // bridge below needs the former, while the draft chain needs the latter.
    std::vector<float> mtp_input_h = mtp_hidden;
    const int last_prompt_token = prompt_ids.back();
    std::vector<float> mtp_logits((size_t)V);
    if (!glm_mtp_forward(engine_, ram_mgr_, prefetcher_, &last_prompt_token,
                         mtp_input_h.data(), 1,
                         (int)prompt_ids.size() - 1, mtp_hidden.data(),
                         mtp_logits.data()))
        return false;

    struct TargetSnapshot {
        std::vector<std::vector<float>> recur;
        std::vector<std::vector<float>> conv;
    };
    auto take_snapshot = [&]() {
        sync_glm5next_kda_gpu_states(engine_);
        TargetSnapshot snap;
        snap.recur = engine_.glm5next_recur_state;
        snap.conv = engine_.glm5next_conv_history;
        return snap;
    };
    auto restore_snapshot = [&](const TargetSnapshot &snap) {
        engine_.glm5next_recur_state = snap.recur;
        engine_.glm5next_conv_history = snap.conv;
        invalidate_glm5next_kda_gpu_states(engine_);
    };

    std::vector<float> current_raw(prompt_hidden.end() - D, prompt_hidden.end());
    int current_pos = (int)prompt_ids.size() - 1;

    // A bounded direct-target tail is used if a later MTP proposal cannot be
    // formed (for example, a best-effort VRAM allocation was reclaimed). It
    // keeps the already-committed target state valid instead of dropping into
    // the caller's normal loop with a hidden state that no longer matches it.
    auto direct_tail = [&]() {
        while ((int)sequence.size() - (int)prompt_ids.size() < max_new_tokens) {
            std::vector<float> one_logits;
            if (!target_logits_batch(current_raw, 1, one_logits)) return false;
            std::vector<float> probs;
            if (!make_distribution(one_logits.data(), V, sequence, temperature,
                                   top_p, probs))
                return false;
            const int token = sample_distribution(probs, rng);
            if (!emit_token(sequence, token, eos_id, emit)) return true;
            if ((int)sequence.size() - (int)prompt_ids.size() >= max_new_tokens)
                return true;
            std::vector<float> next_raw;
            const std::vector<int> one{token};
            if (!target_forward(one, current_pos + 1, next_raw)) return false;
            current_raw = std::move(next_raw);
            current_pos++;
        }
        return true;
    };

    while ((int)sequence.size() - (int)prompt_ids.size() < max_new_tokens) {
        const int remaining = max_new_tokens -
                              ((int)sequence.size() - (int)prompt_ids.size());
        if (remaining <= 1) {
            return direct_tail();
        }
        const int K = std::min(std::max(1, engine_.mtp_draft_tokens), remaining - 1);
        perf().n_mtp_cycles.fetch_add(1, std::memory_order_relaxed);

        std::vector<int> drafts;
        drafts.reserve((size_t)K);
        std::vector<std::vector<float>> q_probs;
        q_probs.reserve((size_t)K);
        std::vector<float> proposal_h = mtp_hidden;
        std::vector<float> proposal_logits = mtp_logits;

        bool proposal_ok = true;
        for (int j = 0; j < K; ++j) {
            std::vector<int> context = sequence;
            context.insert(context.end(), drafts.begin(), drafts.end());
            std::vector<float> q;
            if (!make_distribution(proposal_logits.data(), V, context, temperature,
                                   top_p, q)) {
                proposal_ok = false;
                break;
            }
            const int draft = sample_distribution(q, rng);
            if (draft < 0) {
                proposal_ok = false;
                break;
            }
            q_probs.push_back(std::move(q));
            drafts.push_back(draft);
            if (j + 1 < K) {
                std::vector<float> next_h(D);
                std::vector<float> next_logits(V);
                const int token = draft;
                if (!glm_mtp_forward(engine_, ram_mgr_, prefetcher_, &token,
                                     proposal_h.data(), 1,
                                     current_pos + j + 1, next_h.data(),
                                     next_logits.data())) {
                    proposal_ok = false;
                    break;
                }
                proposal_h = std::move(next_h);
                proposal_logits = std::move(next_logits);
            }
        }
        if (!proposal_ok) {
            if (drafts.empty()) return false; // target state is still untouched
            return direct_tail();
        }
        perf().n_mtp_drafts_proposed.fetch_add((uint64_t)drafts.size(),
                                               std::memory_order_relaxed);

        const TargetSnapshot snapshot = take_snapshot();
        std::vector<float> verified_raw;
        if (!target_forward(drafts, current_pos + 1, verified_raw)) {
            restore_snapshot(snapshot);
            return false; // caller may safely use the ordinary target loop
        }

        // Mirror common_speculative_impl_draft_mtp::process(): after the
        // target verifies [current_token, drafts...], refresh the MTP block
        // with the target hidden rows shifted onto the following tokens. The
        // proposal chain above is speculative, so leaving it in place would
        // make the next cycle depend on rejected MTP state. Row 0 is the
        // current committed token paired with the previous MTP output; row j
        // (j>0) is draft[j-1] paired with the target hidden state after the
        // preceding token.
        std::vector<int> sync_tokens;
        sync_tokens.reserve((size_t)K + 1);
        sync_tokens.push_back(sequence.back());
        sync_tokens.insert(sync_tokens.end(), drafts.begin(), drafts.end());
        std::vector<float> sync_input((size_t)(K + 1) * D, 0.0f);
        std::memcpy(sync_input.data(), mtp_input_h.data(),
                    (size_t)D * sizeof(float));
        if (!normalized_target_hidden(current_raw.data(),
                                      sync_input.data() + (size_t)D)) {
            restore_snapshot(snapshot);
            return false;
        }
        for (int j = 1; j < K; ++j) {
            if (!normalized_target_hidden(
                    verified_raw.data() + (size_t)(j - 1) * D,
                    sync_input.data() + (size_t)(j + 1) * D)) {
                restore_snapshot(snapshot);
                return false;
            }
        }
        std::vector<float> sync_hidden((size_t)(K + 1) * D);
        if (!glm_mtp_forward(engine_, ram_mgr_, prefetcher_, sync_tokens.data(),
                             sync_input.data(), K + 1, current_pos,
                             sync_hidden.data(), nullptr)) {
            restore_snapshot(snapshot);
            return false;
        }

        // One batched target-head call produces p_0 ... p_K.  p_0 is the
        // distribution after the current committed token; p_j (j>0) is after
        // draft j-1; p_K is after the final draft.
        std::vector<float> target_input((size_t)(K + 1) * D);
        std::memcpy(target_input.data(), current_raw.data(),
                    (size_t)D * sizeof(float));
        for (int j = 1; j <= K; ++j) {
            std::memcpy(target_input.data() + (size_t)j * D,
                        verified_raw.data() + (size_t)(j - 1) * D,
                        (size_t)D * sizeof(float));
        }
        std::vector<float> target_logits;
        if (!target_logits_batch(target_input, K + 1, target_logits)) {
            restore_snapshot(snapshot);
            return false;
        }

        std::vector<std::vector<float>> p_probs;
        p_probs.reserve((size_t)K + 1);
        for (int j = 0; j <= K; ++j) {
            std::vector<int> context = sequence;
            const int count = std::min(j, K);
            context.insert(context.end(), drafts.begin(), drafts.begin() + count);
            std::vector<float> p;
            if (!make_distribution(target_logits.data() + (size_t)j * V, V,
                                   context, temperature, top_p, p)) {
                restore_snapshot(snapshot);
                return false;
            }
            p_probs.push_back(std::move(p));
        }

        // DIAGNOSTIC (2026-09-09): direct q-vs-p comparison at the first
        // draft of every cycle, added while chasing an abnormally low
        // acceptance rate (4-12% observed vs. the 30-70%+ a calibrated MTP
        // head should reach). Prints enough to tell "q and p genuinely
        // disagree on the top choice" (a real drafting-quality problem, or
        // q/p miscalibration) apart from "q's own top token gets a
        // near-zero probability under p" (q is closer to uncorrelated
        // noise than an approximation of p) -- these point at different
        // root causes. Cheap (one argmax scan over V per cycle, not per
        // draft) and unconditional so it survives into the next debugging
        // session without needing a rebuild to re-enable.
        {
            const int q_top = (int)std::distance(
                q_probs[0].begin(),
                std::max_element(q_probs[0].begin(), q_probs[0].end()));
            const int p_top = (int)std::distance(
                p_probs[0].begin(),
                std::max_element(p_probs[0].begin(), p_probs[0].end()));
            LOG_STREAM_INFO()
                << "[MTP-DIAG] cycle draft0=" << drafts[0]
                << " q[draft0]=" << q_probs[0][(size_t)drafts[0]]
                << " p[draft0]=" << p_probs[0][(size_t)drafts[0]]
                << " q_top=" << q_top << " q[q_top]=" << q_probs[0][(size_t)q_top]
                << " p[q_top]=" << p_probs[0][(size_t)q_top]
                << " p_top=" << p_top << " p[p_top]=" << p_probs[0][(size_t)p_top]
                << " q[p_top]=" << q_probs[0][(size_t)p_top]
                << " agree=" << (q_top == p_top ? 1 : 0) << "\n";
        }

        int accepted = 0;
        int correction = -1;
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        for (int j = 0; j < K; ++j) {
            const float qprob = q_probs[(size_t)j][(size_t)drafts[(size_t)j]];
            const float pprob = p_probs[(size_t)j][(size_t)drafts[(size_t)j]];
            const float accept = qprob > 0.0f
                                     ? std::min(1.0f, pprob / qprob)
                                     : (pprob > 0.0f ? 1.0f : 0.0f);
            if (uniform(rng) <= accept) {
                ++accepted;
                continue;
            }

            // Exact speculative sampling's residual distribution q is
            // max(0, p-q), renormalized.  With top-p distributions this is
            // still a valid correction distribution and preserves the target
            // support on the rejection branch.
            std::vector<float> residual((size_t)V, 0.0f);
            double sum = 0.0;
            for (int v = 0; v < V; ++v) {
                residual[(size_t)v] = std::max(
                    0.0f, p_probs[(size_t)j][(size_t)v] -
                              q_probs[(size_t)j][(size_t)v]);
                sum += residual[(size_t)v];
            }
            if (!(sum > 0.0)) residual = p_probs[(size_t)j];
            else for (float &v : residual) v = (float)(v / sum);
            correction = sample_distribution(residual, rng);
            break;
        }
        if (accepted == K) {
            correction = sample_distribution(p_probs[(size_t)K], rng);
        }
        if (correction < 0) {
            restore_snapshot(snapshot);
            return false;
        }

        const int correction_pos = current_pos + accepted + 1;
        std::vector<int> committed(drafts.begin(), drafts.begin() + accepted);
        committed.push_back(correction);
        std::vector<float> correction_raw;
        if (accepted == K) {
            // The target state already includes all K drafts.  Only the
            // sampled post-draft correction needs to advance it.
            const std::vector<int> one{correction};
            if (!target_forward(one, correction_pos, correction_raw)) {
                restore_snapshot(snapshot);
                return false;
            }
        } else {
            // Rejected target rows may have updated KDA state.  Restore the
            // state at the current committed position, then replay exactly
            // the accepted prefix and correction token.
            restore_snapshot(snapshot);
            if (!target_forward(committed, current_pos + 1, correction_raw)) {
                restore_snapshot(snapshot);
                return false;
            }
        }

        perf().n_mtp_drafts_accepted.fetch_add((uint64_t)accepted,
                                               std::memory_order_relaxed);
        // emit_token always pushes onto `sequence` before checking the
        // caller's emit()/eos_id stop condition, so a size delta counts
        // every token actually emitted regardless of which branch below
        // stops early -- one fetch_add on the way out covers all of them.
        const size_t seq_before_emit = sequence.size();
        bool stop_early = false;
        for (int j = 0; j < accepted && !stop_early; ++j) {
            if (!emit_token(sequence, drafts[(size_t)j], eos_id, emit))
                stop_early = true;
        }
        if (!stop_early && !emit_token(sequence, correction, eos_id, emit))
            stop_early = true;
        perf().n_mtp_tokens_emitted.fetch_add(
            (uint64_t)(sequence.size() - seq_before_emit),
            std::memory_order_relaxed);
        if (stop_early) return true;
        current_raw = std::move(correction_raw);
        current_pos = correction_pos;

        // Advance the MTP block on the correction token.  The input hidden is
        // the MTP output for the last committed token in the synchronized
        // [current, drafts...] bridge above, matching the reference driver's
        // pending_h handoff. It is deliberately NOT the target output for
        // `correction`: that target row is not the MTP input convention.
        std::vector<float> next_h(D);
        std::vector<float> next_logits(V);
        const int token = correction;
        const float * correction_mtp_input =
            sync_hidden.data() + (size_t)accepted * D;
        if (!glm_mtp_forward(engine_, ram_mgr_, prefetcher_, &token,
                             correction_mtp_input, 1, correction_pos,
                             next_h.data(), next_logits.data()))
            return direct_tail();
        mtp_input_h.assign(correction_mtp_input,
                           correction_mtp_input + D);
        mtp_hidden = std::move(next_h);
        mtp_logits = std::move(next_logits);
    }
    return true;
}
