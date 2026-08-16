#pragma once
#include "engine.hpp"
#include "tensor_trace.hpp"
#include "ram_manager.hpp"
#include "kernels_cpu.hpp"
#include "dense_dispatch.hpp"
#include "kquant_dequant.hpp"     // dequant_row, row_bytes (embed_token)
#include "gguf_tokenizer.hpp"
#include "perf_timer.hpp"
#include "async_prefetcher.hpp"
#include "shutdown.hpp"
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <iostream>
#include <cstring>
#include <cmath>

extern void matmul(float* y, const float* x, const float* W, int S, int I, int O);
extern void rmsnorm_row(float* out, const float* x, const float* w, int D, float eps);

// Diagnostic: count NaN/Inf in a hidden-state buffer, for bisecting WHERE in
// the pipeline corruption first appears (prefill vs. a specific decode step)
// rather than only ever seeing it downstream at the LM-head logits, by which
// point many possible causes have already had a chance to introduce it.
inline void check_hidden_finite(const char* label, const float* data, size_t n) {
    int nan_count = 0, inf_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i])) ++nan_count;
        else if (std::isinf(data[i])) ++inf_count;
    }
    if (nan_count > 0 || inf_count > 0) {
        LOG_ERROR("[HIDDEN] %s: hidden_states corrupted -- %d/%zu NaN, %d/%zu Inf.\n",
                  label, nan_count, n, inf_count, n);
        trace_abort_if_nan("HIDDEN", true);
        return;
    }
    if (trace_gate("HIDDEN")) trace_row("HIDDEN", -1, label, data, n);
}
// token_ids: the S current input token ids that produced hidden_states'
// S rows (one per row, same order) -- needed by DeepSeek-V4 hash-routed MoE
// layers (moe_forward_gguf's ffn_gate_tid2eid lookup, per-token, per-layer
// direct token-id->expert-id table). nullptr/ignored for every other
// family and for non-hash-routed layers within a DeepSeek-V4 file.
extern void layers_forward_gguf(StreamingEngine& engine, RAMSafetyManager& ram_mgr,
                                LockFreePrefetcher& async_io_queue, float* hidden_states, int S, int pos_base,
                                const int* token_ids);
// Hyper-Connections (DeepSeek-V4, hyper_connections.cpp) model-level hooks --
// bracket EVERY layers_forward_gguf call, converting the caller's plain
// single-stream hidden_states into engine.hc_stream_buf's multi-stream form
// before the call and back after, so the multi-stream shape never leaks
// past this pair of call sites (see hyper_connections.cpp's own header
// comment). Gated on architecture_config.hc_mult > 1 at each call site --
// a strict no-op (not even called) for every other family.
extern void hc_expand_embedding(StreamingEngine& engine, const float* hidden_states, int S, int D);
extern void hc_collapse_output(StreamingEngine& engine, float* hidden_states, int S, int D);
// Frees the expert-FFN GPU scratch pool's device buffers back to empty --
// see execution_layer.cpp's definition for why this reclaims little in
// practice (expert dispatch is always S=1) but is kept for consistency
// with CudaManager::shrink_dense_scratch, the one that matters (see
// CLAUDE.md's "prefill scratch reclaim" note).
extern void shrink_moe_gpu_scratch();

// Embeds one token: dequantizes token_embd.weight's row `tok` into out[0..D).
// ROOT CAUSE (found this session): both call sites below used to
// reinterpret_cast host_ptr straight to float* and memcpy at a flat
// D*sizeof(float) stride -- correct ONLY if token_embd.weight is F32. It's a
// large [vocab, hidden] matrix that quantization schemes (e.g. Unsloth
// Dynamic) compress just like any other weight, and unlike norm/bias
// tensors it isn't named "*norm*"/"*.bias" so graph_initializer.cpp's
// verify_float_reinterpret_tensors() load-time scan never covered it. A raw
// float reinterpret of non-F32 bytes at the wrong per-row stride produces
// immediate garbage -- this is what made hidden_states 100% NaN from layer
// 0, token 0, in every generation run this session; every downstream
// diagnostic (router weights, expert FFN, YaRN attention, Q3_K's struct
// layout) was correctly reporting a real symptom of THIS.
inline void embed_token(StreamingEngine& engine, int tok, int D, float* out) {
    auto emb_it = engine.resident_layers.find("token_embd.weight");
    if (emb_it == engine.resident_layers.end() || !emb_it->second.host_ptr) {
        std::fill(out, out + D, 0.0f);
        return;
    }
    int rb = row_bytes(emb_it->second.meta.type, D);
    const uint8_t* row_ptr = emb_it->second.host_ptr + (int64_t)tok * rb;
    dequant_row(emb_it->second.meta.type, row_ptr, out, D);
}

// ============================================================
// GLM-5.2 Reasoning State Machine
// Handles <think>...</think> tags in the token stream.
// Reasoning tokens are kept in KV cache but streamed separately.
// ============================================================
enum class GenerationState { Thinking, Outputting };

struct ReasoningParser {
    GenerationState state = GenerationState::Outputting;
    std::string pending;  // partial tag accumulation
    int think_tokens = 0;
    int output_tokens = 0;

    // Returns: 0 = normal output, 1 = thinking content, 2 = tag consumed (don't print)
    int feed(const std::string& token_text) {
        pending += token_text;

        // Check for tag transitions
        if (state == GenerationState::Outputting) {
            size_t pos = pending.find("<think>");
            if (pos != std::string::npos) {
                state = GenerationState::Thinking;
                pending.clear();
                return 2; // tag consumed
            }
            // Partial match detection: if pending ends with partial "<think" prefix
            if (pending.size() < 7 && std::string("<think>").find(pending) == 0) {
                return 2; // wait for more
            }
            pending.clear();
            output_tokens++;
            return 0;
        } else { // Thinking
            size_t pos = pending.find("</think>");
            if (pos != std::string::npos) {
                state = GenerationState::Outputting;
                pending.clear();
                return 2; // tag consumed
            }
            if (pending.size() < 8 && std::string("</think>").find(pending) == 0) {
                return 2; // wait for more
            }
            pending.clear();
            think_tokens++;
            return 1;
        }
    }

    void reset() {
        state = GenerationState::Outputting;
        pending.clear();
        think_tokens = 0;
        output_tokens = 0;
    }
};

class Sampler {
public:
    std::mt19937 rng;
    Sampler() { rng.seed(std::random_device{}()); }

    int sample(std::vector<float>& logits, float temperature, float top_p) {
        int V = logits.size();
        if (temperature <= 0.0f) {
            return std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        }

        float max_logit = *std::max_element(logits.begin(), logits.end());
        std::vector<float> probs(V);
        float sum_probs = 0.0f;
        float inv_temp = 1.0f / std::max(temperature, 1e-4f);

        for (int i = 0; i < V; ++i) {
            probs[i] = std::exp((logits[i] - max_logit) * inv_temp);
            sum_probs += probs[i];
        }
        for (int i = 0; i < V; ++i) probs[i] /= sum_probs;

        if (top_p > 0.0f && top_p < 1.0f) {
            std::vector<int> indices(V);
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), [&](int a, int b) { return probs[a] > probs[b]; });

            float cumulative_prob = 0.0f;
            int keep_count = V;
            for (int i = 0; i < V; ++i) {
                cumulative_prob += probs[indices[i]];
                if (cumulative_prob >= top_p) {
                    keep_count = i + 1;
                    break;
                }
            }

            float new_sum = 0.0f;
            for (int i = keep_count; i < V; ++i) probs[indices[i]] = 0.0f;
            for (int i = 0; i < keep_count; ++i) new_sum += probs[indices[i]];
            for (int i = 0; i < keep_count; ++i) probs[indices[i]] /= new_sum;
        }

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float target = dist(rng);
        float cdf = 0.0f;
        for (int i = 0; i < V; ++i) {
            if (probs[i] > 0.0f) {
                cdf += probs[i];
                if (cdf >= target) return i;
            }
        }
        // Every prob ended up non-positive (typically NaN propagated in from
        // corrupted logits, since NaN comparisons are always false and skip
        // the loop body above entirely). Previously silent — always returned
        // the same fixed index (V-1), indistinguishable from a real sampling
        // decision, which reads as "the model produced bad output" instead
        // of "something upstream is broken". Loud now; see the [SAMPLER]
        // logits-corruption check in generate_text (generation_loop.hpp) for
        // the earlier, more specific diagnostic.
        LOG_ERROR("[SAMPLER] All %d probabilities were non-positive (NaN or "
                  "<=0) -- falling back to a fixed index (V-1). This is NOT a "
                  "real sampling decision.\n", V);
        return V - 1;
    }
};

// `prefetcher` is caller-owned (main.cpp starts it ONCE for the whole
// session with the real VRAM topology and a pointer back to ram_mgr --
// prefetcher.ram_mgr, set right after prefetcher.start() -- see main.cpp's
// own comment on that assignment) instead of generate_text creating and
// starting its OWN local, topology-less instance per request. That local
// instance was a real structural bug (an earlier investigation): it
// received pilot-prefetch hints and did real disk I/O, but with
// gpu_topo=nullptr its (since-removed) finish_prefetch() could never fire
// an async VRAM promotion -- and meanwhile main.cpp's OWN topology-enabled
// prefetcher instance sat idle, receiving zero hints. Net effect:
// speculative disk reads whose result was consumed by no one. Passing the
// SAME instance everywhere (this function, and prefetcher.ram_mgr for
// LockFreePrefetcher::active_prefetch's own claim_or_wait calls into the
// RAM tier) makes hints and claims all flow through one prefetcher talking
// to one RAM tier.
void generate_text(StreamingEngine& engine, RAMSafetyManager& ram_mgr, GGUFTokenizer& tokenizer,
                   LockFreePrefetcher& prefetcher,
                   const std::vector<int>& prompt_ids, int max_new_tokens, int eos_id,
                   float temperature = 0.7f, float top_p = 0.90f) {
    int max_ctx = prompt_ids.size() + max_new_tokens;
    ram_mgr.calculate_safe_cache_capacity(engine, max_ctx);

    // Snapshot for the whole-request latency breakdown, printed when this
    // request finishes (see bottom of function).
    auto request_snap_start = perf().snapshot();
    auto request_wall_start = std::chrono::high_resolution_clock::now();

    std::vector<int> sequence = prompt_ids;
    Sampler sampler;
    int D = engine.model_file.architecture_config.hidden;
    int V = engine.model_file.architecture_config.vocab;

    // Embed the prompt tokens
    std::vector<float> hidden_states(prompt_ids.size() * D, 0.0f);
    auto emb_it = engine.resident_layers.find("token_embd.weight");
    if (emb_it != engine.resident_layers.end() && emb_it->second.host_ptr) {
        LOG_STREAM_INFO() << "[EMBED] token_embd.weight type=" << (unsigned)emb_it->second.meta.type
                           << " (" << gpu_type_name(emb_it->second.meta.type) << ")\n";
        for (size_t t = 0; t < prompt_ids.size(); ++t) {
            int tok = prompt_ids[t];
            if (tok >= 0 && tok < V) {
                embed_token(engine, tok, D, hidden_states.data() + t * D);
            }
        }
    } else {
        LOG_STREAM_WARN() << "[WARN] token_embd.weight not loaded — cannot embed tokens.\n";
        return;
    }
    // Definitively answers "is embedding the source" -- if this is clean,
    // corruption enters somewhere inside layer 0's own attention computation
    // (mla_attention_absorbed_gguf), which currently has zero diagnostics.
    check_hidden_finite("post-embedding (pre-layers)", hidden_states.data(), hidden_states.size());

    if (D == 0 || V == 0) {
        LOG_STREAM_WARN() << "[WARN] Architecture not configured (D=" << D << " V=" << V << ") — aborting generation.\n";
        return;
    }

    std::cout << "\n[Prefill] Processing " << prompt_ids.size() << " tokens..." << std::flush;
    // Brackets the SAME window PERF_START(prefill)/PERF_END(prefill, ...)
    // measures, reusing the exact Snapshot/operator-/print_summary
    // machinery the whole-REQUEST line already uses (request_snap_start
    // below) -- gives a full attn=/gpu=/disk=/ram_scan=/etc. breakdown
    // scoped to ONLY this prefill call, not the whole request. Added
    // specifically because Phase 2 (prefill_warm/opportunistic-promotion
    // tuning) turned out to cost well under 1s while "Prefill (whole
    // pass)" stayed ~19s -- meaning most of prefill's own wall time was
    // going somewhere never actually broken down (attention? real disk
    // wait? GPU dispatch?), and guessing further at VRAM-placement timing
    // without that breakdown was no longer well-founded. See CLAUDE.md.
    auto prefill_snap_start = perf().snapshot();
    PERF_START(prefill);
    // Hyper-Connections (DeepSeek-V4): expand into engine.hc_stream_buf
    // before, collapse back into hidden_states after -- see
    // hc_expand_embedding/hc_collapse_output's own declarations above.
    bool hc_active = engine.model_file.architecture_config.hc_mult > 1;
    if (hc_active)
        hc_expand_embedding(engine, hidden_states.data(), (int)prompt_ids.size(), D);
    layers_forward_gguf(engine, ram_mgr, prefetcher, hidden_states.data(), prompt_ids.size(), 0, prompt_ids.data());
    if (hc_active)
        hc_collapse_output(engine, hidden_states.data(), (int)prompt_ids.size(), D);
    // Was `PERF_END(prefill, t_attn)` -- dumped this WHOLE prefill pass's
    // wall-clock (attention + MoE routing + MoE expert FFN + cold-cache
    // disk I/O across all 78 layers) into the attention-only t_attn
    // counter, corrupting dump_report()'s "Attention (MLA)" breakdown and
    // masking real attention cost behind unrelated prefill overhead -- see
    // t_prefill_total's comment in perf_timer.hpp for the full story. Own
    // counter now; mla_attention_absorbed_gguf's own per-layer
    // PERF_SCOPE(t_attn) already correctly captures attention's real
    // contribution during this same call, so nothing here needs to also
    // feed t_attn.
    PERF_END(prefill, t_prefill_total);
    // THIS call's own wall time, not t_prefill_total's cumulative-across-
    // requests total -- reuses PERF_END(prefill, ...)'s own start-time local
    // (_perf_prefill, from PERF_START(prefill) above) rather than adding a
    // second independent timer, so this is the exact same window PERF_END
    // just measured (re-reading now() a few cheap instructions later, not a
    // second clock source).
    double prefill_wall_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - _perf_prefill).count();
    perf().n_prompt_tokens.fetch_add(prompt_ids.size(), std::memory_order_relaxed);
    // Full attn=/gpu=/disk=/ram_scan=/etc. breakdown for JUST this prefill
    // call -- same Snapshot-diff + print_summary machinery the whole-
    // REQUEST line uses (request_snap_start/print_summary("REQUEST", ...)
    // below), just scoped to prefill_snap_start's own narrower window
    // instead of the whole request. Label "PREFILL" so it's visually
    // distinct from "REQUEST" and "TOKEN N" in the log.
    auto prefill_snap_end = perf().snapshot();
    auto prefill_delta = prefill_snap_end - prefill_snap_start;
    PerfCounters::print_summary("PREFILL", prefill_wall_s * 1000.0, (int)prompt_ids.size(), prefill_delta);
    // Answers "is phase_b's remaining cost a coverage gap (never hinted)
    // or a worker-capacity gap (hinted, but the decode thread's own
    // dispatch got there before any prefetch worker did)" -- see
    // PerfCounters::Snapshot's own comment on these fields. by_decode
    // means the decode thread itself won the claim (no prefetch worker
    // had even started on it yet); by_prefetch means a worker won it
    // first (the intended case). A high by_decode share alongside a real
    // avg wait time on the SEPARATE "Unified RAM wait" counter (dump_
    // report(), whole-run) would point at prefetch_worker_threads being
    // undersized relative to prefill's own hint volume, not a coverage
    // gap in what gets hinted.
    LOG_STREAM_INFO() << "[PREFILL] RAM claim: " << prefill_delta.n_ram_claim_hit << " hit / "
                       << prefill_delta.n_ram_claim_by_decode << " by-decode (raced ahead of any "
                          "prefetch worker -- a real disk read on the decode thread, phase_b's own cost) / "
                       << prefill_delta.n_ram_claim_by_prefetch << " by-prefetch (a worker won the claim "
                          "first, as intended), " << prefill_delta.n_ram_wait_inflight_decode
                       << " decode wait(s) for an in-flight prefetch, avg "
                       << (prefill_delta.n_ram_wait_inflight_decode > 0
                               ? (prefill_delta.t_ram_wait_inflight_ms_sum / prefill_delta.n_ram_wait_inflight_decode)
                               : 0.0)
                       << "ms\n";
    check_hidden_finite("after prefill", hidden_states.data(), hidden_states.size());

    // Snapshot VRAM slot occupancy RIGHT HERE, at the exact prefill/decode
    // boundary -- directly comparable against the SAME [VRAM-SLOTS] line
    // printed again at shutdown (report_slot_summary, gpu_topology.hpp),
    // to see how much filling happened during prefill (startup preload +
    // Phase 2's warm-up) vs. during decode's own reactive promotions
    // afterward. Previously the only way to see prefill's own contribution
    // was to infer it indirectly.
    LOG_STREAM_INFO() << "[PREFILL-DONE] " << prompt_ids.size() << " tokens in " << prefill_wall_s
                       << "s (" << ((prefill_wall_s > 0.0) ? (prompt_ids.size() / prefill_wall_s) : 0.0)
                       << " tok/s) -- this request's own prefill pass only, matches the 'prefill=' figure "
                          "in the shutdown report's Rates line but available here at the actual "
                          "prefill/decode boundary instead of only after the whole request finishes.\n";
    if (ram_mgr.vram_cache) {
        LOG_STREAM_INFO() << "[PREFILL-DONE] VRAM slot occupancy immediately after prefill (before any decode "
                              "token) -- compare against the same [VRAM-SLOTS] line at shutdown:\n";
        ram_mgr.vram_cache->report_slot_summary();
    }

    // Reclaim GPU scratch grown to fit this prefill's full S -- decode
    // (below) only ever calls with S=1 from here on, so anything grown
    // above that is dead weight for the rest of this request. Real,
    // measured impact on CudaManager's dense-projection scratch (~120 MB
    // at S=1000, ~240 MB at S=2000 for this project's GLM-5.2 dims) given
    // how little free VRAM the CoreEngine GPU can be left with; the
    // expert-FFN scratch shrink reclaims far less in practice (that path
    // is always called with S=1 even during prefill) but costs nothing to
    // include. See CLAUDE.md's "prefill scratch reclaim" note.
    if (engine.gpu) engine.gpu->shrink_dense_scratch();
    shrink_moe_gpu_scratch();

    std::cout << " Done.\n\nOutput: ";

    // Pre-allocate decode buffers (single token at a time)
    std::vector<float> decode_hidden(D);
    std::vector<float> logits(V);
    std::vector<float> normed_last(D);

    // Reasoning state machine for <think>...</think> tag parsing
    ReasoningParser reasoning;

    for (int step = 0; step < max_new_tokens; ++step) {
        // Cooperative shutdown checkpoint: a decode step can take many
        // seconds (CPU-bound expert FFN with no cancellation point inside
        // it), so this is the earliest point a Ctrl-C from the previous
        // step can actually be noticed. Breaking here flows into the exact
        // same report cleanup below as a normal EOS stop, so callers don't
        // need special-casing. (The prefetcher itself is caller-owned now --
        // see the header comment above -- so there's nothing to stop here.)
        if (g_shutdown_requested.load(std::memory_order_relaxed)) {
            std::cout << "\n[Interrupted]\n";
            break;
        }

        const float* last_hidden = hidden_states.data() + ((hidden_states.size() / D - 1) * D);

        // Final norm + lm_head projection
        auto norm_it = engine.resident_layers.find("output_norm.weight");
        auto lm_it = engine.resident_layers.find("output.weight");
        if (norm_it == engine.resident_layers.end() || lm_it == engine.resident_layers.end()) break;

        rmsnorm_proj(engine, "output_norm.weight", last_hidden, normed_last.data(), 1, D,
                     engine.model_file.architecture_config.eps);
        dense_proj(engine, "output.weight", normed_last.data(), logits.data(), 1, D, V);

        // Diagnostic: catch corrupted (NaN/Inf) logits at the source, right
        // where they're produced, rather than downstream where Sampler::sample
        // silently falls back to a fixed index (V-1) when every probability
        // ends up non-positive — a corrupted-but-plausible-looking failure
        // mode that's easy to mistake for "the model is just bad" rather
        // than "something upstream (attention, an expert FFN, or a dense/
        // LM-head GPU kernel) produced garbage." Logged once per occurrence,
        // not every token, so a real problem stays visible without flooding.
        {
            int nan_count = 0, inf_count = 0;
            for (float v : logits) {
                if (std::isnan(v)) ++nan_count;
                else if (std::isinf(v)) ++inf_count;
            }
            if (nan_count > 0 || inf_count > 0) {
                LOG_ERROR("[SAMPLER] step=%d: logits corrupted -- %d/%d NaN, %d/%d Inf. "
                          "Sampler will silently fall back to token V-1 for every NaN case. "
                          "Root cause is upstream of sampling (attention, expert FFN, or the "
                          "output_norm/output.weight dense projection) -- check whether "
                          "output.weight's GPU dispatch (dense_matmul_host, cuda_manager.hpp) "
                          "is implicated by comparing against forcing CPU-only dequant-matmul.\n",
                          step, nan_count, (int)logits.size(), inf_count, (int)logits.size());
            }
        }

        // First-token diagnostic: top-5 logits + norms. Localizes whether the
        // lm_head distribution is peaked-but-wrong (forward pass OK -> look at
        // chat template / tokenization) or near-uniform (forward pass producing
        // noise -> attention/FFN/dim bug). Printed only for the first couple of
        // tokens. This is the fastest way to tell "model is confidently wrong"
        // from "model has no signal at all" without a full llama.cpp parity run.
        if (step < 2) {
            std::vector<int> idx(V);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + std::min(5, V), idx.end(),
                              [&](int a, int b) { return logits[a] > logits[b]; });
            double l2 = 0.0; for (float v : logits) l2 += (double)v * v;
            float maxl = *std::max_element(logits.begin(), logits.end());
            float minl = *std::min_element(logits.begin(), logits.end());
            LOG_STREAM_INFO() << "[LOGITS] step=" << step << " |logits|=" << std::sqrt(l2)
                              << " min=" << minl << " max=" << maxl << " (spread=" << (maxl - minl)
                              << " -- small spread => near-uniform/noise) top5:";
            for (int i = 0; i < std::min(5, V); ++i) {
                int id = idx[i];
                LOG_STREAM_INFO() << " [" << id << "]='" << tokenizer.decode(id)
                                  << "' (" << logits[id] << ")";
            }
            LOG_STREAM_INFO() << "\n";
        }

        // Repetition penalty: no mechanism for this existed before -- with
        // no chat template applied either (main.cpp sends raw BOS+text, not
        // an instruct-formatted prompt; see CLAUDE.md), a low-confidence,
        // unfocused generation can easily drift onto one token and then
        // repeat it forever, since nothing in Sampler::sample discourages
        // it. Standard CTRL-style penalty (Keskar et al. 2019): scale down
        // the logit of every token seen recently, whichever direction makes
        // it less likely (dividing a positive logit or multiplying a
        // negative one both reduce post-softmax probability).
        {
            constexpr float REPETITION_PENALTY = 1.3f;
            constexpr int REPETITION_WINDOW = 64;
            int start = std::max(0, (int)sequence.size() - REPETITION_WINDOW);
            for (int i = start; i < (int)sequence.size(); ++i) {
                int tok = sequence[i];
                if (tok < 0 || tok >= V) continue;
                float& l = logits[tok];
                l = (l > 0.0f) ? (l / REPETITION_PENALTY) : (l * REPETITION_PENALTY);
            }
        }

        int next_token = sampler.sample(logits, temperature, top_p);
        sequence.push_back(next_token);
        perf().n_tokens_generated.fetch_add(1, std::memory_order_relaxed);

        // Reasoning-aware output: decode to real text, then parse
        // <think>...</think> tags out of the decoded stream (the tag
        // detection only works against actual text -- it never matched
        // anything when fed stringified token IDs).
        std::string token_str = tokenizer.decode(next_token);
        int parse_result = reasoning.feed(token_str);
        if (parse_result == 0) {
            // Normal output token
            std::cout << token_str << std::flush;
        } else if (parse_result == 1) {
            // Thinking token — print dimmed (ANSI gray)
            std::cout << "\033[90m" << token_str << "\033[0m" << std::flush;
        }
        // parse_result == 2: tag consumed, don't print

        // GLM-5.2's real eos_token_id is a 3-entry array (<|endoftext|>,
        // <|user|>, <|observation|>) -- checking only the single `eos_id`
        // param means generation runs past the model's real turn-boundary
        // markers into undefined continuation territory
        // tokenizer.is_stop_token covers all of them; the `eos_id` check
        // stays as a redundant safety net.
        if (next_token == eos_id || tokenizer.is_stop_token(next_token)) break;

        // Embed the new token and run one forward step
        if (emb_it != engine.resident_layers.end()) {
            embed_token(engine, next_token, D, decode_hidden.data());
        }
        
        int pos = prompt_ids.size() + step;

        // Per-token latency breakdown: diff perf() counters across just this
        // one decode step so each line reflects that token's own cost, not
        // the cumulative run total.
        auto token_snap_before = perf().snapshot();
        auto token_wall_start = std::chrono::high_resolution_clock::now();
        // Hyper-Connections (DeepSeek-V4): same expand/collapse bracket as
        // the prefill call site above -- hc_active computed once outside
        // this loop.
        if (hc_active)
            hc_expand_embedding(engine, decode_hidden.data(), 1, D);
        layers_forward_gguf(engine, ram_mgr, prefetcher, decode_hidden.data(), 1, pos, &next_token);
        if (hc_active)
            hc_collapse_output(engine, decode_hidden.data(), 1, D);
        double token_wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - token_wall_start).count();
        auto token_snap_after = perf().snapshot();
        char token_label[32];
        snprintf(token_label, sizeof(token_label), "TOKEN %5d", step);
        PerfCounters::print_delta(token_label, token_wall_ms, token_snap_after - token_snap_before);
        check_hidden_finite(token_label, decode_hidden.data(), decode_hidden.size());
        // Real per-GPU VRAM usage (cudaMemGetInfo), sampled once per token
        // to track actual usage over the run -- distinct from the one-time
        // startup topology dump and from the logical cache-fill heatmaps
        // below (this is raw memory, those are "what's actually resident").
        if (ram_mgr.vram_cache) ram_mgr.vram_cache->log_vram_usage(token_label);

        // Replace hidden_states with the single-token output for next iteration
        hidden_states.resize(D);
        std::memcpy(hidden_states.data(), decode_hidden.data(), D * sizeof(float));
    }
    
    // prefetcher is caller-owned (started once for the whole session by
    // main.cpp) -- not stopped here.
    if (reasoning.think_tokens > 0) {
        std::cout << "\n[Reasoning: " << reasoning.think_tokens << " tokens | Output: "
                  << reasoning.output_tokens << " tokens]\n";
    }

    // Whole-request summary (this request's share only, not the program-
    // lifetime total printed by perf().dump_report() at exit) -- always
    // visible at INFO, since this is the actual answer to "how fast was
    // that" that per-token DEBUG lines don't surface on their own.
    double request_wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - request_wall_start).count();
    auto request_snap_end = perf().snapshot();
    auto request_delta = request_snap_end - request_snap_start;
    PerfCounters::print_summary("REQUEST", request_wall_ms, (int)request_delta.n_tokens_generated, request_delta);

    // Accumulate this request's own wall time into the session-wide
    // active-processing total -- deliberately separate from wall-clock
    // session_elapsed_s (main.cpp), which would otherwise silently include
    // idle time between requests (REPL waiting on the next prompt) as if
    // it were processing time. See t_active_processing's declaration.
    PerfCounters::atomic_add(perf().t_active_processing, request_wall_ms / 1000.0);
    perf().n_requests.fetch_add(1, std::memory_order_relaxed);

    // Per-layer cache heatmaps -- once per request (not per token, to avoid
    // flooding), showing which layers ended up hot in VRAM vs host RAM.
    // Requires log_level = "DEBUG" (same as print_delta's per-token lines).
    if (ram_mgr.vram_cache) ram_mgr.vram_cache->log_expert_heatmap(engine.model_file.architecture_config.n_layers);
    ram_mgr.log_host_cache_heatmap(engine);

    std::cout << "\n[Generation Complete]\n";
}