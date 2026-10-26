#pragma once
#include "engine.hpp"
#include "full_graph_model.hpp"
#include "tensor_trace.hpp"
#include "ram_manager.hpp"
#include "kernels_cpu.hpp"
#include "dense_dispatch.hpp"
#include "kquant_dequant.hpp"     // dequant_row, row_bytes (embed_token)
#include "gguf_tokenizer.hpp"
#include "perf_timer.hpp"
#include "async_prefetcher.hpp"
#include "ggml_qwen4exp_graph.hpp"
#include "ggml_glm5next_graph.hpp"
#include "mtp_speculative.hpp"
#include "shutdown.hpp"
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include "request_stats.hpp"
#include "generation_control.hpp"

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

// Diagnostic-only oracle support.  When set, write the unmodified decode
// logits (before repetition penalty and sampling) as a small binary record
// per generation step.  This is intentionally opt-in and outside the hot
// path when unset; it lets a baseline and a hybrid run be compared by max
// absolute logit delta instead of relying only on sampled text.
inline void dump_decode_logits_if_enabled(int step, const float *logits,
                                          size_t n) {
    static const char *path = std::getenv("ZEROMOE_DECODE_LOGITS_DUMP");
    if (!path || !path[0] || !logits || n == 0)
        return;

    std::ofstream out(path, step == 0
                               ? (std::ios::binary | std::ios::trunc)
                               : (std::ios::binary | std::ios::app));
    if (!out)
        return;
    const uint32_t step_u = static_cast<uint32_t>(step);
    const uint32_t n_u = static_cast<uint32_t>(n);
    out.write(reinterpret_cast<const char *>(&step_u), sizeof(step_u));
    out.write(reinterpret_cast<const char *>(&n_u), sizeof(n_u));
    out.write(reinterpret_cast<const char *>(logits),
              static_cast<std::streamsize>(n * sizeof(float)));
}

// Diagnostic-only capture of the final collapsed hidden row immediately
// before output.weight. This separates prefill/trunk drift from an
// output-projection-only difference during llama.cpp parity checks.
inline void dump_prefill_final_hidden_if_enabled(int step, const float *hidden,
                                                  size_t n) {
    static const char *path = std::getenv("ZEROMOE_PREFILL_FINAL_HIDDEN_DUMP");
    if (step != 0 || !path || !path[0] || !hidden || n == 0)
        return;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (out)
        out.write(reinterpret_cast<const char *>(hidden),
                  static_cast<std::streamsize>(n * sizeof(float)));
}

// Diagnostic-only full-vector checksum.  Hash the raw float bytes before the
// repetition penalty mutates the working logits, so baseline and teacher-
// forced runs compare the actual model output rather than sampler state.
// FNV-1a is intentionally implemented over bytes instead of using
// std::hash<float>, whose representation/contract is not a cross-process
// diagnostic format.
inline uint64_t decode_logits_fnv64(const float *logits, size_t n) {
    constexpr uint64_t offset = 1469598103934665603ULL;
    constexpr uint64_t prime = 1099511628211ULL;
    uint64_t hash = offset;
    const auto *bytes = reinterpret_cast<const unsigned char *>(logits);
    for (size_t i = 0; i < n * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= prime;
    }
    return hash;
}

inline bool decode_logits_hash_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("ZEROMOE_DECODE_LOGITS_HASH");
        return value && value[0] == '1';
    }();
    return enabled;
}

// Diagnostic-only prefill parity switch.  When enabled, process the prompt
// one token at a time through the same S==1 path used by autoregressive
// decode.  This is intentionally separate from moe_prefill_batched: that
// option controls expert regrouping inside an S>1 forward, whereas this
// switch changes the model-level prompt batch geometry so it can be compared
// with llama.cpp launched using -b 1 -ub 1.
inline bool prefill_unbatched_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("ZEROMOE_PREFILL_UNBATCHED");
        return value && value[0] == '1';
    }();
    return enabled;
}

inline void log_decode_logits_hash(int step, uint64_t hash, int sampled_token,
                                   int forced_token) {
    if (!decode_logits_hash_enabled())
        return;
    LOG_STREAM_INFO() << "[TEACHER-FORCE-LOGITS] step=" << step
                      << " sampled_token=" << sampled_token
                      << " forced_token=" << forced_token
                      << " fnv64=0x" << std::hex << hash << std::dec << "\n";
}

// Diagnostic-only teacher-forcing support.  A baseline run can write the
// sampled token sequence with ZEROMOE_GENERATED_TOKEN_FILE; a comparison
// run can then consume that sequence through
// ZEROMOE_TEACHER_FORCE_TOKEN_FILE.  The forward pass still produces and
// dumps its own logits, but the next input token is taken from the reference
// sequence so autoregressive divergence cannot hide a numerical mismatch.
inline void dump_generated_token_if_enabled(int step, int token) {
    static const char *path = std::getenv("ZEROMOE_GENERATED_TOKEN_FILE");
    if (!path || !path[0]) {
        return;
    }

    std::ofstream out(path, step == 0 ? std::ios::trunc : std::ios::app);
    if (out) {
        out << token << '\n';
    }
}

inline const std::vector<int> & teacher_force_tokens_if_enabled(int step) {
    // A multi-request diagnostic needs a distinct authoritative token stream
    // per prompt. The original single-file mode remains unchanged.
    static const char *directory = std::getenv("ZEROMOE_TEACHER_FORCE_REQUEST_DIR");
    if (directory && directory[0]) {
        static size_t request = 0;
        static std::vector<int> request_tokens;
        if (step == 0) {
            const std::string path = std::string(directory) + "/request-" + std::to_string(request++) + ".tokens";
            std::ifstream in(path);
            if (!in) throw std::runtime_error("cannot open teacher-forced request tokens: " + path);
            request_tokens.clear();
            int token = 0;
            while (in >> token) request_tokens.push_back(token);
            if (request_tokens.empty()) throw std::runtime_error("empty teacher-forced request: " + path);
        }
        return request_tokens;
    }
    static const std::vector<int> tokens = [] {
        std::vector<int> result;
        const char *path = std::getenv("ZEROMOE_TEACHER_FORCE_TOKEN_FILE");
        if (!path || !path[0]) {
            return result;
        }

        std::ifstream in(path);
        int token = 0;
        while (in >> token) {
            result.push_back(token);
        }
        return result;
    }();
    return tokens;
}

inline bool teacher_force_skip_sampling_enabled() {
    static const bool enabled = [] {
        const char *value =
            std::getenv("ZEROMOE_TEACHER_FORCE_SKIP_SAMPLING");
        return value && value[0] == '1';
    }();
    return enabled;
}
// token_ids: the S current input token ids that produced hidden_states'
// S rows (one per row, same order) -- needed by DeepSeek-V4 hash-routed MoE
// layers (moe_forward_gguf's ffn_gate_tid2eid lookup, per-token, per-layer
// direct token-id->expert-id table). nullptr/ignored for every other
// family and for non-hash-routed layers within a DeepSeek-V4 file.
extern void layers_forward_gguf(StreamingEngine& engine, RAMSafetyManager& ram_mgr,
                                LockFreePrefetcher& async_io_queue, float* hidden_states, int S, int pos_base,
                                const int* token_ids);
// TEMPORARY DIAGNOSTIC (2026-09-12) -- see g_diag_layer_loop_total_ms's own
// comment in execution_layer.cpp. Accumulates layers_forward_gguf's
// per-layer loop's OWN full wall-clock time (every layer, across the whole
// forward call) so it can be compared directly against the [REQUEST] line's
// wall_ms and against the sum of that line's named PERF_SCOPE buckets.
extern void diag_reset_layer_loop_total();
extern double diag_get_layer_loop_total_ms();
extern double diag_get_attn_call_ms();
extern double diag_get_ffn_call_ms();
extern double diag_get_decode_layer_loop_ms();
extern double diag_get_decode_attn_call_ms();
extern double diag_get_decode_ffn_call_ms();
extern double diag_get_decode_pilot_ms();
extern double diag_get_decode_hc_ms();
extern double diag_get_decode_hc_pre_ms();
extern double diag_get_decode_hc_attn_pre_ms();
extern double diag_get_decode_hc_ffn_pre_ms();
extern double diag_get_decode_hc_post_ms();
extern double diag_get_decode_layer_input_ms();
extern double diag_get_decode_qwen_cpu_graph_ms();
extern double diag_get_decode_qwen_shared_submit_ms();
extern double diag_get_decode_qwen_cpu_offload_wait_ms();
extern double diag_get_decode_qwen_cpu_offload_total_ms();
extern double diag_get_decode_qwen_cpu_offload_fence_wait_ms();
extern double diag_get_decode_qwen_cpu_offload_h2d_ms();
extern double diag_get_decode_qwen_cpu_offload_ring_reuse_wait_ms();
extern uint64_t diag_get_decode_qwen_cpu_offload_cold_slots();
extern uint64_t diag_get_decode_qwen_cpu_offload_worker_calls();
extern uint64_t diag_get_decode_qwen_cpu_offload_worker_successes();
extern uint64_t diag_get_decode_qwen_cpu_offload_worker_failures();
extern uint64_t diag_get_decode_qwen_cpu_offload_fenced_calls();
extern uint64_t diag_get_decode_qwen_cpu_offload_fence_failures();
// TEMPORARY DIAGNOSTIC (2026-09-12) -- phase-by-phase bisection of
// dsv41_attention_gguf's ratio!=0 branch (src/attention_layer.cpp), the
// function g_diag_attn_call_ms proved costs ~20-21s/request. See the
// g_diag_dsv41_* accumulators' own comment there for what each phase
// covers and why the conditional ones (owns_kv/owns_index) also track a
// call count.
extern void diag_reset_dsv41_attn_phases();
extern double diag_get_dsv41_calls();
extern double diag_get_dsv41_qrope_ms();
extern double diag_get_dsv41_kvrope_ms();
extern double diag_get_dsv41_ownskv_ms();
extern double diag_get_dsv41_ownskv_calls();
extern double diag_get_dsv41_indexer_qk_ms();
extern double diag_get_dsv41_indexer_score_ms();
extern double diag_get_dsv41_ownsindex_calls();
extern double diag_get_dsv41_perhead_ms();
extern double diag_get_dsv41_outproj_ms();
extern double diag_get_dsv41_woa_dequant_ms();
extern double diag_get_dsv41_woa_dequant_calls();
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

// Keep the historical default unchanged, but allow controlled benchmark runs
// to match another runtime's sampling policy.  In particular, applying a
// 1.3 penalty to the prompt's special tokens can change the very first decode
// token, so it must not be silently mixed into a llama.cpp parity comparison.
// This is intentionally an environment override rather than a model-wide
// default/config change: existing models retain exactly the old behavior.
inline float zeromoe_repetition_penalty() {
    static const float penalty = [] {
        // Default matches llama.cpp's official `repeat_penalty` behavior.
        // default (1.0 == disabled). The prior default (1.3) does not exist
        // upstream and was confirmed as the root cause of a real, reproducible
        // switch-to-non-English defect on Qwen4Exp/UD-IQ1_M: the penalty
        // window (see its call site below) included the PROMPT's own
        // tokens, so a common word merely appearing once in the user's own
        // question (e.g. "and") got its logit suppressed in later decode
        // steps -- confirmed via raw pre-penalty logit dumps
        // (ZEROMOE_DECODE_LOGITS_DUMP) showing the correct English
        // continuation winning by a clear margin pre-penalty, then losing to
        // a non-English token post-penalty, and via llama.cpp itself
        // (same weights, same quant, same prompt, temp=0) staying coherent
        // throughout where ZeroMoE did not. Still available and
        // configurable via ZEROMOE_REPETITION_PENALTY for anyone who wants
        // the old anti-loop behavior back.
        constexpr float default_penalty = 1.0f;
        const char* text = std::getenv("ZEROMOE_REPETITION_PENALTY");
        if (!text || !*text) return default_penalty;

        char* end = nullptr;
        const float value = std::strtof(text, &end);
        if (end == text || *end != '\0' || !std::isfinite(value) || value <= 0.0f) {
            LOG_WARN("[GEN] ignoring invalid ZEROMOE_REPETITION_PENALTY='%s'\n", text);
            return default_penalty;
        }
        LOG_INFO("[GEN] ZEROMOE_REPETITION_PENALTY=%.6g (default=%.6g)\n",
                 value, default_penalty);
        return value;
    }();
    return penalty;
}

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
// Fills one zeromoe::stats::RequestStats for the request and publishes it when the request leaves generate_text(),
// by any path. A request that returns early (error, refused graph) is published with status "error".
struct RequestStatsScope {
    zeromoe::stats::RequestStats s;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    explicit RequestStatsScope(const std::string &family) {
        auto &sink = zeromoe::stats::Sink::instance();
        sink.take_notes();  // drop notes left by startup work; only this request's notes belong in its record
        s.seq = sink.next_seq();
        s.start_unix_ms = zeromoe::stats::unix_ms_now();
        s.model = sink.model();
        s.family = family;
    }
    ~RequestStatsScope() {
        auto &sink = zeromoe::stats::Sink::instance();
        if (s.total_ms < 0)
            s.total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
        s.end_unix_ms = zeromoe::stats::unix_ms_now();
        s.extra = sink.take_notes();
        const std::string json = sink.publish(s);
        LOG_INFO("[STATS] %s\n", json.c_str());
    }
    RequestStatsScope(const RequestStatsScope &) = delete;
    RequestStatsScope &operator=(const RequestStatsScope &) = delete;
};

inline void generate_text(StreamingEngine& engine, RAMSafetyManager& ram_mgr, GGUFTokenizer& tokenizer,
                   LockFreePrefetcher& prefetcher,
                   const std::vector<int>& prompt_ids, int max_new_tokens, int eos_id,
                   float temperature = 0.7f, float top_p = 0.90f,
                   GenerationControl *control = nullptr) {
    auto cancelled = [&] {
        const bool stop = g_shutdown_requested.load(std::memory_order_relaxed) ||
                          (control && control->cancelled());
        if (stop && control) control->interrupted = true;
        return stop;
    };
    if (cancelled()) return;
    engine.qwen4exp_full_graph_failure.clear();
    int max_ctx = prompt_ids.size() + max_new_tokens;
    ram_mgr.calculate_safe_cache_capacity(engine, max_ctx);

    // Benchmark-only mode: max_new_tokens is normally an upper bound because
    // EOS/turn-boundary tokens terminate generation early.  For controlled
    // throughput comparisons, allow the caller to request exactly the full
    // configured decode length.  This is opt-in, applies to the normal
    // non-MTP loop only, and leaves normal inference behavior unchanged.
    const bool bench_ignore_stop = [] {
        const char *env = std::getenv("ZEROMOE_BENCH_IGNORE_STOP");
        return env && env[0] == '1';
    }();
    // llama-bench measures token generation without decoding/printing every
    // token or sampling device memory on every step. This opt-in mode keeps
    // ZeroMoE's compute path and request summary while removing that CLI
    // streaming work from apples-to-apples throughput measurements.
    const bool bench_silent = control || [] {
        const char *env = std::getenv("ZEROMOE_BENCH_SILENT");
        return env && env[0] == '1';
    }();
    const bool token_loop_diag_enabled = [] {
        const char *env = std::getenv("ZEROMOE_TOKEN_LOOP_DIAG");
        return env && env[0] == '1';
    }();
    if (bench_ignore_stop) {
        LOG_STREAM_INFO() << "[BENCH] ZEROMOE_BENCH_IGNORE_STOP=1: "
                             "EOS/stop tokens will not end this bounded decode\n";
    }
    if (bench_silent) {
        LOG_STREAM_INFO() << "[BENCH] ZEROMOE_BENCH_SILENT=1: token text output, "
                             "per-token VRAM sampling, and token diagnostics are disabled\n";
    }

    // TEMPORARY DIAGNOSTIC (2026-09-12, deepseek-v41-decode-speed-session):
    // bisecting the confirmed ~35-50% [REQUEST]-wall-vs-named-buckets gap
    // (present in both prefill and decode, ruled OUT: HC-drain hooks --
    // measured microseconds via [QWEN4EXP-DRAIN], and Sampler::sample's
    // O(V log V) sort -- temperature=0.0 takes the cheap max_element path).
    // Remaining candidates measured directly below instead of guessed at:
    // lm-head norm+proj+NaN-scan+repetition-penalty, and log_vram_usage's
    // per-token cudaSetDevice+cudaMemGetInfo (runs unconditionally every
    // token regardless of log level -- only the string format is
    // LOG_DEBUG-gated, the CUDA calls are not). Remove once the real
    // source is found and fixed.
    double diag_lmhead_ms = 0.0, diag_vramlog_ms = 0.0, diag_vramlog_ms_prev = 0.0;
    // TEMPORARY DIAGNOSTIC (2026-09-12) -- reset layers_forward_gguf's
    // per-layer-loop wall-clock accumulator once per request, so the total
    // printed alongside [REQUEST] below reflects only THIS request's
    // prefill call plus all of its decode-step calls. See
    // g_diag_layer_loop_total_ms's own comment (execution_layer.cpp).
    diag_reset_layer_loop_total();
    // TEMPORARY DIAGNOSTIC (2026-09-12) -- reset dsv41_attention_gguf's
    // per-phase accumulators once per request too, same rationale as
    // diag_reset_layer_loop_total() above. See g_diag_dsv41_* comment
    // (attention_layer.cpp). No-op (extra function call only) for every
    // non-DeepSeek-V4.1 family, since the accumulators are only ever
    // incremented from inside dsv41_attention_gguf's ratio!=0 branch.
    diag_reset_dsv41_attn_phases();
    // TEMPORARY DIAGNOSTIC (2026-09-12) -- t_hc_hooks/t_layer_input_hook are
    // real PerfCounters atomics but are NOT part of Snapshot/print_summary's
    // per-request accounting (only ever printed as whole-session totals by
    // dump_report's "Per-layer hooks:" line) -- captured here by hand so
    // this request's own share is visible alongside the other diagnostics.
    double diag_hc_hooks_start = perf().t_hc_hooks.load();
    double diag_layer_input_hook_start = perf().t_layer_input_hook.load();

    // Snapshot for the whole-request latency breakdown, printed when this
    // request finishes (see bottom of function).
    auto request_snap_start = perf().snapshot();
    auto request_wall_start = std::chrono::high_resolution_clock::now();
    RequestStatsScope stats_scope(engine.family_ops.graph_model ? engine.family_ops.graph_model->name() : "");

    // ---- Prompt (prefix) state cache -------------------------------------
    // Restricted to the two families whose carried state is fully enumerated
    // in StreamingEngine::PromptStateCache. Anything else re-prefills as
    // before: a family with carried state we have NOT enumerated would
    // silently produce wrong output, not fail, so the default is to refuse.
    const ModelFamily pc_fam = detect_model_family(engine.model_file);
    const bool pc_supported = (pc_fam == ModelFamily::Qwen4Exp ||
                               pc_fam == ModelFamily::GLM5NEXT);
    auto pc_save = [&](const std::vector<int>& toks) {
        // The live GLM5NEXT recurrent matrix normally remains on CUDA across
        // every KDA layer call. Snapshot it once before copying prompt state
        // into the host-side prefix cache.
        sync_glm5next_kda_gpu_states(engine);
        auto& pc = engine.prompt_state_cache;
        pc.tokens = toks;
        pc.recur_state = engine.qwen4exp_recur_state;
        pc.conv_history = engine.qwen4exp_conv_history;
        pc.ple_conv_history = engine.qwen4exp_ple_conv_history;
        pc.ple_tok_hist = engine.qwen4exp_ple_tok_hist;
        pc.ple_next_pos = engine.qwen4exp_ple_next_pos;
        pc.glm_recur_state = engine.glm5next_recur_state;
        pc.glm_conv_history = engine.glm5next_conv_history;
        pc.valid = true;
    };
    auto pc_restore = [&]() {
        const auto& pc = engine.prompt_state_cache;
        engine.qwen4exp_recur_state = pc.recur_state;
        engine.qwen4exp_conv_history = pc.conv_history;
        engine.qwen4exp_ple_conv_history = pc.ple_conv_history;
        engine.qwen4exp_ple_tok_hist = pc.ple_tok_hist;
        engine.qwen4exp_ple_next_pos = pc.ple_next_pos;
        engine.glm5next_recur_state = pc.glm_recur_state;
        engine.glm5next_conv_history = pc.glm_conv_history;
        // The restored host vectors are now authoritative; the next KDA call
        // must seed its device-resident matrix from them.
        invalidate_glm5next_kda_gpu_states(engine);
    };

    // A hit needs the cached tokens to be a STRICT prefix of this prompt:
    // only the end-of-prompt state was retained, so an earlier divergence is
    // unusable, and at least one token must remain to produce logits from.
    int pc_reuse = 0;
    if (engine.prompt_cache && pc_supported && engine.prompt_state_cache.valid) {
        const auto& ct = engine.prompt_state_cache.tokens;
        // <= not <: an EXACT repeat (ct.size()==N) is the benchmark case and
        // is handled by replaying the stored last hidden row instead of
        // running a zero-token forward pass.
        if (!ct.empty() && ct.size() <= prompt_ids.size() &&
            std::equal(ct.begin(), ct.end(), prompt_ids.begin()) &&
            !engine.prompt_state_cache.last_hidden.empty())
            pc_reuse = (int)ct.size();
    }
    if (engine.prompt_cache && !pc_supported)
        LOG_STREAM_WARN() << "[PROMPT-CACHE] disabled: this model family's carried "
                             "recurrent/conv state is not enumerated in "
                             "PromptStateCache -- reusing it could silently corrupt output.\n";
    if (pc_reuse > 0) {
        pc_restore();
        LOG_STREAM_INFO() << "[PROMPT-CACHE] hit: reusing " << pc_reuse << " of "
                          << prompt_ids.size() << " prompt token(s); prefilling "
                          << (prompt_ids.size() - pc_reuse) << "\n";
        std::cout << "\n[Prompt cache] reusing " << pc_reuse << "/" << prompt_ids.size()
                  << " tokens" << std::flush;
        perf().n_prompt_cache_hits.fetch_add(1, std::memory_order_relaxed);
        perf().n_prompt_cache_tokens_reused.fetch_add((uint64_t)pc_reuse, std::memory_order_relaxed);
    } else if (engine.prompt_cache && pc_supported) {
        perf().n_prompt_cache_misses.fetch_add(1, std::memory_order_relaxed);
    }
    if (control) control->cached_tokens = pc_reuse;

    std::vector<int> sequence = prompt_ids;
    Sampler sampler;
    int D = engine.model_file.architecture_config.hidden;
    int V = engine.model_file.architecture_config.vocab;
    const size_t pc_n_new = prompt_ids.size() - (size_t)pc_reuse;

    // The model/cache allocations are not the only host consumers during a
    // request.  Prompt hidden rows, decode buffers, sampler work arrays, and
    // the layer scratch vectors are ordinary STL allocations, so they do not
    // individually pass through the RAM-slab allocator.  Reserve a bounded
    // request envelope before creating them; MTP takes a second, more exact
    // transient reservation while it is running.  This keeps both paths
    // inside host_ram_gb without charging prompt_hidden twice in MTP.
    size_t request_bytes = 0;
    bool request_size_overflow = false;
    const auto add_request_bytes = [&](size_t elems, size_t elem_bytes) {
        if (request_size_overflow ||
            elems > std::numeric_limits<size_t>::max() / elem_bytes) {
            request_size_overflow = true;
            return;
        }
        const size_t bytes = elems * elem_bytes;
        if (bytes > std::numeric_limits<size_t>::max() - request_bytes)
            request_size_overflow = true;
        else
            request_bytes += bytes;
    };
    // Pass matrix dimensions separately so the size check also covers the
    // rows*columns multiplication itself.  The request envelope is the last
    // line of defence before ordinary STL vectors are constructed; an
    // invalid/overflowed model shape must fail closed rather than wrap to a
    // tiny allocation and bypass host_ram_gb.
    const auto add_request_matrix = [&](size_t rows, size_t cols) {
        if (rows == 0 || cols == 0) return;
        if (request_size_overflow ||
            cols > std::numeric_limits<size_t>::max() / sizeof(float) ||
            rows > std::numeric_limits<size_t>::max() /
                         (cols * sizeof(float))) {
            request_size_overflow = true;
            return;
        }
        add_request_bytes(rows * cols, sizeof(float));
    };
    add_request_matrix(std::max<size_t>(1, pc_n_new), (size_t)D);
    add_request_matrix(2, (size_t)D); // decode hidden + normed
    add_request_matrix(3, (size_t)V); // logits + sampler probs/index room
    constexpr size_t kRequestScratchMargin = 512ull * 1024ull * 1024ull;
    if (request_bytes > std::numeric_limits<size_t>::max() -
                            kRequestScratchMargin)
        request_size_overflow = true;
    else
        request_bytes += kRequestScratchMargin;
    if (request_size_overflow ||
        !MemoryGuard::try_reserve_host_bytes(request_bytes)) {
        LOG_STREAM_WARN() << "[MEM-GUARD] request host envelope refused: need ~"
                          << (request_size_overflow ? -1.0
                                                     : (double)request_bytes / 1.0e9)
                          << " GB, reserved="
                          << (double)MemoryGuard::reserved_host_bytes() / 1.0e9
                          << " GB, budget="
                          << (double)MemoryGuard::host_budget_bytes() / 1.0e9
                          << " GB; request not started\n";
        return;
    }
    struct RequestHostReservation {
        size_t bytes;
        ~RequestHostReservation() { MemoryGuard::release_host_bytes(bytes); }
    } request_host_reservation{request_bytes};

    // Embed the prompt tokens
    std::vector<float> hidden_states(pc_n_new * D, 0.0f);
    std::vector<float> logits(V);
    engine.qwen4exp_full_graph_logits_ready = false;
    auto emb_it = engine.resident_layers.find("token_embd.weight");
    if (emb_it != engine.resident_layers.end() && emb_it->second.host_ptr) {
        LOG_STREAM_INFO() << "[EMBED] token_embd.weight type=" << (unsigned)emb_it->second.meta.type
                           << " (" << gpu_type_name(emb_it->second.meta.type) << ")\n";
        for (size_t t = 0; t < pc_n_new; ++t) {
            if (cancelled()) return;
            int tok = prompt_ids[(size_t)pc_reuse + t];
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

    // llama.cpp builds its complete model graph before entering the measured
    // prefill.  The prompt IDs are needed only by Qwen's PLE path; graph
    // construction and static weight upload are outside PERF_START.
    const bool unbatched_prefill = prefill_unbatched_enabled();
    const FullGraphModel *graph_model = engine.family_ops.graph_model;
    // Tokens run as one batch through the prefill forward below. A family whose prefill graph has
    // bounded buffers (max_prefill_tokens) runs only the first N tokens through the graph and the
    // rest as S=1 steps; it applies only once the graph actually built, so a failed build keeps
    // the whole-prompt fallback.
    auto supports_continuation_for_tail = [](const FullGraphModel *m, const StreamingEngine &e) {
        return m && m->supports_prefill_continuation(e);
    };
    size_t pc_n_batched = pc_n_new;
    int pc_graph_cap = 0;
    // The KV/state caches hold engine.max_t positions (the config's context_size). A longer prompt would write past them,
    // so it is refused here with the value to raise instead of being run.
    if (engine.max_t > 0 && prompt_ids.size() + 1 > (size_t)engine.max_t) {
        LOG_STREAM_ERROR() << "[PREFILL] the prompt has " << prompt_ids.size() << " tokens but the configured context_size "
                           << "holds " << engine.max_t << " positions; raise context_size in the config (the KV/state "
                           << "caches are sized from it). Refusing the request.\n";
        if (!bench_silent)
            std::cerr << "\n[ERROR] prompt of " << prompt_ids.size() << " tokens exceeds context_size="
                      << engine.max_t << "; raise context_size in the config\n";
        return;
    }
    if (pc_n_new > 1 && pc_reuse == 0 && !unbatched_prefill &&
        graph_model &&
        graph_model->enabled(engine, ModelGraphPhase::Prefill)) {
        size_t pc_n_graph = pc_n_new;
        int cap = graph_model->max_prefill_tokens(engine);
        if (cap > 1 && (size_t)cap < pc_n_new) {
            pc_n_graph = (size_t)cap;
            // A tail smaller than the family's minimum batch would be stepped one token at a time (2083 tokens at a batch
            // of 2048: 35 stepped tokens, 14.3 s instead of ~10 s); shorten the first batch so the tail is a real batch.
            const size_t min_tail = (size_t)std::max(8, graph_model->min_prefill_batch());
            if (supports_continuation_for_tail(graph_model, engine) && pc_n_new - pc_n_graph < min_tail &&
                pc_n_new > min_tail + 64)
                pc_n_graph = ((pc_n_new - min_tail) / 8) * 8;
        }
        auto build_first = [&]() {
            const ModelGraphRequest request{
                ModelGraphPhase::Prefill,
                (int)pc_n_graph,
                (int)pc_reuse,
                prompt_ids.data() + pc_reuse,
                nullptr,
            };
            return graph_model->build(engine, request);
        };
        bool graph_built = build_first();
        // A batch whose workspace does not fit is retried at half the size while the family can continue a prompt
        // from its earlier chunk; the remaining chunks then use the smaller size too.
        int halvings = 0;
        const int min_batch = std::max(8, graph_model->min_prefill_batch());
        while (!graph_built && pc_n_graph > (size_t)min_batch && ++halvings <= 6 &&
               (graph_model->supports_prefill_continuation(engine) ||
                graph_model->suggested_prefill_batch(engine) >= min_batch)) {
            const int hint = graph_model->suggested_prefill_batch(engine);
            pc_n_graph = (hint >= min_batch && (size_t)hint < pc_n_graph) ? (size_t)hint
                                                                           : std::max<size_t>(min_batch, ((pc_n_graph / 2) / 8) * 8);
            cap = (int)pc_n_graph;
            LOG_STREAM_INFO() << "[PREFILL] graph declined; retrying with a batch of " << pc_n_graph << " tokens\n";
            graph_built = build_first();
        }
        pc_graph_cap = cap;
        // An explicitly requested DeepSeek graph must fail visibly.
        if (!engine.qwen4exp_full_graph_failure.empty())
            return;
        if (!graph_built &&
            (engine.qwen4exp_full_graph_required || graph_model->graph_only())) {
            // Smaller batches were tried above (memory); the layered forward cannot take over here: with the graph-only
            // expert placement it finds no staged expert blocks and would silently drop their tokens.
            engine.qwen4exp_full_graph_failure =
                std::string(graph_model->name()) + " prefill graph construction failed";
            LOG_ERROR("[GRAPH-POLICY] %s even with the smallest batch; aborting request\n",
                      engine.qwen4exp_full_graph_failure.c_str());
            return;
        }
        if (graph_built) {
            pc_n_batched = pc_n_graph;
        } else if (!bench_silent) {
            std::cerr << "\n[WARN] prefill graph declined (see the engine log); falling back to the "
                         "legacy whole-prompt prefill, which is much slower for long prompts\n";
        }
    } else if (engine.qwen4exp_full_graph_required && pc_n_new > 1 &&
               !unbatched_prefill && pc_reuse == 0) {
        engine.qwen4exp_full_graph_failure =
            "prefill graph operation is unavailable or disabled";
        LOG_ERROR("[QWEN4EXP-GRAPH-POLICY] %s; aborting request\n",
                  engine.qwen4exp_full_graph_failure.c_str());
        return;
    }

    if (!bench_silent) {
        std::cout << "\n[Prefill] Processing " << prompt_ids.size() << " tokens..." << std::flush;
        if (const char *dump = std::getenv("ZEROMOE_DUMP_PROMPT_TOKENS"); dump && dump[0]) {
            std::ofstream pf(dump, std::ios::trunc);
            for (auto id : prompt_ids) pf << id << '\n';
        }
    }
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
    if (pc_n_new == 0) {
        // Exact repeat: every layer's state is already exactly what it was at
        // the end of this same prompt, so there is no token left to process --
        // replay the stored final hidden row. Bit-identical by construction.
        hidden_states.assign(engine.prompt_state_cache.last_hidden.begin(),
                             engine.prompt_state_cache.last_hidden.end());
    } else {
        // llama.cpp's -b 1 -ub 1 parity mode: run each prompt token through
        // the existing S==1 path.  This preserves the normal recurrent/KV
        // update order while avoiding the batched HC projection and batched
        // activation-quantization kernels.  The default prompt path below is
        // unchanged when ZEROMOE_PREFILL_UNBATCHED is not enabled.
        if (unbatched_prefill && pc_n_new > 1) {
            LOG_STREAM_INFO() << "[PREFILL] unbatched parity mode: "
                              << pc_n_new << " x S=1 forward calls\n";
            for (size_t t = 0; t < pc_n_new; ++t) {
                if (cancelled()) return;
                float *row = hidden_states.data() + t * (size_t)D;
                if (hc_active)
                    hc_expand_embedding(engine, row, 1, D);
                // pos_base = pc_reuse + t: cached-prefix KV rows remain in
                // place and each prompt token advances the recurrent state.
                layers_forward_gguf(engine, ram_mgr, prefetcher, row, 1,
                                    pc_reuse + (int)t,
                                    prompt_ids.data() + pc_reuse + t);
                if (!engine.qwen4exp_full_graph_failure.empty())
                    return;
                if (hc_active && !engine.full_graph_output_ready && !engine.qwen4exp_full_graph_output_ready &&
                    !engine.glm5next_full_graph_output_ready &&
                    !engine.deepseek_v4_full_graph_output_ready)
                    hc_collapse_output(engine, row, 1, D);
                engine.full_graph_output_ready = false;
                engine.qwen4exp_full_graph_output_ready = false;
                engine.glm5next_full_graph_output_ready = false;
                engine.deepseek_v4_full_graph_output_ready = false;
            }
        } else {
            if (hc_active)
                hc_expand_embedding(engine, hidden_states.data(), (int)pc_n_batched, D);
            // pos_base = pc_reuse: the reused prefix's dense KV rows are already in
            // place (position-indexed and untouched since they were written), so the
            // causal attention below still sees the full context.
            engine.qwen4exp_full_graph_logits_target = logits.data();
            engine.qwen4exp_full_graph_logits_target_size = V;
            layers_forward_gguf(engine, ram_mgr, prefetcher, hidden_states.data(), pc_n_batched,
                                pc_reuse, prompt_ids.data() + pc_reuse);
            engine.qwen4exp_full_graph_logits_target = nullptr;
            engine.qwen4exp_full_graph_logits_target_size = 0;
            if (!engine.qwen4exp_full_graph_failure.empty())
                return;
            if (hc_active && !engine.full_graph_output_ready && !engine.qwen4exp_full_graph_output_ready &&
                !engine.glm5next_full_graph_output_ready &&
                !engine.deepseek_v4_full_graph_output_ready)
                hc_collapse_output(engine, hidden_states.data(), (int)pc_n_batched, D);
            engine.full_graph_output_ready = false;
            engine.qwen4exp_full_graph_output_ready = false;
            engine.glm5next_full_graph_output_ready = false;
            engine.deepseek_v4_full_graph_output_ready = false;

            // Tokens beyond the bounded graph batch: the same S=1 steps the unbatched parity mode
            // uses (the decode graph reads the KV/recurrent state the batched part just wrote).
            // Further chunks through the graph while the family can continue from the state the previous chunk left;
            // whatever it declines (context beyond the sparse top-k, a failed build) is stepped below.
            if (pc_n_batched < pc_n_new && graph_model &&
                graph_model->supports_prefill_continuation(engine)) {
                const int cap = std::max(1, pc_graph_cap > 0 ? pc_graph_cap : graph_model->max_prefill_tokens(engine));
                while (pc_n_batched < pc_n_new) {
                    if (cancelled()) return;
                    const size_t remaining = pc_n_new - pc_n_batched;
                    size_t take = std::min<size_t>((size_t)cap, remaining);
                    // Keep the last chunk at 64+ tokens (smaller graph batches take the slow cold-expert path) and every
                    // boundary a multiple of 8 (the indexer's pooling blocks).
                    const int min_chunk = std::max(8, graph_model->min_prefill_batch());
                    if (remaining > (size_t)cap && remaining - (size_t)cap < (size_t)min_chunk)
                        take = ((remaining - (size_t)min_chunk) / 8) * 8;
                    int n = (int)take;
                    if (n < min_chunk)
                        break;
                    const int pos = pc_reuse + (int)pc_n_batched;
                    const ModelGraphRequest request{ModelGraphPhase::Prefill, n, pos,
                                                    prompt_ids.data() + pos, nullptr};
                    int n_run = n;
                    ModelGraphRequest run_request = request;
                    if (!graph_model->build(engine, request)) {
                        // The free memory moved since the first batch: retry once with the size the family now suggests.
                        const int hint = graph_model->suggested_prefill_batch(engine);
                        if (hint < min_chunk || hint >= n)
                            break;
                        n_run = hint;
                        run_request = ModelGraphRequest{ModelGraphPhase::Prefill, n_run, pos, prompt_ids.data() + pos, nullptr};
                        if (!graph_model->build(engine, run_request))
                            break;
                        pc_graph_cap = n_run;
                    }
                    n = n_run;
                    float *rows = hidden_states.data() + pc_n_batched * (size_t)D;
                    if (hc_active)
                        hc_expand_embedding(engine, rows, n, D);
                    layers_forward_gguf(engine, ram_mgr, prefetcher, rows, n, pos, prompt_ids.data() + pos);
                    if (!engine.qwen4exp_full_graph_failure.empty())
                        return;
                    const bool graph_ran = engine.full_graph_output_ready || engine.glm5next_full_graph_output_ready ||
                                           engine.qwen4exp_full_graph_output_ready ||
                                           engine.deepseek_v4_full_graph_output_ready;
                    if (hc_active && !graph_ran)
                        hc_collapse_output(engine, rows, n, D);
                    engine.full_graph_output_ready = false;
                    engine.qwen4exp_full_graph_output_ready = false;
                    engine.glm5next_full_graph_output_ready = false;
                    engine.deepseek_v4_full_graph_output_ready = false;
                    pc_n_batched += (size_t)n; // the chunk is done either way (layered path advances state too)
                    if (!graph_ran)
                        break;
                }
            }
            if (pc_n_batched < pc_n_new) {
                LOG_STREAM_INFO() << "[PREFILL] graph batch covers the first " << pc_n_batched
                                  << " tokens; stepping the remaining "
                                  << (pc_n_new - pc_n_batched) << " one token at a time\n";
                for (size_t t = pc_n_batched; t < pc_n_new; ++t) {
                    if (cancelled()) return;
                    float *row = hidden_states.data() + t * (size_t)D;
                    if (hc_active)
                        hc_expand_embedding(engine, row, 1, D);
                    layers_forward_gguf(engine, ram_mgr, prefetcher, row, 1,
                                        pc_reuse + (int)t,
                                        prompt_ids.data() + pc_reuse + t);
                    if (!engine.qwen4exp_full_graph_failure.empty())
                        return;
                    if (hc_active && !engine.full_graph_output_ready && !engine.qwen4exp_full_graph_output_ready &&
                        !engine.glm5next_full_graph_output_ready &&
                        !engine.deepseek_v4_full_graph_output_ready)
                        hc_collapse_output(engine, row, 1, D);
                    engine.full_graph_output_ready = false;
                    engine.qwen4exp_full_graph_output_ready = false;
                    engine.glm5next_full_graph_output_ready = false;
                    engine.deepseek_v4_full_graph_output_ready = false;
                }
            }
        }
    }
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
    // or a worker-capacity gap (hinted, but the main thread's own dispatch
    // got there before any prefetch worker did)" -- see PerfCounters::
    // Snapshot's own comment on these fields. This whole line is scoped
    // to the prefill call ONLY (a Snapshot delta taken right before/after
    // it, see prefill_snap_start/prefill_snap_end above) -- so every count
    // below is real prefill activity, even though "by-main-thread" (below)
    // is labeled by CALLER THREAD, not generation phase (see RamClaimCaller's
    // own comment, engine.hpp, for why -- it fires from the SAME per-token
    // dispatch loop during both prefill and decode, so the label alone
    // can't tell you which phase; reading it from THIS [PREFILL]-scoped
    // line is what tells you that). by-main-thread means the main thread
    // itself won the claim here, during prefill (no prefetch worker had
    // even started on it yet); by-prefetch means a worker won it first (the
    // intended case). A high by-main-thread share alongside a real avg wait
    // time on the SEPARATE "Unified RAM wait" counter (dump_report(),
    // whole-run) would point at prefetch_worker_threads being undersized
    // relative to prefill's own hint volume, not a coverage gap in what
    // gets hinted.
    LOG_STREAM_INFO() << "[PREFILL] RAM claim: " << prefill_delta.n_ram_claim_hit << " hit / "
                       << prefill_delta.n_ram_claim_by_main_thread << " by-main-thread (raced ahead of any "
                          "prefetch worker -- a real disk read on the main thread, phase_b's own cost) / "
                       << prefill_delta.n_ram_claim_by_prefetch << " by-prefetch (a worker won the claim "
                          "first, as intended), " << prefill_delta.n_ram_wait_inflight_main_thread
                       << " main-thread wait(s) for an in-flight prefetch, avg "
                       << (prefill_delta.n_ram_wait_inflight_main_thread > 0
                               ? (prefill_delta.t_ram_wait_inflight_ms_sum / prefill_delta.n_ram_wait_inflight_main_thread)
                               : 0.0)
                       << "ms\n";
    check_hidden_finite("after prefill", hidden_states.data(), hidden_states.size());
    if (const char *dump = std::getenv("ZEROMOE_DUMP_LAST_HIDDEN"); dump && dump[0] && hidden_states.size() >= (size_t)D) {
        std::ofstream hf(dump, std::ios::binary | std::ios::trunc);
        hf.write(reinterpret_cast<const char *>(hidden_states.data() + hidden_states.size() - D), (std::streamsize)D * sizeof(float));
    }
    // Snapshot the end-of-prompt state for the next request. Saved AFTER
    // prefill and before any decode token, so the cache always corresponds to
    // exactly this prompt -- decode writes dense KV at positions >= N and
    // advances the recurrent state, neither of which may be folded in here.
    if (cancelled()) return;
    if (engine.prompt_cache && pc_supported) {
        pc_save(prompt_ids);
        engine.prompt_state_cache.last_hidden.assign(
            hidden_states.end() - D, hidden_states.end());
    }

    // Snapshot VRAM slot occupancy RIGHT HERE, at the exact prefill/decode
    // boundary -- directly comparable against the SAME [VRAM-SLOTS] line
    // printed again at shutdown (report_slot_summary, gpu_topology.hpp),
    // to see how much filling happened during prefill (startup preload +
    // Phase 2's warm-up) vs. during decode's own reactive promotions
    // afterward. Previously the only way to see prefill's own contribution
    // was to infer it indirectly.
    // Exact decode-only timing boundary.  The legacy aggregate Rates line
    // subtracts t_prefill_total from the whole request wall time, which also
    // contains prompt embedding and full-graph setup before PERF_START.
    const auto decode_boundary_start = std::chrono::high_resolution_clock::now();
    stats_scope.s.prompt_tokens = static_cast<int>(prompt_ids.size());
    stats_scope.s.prompt_cached_tokens = pc_reuse;
    stats_scope.s.prefill_ms = prefill_wall_s * 1000.0;
    stats_scope.s.decode_start_unix_ms = zeromoe::stats::unix_ms_now();
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
    // 2026-09-04: the RAM-tier equivalent of the VRAM snapshot right
    // above, added to directly test a real question -- does the RAM tier
    // (already ~50% pre-filled by static Tier-3 startup preload before
    // token 1) reach its own capacity ceiling DURING prefill, or does
    // decode have real free room to grow into for at least its first few
    // tokens? Compare against the same [RAM-SLOTS] line at shutdown.
    LOG_STREAM_INFO() << "[PREFILL-DONE] Host RAM tier occupancy immediately after prefill (before any decode "
                          "token) -- compare against the same [RAM-SLOTS] line at shutdown:\n";
    ram_mgr.report_slot_summary(engine);

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

    if (!bench_silent) {
        std::cout << " Done.\n\nOutput: ";
    }

    // Pre-allocate decode buffers (single token at a time)
    std::vector<float> decode_hidden(D);
    std::vector<float> normed_last(D);

    // Reasoning state machine for <think>...</think> tag parsing
    ReasoningParser reasoning;

    size_t timed_output_tokens = 0;
    std::chrono::steady_clock::time_point first_token_output;
    std::chrono::steady_clock::time_point last_token_output;
    bool mtp_started = false;
    const bool mtp_context_ready =
        engine.mtp_enabled && pc_reuse == 0 && pc_n_new == prompt_ids.size() &&
        !prompt_ids.empty() && !bench_ignore_stop;
    if (mtp_context_ready) {
        auto emit_mtp = [&](int next_token) {
            if (cancelled()) return false;
            perf().n_tokens_generated.fetch_add(1, std::memory_order_relaxed);
            // Same ordering fix as the main decode loop above: check for a
            // stop token before printing its decoded text, so a correctly-
            // detected stop token's own string isn't echoed to the user.
            const bool is_stop =
                next_token == eos_id || tokenizer.is_stop_token(next_token);
            if (control) {
                ++control->generated_tokens;
                if (is_stop) control->stopped = true;
                else if (control->emit && !control->emit(tokenizer.decode(next_token))) {
                    control->stopped = true;
                    return false;
                }
            }
            if (!is_stop && !bench_silent) {
                std::string token_str = tokenizer.decode(next_token);
                int parse_result = reasoning.feed(token_str);
                if (parse_result == 0) {
                    std::cout << token_str << std::flush;
                } else if (parse_result == 1) {
                    std::cout << "\033[90m" << token_str << "\033[0m" << std::flush;
                }
            }
            if (is_stop)
                return false;
            return !cancelled();
        };
        // Qwen4Exp's "shared" MTP checkpoint needs its own driver
        // (Qwen4ExpMtpSpeculator, mtp_speculative.cpp) -- its NextN state
        // is hc_dim wide, not n_embd wide, and GLM's MtpSpeculator is kept
        // untouched by this family's own, separately-verified path (see
        // Qwen4ExpMtpSpeculator's own header comment).
        if (engine.family_ops.family == ModelFamily::Qwen4Exp) {
            Qwen4ExpMtpSpeculator mtp(engine, ram_mgr, prefetcher);
            mtp_started = mtp.run(prompt_ids, hidden_states, max_new_tokens,
                                  eos_id, temperature, top_p, sequence,
                                  sampler.rng, emit_mtp);
        } else {
            MtpSpeculator mtp(engine, ram_mgr, prefetcher);
            mtp_started = mtp.run(prompt_ids, hidden_states, max_new_tokens,
                                  eos_id, temperature, top_p, sequence,
                                  sampler.rng, emit_mtp);
        }
        if (mtp_started) {
            LOG_STREAM_INFO() << "[MTP] speculative decoding active (draft tokens="
                              << engine.mtp_draft_tokens << ")\n";
        }
    }

    if (control && !mtp_started && control->generated_tokens > 0) {
        engine.qwen4exp_full_graph_failure = "MTP failed after emitting output";
        return;
    }
    if (!mtp_started) for (int step = 0; step < max_new_tokens; ++step) {
        auto diag_loop_iter_start = std::chrono::high_resolution_clock::now();
        // Cooperative shutdown checkpoint: a decode step can take many
        // seconds (CPU-bound expert FFN with no cancellation point inside
        // it), so this is the earliest point a Ctrl-C from the previous
        // step can actually be noticed. Breaking here flows into the exact
        // same report cleanup below as a normal EOS stop, so callers don't
        // need special-casing. (The prefetcher itself is caller-owned now --
        // see the header comment above -- so there's nothing to stop here.)
        if (cancelled()) {
            if (!control) std::cout << "\n[Interrupted]\n";
            break;
        }

        const float* last_hidden = hidden_states.data() + ((hidden_states.size() / D - 1) * D);
        dump_prefill_final_hidden_if_enabled(step, last_hidden, (size_t)D);

        // Full decode graphs can publish next-token logits while executing
        // the trunk. Qwen and DeepSeek V4.1 use this shared destination;
        // consume it once before the normal layered lm-head dispatch.
        const bool have_full_graph_logits =
            engine.qwen4exp_full_graph_logits_ready;
        if (have_full_graph_logits) {
            engine.qwen4exp_full_graph_logits_ready = false;
        } else {
            // output_norm is optional. Qwen4Exp has no output_norm tensor;
            // its low-rank HC output mixer is already the final collapse.
            auto norm_it = engine.resident_layers.find("output_norm.weight");
            auto lm_it = engine.resident_layers.find("output.weight");
            if (lm_it == engine.resident_layers.end()) {
                if (control) {
                    engine.qwen4exp_full_graph_failure = "output.weight is missing";
                    return;
                }
                break;
            }

            auto diag_lmhead_start = std::chrono::high_resolution_clock::now();
            const float* lm_in = last_hidden;
            if (norm_it != engine.resident_layers.end()) {
                rmsnorm_proj(engine, "output_norm.weight", last_hidden,
                             normed_last.data(), 1, D,
                             engine.model_file.architecture_config.eps);
                lm_in = normed_last.data();
            }
            dense_proj(engine, "output.weight", lm_in, logits.data(), 1, D, V);
            diag_lmhead_ms += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - diag_lmhead_start).count();
        }

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
                if (control) {
                    engine.qwen4exp_full_graph_failure = "non-finite generation logits";
                    return;
                }
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
        if (step < 2 && !bench_silent) {
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

        dump_decode_logits_if_enabled(step, logits.data(), logits.size());
        const uint64_t raw_logits_hash = decode_logits_hash_enabled()
                                             ? decode_logits_fnv64(logits.data(), logits.size())
                                             : 0;

        // Repetition penalty: no mechanism for this existed before -- with
        // no chat template applied either (main.cpp sends raw BOS+text, not
        // an instruct-formatted prompt; see CLAUDE.md), a low-confidence,
        // unfocused generation can easily drift onto one token and then
        // repeat it forever, since nothing in Sampler::sample discourages
        // it. Standard CTRL-style penalty (Keskar et al. 2019): scale down
        // the logit of every token seen recently, whichever direction makes
        // it less likely (dividing a positive logit or multiplying a
        // negative one both reduce post-softmax probability).
        //
        // The window must never reach into the PROMPT: `sequence` starts as
        // a copy of `prompt_ids`, so scanning from `sequence.size() -
        // REPETITION_WINDOW` with no floor at the prompt boundary penalizes
        // the model for wanting to reuse a word the USER's own question
        // happened to contain -- confirmed as the real mechanism behind a
        // reproducible switch-to-non-English defect (see
        // zeromoe_repetition_penalty's own comment). Only tokens the
        // model itself generated should ever be penalized.
        if (const float repetition_penalty = zeromoe_repetition_penalty();
            repetition_penalty != 1.0f) {
            constexpr int REPETITION_WINDOW = 64;
            int start = std::max((int)prompt_ids.size(),
                                  (int)sequence.size() - REPETITION_WINDOW);
            for (int i = start; i < (int)sequence.size(); ++i) {
                int tok = sequence[i];
                if (tok < 0 || tok >= V) continue;
                float& l = logits[tok];
                l = (l > 0.0f) ? (l / repetition_penalty) : (l * repetition_penalty);
            }
        }

        const auto &teacher_tokens = teacher_force_tokens_if_enabled(step);
        const bool has_teacher_token =
            step < static_cast<int>(teacher_tokens.size());
        const bool skip_teacher_sampling =
            has_teacher_token && teacher_force_skip_sampling_enabled();
        // ZEROMOE_TEACHER_FORCE_NLL=1: log -log p(forced token) under this
        // step's raw logits (natural log), so two configs can be compared on
        // a fixed text by mean NLL (quality) as well as by speed.
        if (has_teacher_token) {
            static const bool log_nll = [] {
                const char *e = std::getenv("ZEROMOE_TEACHER_FORCE_NLL");
                return e && e[0] == '1';
            }();
            const int ft = teacher_tokens[step];
            if (log_nll && ft >= 0 && ft < V) {
                float mx = logits[0];
                for (int i = 1; i < V; ++i) mx = std::max(mx, logits[i]);
                double sum = 0.0;
                for (int i = 0; i < V; ++i) sum += std::exp((double)logits[i] - mx);
                const double nll = (double)mx + std::log(sum) - (double)logits[ft];
                LOG_STREAM_INFO() << "[TF-NLL] step=" << step << " nll=" << nll << "\n";
            }
        }
        const int sampled_token = skip_teacher_sampling
                                      ? teacher_tokens[step]
                                      : sampler.sample(logits, temperature, top_p);
        dump_generated_token_if_enabled(step, sampled_token);

        int next_token = sampled_token;
        if (has_teacher_token) {
            next_token = teacher_tokens[step];
            if (step == 0) {
                LOG_STREAM_INFO() << "[TEACHER-FORCE] using " << teacher_tokens.size()
                                  << " reference decode token(s) from "
                                  << "ZEROMOE_TEACHER_FORCE_TOKEN_FILE\n";
            }
        }
        log_decode_logits_hash(step, raw_logits_hash, sampled_token, next_token);
        sequence.push_back(next_token);
        perf().n_tokens_generated.fetch_add(1, std::memory_order_relaxed);

        // GLM-5.2's real eos_token_id is a 3-entry array (<|endoftext|>,
        // <|user|>, <|observation|>) -- checking only the single `eos_id`
        // param means generation runs past the model's real turn-boundary
        // markers into undefined continuation territory
        // tokenizer.is_stop_token covers all of them; the `eos_id` check
        // stays as a redundant safety net.
        //
        // This check must run BEFORE printing: the original ordering
        // printed every token's decoded text first and checked for a stop
        // token second, so a real, correctly-detected stop token's own
        // string (e.g. "<|im_end|>", "<|endoftext|>") was echoed to the
        // user right as generation correctly ended -- confirmed via
        // ZEROMOE_GENERATED_TOKEN_FILE (dump_generated_token_if_enabled)
        // showing the model's actual last sampled token WAS the genuine
        // stop-token ID, not ordinary vocabulary tokens that merely spell
        // out that text. The model was stopping correctly; only the
        // display was wrong.
        const bool is_stop_now = !bench_ignore_stop &&
            (next_token == eos_id || tokenizer.is_stop_token(next_token));
        if (control) {
            ++control->generated_tokens;
            control->stopped = is_stop_now;
        }

        if (step == 0 && !is_stop_now && max_new_tokens > 1 &&
            engine.family_ops.family == ModelFamily::Qwen4Exp &&
            graph_model &&
            graph_model->enabled(engine, ModelGraphPhase::Decode)) {
            const ModelGraphRequest first_decode_request{
                ModelGraphPhase::Decode, 1, static_cast<int>(prompt_ids.size()),
                &next_token, decode_hidden.data(), logits.data(), V};
            const auto decode_setup_start = std::chrono::high_resolution_clock::now();
            const bool decode_graph_built = graph_model->build(engine, first_decode_request);
            stats_scope.s.decode_setup_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - decode_setup_start).count();
            if (!decode_graph_built &&
                engine.qwen4exp_full_graph_required) {
                engine.qwen4exp_full_graph_failure =
                    "decode graph construction failed before first token";
                LOG_ERROR("[QWEN4EXP-GRAPH-POLICY] %s; aborting request\n",
                          engine.qwen4exp_full_graph_failure.c_str());
                return;
            }
        }

        if (!is_stop_now && control && control->emit &&
            !control->emit(tokenizer.decode(next_token))) {
            control->stopped = true;
            break;
        }
        if (!is_stop_now && !bench_silent) {
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
        }

        if (!is_stop_now) {
            last_token_output = std::chrono::steady_clock::now();
            if (timed_output_tokens++ == 0) {
                first_token_output = last_token_output;
                stats_scope.s.ttft_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - request_wall_start).count();
            }
        }

        if (is_stop_now || step + 1 >= max_new_tokens)
            break;

        double diag_preproc_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - diag_loop_iter_start).count();

        // Embed the new token and run one forward step
        auto diag_embed_start = std::chrono::high_resolution_clock::now();
        if (emb_it != engine.resident_layers.end()) {
            embed_token(engine, next_token, D, decode_hidden.data());
        }
        double diag_embed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - diag_embed_start).count();

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
        // This destination is valid only for the immediately following
        // forward call. It is consumed by the Qwen and DeepSeek V4.1 full
        // decode bridges; other model families ignore it.
        engine.qwen4exp_full_graph_logits_target = logits.data();
        engine.qwen4exp_full_graph_logits_target_size = V;
        layers_forward_gguf(engine, ram_mgr, prefetcher, decode_hidden.data(), 1, pos, &next_token);
        engine.qwen4exp_full_graph_logits_target = nullptr;
        engine.qwen4exp_full_graph_logits_target_size = 0;
        if (!engine.qwen4exp_full_graph_failure.empty())
            return;
        if (hc_active && !engine.full_graph_output_ready && !engine.qwen4exp_full_graph_output_ready &&
            !engine.glm5next_full_graph_output_ready &&
            !engine.deepseek_v4_full_graph_output_ready)
            hc_collapse_output(engine, decode_hidden.data(), 1, D);
        double token_wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - token_wall_start).count();
        auto token_snap_after = perf().snapshot();
        char token_label[32];
        snprintf(token_label, sizeof(token_label), "TOKEN %5d", step);
        if (!bench_silent) {
            PerfCounters::print_delta(token_label, token_wall_ms, token_snap_after - token_snap_before);
        }
        check_hidden_finite(token_label, decode_hidden.data(), decode_hidden.size());
        // Real per-GPU VRAM usage (cudaMemGetInfo), sampled once per token
        // to track actual usage over the run -- distinct from the one-time
        // startup topology dump and from the logical cache-fill heatmaps
        // below (this is raw memory, those are "what's actually resident").
        if (!bench_silent) {
            auto diag_vramlog_start = std::chrono::high_resolution_clock::now();
            if (ram_mgr.vram_cache) ram_mgr.vram_cache->log_vram_usage(token_label);
            diag_vramlog_ms += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - diag_vramlog_start).count();
        }

        // Replace hidden_states with the single-token output for next iteration
        hidden_states.resize(D);
        std::memcpy(hidden_states.data(), decode_hidden.data(), D * sizeof(float));

        double diag_loop_iter_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - diag_loop_iter_start).count();
        stats_scope.s.step_ms.push_back(static_cast<float>(diag_loop_iter_ms));
        double diag_vramlog_delta_ms = diag_vramlog_ms - diag_vramlog_ms_prev;
        diag_vramlog_ms_prev = diag_vramlog_ms;
        if (token_loop_diag_enabled && !bench_silent) {
            LOG_INFO("[TOKEN-LOOP-DIAG] step=%d iter_total=%.3fms preproc(sample+decode+print)=%.3fms "
                     "embed=%.3fms core(hc+forward)=%.3fms vramlog=%.3fms tail=%.3fms\n",
                     step, diag_loop_iter_ms, diag_preproc_ms, diag_embed_ms, token_wall_ms,
                     diag_vramlog_delta_ms,
                     diag_loop_iter_ms - diag_preproc_ms - diag_embed_ms - token_wall_ms - diag_vramlog_delta_ms);
        }
    }
    
    // prefetcher is caller-owned (started once for the whole session by
    // main.cpp) -- not stopped here.
    if (!bench_silent && reasoning.think_tokens > 0) {
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
    const double decode_boundary_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - decode_boundary_start).count();
    stats_scope.s.generated_tokens = static_cast<int>(request_delta.n_tokens_generated);
    stats_scope.s.decode_wall_ms = decode_boundary_s * 1000.0;
    stats_scope.s.total_ms = request_wall_ms;
    stats_scope.s.status = "ok";
    if (control) control->completed = !control->interrupted;
    if (request_delta.n_tokens_generated > 0 && decode_boundary_s > 0.0) {
        LOG_INFO("[REQUEST-DECODE-WALL] tokens=%lu wall=%.3fs rate=%.2f tok/s "
                 "(starts immediately after PREFILL-DONE; includes one-time "
                 "decode-graph setup)\n",
                 (unsigned long)request_delta.n_tokens_generated,
                 decode_boundary_s,
                 (double)request_delta.n_tokens_generated / decode_boundary_s);
    }
    if (timed_output_tokens > 1) {
        const double output_wall_s = std::chrono::duration<double>(
            last_token_output - first_token_output).count();
        if (output_wall_s > 0.0) {
            LOG_INFO("[REQUEST-DECODE-OUTPUT] tokens=%zu intervals=%zu "
                     "wall=%.6fs rate=%.2f tok/s "
                     "(first output to last output; Qwen graph setup "
                     "precedes first output)\n",
                     timed_output_tokens, timed_output_tokens - 1,
                     output_wall_s,
                     static_cast<double>(timed_output_tokens - 1) / output_wall_s);
        }
    }
    // TEMPORARY DIAGNOSTIC -- see this function's top-of-function comment.
    LOG_INFO("[REQUEST-DIAG] lmhead(norm+proj)=%.3fs vramlog(cudaMemGetInfo)=%.3fs\n",
             diag_lmhead_ms / 1000.0, diag_vramlog_ms / 1000.0);
    // TEMPORARY DIAGNOSTIC (2026-09-12) -- layer_loop_total is the FULL
    // wall-clock time of layers_forward_gguf's per-layer loop, every layer,
    // summed across this request's prefill call + all decode-step calls
    // (see g_diag_layer_loop_total_ms's own comment, execution_layer.cpp).
    // Compare against request_wall_ms above: if it's close, the ~21s/
    // request gap between [REQUEST]'s wall_ms and its named buckets
    // (attn=/route=/expert=/dense_gpu=) is genuinely inside this loop's own
    // orchestration. t_prefetch is printed alongside it because it is
    // ALREADY an existing, real PERF_START/PERF_END(t_prefetch) counter
    // (execution_layer.cpp's DEPTH-1/ATTN-PILOT/DEPTH-2 speculative pilot
    // block) that is captured correctly in every Snapshot diff but was
    // never actually printed in print_summary/print_delta's format
    // strings -- so it has been silently invisible in every [REQUEST] line
    // to date despite running substantial CPU work (rmsnorm + a full
    // dense_proj router matvec x3, plus RAM/VRAM residency scans) every
    // single layer transition, unconditionally, regardless of whether any
    // of the actual prefetch/promotion flags it feeds are even enabled.
    // attn_call/ffn_call: wall-clock measured around the SAME calls that
    // feed t_attn/t_moe_expert (request_delta.t_attn / .t_moe_expert) --
    // see g_diag_attn_call_ms/g_diag_ffn_call_ms's own comment
    // (execution_layer.cpp). Large wall vs. named-counter gaps here would
    // mean async GPU dispatch returns before the work actually lands, with
    // PERF_SCOPE's RAII destructor stopping the clock too early.
    // hc_hooks/layer_input_hook: this request's own share of two
    // PerfCounters atomics that exist but were never wired into
    // Snapshot/print_summary at all (see the capture site above).
    LOG_INFO("[REQUEST-DIAG2] layer_loop_total=%.3fs t_prefetch(pilot, already-collected-but-never-printed)=%.3fs "
             "attn_call_wall=%.3fs(t_attn=%.3fs) ffn_call_wall=%.3fs(t_moe_expert=%.3fs) "
             "hc_hooks=%.3fs layer_input_hook=%.3fs\n",
             diag_get_layer_loop_total_ms() / 1000.0, request_delta.t_prefetch,
             diag_get_attn_call_ms() / 1000.0, request_delta.t_attn,
             diag_get_ffn_call_ms() / 1000.0, request_delta.t_moe_expert,
             perf().t_hc_hooks.load() - diag_hc_hooks_start,
             perf().t_layer_input_hook.load() - diag_layer_input_hook_start);
    const double decode_layer_ms = diag_get_decode_layer_loop_ms();
    const double decode_attn_ms = diag_get_decode_attn_call_ms();
    const double decode_ffn_ms = diag_get_decode_ffn_call_ms();
    const double decode_qwen_cpu_ms = diag_get_decode_qwen_cpu_graph_ms();
    const double decode_qwen_shared_submit_ms =
        diag_get_decode_qwen_shared_submit_ms();
    const double decode_qwen_cpu_offload_wait_ms =
        diag_get_decode_qwen_cpu_offload_wait_ms();
    const double decode_qwen_cpu_offload_total_ms =
        diag_get_decode_qwen_cpu_offload_total_ms();
    const double decode_qwen_cpu_offload_fence_wait_ms =
        diag_get_decode_qwen_cpu_offload_fence_wait_ms();
    const double decode_qwen_cpu_offload_h2d_ms =
        diag_get_decode_qwen_cpu_offload_h2d_ms();
    const double decode_qwen_cpu_offload_ring_reuse_wait_ms =
        diag_get_decode_qwen_cpu_offload_ring_reuse_wait_ms();
    const uint64_t decode_qwen_cpu_offload_cold_slots =
        diag_get_decode_qwen_cpu_offload_cold_slots();
    const uint64_t decode_qwen_cpu_offload_worker_calls =
        diag_get_decode_qwen_cpu_offload_worker_calls();
    const uint64_t decode_qwen_cpu_offload_worker_successes =
        diag_get_decode_qwen_cpu_offload_worker_successes();
    const uint64_t decode_qwen_cpu_offload_worker_failures =
        diag_get_decode_qwen_cpu_offload_worker_failures();
    const uint64_t decode_qwen_cpu_offload_fenced_calls =
        diag_get_decode_qwen_cpu_offload_fenced_calls();
    const uint64_t decode_qwen_cpu_offload_fence_failures =
        diag_get_decode_qwen_cpu_offload_fence_failures();
    const double decode_residual_ms =
        std::max(0.0, decode_layer_ms - decode_attn_ms - decode_ffn_ms);
    const double decode_hc_ms = diag_get_decode_hc_ms();
    const double decode_hc_pre_ms = diag_get_decode_hc_pre_ms();
    const double decode_hc_post_ms = diag_get_decode_hc_post_ms();
    const double decode_layer_input_ms = diag_get_decode_layer_input_ms();
    const double decode_other_ms = std::max(
        0.0, decode_residual_ms - decode_hc_ms - decode_layer_input_ms -
                 diag_get_decode_pilot_ms());
    const double decode_tokens =
        std::max(1.0, (double)request_delta.n_tokens_generated);
    LOG_INFO("[REQUEST-DECODE-DIAG] tokens=%.0f layer_loop=%.3fs (%.3fms/token) "
             "attention_call=%.3fs (%.3fms/token) "
             "ffn_call=%.3fs (%.3fms/token) "
             "qwen_cpu_routed=%.3fs (%.3fms/token) "
             "qwen_shared_submit=%.3fs (%.3fms/token) "
             "qwen_cpu_offload_wait=%.3fs (%.3fms/token) "
             "qwen_cpu_offload_total=%.3fs (%.3fms/token) "
             "qwen_cpu_offload_fence_wait=%.3fs (%.3fms/token) "
             "qwen_cpu_offload_h2d=%.3fs (%.3fms/token) "
             "qwen_cpu_offload_ring_reuse=%.3fs (%.3fms/token) "
             "qwen_cpu_offload_cold_slots=%llu (%.3f/token) "
             "qwen_cpu_offload_workers=%llu success=%llu failure=%llu "
             "fenced=%llu fence_failures=%llu "
             "residual_scheduler_hc_route=%.3fs (%.3fms/token), "
             "hc=%.3fs (%.3fms/token; pre=%.3f, post=%.3f), "
             "pre_attn=%.3f, pre_ffn=%.3f, "
             "layer_input=%.3fs (%.3fms/token), "
             "other=%.3fs (%.3fms/token) "
             "pilot=%.3fs (%.3fms/token)\n",
             decode_tokens, decode_layer_ms / 1000.0,
             decode_layer_ms / decode_tokens,
             decode_attn_ms / 1000.0, decode_attn_ms / decode_tokens,
             decode_ffn_ms / 1000.0, decode_ffn_ms / decode_tokens,
             decode_qwen_cpu_ms / 1000.0,
             decode_qwen_cpu_ms / decode_tokens,
             decode_qwen_shared_submit_ms / 1000.0,
             decode_qwen_shared_submit_ms / decode_tokens,
             decode_qwen_cpu_offload_wait_ms / 1000.0,
             decode_qwen_cpu_offload_wait_ms / decode_tokens,
             decode_qwen_cpu_offload_total_ms / 1000.0,
             decode_qwen_cpu_offload_total_ms / decode_tokens,
             decode_qwen_cpu_offload_fence_wait_ms / 1000.0,
             decode_qwen_cpu_offload_fence_wait_ms / decode_tokens,
             decode_qwen_cpu_offload_h2d_ms / 1000.0,
             decode_qwen_cpu_offload_h2d_ms / decode_tokens,
             decode_qwen_cpu_offload_ring_reuse_wait_ms / 1000.0,
             decode_qwen_cpu_offload_ring_reuse_wait_ms / decode_tokens,
             (unsigned long long)decode_qwen_cpu_offload_cold_slots,
             (double)decode_qwen_cpu_offload_cold_slots / decode_tokens,
             (unsigned long long)decode_qwen_cpu_offload_worker_calls,
             (unsigned long long)decode_qwen_cpu_offload_worker_successes,
             (unsigned long long)decode_qwen_cpu_offload_worker_failures,
             (unsigned long long)decode_qwen_cpu_offload_fenced_calls,
             (unsigned long long)decode_qwen_cpu_offload_fence_failures,
             decode_residual_ms / 1000.0,
             decode_residual_ms / decode_tokens,
             decode_hc_ms / 1000.0, decode_hc_ms / decode_tokens,
             decode_hc_pre_ms / decode_tokens,
             decode_hc_post_ms / decode_tokens,
             diag_get_decode_hc_attn_pre_ms() / decode_tokens,
             diag_get_decode_hc_ffn_pre_ms() / decode_tokens,
             decode_layer_input_ms / 1000.0,
             decode_layer_input_ms / decode_tokens,
             decode_other_ms / 1000.0, decode_other_ms / decode_tokens,
             diag_get_decode_pilot_ms() / 1000.0,
             diag_get_decode_pilot_ms() / decode_tokens);
    // TEMPORARY DIAGNOSTIC (2026-09-12) -- phase-by-phase bisection of
    // dsv41_attention_gguf's ratio!=0 branch (the thing attn_call_wall/
    // t_attn above already proved costs ~20-21s/request for this model).
    // qrope/kvrope/perhead/outproj run on EVERY call (all layers, all
    // tokens); ownskv/indexer_qk/indexer_score only run on owns_kv/
    // owns_index layers respectively -- their own *_calls counts are
    // printed alongside them so a small per-request total isn't misread as
    // "cheap per call" when few layers ever take that branch (see
    // g_diag_dsv41_* comment, attention_layer.cpp). woa_dequant is a
    // subset of outproj isolating the one-time WOA weight dequant.
    LOG_INFO("[REQUEST-DIAG3] dsv41_attn_calls=%.0f qrope=%.3fs kvrope=%.3fs "
             "ownskv=%.3fs(calls=%.0f) indexer_qk=%.3fs indexer_score=%.3fs(idx_calls=%.0f) "
             "perhead=%.3fs outproj=%.3fs(woa_dequant=%.3fs,calls=%.0f)\n",
             diag_get_dsv41_calls(),
             diag_get_dsv41_qrope_ms() / 1000.0,
             diag_get_dsv41_kvrope_ms() / 1000.0,
             diag_get_dsv41_ownskv_ms() / 1000.0, diag_get_dsv41_ownskv_calls(),
             diag_get_dsv41_indexer_qk_ms() / 1000.0,
             diag_get_dsv41_indexer_score_ms() / 1000.0,
             diag_get_dsv41_ownsindex_calls(),
             diag_get_dsv41_perhead_ms() / 1000.0,
             diag_get_dsv41_outproj_ms() / 1000.0,
             diag_get_dsv41_woa_dequant_ms() / 1000.0,
             diag_get_dsv41_woa_dequant_calls());

    // Inter-request decay of prefill->decode retention stamps. Fires once
    // per request, right after the REQUEST line, so the numbers above still
    // describe the request that just ran. See
    // RAMSafetyManager::decay_prefill_demand for why this is a decay rather
    // than a bulk release of the retained set.
    // No-ops internally when prefill_retention_weight is 0, so this costs
    // nothing when the feature is off.
    if (engine.prefill_retention_decay) {
        ram_mgr.decay_prefill_demand(engine);
    }

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

    if (!bench_silent) {
        std::cout << "\n[Generation Complete]\n";
    }
}
