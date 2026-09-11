#pragma once

#include <functional>
#include <random>
#include <vector>

class StreamingEngine;
class RAMSafetyManager;
class LockFreePrefetcher;

// Runs the single GLM NextN block.  `h_input` is the hidden state supplied by
// the target for each row (the first row may be the pending zero state), and
// `h_nextn` receives the block's shared-head-normalized hidden rows.  Only the
// final row's logits are returned because proposal steps consume one row at a
// time.  The function lives beside the normal execution layer so it can reuse
// the existing attention, MoE cache, and dense dispatch implementations.
bool glm_mtp_forward(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                     LockFreePrefetcher &prefetcher, const int *token_ids,
                     const float *h_input, int S, int pos_base,
                     float *h_nextn, float *logits_last);

// Runs the single Qwen4Exp ("shared" checkpoint) NextN block. Same overall
// bridge contract as glm_mtp_forward (h_input -> h_nextn is the state
// carried between successive draft/correction calls, logits_last covers
// only the final row), but h_input/h_nextn are hc_dim = hc_mult * n_embd
// WIDE (the block's own local multi-stream residual, [S, hc_mult, n_embd]
// stream-major -- see qwen4exp_hc_mix_impl's own comment), not n_embd
// wide: Qwen4Exp's NextN block keeps its own independent Hyper-Connections
// state across calls, verified against /home/ray/Projects/unsloth/
// llama.cpp's qwen4exp.cpp (branch mtp/qwen4exp-nextn), graph_mtp's own
// `res->t_h_nextn = res_hc` (the WIDE, pre-collapse state). Caller is
// responsible for sizing h_input/h_nextn to S*hc_dim floats.
bool qwen4exp_mtp_forward(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                          LockFreePrefetcher &prefetcher, const int *token_ids,
                          const float *h_input, int S, int pos_base,
                          float *h_nextn, float *logits_last);

using MtpEmitFn = std::function<bool(int token)>;

// Small, model-gated speculative-decoding driver for GLM5NEXT/GLM_DSA.
// The target trunk remains authoritative: every proposed token is verified by
// a target batch, and rejected batches restore the recurrent target state then
// replay only the committed prefix plus the correction token.
class MtpSpeculator {
public:
    MtpSpeculator(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                  LockFreePrefetcher &prefetcher);

    bool supported() const;

    // Returns true once the MTP path has started and owns generation. Returns
    // false before mutating target state when the path is unavailable, so the
    // caller can safely fall back to the ordinary one-token loop.
    bool run(const std::vector<int> &prompt_ids,
             const std::vector<float> &prompt_hidden, int max_new_tokens,
             int eos_id, float temperature, float top_p,
             std::vector<int> &sequence, std::mt19937 &rng,
             const MtpEmitFn &emit);

    // Public (not just for MtpSpeculator's own run()): Qwen4ExpMtpSpeculator
    // below reuses embed/target_forward/target_logits_batch/
    // make_distribution/sample_distribution/emit_token as-is via
    // composition (holds its own MtpSpeculator instance) rather than
    // duplicating this family-agnostic sampling/verification math --
    // target_logits_batch's own family branch is the only place that
    // actually differs per family; see its .cpp comment.
    bool embed(int token, float *out) const;
    bool target_forward(const std::vector<int> &tokens, int pos_base,
                        std::vector<float> &raw_hidden);
    bool target_logits_batch(const std::vector<float> &raw_hidden, int rows,
                             std::vector<float> &logits);
    bool normalized_target_hidden(const float *raw, float *out) const;
    bool make_distribution(const float *logits, int vocab,
                           const std::vector<int> &context, float temperature,
                           float top_p, std::vector<float> &probs) const;
    int sample_distribution(const std::vector<float> &probs,
                            std::mt19937 &rng) const;
    bool emit_token(std::vector<int> &sequence, int token, int eos_id,
                    const MtpEmitFn &emit) const;

private:
    StreamingEngine &engine_;
    RAMSafetyManager &ram_mgr_;
    LockFreePrefetcher &prefetcher_;
};

// Speculative-decoding driver for Qwen4Exp's "shared" MTP checkpoint.
// Mirrors MtpSpeculator::run()'s overall structure (same exact speculative-
// sampling accept/residual-correction math) but the MTP bridge state is
// hc_dim = hc_mult*n_embd WIDE (qwen4exp_mtp_forward's own h_input/
// h_nextn), not n_embd wide -- kept as a SEPARATE class rather than a
// branch inside MtpSpeculator so GLM's already-verified driver code is
// never touched by this family's own, less-tested path.
class Qwen4ExpMtpSpeculator {
public:
    Qwen4ExpMtpSpeculator(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                          LockFreePrefetcher &prefetcher);

    bool supported() const;

    bool run(const std::vector<int> &prompt_ids,
             const std::vector<float> &prompt_hidden, int max_new_tokens,
             int eos_id, float temperature, float top_p,
             std::vector<int> &sequence, std::mt19937 &rng,
             const MtpEmitFn &emit);

private:
    StreamingEngine &engine_;
    RAMSafetyManager &ram_mgr_;
    LockFreePrefetcher &prefetcher_;
    // Composition, not inheritance: reuses its family-agnostic public
    // methods (embed/target_forward/target_logits_batch/make_distribution/
    // sample_distribution/emit_token) without duplicating that math.
    MtpSpeculator base_;
};

