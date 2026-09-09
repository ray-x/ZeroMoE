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

private:
    StreamingEngine &engine_;
    RAMSafetyManager &ram_mgr_;
    LockFreePrefetcher &prefetcher_;

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
};

