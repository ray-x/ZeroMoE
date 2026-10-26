#pragma once

// Target-verified speculative generation for graph-native MTP.
//
// The driver is model-agnostic and has no CUDA or ggml dependency: the target
// and draft are reached through two narrow interfaces so the accept/reject
// math and the state reconciliation can be proven on the CPU with a
// deterministic fake before any GPU run.
//
// Protocol (matches docs/design/qwen4exp-mtp-graph.md):
//  * The target is authoritative. Every emitted token is either sampled from
//    the target distribution or accepted by exact speculative sampling
//    (accept with min(1, p/q), otherwise resample from norm(max(0, p - q))).
//  * A cycle starts with a *pending* token y that the target has not yet
//    processed. The draft pairs y with the target hidden of the position
//    before it, then feeds its own residual back for later drafts.
//  * Verification steps the target one token at a time over y, d1..dk. A
//    state checkpoint is taken before every step after the first, so a
//    rejection at draft a restores exactly the state after processing d_a
//    without replaying anything.
//  * The draft's K/V for accepted positions is rebuilt from the true target
//    hiddens (its drafted rows used its own residual), then the next cycle's
//    pending token is the correction.
//
// On any failure the target and draft states are unspecified and the caller
// must reset both; checkpoints are always released.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace zeromoe::graph {

class SpecTarget {
 public:
  virtual ~SpecTarget() = default;
  // Process `token` at `position`. `wide` receives the pre-head residual after
  // the token (the hidden the draft consumes); `logits` the next-token logits.
  virtual bool step(int token, int position, std::span<float> wide,
                    std::span<float> logits, std::string &error) = 0;
  // Checkpoint the live recurrent state. Handles stay valid until released.
  virtual bool save(int &handle, std::string &error) = 0;
  // Make the checkpoint's state the live state.
  virtual bool restore(int handle, std::string &error) = 0;
  virtual void release(int handle) = 0;
};

class SpecDraft {
 public:
  virtual ~SpecDraft() = default;
  // Append one draft row at next_position() == `position`.
  virtual bool step(int token, int position, std::span<const float> hidden,
                    std::span<float> next_hidden, std::span<float> logits,
                    std::string &error) = 0;
  // Drop rows at or after `position`.
  virtual bool rewind(int position, std::string &error) = 0;
  virtual int next_position() const = 0;
};

// Turns logits plus the token context into a probability distribution
// (temperature, top-p, penalties). Greedy decoding is a one-hot result.
using SpecDistribution = std::function<bool(
    std::span<const float> logits, const std::vector<int> &context,
    std::vector<float> &probs)>;

struct SpecConfig {
  size_t vocab = 0;
  size_t wide = 0;        // draft/target hidden width (D * HC for Qwen4Exp)
  int max_draft = 3;      // K
  int max_new_tokens = 0;
  int eos_id = -1;
  int max_position = 0;   // exclusive capacity of both target and draft
};

struct SpecPrompt {
  std::vector<int> tokens;
  // Target pre-head residual after each prompt token, tokens.size() * wide.
  std::vector<float> wide;
  // Target next-token logits after the last prompt token.
  std::vector<float> last_logits;
};

struct SpecStats {
  uint64_t cycles = 0, drafted = 0, accepted = 0, emitted = 0;
  uint64_t target_steps = 0, draft_steps = 0, restores = 0;
  bool truncated = false;  // stopped because the context capacity was reached
};

namespace detail {

inline int sample_probs(const std::vector<float> &probs, std::mt19937 &rng) {
  double total = 0.0;
  for (float p : probs) total += p;
  if (!(total > 0.0) || !std::isfinite(total)) return -1;
  std::uniform_real_distribution<double> uniform(0.0, total);
  const double target = uniform(rng);
  double running = 0.0;
  int last_positive = -1;
  for (size_t i = 0; i < probs.size(); ++i) {
    if (probs[i] <= 0.0f) continue;
    last_positive = static_cast<int>(i);
    running += probs[i];
    if (target < running) return last_positive;
  }
  return last_positive;
}

struct CheckpointSet {
  SpecTarget &target;
  std::vector<int> handles;
  explicit CheckpointSet(SpecTarget &t) : target(t) {}
  ~CheckpointSet() { for (int h : handles) target.release(h); }
  CheckpointSet(const CheckpointSet &) = delete;
  CheckpointSet &operator=(const CheckpointSet &) = delete;
};

}  // namespace detail

// Returns true on success (including a caller-requested stop, EOS, the token
// budget, or reaching capacity). Returns false with `error` set on failure.
inline bool speculate(SpecTarget &target, SpecDraft &draft,
                      const SpecConfig &cfg, const SpecPrompt &prompt,
                      const SpecDistribution &distribution, std::mt19937 &rng,
                      const std::function<bool(int)> &emit, SpecStats &stats,
                      std::string &error) {
  error.clear();
  const size_t V = cfg.vocab, W = cfg.wide;
  const int n = static_cast<int>(prompt.tokens.size());
  if (!V || !W || n <= 0 || cfg.max_new_tokens <= 0 || cfg.max_draft < 0 ||
      cfg.max_position <= n || prompt.wide.size() != size_t(n) * W ||
      prompt.last_logits.size() != V) {
    error = "invalid speculation configuration or prompt";
    return false;
  }
  if (draft.next_position() != 0) {
    error = "draft must be reset before speculation";
    return false;
  }

  std::vector<float> scratch_h(W), scratch_l(V), zeros(W, 0.0f);
  const auto draft_step = [&](int token, int position,
                              std::span<const float> hidden,
                              std::span<float> next_h,
                              std::span<float> logits) {
    ++stats.draft_steps;
    return draft.step(token, position, hidden, next_h, logits, error);
  };
  const auto make_probs = [&](std::span<const float> logits,
                              const std::vector<int> &context,
                              std::vector<float> &probs) {
    if (!distribution(logits, context, probs) || probs.size() != V) {
      error = "distribution callback failed or returned the wrong size";
      return false;
    }
    for (float p : probs)
      if (!(p >= 0.0f) || !std::isfinite(p)) {
        error = "distribution callback returned an invalid probability";
        return false;
      }
    return true;
  };

  // Catch the draft up on the prompt: x[p] pairs with the target hidden of p-1.
  for (int p = 0; p < n; ++p) {
    const std::span<const float> hidden =
        p == 0 ? std::span<const float>(zeros)
               : std::span<const float>(prompt.wide.data() + size_t(p - 1) * W, W);
    if (!draft_step(prompt.tokens[p], p, hidden, scratch_h, scratch_l))
      return false;
  }

  std::vector<int> seq = prompt.tokens;
  int generated = 0;
  // Commit one token. Returns false when generation must stop.
  const auto commit = [&](int token) {
    seq.push_back(token);
    ++generated;
    ++stats.emitted;
    const bool keep_going = emit(token);
    return keep_going && token != cfg.eos_id && generated < cfg.max_new_tokens;
  };

  {  // The first token is sampled by the target from the prompt's logits.
    std::vector<float> probs;
    if (!make_probs(prompt.last_logits, seq, probs)) return false;
    const int y = detail::sample_probs(probs, rng);
    if (y < 0) { error = "target distribution has no mass"; return false; }
    if (!commit(y)) return true;
  }

  std::vector<float> hidden_prev(prompt.wide.end() - W, prompt.wide.end());
  while (true) {
    const int pos = static_cast<int>(seq.size()) - 1;  // pending token's position
    if (pos >= cfg.max_position) { stats.truncated = true; return true; }
    if (draft.next_position() != pos) {
      error = "draft/target position invariant violated";
      return false;
    }
    const int remaining = cfg.max_new_tokens - generated;
    const int k = std::max(0, std::min({cfg.max_draft, remaining - 1,
                                        cfg.max_position - 1 - pos}));
    ++stats.cycles;

    // ---- draft k tokens after the pending token ----
    std::vector<int> drafts;
    std::vector<std::vector<float>> q;
    std::vector<float> cur_h = hidden_prev, next_h(W), logits(V);
    int cur_tok = seq.back();
    for (int j = 1; j <= k; ++j) {
      if (!draft_step(cur_tok, pos + j - 1, cur_h, next_h, logits)) return false;
      std::vector<int> context = seq;
      context.insert(context.end(), drafts.begin(), drafts.end());
      std::vector<float> probs;
      if (!make_probs(logits, context, probs)) return false;
      const int d = detail::sample_probs(probs, rng);
      if (d < 0) { error = "draft distribution has no mass"; return false; }
      drafts.push_back(d);
      q.push_back(std::move(probs));
      cur_tok = d;
      cur_h = next_h;
    }
    stats.drafted += drafts.size();

    // ---- verify: step the target over y, d1..dk with checkpoints ----
    detail::CheckpointSet checkpoints(target);
    std::vector<float> wides(size_t(k + 1) * W), plogits(size_t(k + 1) * V);
    for (int j = 0; j <= k; ++j) {
      if (j >= 1) {
        int handle = 0;
        if (!target.save(handle, error)) return false;
        checkpoints.handles.push_back(handle);  // handles[j-1] = state before step j
      }
      const int token = j == 0 ? seq.back() : drafts[size_t(j - 1)];
      ++stats.target_steps;
      if (!target.step(token, pos + j,
                       std::span<float>(wides.data() + size_t(j) * W, W),
                       std::span<float>(plogits.data() + size_t(j) * V, V), error))
        return false;
    }

    std::vector<std::vector<float>> p(size_t(k + 1));
    for (int j = 0; j <= k; ++j) {
      std::vector<int> context = seq;
      context.insert(context.end(), drafts.begin(), drafts.begin() + j);
      if (!make_probs(std::span<const float>(plogits.data() + size_t(j) * V, V),
                      context, p[size_t(j)]))
        return false;
    }

    // ---- exact speculative-sampling accept / correct ----
    int accepted = 0, correction = -1;
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (int j = 0; j < k; ++j) {
      const size_t d = size_t(drafts[size_t(j)]);
      const float qp = q[size_t(j)][d], pp = p[size_t(j)][d];
      const float accept = qp > 0.0f ? std::min(1.0f, pp / qp) : (pp > 0.0f ? 1.0f : 0.0f);
      if (uniform(rng) <= accept) { ++accepted; continue; }
      std::vector<float> residual(V);
      double sum = 0.0;
      for (size_t v = 0; v < V; ++v) {
        residual[v] = std::max(0.0f, p[size_t(j)][v] - q[size_t(j)][v]);
        sum += residual[v];
      }
      if (sum > 0.0) for (float &r : residual) r = static_cast<float>(r / sum);
      else residual = p[size_t(j)];
      correction = detail::sample_probs(residual, rng);
      break;
    }
    if (accepted == k) correction = detail::sample_probs(p[size_t(k)], rng);
    if (correction < 0) { error = "correction distribution has no mass"; return false; }
    stats.accepted += accepted;

    // ---- reconcile target state: it must stand after processing d_accepted ----
    if (accepted < k) {
      if (!target.restore(checkpoints.handles[size_t(accepted)], error)) return false;
      ++stats.restores;
    }

    // ---- rebuild the draft rows for t_0..t_accepted from true target hiddens ----
    if (draft.next_position() > pos + 1 && !draft.rewind(pos + 1, error)) return false;
    for (int j = draft.next_position() - pos; j <= accepted; ++j) {
      const std::span<const float> hidden =
          j == 0 ? std::span<const float>(hidden_prev)
                 : std::span<const float>(wides.data() + size_t(j - 1) * W, W);
      const int token = j == 0 ? seq.back() : drafts[size_t(j - 1)];
      if (!draft_step(token, pos + j, hidden, scratch_h, scratch_l)) return false;
    }

    // ---- commit ----
    hidden_prev.assign(wides.begin() + size_t(accepted) * W,
                       wides.begin() + size_t(accepted + 1) * W);
    for (int j = 0; j < accepted; ++j)
      if (!commit(drafts[size_t(j)])) return true;
    if (!commit(correction)) return true;
  }
}

}  // namespace zeromoe::graph
