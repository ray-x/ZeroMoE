// CPU-only proof of the speculative verification driver (no CUDA, no ggml).
//
//   c++ -std=c++23 -O2 -Imodels/graph tests/test_mtp_speculation.cpp -o /tmp/zeromoe-mtp-spec
//
// The fake target carries a hash chain as its recurrent state, so any state
// rollback error changes every later logit and is caught by comparing against
// plain one-token-at-a-time decoding.

#include "mtp_speculation.hpp"

#include <cstdio>
#include <map>

using namespace zeromoe::graph;

namespace {

constexpr size_t W = 4;

uint32_t mix(uint32_t a, uint32_t b, uint32_t c = 0) {
  uint32_t x = a * 2654435761u ^ (b + 0x9e3779b9u) * 40503u ^ (c + 77u) * 2246822519u;
  x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
  return x;
}

// Next-token logits as a pure function of the recurrent state.
void model_logits(uint32_t h, size_t V, std::span<float> out, uint32_t noise_salt = 0,
                  float noise = 0.0f) {
  const size_t preferred = mix(h, 1) % V;
  for (size_t v = 0; v < V; ++v) {
    float l = (mix(h, uint32_t(v) + 100) % 1000) / 500.0f;
    if (v == preferred) l += 3.0f;
    if (noise > 0.0f) l += noise * ((mix(h, uint32_t(v), noise_salt) % 1000) / 500.0f - 1.0f) * 4.0f;
    out[v] = l;
  }
}

struct FakeTarget : SpecTarget {
  size_t V;
  uint32_t h = 12345;
  int steps = 0, live = 0, next_handle = 1, fail_at = -1, restores = 0;
  std::map<int, uint32_t> checkpoints;
  explicit FakeTarget(size_t vocab) : V(vocab) {}
  bool step(int token, int position, std::span<float> wide, std::span<float> logits,
            std::string &error) override {
    if (steps++ == fail_at) { error = "injected target failure"; return false; }
    h = mix(h, uint32_t(token), uint32_t(position)) & 0xFFFFFF;  // exact in a float
    wide[0] = float(h); wide[1] = float(position); wide[2] = float(token); wide[3] = 0;
    model_logits(h, V, logits);
    return true;
  }
  bool save(int &handle, std::string &) override {
    handle = next_handle++;
    checkpoints[handle] = h;
    ++live;
    return true;
  }
  bool restore(int handle, std::string &error) override {
    auto it = checkpoints.find(handle);
    if (it == checkpoints.end()) { error = "unknown checkpoint"; return false; }
    h = it->second;
    ++restores;
    return true;
  }
  void release(int handle) override {
    if (checkpoints.erase(handle)) --live;
  }
};

struct FakeDraft : SpecDraft {
  size_t V;
  float noise;
  uint32_t salt;
  int fail_at = -1, steps = 0;
  std::vector<std::pair<int, int>> rows;  // (token, position)
  std::vector<float> row_hidden0;         // hidden[0] consumed by each row
  bool order_violation = false;
  FakeDraft(size_t vocab, float noise_, uint32_t salt_) : V(vocab), noise(noise_), salt(salt_) {}
  bool step(int token, int position, std::span<const float> hidden,
            std::span<float> next_hidden, std::span<float> logits,
            std::string &error) override {
    if (steps++ == fail_at) { error = "injected draft failure"; return false; }
    if (position != int(rows.size())) { order_violation = true; error = "draft position gap"; return false; }
    rows.push_back({token, position});
    row_hidden0.push_back(hidden[0]);
    const uint32_t h = mix(uint32_t(hidden[0]), uint32_t(token), uint32_t(position)) & 0xFFFFFF;
    next_hidden[0] = float(h);
    for (size_t i = 1; i < next_hidden.size(); ++i) next_hidden[i] = 0;
    model_logits(h, V, logits, salt, noise);
    return true;
  }
  bool rewind(int position, std::string &error) override {
    if (position < 0 || position > int(rows.size())) { error = "bad rewind"; return false; }
    rows.resize(size_t(position));
    row_hidden0.resize(size_t(position));
    return true;
  }
  int next_position() const override { return int(rows.size()); }
};

SpecDistribution greedy() {
  return [](std::span<const float> logits, const std::vector<int> &, std::vector<float> &probs) {
    probs.assign(logits.size(), 0.0f);
    probs[size_t(std::max_element(logits.begin(), logits.end()) - logits.begin())] = 1.0f;
    return true;
  };
}

SpecDistribution softmax(float temperature) {
  return [temperature](std::span<const float> logits, const std::vector<int> &,
                       std::vector<float> &probs) {
    probs.resize(logits.size());
    const float mx = *std::max_element(logits.begin(), logits.end());
    double sum = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
      probs[i] = std::exp((logits[i] - mx) / temperature);
      sum += probs[i];
    }
    for (float &p : probs) p = float(p / sum);
    return true;
  };
}

// Prefill the fake target token by token, producing what a real prefill exports.
SpecPrompt make_prompt(FakeTarget &t, const std::vector<int> &tokens) {
  SpecPrompt p;
  p.tokens = tokens;
  p.last_logits.resize(t.V);
  std::string e;
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::vector<float> wide(W), logits(t.V);
    t.step(tokens[i], int(i), wide, logits, e);
    p.wide.insert(p.wide.end(), wide.begin(), wide.end());
    if (i + 1 == tokens.size()) p.last_logits = logits;
  }
  return p;
}

// Plain decoding: one target step per token, no draft.
std::vector<int> plain_decode(size_t V, const std::vector<int> &prompt, int max_new,
                              int eos, int max_position, const SpecDistribution &dist,
                              std::mt19937 &rng) {
  FakeTarget t(V);
  SpecPrompt p = make_prompt(t, prompt);
  std::vector<int> out, seq = prompt;
  std::vector<float> logits = p.last_logits, wide(W);
  std::string e;
  while (int(out.size()) < max_new) {
    const int pos = int(seq.size());
    std::vector<float> probs;
    dist(logits, seq, probs);
    double total = 0;
    for (float x : probs) total += x;
    std::uniform_real_distribution<double> u(0.0, total);
    double target = u(rng), run = 0;
    int tok = -1;
    for (size_t i = 0; i < probs.size(); ++i) {
      if (probs[i] <= 0) continue;
      tok = int(i);
      run += probs[i];
      if (target < run) break;
    }
    out.push_back(tok);
    seq.push_back(tok);
    if (tok == eos) break;
    if (pos >= max_position) break;
    t.step(tok, pos, wide, logits, e);
  }
  return out;
}

}  // namespace

int main() {
  int failures = 0;
  const auto check = [&](bool ok, const char *name) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", name); ++failures; }
  };

  // 1) Greedy speculation is bit-identical to plain decoding for every draft
  //    quality (perfect, noisy, adversarial) and draft length, including k=0.
  {
    const size_t V = 24;
    bool all_equal = true, no_leak = true, ordered = true, steps_ok = true;
    uint64_t total_accepted = 0, total_drafted = 0;
    for (float noise : {0.0f, 0.15f, 0.6f, 3.0f})
      for (int K : {0, 1, 2, 3, 5})
        for (uint32_t seed = 1; seed <= 12; ++seed) {
          std::mt19937 rng(seed);
          std::vector<int> prompt;
          const int plen = 1 + int(seed % 7);
          for (int i = 0; i < plen; ++i) prompt.push_back(int((seed * 31 + i * 17) % V));
          const int max_new = 3 + int(seed % 23);
          const int max_position = 200;
          std::mt19937 plain_rng(1);
          const auto ref = plain_decode(V, prompt, max_new, -1, max_position, greedy(), plain_rng);

          FakeTarget target(V);
          FakeDraft draft(V, noise, seed);
          SpecPrompt sp = make_prompt(target, prompt);
          std::vector<int> got;
          SpecStats stats;
          std::string error;
          SpecConfig cfg{V, W, K, max_new, -1, max_position};
          const bool ok = speculate(target, draft, cfg, sp, greedy(), rng,
                                    [&](int t) { got.push_back(t); return true; }, stats, error);
          if (!ok || got != ref) {
            all_equal = false;
            std::fprintf(stderr, "  mismatch noise=%g K=%d seed=%u ok=%d err=%s\n", noise, K, seed, ok, error.c_str());
          }
          no_leak = no_leak && target.live == 0;
          ordered = ordered && !draft.order_violation && int(got.size()) <= max_new;
          steps_ok = steps_ok && stats.target_steps == stats.cycles + stats.drafted;
          total_accepted += stats.accepted;
          total_drafted += stats.drafted;
        }
    check(all_equal, "greedy speculation equals plain decoding for all draft qualities/lengths");
    check(no_leak, "every checkpoint released");
    check(ordered, "draft positions stay sequential and token budget respected");
    check(steps_ok, "target steps == cycles + drafted");
    check(total_drafted > 0 && total_accepted > 0 && total_accepted < total_drafted,
          "test exercises both acceptance and rejection");
  }

  // 2) The draft's rebuilt rows are paired with the TRUE target hidden.
  {
    const size_t V = 24;
    FakeTarget target(V);
    FakeDraft draft(V, 0.6f, 5);
    std::mt19937 rng(9);
    const std::vector<int> prompt{3, 9, 1};
    SpecPrompt sp = make_prompt(target, prompt);
    std::vector<int> got;
    SpecStats stats;
    std::string error;
    SpecConfig cfg{V, W, 3, 30, -1, 200};
    check(speculate(target, draft, cfg, sp, greedy(), rng,
                    [&](int t) { got.push_back(t); return true; }, stats, error),
          "rebuild run succeeds");
    // Replay the committed sequence on a clean target and compare hidden[0]
    // consumed by each draft row below the live position.
    FakeTarget clean(V);
    std::vector<int> seq = prompt;
    seq.insert(seq.end(), got.begin(), got.end());
    bool paired = true;
    uint32_t h = 12345;
    std::vector<uint32_t> true_h;  // true_h[p] = state after processing position p
    for (size_t p = 0; p < seq.size(); ++p) {
      h = mix(h, uint32_t(seq[p]), uint32_t(p)) & 0xFFFFFF;
      true_h.push_back(h);
    }
    for (int p = 1; p < draft.next_position() && size_t(p) < seq.size(); ++p) {
      if (draft.rows[size_t(p)].first != seq[size_t(p)]) paired = false;
      if (uint32_t(draft.row_hidden0[size_t(p)]) != true_h[size_t(p - 1)]) paired = false;
    }
    check(paired, "draft rows hold committed tokens paired with true target hidden");
    check(stats.restores > 0, "rollback path exercised");
  }

  // 3) Sampling is exact: the joint of the first three tokens matches the
  //    target's exact joint, for good and bad drafts, with real multi-token drafts.
  {
    const size_t V = 4;
    const float T = 0.9f;
    const int L = 3;
    const std::vector<int> prompt{2, 1};
    // Exact joint by recursion over the fake target.
    std::vector<double> exact(size_t(V * V * V), 0.0);
    {
      FakeTarget root(V);
      SpecPrompt sp0 = make_prompt(root, prompt);
      std::function<void(FakeTarget, std::vector<float>, std::vector<int>, double, int)> walk =
          [&](FakeTarget t, std::vector<float> logits, std::vector<int> toks, double pr, int depth) {
            if (depth == L) {
              size_t idx = 0;
              for (int x : toks) idx = idx * V + size_t(x);
              exact[idx] += pr;
              return;
            }
            std::vector<float> probs;
            softmax(T)(logits, prompt, probs);
            for (size_t a = 0; a < V; ++a) {
              FakeTarget next = t;
              std::vector<float> w(W), l(V);
              std::string e;
              next.step(int(a), int(prompt.size()) + depth, w, l, e);
              auto nt = toks;
              nt.push_back(int(a));
              walk(next, l, nt, pr * probs[a], depth + 1);
            }
          };
      walk(root, sp0.last_logits, {}, 1.0, 0);
    }
    for (float noise : {0.0f, 0.5f, 3.0f}) {
      const int N = 120000;
      std::vector<double> hist(exact.size(), 0.0);
      std::mt19937 rng(2024);
      uint64_t drafted = 0, accepted = 0;
      for (int i = 0; i < N; ++i) {
        FakeTarget target(V);
        FakeDraft draft(V, noise, 3);
        SpecPrompt sp = make_prompt(target, prompt);
        std::vector<int> got;
        SpecStats stats;
        std::string error;
        SpecConfig cfg{V, W, 3, 6, -1, 100};  // budget 6 => real k=3 drafts
        speculate(target, draft, cfg, sp, softmax(T), rng,
                  [&](int t) { got.push_back(t); return true; }, stats, error);
        drafted += stats.drafted;
        accepted += stats.accepted;
        size_t idx = 0;
        for (int j = 0; j < L; ++j) idx = idx * V + size_t(got[size_t(j)]);
        hist[idx] += 1.0;
      }
      double tv = 0.0;
      for (size_t i = 0; i < exact.size(); ++i) tv += std::abs(hist[i] / N - exact[i]);
      tv *= 0.5;
      const double rate = drafted ? double(accepted) / double(drafted) : 0.0;
      std::fprintf(stderr, "  joint TV distance noise=%g: %.4f (accept rate %.3f)\n", noise, tv, rate);
      check(drafted > 0, "sampling test really drafts");
      check(tv < 0.02, "speculative sampling preserves the target joint distribution");
      if (noise == 0.0f) check(rate > 0.999, "a perfect draft is mostly accepted");
      if (noise == 3.0f) check(rate < 0.6, "a bad draft is mostly rejected");
    }
  }

  // 4) EOS, stop-from-emit, and capacity truncation.
  {
    const size_t V = 24;
    std::mt19937 plain_rng(1);
    const std::vector<int> prompt{5, 6};
    const auto ref = plain_decode(V, prompt, 40, -1, 200, greedy(), plain_rng);
    const int eos = ref[7];
    std::mt19937 p2(1);
    const auto ref_eos = plain_decode(V, prompt, 40, eos, 200, greedy(), p2);
    FakeTarget target(V);
    FakeDraft draft(V, 0.3f, 1);
    SpecPrompt sp = make_prompt(target, prompt);
    std::vector<int> got;
    SpecStats stats;
    std::string error;
    std::mt19937 rng(1);
    SpecConfig cfg{V, W, 4, 40, eos, 200};
    check(speculate(target, draft, cfg, sp, greedy(), rng,
                    [&](int t) { got.push_back(t); return true; }, stats, error) &&
              got == ref_eos && got.back() == eos,
          "stops exactly at EOS with plain-decoding output");

    FakeTarget t2(V);
    FakeDraft d2(V, 0.3f, 1);
    SpecPrompt sp2 = make_prompt(t2, prompt);
    got.clear();
    SpecStats s2;
    SpecConfig cfg2{V, W, 4, 40, -1, 200};
    check(speculate(t2, d2, cfg2, sp2, greedy(), rng,
                    [&](int t) { got.push_back(t); return got.size() < 5; }, s2, error) &&
              got.size() == 5 && t2.live == 0,
          "emit callback can stop generation");

    FakeTarget t3(V);
    FakeDraft d3(V, 0.3f, 1);
    SpecPrompt sp3 = make_prompt(t3, prompt);
    got.clear();
    SpecStats s3;
    SpecConfig cfg3{V, W, 4, 40, -1, 9};  // capacity 9 positions
    check(speculate(t3, d3, cfg3, sp3, greedy(), rng,
                    [&](int t) { got.push_back(t); return true; }, s3, error) &&
              s3.truncated && got.size() < 40 && int(prompt.size() + got.size()) <= 10 &&
              std::equal(got.begin(), got.end(), ref.begin()),
          "truncates at capacity with a correct prefix");
  }

  // 5) Failures propagate, checkpoints are released, bad input is rejected.
  {
    const size_t V = 24;
    const std::vector<int> prompt{1, 2, 3};
    std::mt19937 rng(5);
    for (int fail_at : {3, 4, 5, 8}) {
      FakeTarget target(V);
      FakeDraft draft(V, 0.2f, 1);
      SpecPrompt sp = make_prompt(target, prompt);
      target.fail_at = target.steps + (fail_at - 3);
      SpecStats stats;
      std::string error;
      SpecConfig cfg{V, W, 3, 30, -1, 200};
      const bool ok = speculate(target, draft, cfg, sp, greedy(), rng, [](int) { return true; }, stats, error);
      check(!ok && !error.empty() && target.live == 0, "target failure propagates and releases checkpoints");
    }
    for (int fail_at : {0, 2, 4, 7}) {
      FakeTarget target(V);
      FakeDraft draft(V, 0.2f, 1);
      draft.fail_at = fail_at;
      SpecPrompt sp = make_prompt(target, prompt);
      SpecStats stats;
      std::string error;
      SpecConfig cfg{V, W, 3, 30, -1, 200};
      const bool ok = speculate(target, draft, cfg, sp, greedy(), rng, [](int) { return true; }, stats, error);
      check(!ok && !error.empty() && target.live == 0, "draft failure propagates and releases checkpoints");
    }
    FakeTarget target(V);
    FakeDraft draft(V, 0.2f, 1);
    SpecPrompt sp = make_prompt(target, prompt);
    SpecStats stats;
    std::string error;
    SpecPrompt bad = sp;
    bad.wide.pop_back();
    SpecConfig cfg{V, W, 3, 30, -1, 200};
    check(!speculate(target, draft, cfg, bad, greedy(), rng, [](int) { return true; }, stats, error),
          "rejects malformed prompt hidden rows");
    SpecConfig zero{V, W, 3, 0, -1, 200};
    check(!speculate(target, draft, zero, sp, greedy(), rng, [](int) { return true; }, stats, error),
          "rejects zero token budget");
    draft.rows.push_back({0, 0});
    check(!speculate(target, draft, cfg, sp, greedy(), rng, [](int) { return true; }, stats, error),
          "rejects a draft that was not reset");
  }

  std::printf("MTP speculation: %s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
