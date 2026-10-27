#include "qwen4exp_graph_mtp.hpp"

#include "engine.hpp"
#include "logger.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)
#include "ggml_qwen4exp_graph.hpp"
#include "mtp_graph_draft.hpp"
#include "mtp_graph_report.hpp"
#include "nvtx_range.hpp"
#include "mtp_graph_runtime.hpp"
#include "mtp_speculation.hpp"
#include "qwen4exp_mtp_graph.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <chrono>
#include <memory>

extern void layers_forward_gguf(StreamingEngine &engine,
                                RAMSafetyManager &ram_mgr,
                                LockFreePrefetcher &prefetcher,
                                float *hidden_states, int S, int pos_base,
                                const int *token_ids);
extern void hc_expand_embedding(StreamingEngine &engine,
                                const float *hidden_states, int S, int D);

namespace {

using namespace zeromoe::graph;

constexpr size_t kDraftBudgetBytes = size_t{3} << 30;  // NextN weights + K/V + workspace

// Optional per-phase accounting (ZEROMOE_MTP_PROFILE=1): milliseconds spent in
// each phase of graph-native MTP, reported with the [MTP-GRAPH] line.
struct MtpProfile {
  double embed_ms = 0, prepare_ms = 0, run_ms = 0, copy_ms = 0, commit_ms = 0, draft_ms = 0;
  uint64_t windows = 0, drafts = 0;
};
MtpProfile g_profile;
bool profile_enabled() {
  static const bool on = [] { const char *v = std::getenv("ZEROMOE_MTP_PROFILE"); return v && v[0] == '1'; }();
  return on;
}
double since_ms(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

bool all_finite(std::span<const float> v) {
  return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
}

// One token through the production full-graph decode path. Same bracket the
// generation loop uses (embedding -> HC expand -> layers_forward_gguf), with
// the logits destination installed for exactly this call and the wide
// pre-head residual read back from the graph. Anything that is not the full
// graph is an error: this path never falls back to layered execution.
class GraphTarget final : public SpecTarget {
 public:
  GraphTarget(StreamingEngine &engine, RAMSafetyManager &ram,
              LockFreePrefetcher &prefetcher, const MtpSpeculator &embedder)
      : engine_(engine), ram_(ram), prefetcher_(prefetcher), embedder_(embedder) {}

  bool step(int token, int position, std::span<float> wide,
            std::span<float> logits, std::string &error) override {
    const auto &c = engine_.model_file.architecture_config;
    const size_t D = size_t(c.hidden), W = D * size_t(c.hc_mult);
    if (wide.size() != W || logits.size() != size_t(c.vocab)) {
      error = "target step buffers have the wrong size";
      return false;
    }
    hidden_.assign(D, 0.0f);
    if (!embedder_.embed(token, hidden_.data())) {
      error = "token embedding failed";
      return false;
    }
    hc_expand_embedding(engine_, hidden_.data(), 1, int(D));
    engine_.qwen4exp_full_graph_logits_ready = false;
    engine_.qwen4exp_full_graph_output_ready = false;
    engine_.qwen4exp_full_graph_logits_target = logits.data();
    engine_.qwen4exp_full_graph_logits_target_size = c.vocab;
    layers_forward_gguf(engine_, ram_, prefetcher_, hidden_.data(), 1, position, &token);
    engine_.qwen4exp_full_graph_logits_target = nullptr;
    engine_.qwen4exp_full_graph_logits_target_size = 0;
    if (!engine_.qwen4exp_full_graph_failure.empty()) {
      error = engine_.qwen4exp_full_graph_failure;
      return false;
    }
    if (!engine_.qwen4exp_full_graph_output_ready ||
        !engine_.qwen4exp_full_graph_logits_ready) {
      error = "target step did not execute on the full graph";
      return false;
    }
    engine_.qwen4exp_full_graph_logits_ready = false;
    if (engine_.qwen4exp_wide_out.size() != W) {
      error = "decode graph did not export the wide residual";
      return false;
    }
    std::copy(engine_.qwen4exp_wide_out.begin(), engine_.qwen4exp_wide_out.end(),
              wide.begin());
    if (!all_finite(wide) || !all_finite(logits)) {
      error = "target step produced non-finite output";
      return false;
    }
    return true;
  }

  bool save(int &handle, std::string &error) override {
    if (qwen4exp_full_graph_state_save(engine_, handle)) return true;
    error = "target state checkpoint failed (no live decode graph or out of memory)";
    return false;
  }
  bool restore(int handle, std::string &error) override {
    if (qwen4exp_full_graph_state_restore(engine_, handle)) return true;
    error = "target state restore failed (decode graph was rebuilt, e.g. KV capacity "
            "growth; raise ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE_KV_CAPACITY)";
    return false;
  }
  void release(int handle) override { qwen4exp_full_graph_state_release(handle); }

 private:
  StreamingEngine &engine_;
  RAMSafetyManager &ram_;
  LockFreePrefetcher &prefetcher_;
  const MtpSpeculator &embedder_;
  std::vector<float> hidden_;
};

// Times the draft's steps for ZEROMOE_MTP_PROFILE without touching the shared
// GraphMtpDraft.
class TimedDraft final : public SpecDraft {
 public:
  explicit TimedDraft(SpecDraft &inner) : inner_(inner) {}
  bool step(int token, int position, std::span<const float> hidden,
            std::span<float> next_hidden, std::span<float> logits,
            std::string &error) override {
    ZE_NVTX("mtp.draft_step");
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = inner_.step(token, position, hidden, next_hidden, logits, error);
    g_profile.draft_ms += since_ms(t0);
    ++g_profile.drafts;
    return ok;
  }
  bool rewind(int position, std::string &error) override { return inner_.rewind(position, error); }
  int next_position() const override { return inner_.next_position(); }
 private:
  SpecDraft &inner_;
};

// Target that verifies a whole window of tokens in one pass of the decode graph
// (built for S = max_draft + 1 tokens). Per-token recurrent snapshots inside the
// graph let commit_prefix keep exactly the accepted tokens, so a rejection costs
// a few device copies instead of a restore plus replay.
class GraphWindowTarget final : public SpecTarget {
 public:
  GraphWindowTarget(StreamingEngine &engine, RAMSafetyManager &ram,
                    LockFreePrefetcher &prefetcher, const MtpSpeculator &embedder,
                    int window)
      : engine_(engine), ram_(ram), prefetcher_(prefetcher), embedder_(embedder),
        window_(window) {}

  bool supports_batch() const override { return true; }
  bool supports_prefix_commit() const override { return true; }

  bool step(int token, int position, std::span<float> wide, std::span<float> logits,
            std::string &error) override {
    const int tokens[1] = {token};
    return step_batch(std::span<const int>(tokens, 1), position, wide, logits, error) &&
           commit_prefix(1, error);
  }

  bool step_batch(std::span<const int> tokens, int position, std::span<float> wide,
                  std::span<float> logits, std::string &error) override {
    ZE_NVTX("mtp.verify_window");
    const auto &c = engine_.model_file.architecture_config;
    const size_t D = size_t(c.hidden), W = D * size_t(c.hc_mult), V = size_t(c.vocab);
    const size_t n = tokens.size();
    if (n == 0 || n > size_t(window_) || wide.size() != n * W || logits.size() != n * V) {
      error = "window step buffers have the wrong size";
      return false;
    }
    // The graph always runs `window_` tokens; the tail is padding whose rows
    // are discarded and whose state effects commit_prefix never keeps.
    auto t0 = std::chrono::steady_clock::now();
    padded_.assign(tokens.begin(), tokens.end());
    padded_.resize(size_t(window_), tokens.back());
    hidden_.assign(size_t(window_) * D, 0.0f);
    for (int i = 0; i < window_; ++i)
      if (!embedder_.embed(padded_[size_t(i)], hidden_.data() + size_t(i) * D)) {
        error = "token embedding failed";
        return false;
      }
    hc_expand_embedding(engine_, hidden_.data(), window_, int(D));
    g_profile.embed_ms += since_ms(t0);
    t0 = std::chrono::steady_clock::now();
    if (!qwen4exp_full_graph_decode_prepare(engine_, position, padded_.data()) ||
        !engine_.qwen4exp_full_graph_failure.empty()) {
      error = engine_.qwen4exp_full_graph_failure.empty()
                  ? "window decode graph is not ready" : engine_.qwen4exp_full_graph_failure;
      return false;
    }
    g_profile.prepare_ms += since_ms(t0);
    t0 = std::chrono::steady_clock::now();
    all_logits_.resize(size_t(window_) * V);
    all_wide_.resize(size_t(window_) * W);
    if (!qwen4exp_full_graph_window_run(engine_, window_, position, padded_.data(),
                                        all_logits_.data(), all_wide_.data())) {
      error = "window decode graph execution failed";
      return false;
    }
    g_profile.run_ms += since_ms(t0);
    t0 = std::chrono::steady_clock::now();
    std::copy_n(all_logits_.begin(), n * V, logits.begin());
    std::copy_n(all_wide_.begin(), n * W, wide.begin());
    if (!all_finite(wide) || !all_finite(logits)) {
      error = "window step produced non-finite output";
      return false;
    }
    g_profile.copy_ms += since_ms(t0);
    ++g_profile.windows;
    return true;
  }

  bool commit_prefix(int m, std::string &error) override {
    ZE_NVTX("mtp.commit_prefix");
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = qwen4exp_full_graph_window_commit(engine_, m);
    g_profile.commit_ms += since_ms(t0);
    if (ok) return true;
    error = "window prefix commit failed";
    return false;
  }

  bool save(int &handle, std::string &error) override {
    if (qwen4exp_full_graph_state_save(engine_, handle)) return true;
    error = "target state checkpoint failed";
    return false;
  }
  bool restore(int handle, std::string &error) override {
    if (qwen4exp_full_graph_state_restore(engine_, handle)) return true;
    error = "target state restore failed";
    return false;
  }
  void release(int handle) override { qwen4exp_full_graph_state_release(handle); }

 private:
  StreamingEngine &engine_;
  RAMSafetyManager &ram_;
  LockFreePrefetcher &prefetcher_;
  const MtpSpeculator &embedder_;
  int window_;
  std::vector<int> padded_;
  std::vector<float> hidden_, all_logits_, all_wide_;
};

}  // namespace

bool qwen4exp_graph_mtp_run(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                            LockFreePrefetcher &prefetcher,
                            const std::vector<int> &prompt_ids,
                            const std::vector<float> &last_logits,
                            int max_new_tokens, int eos_id,
                            const zeromoe::graph::SpecDistribution &distribution,
                            std::vector<int> &sequence, std::mt19937 &rng,
                            const MtpEmitFn &emit, std::string &error) {
  error.clear();
  const auto &c = engine.model_file.architecture_config;
  const size_t D = size_t(c.hidden), W = D * size_t(c.hc_mult), V = size_t(c.vocab);
  const size_t n = prompt_ids.size();
  if (engine.family_ops.family != ModelFamily::Qwen4Exp || !engine.has_mtp_model ||
      !engine.mtp_enabled || !engine.qwen4exp_wide_export || c.hc_mult <= 1) {
    error = "graph-native MTP needs Qwen4Exp, the shared MTP checkpoint, mtp_enabled "
            "and the wide-residual export";
    return false;
  }
  if (n == 0 || sequence != prompt_ids || last_logits.size() != V ||
      engine.qwen4exp_wide_prompt.size() != n * W || max_new_tokens <= 0) {
    error = "graph-native MTP needs the exported prompt rows and last-token logits "
            "of a fresh, fully prefilled prompt";
    return false;
  }

  MtpSpeculator base(engine, ram_mgr, prefetcher);  // embedding + sampling math only
  const int max_draft = std::max(1, engine.mtp_draft_tokens);
  const int max_position = int(std::min<size_t>(
      size_t(std::max(engine.max_t, 1)), n + size_t(max_new_tokens) + size_t(max_draft) + 2));


  // Timing probe (ZEROMOE_MTP_WINDOW_PROBE=S): what does a window of S
  // consecutive tokens cost compared with one? Runs the last S prompt tokens
  // as a window at the same position repeatedly (idempotent state writes), so
  // routing is text-like. Not a generation path; always returns false.
  if (const char *probe_env = std::getenv("ZEROMOE_MTP_WINDOW_PROBE")) {
    const int S = std::clamp(std::atoi(probe_env), 1, 8);
    engine.qwen4exp_decode_window = S;
    std::vector<int> tokens(prompt_ids.end() - std::min<size_t>(S, n), prompt_ids.end());
    while (int(tokens.size()) < S) tokens.push_back(tokens.back());
    const int pos = int(n);
    if (!qwen4exp_full_graph_decode_prepare(engine, pos, tokens.data())) {
      error = "probe: window decode graph could not be built";
      return false;
    }
    std::vector<float> hidden(size_t(S) * D);
    for (int i = 0; i < S; ++i)
      if (!base.embed(tokens[size_t(i)], hidden.data() + size_t(i) * D)) {
        error = "probe: embedding failed";
        return false;
      }
    hc_expand_embedding(engine, hidden.data(), S, int(D));
    std::vector<float> plogits(size_t(S) * V), pwide(size_t(S) * W);
    const int reps = 40;
    int checkpoint = 0;
    if (!qwen4exp_full_graph_state_save(engine, checkpoint)) {
      error = "probe: state checkpoint failed";
      return false;
    }
    double total = 0.0, best = 1e30;
    for (int r = 0; r < reps + 5; ++r) {
      if (!qwen4exp_full_graph_decode_prepare(engine, pos, tokens.data())) {
        error = "probe: window prepare failed";
        return false;
      }
      const auto t0 = std::chrono::steady_clock::now();
      if (!qwen4exp_full_graph_window_run(engine, S, pos, tokens.data(), plogits.data(),
                                          pwide.data())) {
        error = "probe: window run failed";
        return false;
      }
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
      if (r >= 5) { total += ms; best = std::min(best, ms); }
      if (!qwen4exp_full_graph_state_restore(engine, checkpoint)) {
        error = "probe: state restore failed";
        return false;
      }
    }
    qwen4exp_full_graph_state_release(checkpoint);
    std::vector<int> amax;
    for (int i = 0; i < S; ++i) {
      const float *row = plogits.data() + size_t(i) * V;
      amax.push_back(int(std::max_element(row, row + V) - row));
    }
    LOG_INFO("[MTP-WINDOW-PROBE] S=%d mean=%.2f ms best=%.2f ms (reps=%d) argmax[0]=%d\n", S,
             total / reps, best, reps, amax[0]);
    error = "window probe finished";
    return false;
  }

  // Admission order matters: the target's decode graph must be resident BEFORE
  // the draft claims device memory, otherwise the draft starves the decode
  // build (which needs a safety floor of free VRAM), the build is declined, and
  // the target silently runs in a different configuration with different
  // numerics. Build it now, exactly as the ordinary loop does before step 0.
  const char *window_env = std::getenv("ZEROMOE_QWEN4EXP_MTP_WINDOW");
  const bool use_window = !(window_env && window_env[0] == '0') && max_draft >= 1 &&
                          max_draft + 1 <= 8 &&
                          engine.family_ops.family == ModelFamily::Qwen4Exp;
  const int window = use_window ? max_draft + 1 : 1;
  engine.qwen4exp_decode_window = window;
  {
    std::vector<int> build_tokens(size_t(window), prompt_ids.back());
    if (!qwen4exp_full_graph_decode_prepare(engine, int(n), build_tokens.data())) {
      error = "target decode graph could not be built before draft admission";
      return false;
    }
  }

  qwen4exp_mtp::Options options;
  options.device = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  options.capacity = max_position;
  options.device_budget = kDraftBudgetBytes;
  // The engine outlives this call; the draft only borrows it.
  std::shared_ptr<StreamingEngine> borrowed(std::shared_ptr<void>(), &engine);
  auto runtime = qwen4exp_mtp::create(borrowed, options, error);
  if (!runtime) return false;

  GraphTarget stepwise_target(engine, ram_mgr, prefetcher, base);
  GraphWindowTarget window_target(engine, ram_mgr, prefetcher, base, window);
  SpecTarget &target = use_window ? static_cast<SpecTarget &>(window_target)
                                  : static_cast<SpecTarget &>(stepwise_target);
  GraphMtpDraft draft(
      std::move(runtime),
      [&base](int token, float *out) { return base.embed(token, out); }, D);

  SpecPrompt prompt;
  prompt.tokens = prompt_ids;
  prompt.wide = engine.qwen4exp_wide_prompt;
  prompt.last_logits = last_logits;

  const SpecConfig config{V, W, max_draft, max_new_tokens, eos_id, max_position};
  SpecStats stats;
  TimedDraft timed_draft(draft);
  const auto speculate_begin = std::chrono::steady_clock::now();
  const bool ok = speculate(
      target, timed_draft, config, prompt, distribution, rng,
      [&](int token) { return base.emit_token(sequence, token, eos_id, emit); },
      stats, error);

  if (profile_enabled()) {
    const double total = since_ms(speculate_begin);
    LOG_INFO("[MTP-PROFILE] speculate=%.0fms windows=%llu: embed+expand=%.0f prepare=%.0f run=%.0f "
             "copy+finite=%.0f commit=%.0f | draft steps=%llu %.0f ms | other (sampling, driver)=%.0f ms\n",
             total, (unsigned long long)g_profile.windows, g_profile.embed_ms,
             g_profile.prepare_ms, g_profile.run_ms, g_profile.copy_ms, g_profile.commit_ms,
             (unsigned long long)g_profile.drafts, g_profile.draft_ms,
             total - g_profile.embed_ms - g_profile.prepare_ms - g_profile.run_ms -
                 g_profile.copy_ms - g_profile.commit_ms - g_profile.draft_ms);
  }
  mtp_graph_report(stats);
  return ok;
}

#else  // !ZEROMOE_USE_GGML_CUDA

bool qwen4exp_graph_mtp_run(StreamingEngine &, RAMSafetyManager &, LockFreePrefetcher &,
                            const std::vector<int> &, const std::vector<float> &, int, int,
                            const zeromoe::graph::SpecDistribution &, std::vector<int> &,
                            std::mt19937 &, const MtpEmitFn &, std::string &error) {
  error = "graph-native MTP requires the CUDA build";
  return false;
}

#endif
