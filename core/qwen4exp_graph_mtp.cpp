#include "qwen4exp_graph_mtp.hpp"

#include "engine.hpp"
#include "logger.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)
#include "ggml_qwen4exp_graph.hpp"
#include "mtp_graph_runtime.hpp"
#include "mtp_speculation.hpp"
#include "qwen4exp_mtp_graph.hpp"

#include <algorithm>
#include <cmath>
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

class GraphDraft final : public SpecDraft {
 public:
  GraphDraft(std::unique_ptr<MtpGraphRuntime> runtime, const MtpSpeculator &embedder,
             size_t D)
      : runtime_(std::move(runtime)), embedder_(embedder), embedding_(D) {}

  bool step(int token, int position, std::span<const float> hidden,
            std::span<float> next_hidden, std::span<float> logits,
            std::string &error) override {
    if (!embedder_.embed(token, embedding_.data())) {
      error = "draft token embedding failed";
      return false;
    }
    return runtime_->replay({position, embedding_, hidden, next_hidden, logits}, error);
  }
  bool rewind(int position, std::string &error) override {
    return runtime_->rewind(position, error);
  }
  int next_position() const override { return runtime_->next_position(); }

 private:
  std::unique_ptr<MtpGraphRuntime> runtime_;
  const MtpSpeculator &embedder_;
  std::vector<float> embedding_;
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

  // Admission order matters: the target's decode graph must be resident BEFORE
  // the draft claims device memory, otherwise the draft starves the decode
  // build (which needs a safety floor of free VRAM), the build is declined, and
  // the target silently runs in a different configuration with different
  // numerics. Build it now, exactly as the ordinary loop does before step 0.
  {
    int build_token = prompt_ids.back();
    if (!qwen4exp_full_graph_decode_prepare(engine, int(n), &build_token)) {
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

  GraphTarget target(engine, ram_mgr, prefetcher, base);
  GraphDraft draft(std::move(runtime), base, D);

  SpecPrompt prompt;
  prompt.tokens = prompt_ids;
  prompt.wide = engine.qwen4exp_wide_prompt;
  prompt.last_logits = last_logits;

  const SpecConfig config{V, W, max_draft, max_new_tokens, eos_id, max_position};
  SpecStats stats;
  const bool ok = speculate(
      target, draft, config, prompt, distribution, rng,
      [&](int token) { return base.emit_token(sequence, token, eos_id, emit); },
      stats, error);

  perf().n_mtp_cycles.fetch_add(stats.cycles, std::memory_order_relaxed);
  perf().n_mtp_drafts_proposed.fetch_add(stats.drafted, std::memory_order_relaxed);
  perf().n_mtp_drafts_accepted.fetch_add(stats.accepted, std::memory_order_relaxed);
  perf().n_mtp_tokens_emitted.fetch_add(stats.emitted, std::memory_order_relaxed);
  LOG_INFO("[MTP-GRAPH] cycles=%llu drafted=%llu accepted=%llu (%.1f%%) emitted=%llu "
           "target_steps=%llu draft_steps=%llu restores=%llu%s\n",
           (unsigned long long)stats.cycles, (unsigned long long)stats.drafted,
           (unsigned long long)stats.accepted,
           stats.drafted ? 100.0 * double(stats.accepted) / double(stats.drafted) : 0.0,
           (unsigned long long)stats.emitted, (unsigned long long)stats.target_steps,
           (unsigned long long)stats.draft_steps, (unsigned long long)stats.restores,
           stats.truncated ? " (truncated at context capacity)" : "");
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
