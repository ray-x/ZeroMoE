#pragma once

#include "mtp_graph_contract.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)
#include "graph_core.hpp"

#include <functional>
#include <map>

namespace zeromoe::graph {

struct MtpGraphState : GraphCore {
  std::shared_ptr<StreamingEngine> owner;
  ggml_context *weight_ctx = nullptr;
  ggml_context *state_ctx = nullptr;
  ggml_tensor *embedding = nullptr;
  ggml_tensor *hidden = nullptr;
  std::map<std::string, ggml_tensor *> diagnostics;
  std::function<std::string(int)> prepare_position;
  std::function<bool()> bindings_valid;
  size_t device_weight_bytes = 0;
  size_t device_state_bytes = 0;

  ~MtpGraphState();
  ggml_context *make_context(size_t bytes);
};

struct MtpGraphMemory {
  size_t weights = 0;
  size_t state = 0;
  size_t device_workspace = 0;
  size_t host_workspace = 0;
  double measure_build_ms = 0.0;
  double execution_build_ms = 0.0;
  double scheduler_alloc_ms = 0.0;
  double reset_ms = 0.0;
  double total_setup_ms = 0.0;
};

class MtpGraphRuntime {
 public:
  using Builder = std::function<std::shared_ptr<MtpGraphState>(std::string &)>;

  static std::unique_ptr<MtpGraphRuntime> create(
      MtpGraphShape shape, size_t device_budget, const Builder &build,
      std::string &error);
  ~MtpGraphRuntime();
  MtpGraphRuntime(const MtpGraphRuntime &) = delete;
  MtpGraphRuntime &operator=(const MtpGraphRuntime &) = delete;

  bool replay(const MtpGraphRequest &request, std::string &error);
  bool reset(std::string &error);
  // Drops draft K/V rows at or after `position` (see MtpGraphCursor::rewind).
  // Invalidates diagnostics from the last replay.
  bool rewind(int position, std::string &error);
  bool read_diagnostic(const std::string &name, std::span<float> output,
                       std::string &error) const;
  bool read_ids(const std::string &name, std::span<int32_t> output,
                std::string &error) const;
  int next_position() const { return cursor_.next_position(); }
  const MtpGraphMemory &memory() const { return memory_; }
  size_t replay_count() const { return replay_count_; }
  size_t build_count() const { return build_count_; }

 private:
  explicit MtpGraphRuntime(MtpGraphShape shape) : cursor_(shape) {}
  MtpGraphCursor cursor_;
  MtpGraphMemory memory_;
  std::shared_ptr<MtpGraphState> state_;
  std::vector<float> hidden_scratch_, logits_scratch_;
  size_t workspace_reservation_ = 0;
  size_t replay_count_ = 0;
  size_t build_count_ = 0;
  bool has_output_ = false;
};

} // namespace zeromoe::graph
#endif
