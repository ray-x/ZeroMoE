#pragma once

// SpecDraft over a graph-native MTP draft runtime (MtpGraphRuntime).
//
// Model independent: the model supplies the runtime (its own graph builder,
// see models/qwen/qwen4exp_mtp_graph.cpp for the Qwen4Exp one) and an
// embedding lookup. Everything else the speculation driver needs from a draft
// (append a row, rewind, report the next position) is the runtime's cursor.

#if defined(ZEROMOE_USE_GGML_CUDA)

#include "mtp_graph_runtime.hpp"
#include "mtp_speculation.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace zeromoe::graph {

class GraphMtpDraft final : public SpecDraft {
 public:
  // Writes the embedding of `token` (`embed_width` floats) to `out`.
  using EmbedFn = std::function<bool(int token, float *out)>;

  GraphMtpDraft(std::unique_ptr<MtpGraphRuntime> runtime, EmbedFn embed,
                size_t embed_width)
      : runtime_(std::move(runtime)), embed_(std::move(embed)),
        embedding_(embed_width) {}

  bool step(int token, int position, std::span<const float> hidden,
            std::span<float> next_hidden, std::span<float> logits,
            std::string &error) override {
    if (!embed_(token, embedding_.data())) {
      error = "draft token embedding failed";
      return false;
    }
    return runtime_->replay({position, embedding_, hidden, next_hidden, logits}, error);
  }
  bool rewind(int position, std::string &error) override {
    return runtime_->rewind(position, error);
  }
  int next_position() const override { return runtime_->next_position(); }
  const MtpGraphMemory &memory() const { return runtime_->memory(); }

 private:
  std::unique_ptr<MtpGraphRuntime> runtime_;
  EmbedFn embed_;
  std::vector<float> embedding_;
};

}  // namespace zeromoe::graph

#endif
