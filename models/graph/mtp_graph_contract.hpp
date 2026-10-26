#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <span>
#include <string>

namespace zeromoe::graph {

struct MtpGraphShape {
  size_t embedding = 0;
  size_t hidden = 0;
  size_t vocab = 0;
  int capacity = 0;
};

struct MtpGraphRequest {
  int position = 0;
  std::span<const float> embedding;
  std::span<const float> hidden;
  std::span<float> next_hidden;
  std::span<float> logits;
};

// Submission failures may already have changed device KV. Only reset makes
// that session reusable; malformed requests rejected before submission do not.
class MtpGraphCursor {
 public:
  explicit MtpGraphCursor(MtpGraphShape shape) : shape_(shape) {}

  std::string validate(const MtpGraphRequest &r) const {
    if (!shape_.embedding || !shape_.hidden || !shape_.vocab ||
        shape_.capacity <= 0)
      return "invalid draft graph shape";
    if (poisoned_) return "draft graph requires reset after execution failure";
    if (r.position != next_position_)
      return "draft graph accepts only the next sequential position";
    if (r.position < 0 || r.position >= shape_.capacity)
      return "draft graph context capacity exceeded";
    if (r.embedding.size() != shape_.embedding ||
        r.hidden.size() != shape_.hidden ||
        r.next_hidden.size() != shape_.hidden ||
        r.logits.size() != shape_.vocab)
      return "draft graph input/output shape mismatch";
    const std::less<const float *> before;
    if (before(r.next_hidden.data(), r.logits.data() + r.logits.size()) &&
        before(r.logits.data(), r.next_hidden.data() + r.next_hidden.size()))
      return "draft graph output buffers overlap";
    const auto finite = [](float v) { return std::isfinite(v); };
    if (!std::all_of(r.embedding.begin(), r.embedding.end(), finite) ||
        !std::all_of(r.hidden.begin(), r.hidden.end(), finite))
      return "non-finite draft graph input";
    return {};
  }

  void commit() { ++next_position_; }
  // Logical rollback of the draft's attention K/V. Valid because the draft
  // owns only K/V at fixed addresses: rows at or beyond the cursor are masked
  // and overwritten by the next append, so no other state can diverge.
  // Never moves forward, and never un-poisons (a failed submission may have
  // left K/V in an unknown partial state; only reset recovers that).
  std::string rewind(int position) {
    if (poisoned_) return "draft graph requires reset after execution failure";
    if (position < 0 || position > next_position_)
      return "draft rewind target outside the committed prefix";
    next_position_ = position;
    return {};
  }
  void poison() { poisoned_ = true; }
  void reset() { next_position_ = 0; poisoned_ = false; }
  int next_position() const { return next_position_; }
  bool poisoned() const { return poisoned_; }
  const MtpGraphShape &shape() const { return shape_; }

 private:
  MtpGraphShape shape_;
  int next_position_ = 0;
  bool poisoned_ = false;
};

} // namespace zeromoe::graph
