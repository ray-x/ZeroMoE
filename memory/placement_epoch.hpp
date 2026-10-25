#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace zeromoe::memory {

// Serial graph-build/phase-transition contract. A graph borrows a snapshot;
// physical storage cannot retire until every borrower has been destroyed.
class PlacementEpoch {
 public:
  using Lease = std::shared_ptr<const uint64_t>;
  uint64_t generation() const { return generation_; }
  bool borrowed() const { return current_ && current_.use_count() > 1; }
  Lease borrow() const { return current_; }
  bool matches(const Lease &lease) const { return lease && lease == current_; }
  void publish() {
    if (current_) throw std::logic_error("placement must retire before republishing");
    current_ = std::make_shared<const uint64_t>(++generation_);
  }
  bool retire() {
    if (borrowed()) return false;
    if (current_) { current_.reset(); ++generation_; }
    return true;
  }
 private:
  uint64_t generation_ = 0;
  Lease current_;
};

} // namespace zeromoe::memory
