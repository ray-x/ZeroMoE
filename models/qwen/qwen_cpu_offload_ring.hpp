#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

// Fixed-buffer device-fenced handoff for the Qwen4Exp CPUOffload path.
// The class owns a small ring of pinned host result vectors, device result
// vectors, one copy stream, and one H2D completion/reuse event pair per slot.
// It deliberately exposes only opaque stream/pointer types so the header is
// harmless in the non-CUDA build.
class QwenCpuOffloadRing {
public:
  static QwenCpuOffloadRing &instance();

  QwenCpuOffloadRing() = default;
  ~QwenCpuOffloadRing();
  QwenCpuOffloadRing(const QwenCpuOffloadRing &) = delete;
  QwenCpuOffloadRing &operator=(const QwenCpuOffloadRing &) = delete;

  bool ensure(int device, int d_model, int requested_slots);

  // Acquires a slot for a new token.  If the previous use of the slot has
  // not been consumed by the compute stream, this is the only intentional
  // ring backpressure point.  It never synchronizes the copy stream.
  bool acquire(int &slot, double &reuse_wait_ms);

  // Must be called before submitting the matching CPU worker request.
  // `compute_stream` is the shared GGML CUDA backend stream that will consume
  // the device result after this ring's callback queues its event wait.
  bool prepare_slot(int slot, void *compute_stream);
  void *completion_user(int slot);
  void abandon(int slot);

  // Completion callback passed to GgmlQwenCpuMoeRunner.  It queues the H2D
  // copy and the compute-stream wait from the CPU worker thread, then
  // publishes the enqueue state to the decode thread.
  static void completion_callback(void *user, bool success);

  // Waits only until the callback has published the copy/event/fence
  // enqueue.  It does not wait for the PCIe transfer or the compute stream.
  bool wait_fence_enqueued(int slot, double &wait_ms, bool &cpu_success);

  float *host_output(int slot);
  const void *device_output(int slot) const;

  // Records the compute-stream consumer event after the final device combine
  // graph has been queued.  The slot cannot be reused before this event.
  bool mark_consumed(int slot, void *compute_stream);

  double h2d_total_ms() const { return h2d_total_ms_; }
  uint64_t h2d_copies() const { return h2d_copies_; }
  uint64_t fence_failures() const { return fence_failures_; }
  uint64_t reuse_waits() const { return reuse_waits_; }

private:
  static constexpr int kMaxSlots = 4;

  struct Slot {
    QwenCpuOffloadRing *owner = nullptr;
    float *host = nullptr;
    void *device = nullptr;
    void *compute_stream = nullptr;
    void *h2d_stream = nullptr;
    void *h2d_start = nullptr;
    void *h2d_done = nullptr;
    void *consumed = nullptr;
    std::atomic<int> callback_state{0}; // 0=pending, 1=fenced, -1=failed
    bool in_use = false;
    bool consumed_recorded = false;
    bool h2d_collected = false;
  };

  void destroy();
  bool enqueue_completion(Slot &slot, bool success);
  void collect_slot_h2d(Slot &slot);

  std::array<Slot, kMaxSlots> slots_{};
  int device_ = -1;
  int d_model_ = 0;
  int slot_count_ = 0;
  int next_slot_ = 0;
  size_t bytes_ = 0;
  void *copy_stream_ = nullptr;
  size_t host_reserved_bytes_ = 0;
  double h2d_total_ms_ = 0.0;
  uint64_t h2d_copies_ = 0;
  uint64_t fence_failures_ = 0;
  uint64_t reuse_waits_ = 0;
};
