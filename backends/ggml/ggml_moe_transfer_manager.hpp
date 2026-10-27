#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace moe {

// Owns the transfer-side state inherited by GgmlMoeRunner. Public runner
// methods remain source-compatible, while their implementations live in
// ggml_moe_transfer_manager.cpp and operate on this isolated state.
class GgmlMoeTransferManager {
 protected:
  static constexpr int kTransferBuffers = 2;
  static constexpr int kWeightBuffers = 2;
  static constexpr int kWeightEvents = 64;
  static constexpr int kH2dTimingEvents = 128;

  GgmlMoeTransferManager() = default;
  ~GgmlMoeTransferManager();
  GgmlMoeTransferManager(const GgmlMoeTransferManager &) = delete;
  GgmlMoeTransferManager &operator=(const GgmlMoeTransferManager &) = delete;

  void reset_transfer_state(int device) noexcept;

  size_t weight_expert_stride_ = 0;
  size_t weight_raw_expert_stride_ = 0;
  size_t weight_tail_padding_ = 0;
  size_t weight_gate_plane_offset_ = 0;
  size_t weight_up_plane_offset_ = 0;
  size_t weight_down_plane_offset_ = 0;
  void *packed_wbuf_[kWeightBuffers] = {};
  size_t packed_wbuf_bytes_ = 0;
  uint8_t *packed_base_ = nullptr;

  uint8_t *h_stage_[kTransferBuffers] = {};
  void *d_stage_slot_map_[kTransferBuffers] = {};
  std::vector<int32_t> stage_slot_map_[kTransferBuffers];
  std::vector<int32_t> packed_h2d_slot_map_[kTransferBuffers];
  std::vector<int32_t> d2d_slot_map_[kTransferBuffers];
  size_t stage_bytes_ = 0;
  size_t stage_host_reservation_bytes_ = 0;
  int stage_slot_capacity_ = 0;
  void *copy_stream_ = nullptr;
  void *copy_backend_ = nullptr;
  void *ev_free_[kTransferBuffers] = {};
  void *ev_t0_[kTransferBuffers] = {};
  void *backend_copy_t0_event_[kTransferBuffers] = {};
  void *backend_copy_t1_event_[kTransferBuffers] = {};
  void *backend_copy_split_event_[kTransferBuffers] = {};
  void *backend_stage_event_[kTransferBuffers] = {};
  void *h2d_t0_event_[kTransferBuffers][kH2dTimingEvents] = {};
  void *h2d_t1_event_[kTransferBuffers][kH2dTimingEvents] = {};
  int h2d_timing_count_[kTransferBuffers] = {};
  void *backend_weight_done_event_[kWeightEvents] = {};
  void *backend_weight_last_event_[kWeightBuffers] = {};
  bool backend_weight_buffer_in_use_[kWeightBuffers] = {};
  int backend_weight_event_next_ = 0;
  int weight_buffer_index_ = 0;
  static constexpr bool backend_stream_copies_ = false;
  int weight_buffer_count_ = 1;
  bool raw_d2d_pending_ = false;
  bool batch_d2d_unpack_ = false;
  std::chrono::steady_clock::time_point
      backend_copy_t0_[kTransferBuffers];
  bool pending_[kTransferBuffers] = {};
  size_t pending_bytes_[kTransferBuffers] = {};
  size_t pending_d2d_bytes_[kTransferBuffers] = {};
  bool copy_split_recorded_[kTransferBuffers] = {};

  struct DeferredVramHold {
    void *topology = nullptr;
    int layer = -1;
    int expert = -1;
    const uint8_t *device_ptr = nullptr;
  };
  std::vector<DeferredVramHold> deferred_vram_holds_;
  std::vector<std::function<void()>> deferred_host_releases_;
  bool copy_batch_started_[kTransferBuffers] = {};
  size_t direct_h2d_bytes_[kTransferBuffers] = {};

  double h2d_seconds_ = 0.0;
  double h2d_dma_seconds_ = 0.0;
  size_t h2d_bytes_ = 0;
  double d2d_seconds_ = 0.0;
  size_t d2d_bytes_ = 0;
};

}  // namespace moe
