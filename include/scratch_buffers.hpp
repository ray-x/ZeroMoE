#pragma once

#include "engine.hpp"
#include "gpu_expert_executor.hpp"
#include "gpu_topology.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

class GgmlMoeRunner;

struct GpuExpertScratch {
  GpuExpertScratch() = default;
  ~GpuExpertScratch();
  GpuExpertScratch(const GpuExpertScratch &) = delete;
  GpuExpertScratch &operator=(const GpuExpertScratch &) = delete;
  GpuExpertScratch(GpuExpertScratch &&other) noexcept;
  GpuExpertScratch &operator=(GpuExpertScratch &&other) noexcept;

  int *d_bad = nullptr;
  float *d_x = nullptr;
  float *d_gate = nullptr;
  float *d_up = nullptr;
  float *d_prod = nullptr;
  float *d_down = nullptr;
  size_t cap_sd = 0;
  size_t cap_si = 0;
  int device_id = -1;

  int *ensure_bad_counter();
  bool ensure(int S, int D, int moe_inter, int target_device);
  void shrink();
};

struct ExpertScore {
  int id;
  float score;
};

class PinnedFloatBuffer {
public:
  PinnedFloatBuffer() = default;
  ~PinnedFloatBuffer();
  PinnedFloatBuffer(const PinnedFloatBuffer &) = delete;
  PinnedFloatBuffer &operator=(const PinnedFloatBuffer &) = delete;

  bool resize(size_t n, const char *name = nullptr);
  float *data() { return ptr_; }
  const float *data() const { return ptr_; }
  size_t size() const { return size_; }
  float &operator[](size_t i) { return ptr_[i]; }
  const float &operator[](size_t i) const { return ptr_[i]; }

private:
  void free_buf();

  float *ptr_ = nullptr;
  size_t size_ = 0;
  size_t capacity_ = 0;
  bool pinned_ = false;
  size_t reserved_bytes_ = 0;
};

struct LayerScratch {
  std::vector<float> attn_out;
  std::vector<float> moe_out;
  const void *moe_device_output = nullptr;
  GgmlMoeRunner *moe_device_runner = nullptr;
  bool moe_shared_device_output = false;
  bool moe_device_output_ready = false;
  PinnedFloatBuffer normed;
  std::vector<float> router_logits;
  PinnedFloatBuffer gate_buf;
  PinnedFloatBuffer up_buf;
  PinnedFloatBuffer down_buf;
  std::vector<float> expert_acc;
  PinnedFloatBuffer qwen_cpu_moe_out;

  // CPUOffload Milestone A: fixed-capacity per-token cold route staging.
  // These vectors are sized with the model's top-k during initialization and
  // are only overwritten during decode; the offload path must not grow them.
  std::vector<int32_t> qwen_cpu_offload_cold_ids;
  std::vector<float> qwen_cpu_offload_cold_weights;
  std::vector<uint8_t> qwen_cpu_offload_cold_slots;

  std::vector<ExpertScore> scores;
  std::vector<MoEExpertBlock *> expert_ptrs;
  std::vector<float> pilot_nrm;
  std::vector<float> pilot_logits;

  static constexpr int kQwenPilotMax = 16;
  std::array<int, kQwenPilotMax> qwen_pilot_pred_ids{};
  std::vector<float> qwen_pilot_logits;
  std::vector<int32_t> qwen_pilot_gpu_ids;
  std::vector<float> qwen_pilot_gpu_weights;
  bool qwen_pilot_gpu_pending = false;
  int qwen_pilot_gpu_pending_layer = -1;
  int qwen_pilot_pred_count = 0;
  int qwen_pilot_pred_layer = -1;
  int qwen_pilot_prefetch_count = 0;

  std::vector<MoEExpertBlock> vram_shadow_blocks;
  std::vector<const ExpertLocation *> vram_loc_hint;
  std::vector<GpuExpertScratch> gpu_pool;
  std::vector<PendingGpuExpert> pending_gpu;

  std::vector<int> tok_topk_ids;
  std::vector<float> tok_topk_weights;
  std::vector<int> expert_freq;
  std::vector<int> first_needed_by_token;

  struct PendingPrefillPromotion {
    int layer_idx;
    int expert_id;
    int age_layers;
  };
  static constexpr int PENDING_PREFILL_CAP = 32;
  std::array<PendingPrefillPromotion, PENDING_PREFILL_CAP>
      pending_prefill_promotions{};
  int pending_prefill_count = 0;

  std::array<int, 8> pilot_pred_ids{};
  int pilot_pred_count = 0;
  int pilot_pred_layer = -1;

  static constexpr int kPilotWideMax = 128;
  std::array<int, kPilotWideMax> pilot_wide_pred_ids{};
  int pilot_wide_pred_count = 0;
  int pilot_wide_pred_layer = -1;
  std::vector<float> pilot_wide_nrm;
  std::vector<float> pilot_wide_logits;
  std::array<bool, 8> pilot_pred_was_hint{};
  std::array<bool, 8> pilot_pred_was_ram_only{};

  struct Pilot2Pending {
    std::array<int, 8> ids{};
    int count = 0;
    int layer = -1;
    std::array<bool, 8> was_hint{};
  };
  std::array<Pilot2Pending, 2> pilot_pred2_slots{};

  std::array<int, 8> pilot_attn_pred_ids{};
  int pilot_attn_pred_count = 0;
  int pilot_attn_pred_layer = -1;
  std::array<bool, 8> pilot_attn_pred_was_hint{};
  std::vector<float> pilot_attn_snapshot;

  bool resize(int S, int D, int n_experts, int moe_inter, int topk,
              int n_expert_shared = 0, int dense_inter = 0);
};
