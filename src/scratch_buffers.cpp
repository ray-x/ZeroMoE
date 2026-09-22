#include "scratch_buffers.hpp"

#include "logger.hpp"
#include "memory_guard.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>

#include <cuda_runtime.h>

GpuExpertScratch::~GpuExpertScratch() { shrink(); }

GpuExpertScratch::GpuExpertScratch(GpuExpertScratch &&other) noexcept
    : d_bad(other.d_bad), d_x(other.d_x), d_gate(other.d_gate),
      d_up(other.d_up), d_prod(other.d_prod), d_down(other.d_down),
      cap_sd(other.cap_sd), cap_si(other.cap_si), device_id(other.device_id) {
  other.d_bad = nullptr;
  other.d_x = other.d_gate = other.d_up = other.d_prod = other.d_down =
      nullptr;
  other.cap_sd = other.cap_si = 0;
  other.device_id = -1;
}

GpuExpertScratch &
GpuExpertScratch::operator=(GpuExpertScratch &&other) noexcept {
  if (this == &other)
    return *this;
  shrink();
  d_bad = other.d_bad;
  d_x = other.d_x;
  d_gate = other.d_gate;
  d_up = other.d_up;
  d_prod = other.d_prod;
  d_down = other.d_down;
  cap_sd = other.cap_sd;
  cap_si = other.cap_si;
  device_id = other.device_id;
  other.d_bad = nullptr;
  other.d_x = other.d_gate = other.d_up = other.d_prod = other.d_down =
      nullptr;
  other.cap_sd = other.cap_si = 0;
  other.device_id = -1;
  return *this;
}

int *GpuExpertScratch::ensure_bad_counter() {
  if (!d_bad) {
    if (cudaMalloc(reinterpret_cast<void **>(&d_bad), 2 * sizeof(int)) !=
        cudaSuccess) {
      d_bad = nullptr;
    } else {
      cudaMemset(d_bad, 0, 2 * sizeof(int));
    }
  }
  return d_bad;
}

bool GpuExpertScratch::ensure(int S, int D, int moe_inter,
                              int target_device) {
  if (target_device != device_id) {
    if (device_id >= 0) {
      cudaSetDevice(device_id);
      if (d_bad)
        cudaFree(d_bad);
      if (d_x)
        cudaFree(d_x);
      if (d_down)
        cudaFree(d_down);
      if (d_gate)
        cudaFree(d_gate);
      if (d_up)
        cudaFree(d_up);
      if (d_prod)
        cudaFree(d_prod);
    }
    d_bad = nullptr;
    d_x = d_down = d_gate = d_up = d_prod = nullptr;
    cap_sd = cap_si = 0;
    device_id = target_device;
  }
  cudaSetDevice(device_id);

  const size_t need_sd = static_cast<size_t>(S) * D;
  const size_t need_si = static_cast<size_t>(S) * moe_inter;
  if (cap_sd < need_sd) {
    if (d_x)
      cudaFree(d_x);
    if (d_down)
      cudaFree(d_down);
    d_x = d_down = nullptr;
    cap_sd = 0;
    cudaError_t err1 =
        cudaMalloc(reinterpret_cast<void **>(&d_x), need_sd * sizeof(float));
    cudaError_t err2 =
        err1 == cudaSuccess
            ? cudaMalloc(reinterpret_cast<void **>(&d_down),
                         need_sd * sizeof(float))
            : err1;
    if (err1 != cudaSuccess || err2 != cudaSuccess) {
      const cudaError_t err = err1 != cudaSuccess ? err1 : err2;
      LOG_ERROR("[EXPERT-GPU] scratch d_x/d_down grow failed on device %d: "
                "%s\n",
                device_id, cudaGetErrorString(err));
      if (d_x)
        cudaFree(d_x);
      if (d_down)
        cudaFree(d_down);
      d_x = d_down = nullptr;
      return false;
    }
    cap_sd = need_sd;
  }
  if (cap_si < need_si) {
    if (d_gate)
      cudaFree(d_gate);
    if (d_up)
      cudaFree(d_up);
    if (d_prod)
      cudaFree(d_prod);
    d_gate = d_up = d_prod = nullptr;
    cap_si = 0;
    cudaError_t err1 = cudaMalloc(reinterpret_cast<void **>(&d_gate),
                                  need_si * sizeof(float));
    cudaError_t err2 =
        err1 == cudaSuccess
            ? cudaMalloc(reinterpret_cast<void **>(&d_up),
                         need_si * sizeof(float))
            : err1;
    cudaError_t err3 =
        err2 == cudaSuccess
            ? cudaMalloc(reinterpret_cast<void **>(&d_prod),
                         need_si * sizeof(float))
            : err2;
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
      const cudaError_t err =
          err1 != cudaSuccess ? err1 : (err2 != cudaSuccess ? err2 : err3);
      LOG_ERROR("[EXPERT-GPU] scratch d_gate/d_up/d_prod grow failed on "
                "device %d: %s\n",
                device_id, cudaGetErrorString(err));
      if (d_gate)
        cudaFree(d_gate);
      if (d_up)
        cudaFree(d_up);
      if (d_prod)
        cudaFree(d_prod);
      d_gate = d_up = d_prod = nullptr;
      return false;
    }
    cap_si = need_si;
  }
  return true;
}

void GpuExpertScratch::shrink() {
  if (device_id < 0)
    return;
  cudaSetDevice(device_id);
  if (d_bad)
    cudaFree(d_bad);
  if (d_x)
    cudaFree(d_x);
  if (d_down)
    cudaFree(d_down);
  if (d_gate)
    cudaFree(d_gate);
  if (d_up)
    cudaFree(d_up);
  if (d_prod)
    cudaFree(d_prod);
  d_bad = nullptr;
  d_x = d_down = d_gate = d_up = d_prod = nullptr;
  cap_sd = cap_si = 0;
  device_id = -1;
}

PinnedFloatBuffer::~PinnedFloatBuffer() { free_buf(); }

bool PinnedFloatBuffer::resize(size_t n, const char *name) {
  if (n > capacity_) {
    const size_t new_cap = std::max(n, capacity_ * 2);
    if (new_cap > std::numeric_limits<size_t>::max() / sizeof(float))
      return false;
    const size_t new_bytes = new_cap * sizeof(float);
    if (!MemoryGuard::try_reserve_host_bytes(new_bytes))
      return false;

    float *new_ptr = nullptr;
    cudaError_t err =
        cudaMallocHost(reinterpret_cast<void **>(&new_ptr), new_bytes);
    const bool now_pinned = err == cudaSuccess && new_ptr;
    if (!now_pinned)
      new_ptr = static_cast<float *>(std::malloc(new_bytes));
    if (!new_ptr) {
      MemoryGuard::release_host_bytes(new_bytes);
      return false;
    }
    if (name) {
      LOG_STREAM_INFO() << "[SCRATCH-PIN] " << name << ": "
                        << (now_pinned ? "pinned (cudaMallocHost)"
                                       : "unpinned fallback")
                        << " bytes=" << new_bytes
                        << (now_pinned
                                ? ""
                                : (std::string(" (") +
                                   cudaGetErrorString(err) + ")"))
                        << "\n";
    }
    if (pinned_ && ptr_)
      cudaDeviceSynchronize();
    free_buf();
    ptr_ = new_ptr;
    capacity_ = new_cap;
    pinned_ = now_pinned;
    reserved_bytes_ = new_bytes;
  }
  size_ = n;
  return true;
}

void PinnedFloatBuffer::free_buf() {
  if (!ptr_)
    return;
  if (pinned_)
    cudaFreeHost(ptr_);
  else
    std::free(ptr_);
  ptr_ = nullptr;
  MemoryGuard::release_host_bytes(reserved_bytes_);
  reserved_bytes_ = 0;
  size_ = capacity_ = 0;
  pinned_ = false;
}

bool LayerScratch::resize(int S, int D, int n_experts, int moe_inter, int topk,
                          int n_expert_shared, int dense_inter) {
  attn_out.resize(static_cast<size_t>(S) * D);
  moe_out.resize(static_cast<size_t>(S) * D);
  if (!normed.resize(static_cast<size_t>(S) * D, "normed"))
    return false;
  router_logits.resize(static_cast<size_t>(S) * n_experts);
  const int shared_inter = moe_inter * std::max(1, n_expert_shared);
  const int buf_inter = std::max({moe_inter, shared_inter, dense_inter});
  if (!gate_buf.resize(static_cast<size_t>(S) * buf_inter, "gate_buf") ||
      !up_buf.resize(static_cast<size_t>(S) * buf_inter, "up_buf") ||
      !down_buf.resize(static_cast<size_t>(S) * D, "down_buf"))
    return false;
  expert_acc.resize(static_cast<size_t>(S) * D);
  scores.resize(n_experts);
  expert_ptrs.resize(topk);
  vram_shadow_blocks.resize(std::max(1, topk));
  vram_loc_hint.assign(std::max(1, topk), nullptr);
  pilot_nrm.resize(D);
  pilot_logits.resize(n_experts);
  qwen_pilot_logits.resize(n_experts);
  qwen_pilot_gpu_ids.resize(std::max(1, topk));
  qwen_pilot_gpu_weights.resize(std::max(1, topk));
  qwen_pilot_pred_count = 0;
  qwen_pilot_pred_layer = -1;
  qwen_pilot_prefetch_count = 0;
  qwen_pilot_gpu_pending = false;
  qwen_pilot_gpu_pending_layer = -1;
  pilot_wide_nrm.resize(D);
  pilot_wide_logits.resize(n_experts);
  pilot_attn_snapshot.resize(D);
  tok_topk_ids.resize(static_cast<size_t>(S) * topk);
  tok_topk_weights.resize(static_cast<size_t>(S) * topk);
  qwen_cpu_offload_cold_ids.resize(static_cast<size_t>(topk));
  qwen_cpu_offload_cold_weights.resize(static_cast<size_t>(topk));
  qwen_cpu_offload_cold_slots.resize(static_cast<size_t>(topk));
  expert_freq.resize(n_experts);
  first_needed_by_token.resize(n_experts);
  if (static_cast<int>(gpu_pool.size()) < std::max(1, topk))
    gpu_pool.resize(std::max(1, topk));
  pending_gpu.reserve(std::max(1, topk));
  return true;
}
