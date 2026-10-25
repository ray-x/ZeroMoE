#include "ggml_moe_runner.hpp"
#include "ggml_moe_transfer_manager.hpp"
#include "ggml_moe_internal.hpp"

#include "logger.hpp"
#include "memory_guard.hpp"
#include "perf_timer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

moe::GgmlMoeTransferManager::~GgmlMoeTransferManager() {
  reset_transfer_state(-1);
}

#if defined(ZEROMOE_USE_GGML_CUDA)
#include <cuda_runtime.h>
#include "gpu_topology.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#if defined(_OPENMP)
#include <omp.h>
#endif
using namespace moe::detail;

extern "C" cudaError_t launch_ggml_moe_unpack(
    const void *src, void *gate, void *up, void *down, int count,
    size_t src_stride, size_t gate_bytes, size_t up_bytes, size_t down_bytes,
    size_t gate_stride, size_t up_stride, size_t down_stride,
    cudaStream_t stream);

extern "C" cudaError_t launch_ggml_moe_unpack_scatter(
    const void *src, void *gate, void *up, void *down,
    const int32_t *dst_slots, int count, size_t src_stride,
    size_t gate_bytes, size_t up_bytes, size_t down_bytes,
    size_t gate_stride, size_t up_stride, size_t down_stride,
    cudaStream_t stream);

void moe::GgmlMoeTransferManager::reset_transfer_state(int device) noexcept {
  if (device >= 0) cudaSetDevice(device);
  if (copy_stream_) cudaStreamSynchronize((cudaStream_t)copy_stream_);
  if (copy_backend_)
    ggml_backend_synchronize((ggml_backend_t)copy_backend_);
  for (int i = 0; i < kTransferBuffers; ++i) {
    if (h_stage_[i]) cudaFreeHost(h_stage_[i]);
    if (d_stage_slot_map_[i]) cudaFree(d_stage_slot_map_[i]);
    if (ev_free_[i]) cudaEventDestroy((cudaEvent_t)ev_free_[i]);
    if (ev_t0_[i]) cudaEventDestroy((cudaEvent_t)ev_t0_[i]);
    if (backend_stage_event_[i])
      ggml_backend_event_free((ggml_backend_event_t)backend_stage_event_[i]);
    if (backend_copy_t0_event_[i])
      cudaEventDestroy((cudaEvent_t)backend_copy_t0_event_[i]);
    if (backend_copy_t1_event_[i])
      cudaEventDestroy((cudaEvent_t)backend_copy_t1_event_[i]);
    if (backend_copy_split_event_[i])
      cudaEventDestroy((cudaEvent_t)backend_copy_split_event_[i]);
    h_stage_[i] = nullptr;
    d_stage_slot_map_[i] = nullptr;
    ev_free_[i] = ev_t0_[i] = nullptr;
    backend_stage_event_[i] = nullptr;
    backend_copy_t0_event_[i] = backend_copy_t1_event_[i] = nullptr;
    backend_copy_split_event_[i] = nullptr;
    for (int j = 0; j < kH2dTimingEvents; ++j) {
      if (h2d_t0_event_[i][j])
        cudaEventDestroy((cudaEvent_t)h2d_t0_event_[i][j]);
      if (h2d_t1_event_[i][j])
        cudaEventDestroy((cudaEvent_t)h2d_t1_event_[i][j]);
      h2d_t0_event_[i][j] = h2d_t1_event_[i][j] = nullptr;
    }
    stage_slot_map_[i].clear();
    packed_h2d_slot_map_[i].clear();
    d2d_slot_map_[i].clear();
    h2d_timing_count_[i] = 0;
    pending_[i] = false;
    pending_bytes_[i] = pending_d2d_bytes_[i] = 0;
    copy_split_recorded_[i] = false;
    copy_batch_started_[i] = false;
    direct_h2d_bytes_[i] = 0;
  }
  for (int i = 0; i < kWeightEvents; ++i) {
    if (backend_weight_done_event_[i])
      ggml_backend_event_free(
          (ggml_backend_event_t)backend_weight_done_event_[i]);
    backend_weight_done_event_[i] = nullptr;
  }
  for (int i = 0; i < kWeightBuffers; ++i) {
    if (packed_wbuf_[i]) cudaFree(packed_wbuf_[i]);
    packed_wbuf_[i] = nullptr;
    backend_weight_last_event_[i] = nullptr;
    backend_weight_buffer_in_use_[i] = false;
  }
  MemoryGuard::release_host_bytes(stage_host_reservation_bytes_);
  stage_host_reservation_bytes_ = 0;
  stage_bytes_ = 0;
  stage_slot_capacity_ = 0;
  backend_weight_event_next_ = 0;
  weight_buffer_index_ = 0;
  backend_stream_copies_ = false;
  weight_buffer_count_ = 1;
  raw_d2d_pending_ = false;
  batch_d2d_unpack_ = false;
  packed_wbuf_bytes_ = 0;
  packed_base_ = nullptr;
  weight_expert_stride_ = 0;
  weight_raw_expert_stride_ = 0;
  weight_tail_padding_ = 0;
  weight_gate_plane_offset_ = 0;
  weight_up_plane_offset_ = 0;
  weight_down_plane_offset_ = 0;
  h2d_seconds_ = h2d_dma_seconds_ = d2d_seconds_ = 0.0;
  h2d_bytes_ = d2d_bytes_ = 0;
  if (copy_stream_) cudaStreamDestroy((cudaStream_t)copy_stream_);
  copy_stream_ = nullptr;
  if (copy_backend_) ggml_backend_free((ggml_backend_t)copy_backend_);
  copy_backend_ = nullptr;
}

bool GgmlMoeRunner::ensure_h2d_timing_events() {
  if (!trace_pure_h2d()) return true;
  for (int buf = 0; buf < NBUF; ++buf) {
    for (int i = 0; i < NH2D_TIMING_EVENTS; ++i) {
      if (!h2d_t0_event_[buf][i] &&
          cudaEventCreate((cudaEvent_t *)&h2d_t0_event_[buf][i]) != cudaSuccess)
        return false;
      if (!h2d_t1_event_[buf][i] &&
          cudaEventCreate((cudaEvent_t *)&h2d_t1_event_[buf][i]) != cudaSuccess)
        return false;
    }
  }
  return true;
}

bool GgmlMoeRunner::begin_h2d_timing(int buf, int &slot) {
  slot = -1;
  if (!trace_pure_h2d()) return true;
  if (buf < 0 || buf >= NBUF) return false;
  const int next = h2d_timing_count_[buf];
  if (next >= NH2D_TIMING_EVENTS) return true; // tracer capacity only
  if (cudaEventRecord((cudaEvent_t)h2d_t0_event_[buf][next],
                      (cudaStream_t)copy_stream_) != cudaSuccess)
    return false;
  h2d_timing_count_[buf] = next + 1;
  slot = next;
  return true;
}

bool GgmlMoeRunner::end_h2d_timing(int buf, int slot) {
  if (!trace_pure_h2d() || slot < 0) return true;
  return cudaEventRecord((cudaEvent_t)h2d_t1_event_[buf][slot],
                         (cudaStream_t)copy_stream_) == cudaSuccess;
}

bool GgmlMoeRunner::ensure_stage_buffers(size_t bytes_per_buf) {
  if (stage_bytes_ >= bytes_per_buf && h_stage_[0] && h_stage_[1] &&
      stage_slot_capacity_ >= plan_.chunk_experts &&
      d_stage_slot_map_[0] && d_stage_slot_map_[1])
    return true;
  if (copy_stream_) cudaStreamSynchronize((cudaStream_t)copy_stream_);
  if (copy_backend_) ggml_backend_synchronize((ggml_backend_t)copy_backend_);

  if (bytes_per_buf == 0 ||
      bytes_per_buf > std::numeric_limits<size_t>::max() / NBUF) {
    LOG_WARN("[GGML-MOE] pinned staging size overflow -- disabled\n");
    return false;
  }
  const size_t host_bytes = bytes_per_buf * NBUF;

  // Drop the old buffers only after all streams have drained, then charge the
  // replacement before asking CUDA for pages.  The old implementation had no
  // host admission at this generic GGML/Qwen path, so a graph run could pass
  // the RAM-tier check and still allocate another multi-GB pinned staging
  // pair outside host_ram_gb.
  for (int i = 0; i < NBUF; ++i) {
    if (h_stage_[i]) {
      cudaFreeHost(h_stage_[i]);
      h_stage_[i] = nullptr;
    }
    if (d_stage_slot_map_[i]) {
      cudaFree(d_stage_slot_map_[i]);
      d_stage_slot_map_[i] = nullptr;
    }
    stage_slot_map_[i].clear();
    d2d_slot_map_[i].clear();
    packed_h2d_slot_map_[i].clear();
    pending_[i] = false;
    copy_batch_started_[i] = false;
    direct_h2d_bytes_[i] = 0;
    pending_d2d_bytes_[i] = 0;
    copy_split_recorded_[i] = false;
  }
  MemoryGuard::release_host_bytes(stage_host_reservation_bytes_);
  stage_host_reservation_bytes_ = 0;
  stage_bytes_ = 0;
  stage_slot_capacity_ = 0;

  if (!MemoryGuard::try_reserve_host_bytes(host_bytes)) {
    LOG_WARN("[MEM-GUARD] GGML-MOE pinned staging refused: %.2f MiB "
             "(reserved=%.2f GiB, budget=%.2f GB)\n",
             (double)host_bytes / (1024.0 * 1024.0),
             (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
             (double)MemoryGuard::host_budget_bytes() / 1.0e9);
    return false;
  }
  stage_host_reservation_bytes_ = host_bytes;

  auto cleanup_failed_stage = [&]() {
    for (int j = 0; j < NBUF; ++j) {
      if (h_stage_[j]) {
        cudaFreeHost(h_stage_[j]);
        h_stage_[j] = nullptr;
      }
      if (d_stage_slot_map_[j]) {
        cudaFree(d_stage_slot_map_[j]);
        d_stage_slot_map_[j] = nullptr;
      }
      stage_slot_map_[j].clear();
    }
    MemoryGuard::release_host_bytes(stage_host_reservation_bytes_);
    stage_host_reservation_bytes_ = 0;
    stage_bytes_ = 0;
    stage_slot_capacity_ = 0;
  };

  for (int i = 0; i < NBUF; ++i) {
    void *p = nullptr;
    if (cudaMallocHost(&p, bytes_per_buf) != cudaSuccess || !p) {
      LOG_ERROR("[GGML-MOE] pinned staging alloc failed (%.2f MB)\n",
                (double)bytes_per_buf / (1024.0 * 1024.0));
      cleanup_failed_stage();
      return false;
    }
    h_stage_[i] = (uint8_t *)p;
  }
  for (int i = 0; i < NBUF; ++i) {
    if (cudaMalloc(&d_stage_slot_map_[i],
                   (size_t)plan_.chunk_experts * sizeof(int32_t)) !=
            cudaSuccess ||
        !d_stage_slot_map_[i]) {
      LOG_ERROR("[GGML-MOE] packed stage slot-map allocation failed\n");
      cleanup_failed_stage();
      return false;
    }
    stage_slot_map_[i].reserve((size_t)plan_.chunk_experts);
    d2d_slot_map_[i].reserve((size_t)plan_.chunk_experts);
    packed_h2d_slot_map_[i].reserve((size_t)plan_.chunk_experts);
  }
  stage_bytes_ = bytes_per_buf;
  stage_slot_capacity_ = plan_.chunk_experts;
  if (!copy_stream_) {
    cudaStream_t s = nullptr;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
      LOG_ERROR("[GGML-MOE] copy stream creation failed\n");
      cleanup_failed_stage();
      return false;
    }
    copy_stream_ = s;
  }
  for (int i = 0; i < NBUF; ++i) {
    if (!backend_stream_copies_ && !ev_free_[i]) {
      cudaEvent_t e = nullptr;
      if (cudaEventCreate(&e) != cudaSuccess) {
        cleanup_failed_stage();
        return false;
      }
      ev_free_[i] = e;
    }
    if (!backend_stream_copies_ && !ev_t0_[i]) {
      cudaEvent_t e = nullptr;
      if (cudaEventCreate(&e) != cudaSuccess) {
        cleanup_failed_stage();
        return false;
      }
      ev_t0_[i] = e;
    }
    if (!backend_copy_split_event_[i]) {
      cudaEvent_t e = nullptr;
      if (cudaEventCreate(&e) != cudaSuccess) {
        cleanup_failed_stage();
        return false;
      }
      backend_copy_split_event_[i] = e;
    }
    if (backend_stream_copies_ && !backend_stage_event_[i]) {
      ggml_backend_event_t e = ggml_backend_event_new(
          ggml_backend_get_device((ggml_backend_t)backend_));
      if (!e) {
        LOG_ERROR("[GGML-MOE] GGML staging event creation failed\n");
        cleanup_failed_stage();
        return false;
      }
      backend_stage_event_[i] = e;
    }
    if (backend_stream_copies_ && !backend_copy_t0_event_[i]) {
      cudaEvent_t e0 = nullptr;
      cudaEvent_t e1 = nullptr;
      if (cudaEventCreate(&e0) != cudaSuccess ||
          cudaEventCreate(&e1) != cudaSuccess) {
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
        LOG_ERROR("[GGML-MOE] backend copy timing event creation failed\n");
        cleanup_failed_stage();
        return false;
      }
      backend_copy_t0_event_[i] = e0;
      backend_copy_t1_event_[i] = e1;
    }
  }
  if (backend_stream_copies_) {
    for (int i = 0; i < NWEIGHT_EVENTS; ++i) {
      if (backend_weight_done_event_[i]) continue;
      ggml_backend_event_t e = ggml_backend_event_new(
          ggml_backend_get_device((ggml_backend_t)backend_));
      if (!e) {
        LOG_ERROR("[GGML-MOE] GGML weight completion event creation failed\n");
        cleanup_failed_stage();
        return false;
      }
      backend_weight_done_event_[i] = e;
    }
  }
  return true;
}

bool GgmlMoeRunner::begin_copy_batch(int buf) {
  if (!layer_ready_ || buf < 0 || buf >= NBUF)
    return false;
  if (backend_stream_copies_) {
    if (!backend_stage_event_[buf]) return false;
    if (copy_batch_started_[buf]) return true;
    cudaSetDevice(device_);
    if (plan_.chunked && chunk_weight_recorded_[buf] &&
        chunk_weight_done_events_[buf]) {
      // The alternating chunk slot is reused only after the previous graph
      // has finished reading it. This dependency stays on the copy stream;
      // the host does not round-trip at the chunk boundary.
      if (cudaStreamWaitEvent(
              (cudaStream_t)copy_stream_,
              (cudaEvent_t)((ggml_backend_event_t)
                                chunk_weight_done_events_[buf])
                  ->context,
              0) != cudaSuccess)
        return false;
    }
    if (plan_.chunked && chunk_scatter_recorded_[buf] &&
        chunk_scatter_done_events_[buf]) {
      // The default-stream scatter reads the per-slot token snapshot.  The
      // copy stream must not reuse that slot until the previous reduction has
      // finished, even though the GGML graph itself finished earlier.
      if (cudaStreamWaitEvent(
              (cudaStream_t)copy_stream_,
              (cudaEvent_t)chunk_scatter_done_events_[buf], 0) != cudaSuccess)
        return false;
    }
    direct_h2d_bytes_[buf] = 0;
    h2d_timing_count_[buf] = 0;
    if (cudaEventRecord((cudaEvent_t)backend_copy_t0_event_[buf],
                        (cudaStream_t)copy_stream_) != cudaSuccess)
      return false;
    backend_copy_t0_[buf] = std::chrono::steady_clock::now();
    copy_batch_started_[buf] = true;
    pending_d2d_bytes_[buf] = 0;
    copy_split_recorded_[buf] = false;
    d2d_slot_map_[buf].clear();
    packed_h2d_slot_map_[buf].clear();
    return true;
  }
  if (!copy_stream_ || !ev_t0_[buf]) return false;
  cudaSetDevice(device_);
  if (copy_batch_started_[buf]) return true;
  if (cudaEventRecord((cudaEvent_t)ev_t0_[buf],
                      (cudaStream_t)copy_stream_) != cudaSuccess)
    return false;
  backend_copy_t0_[buf] = std::chrono::steady_clock::now();
  direct_h2d_bytes_[buf] = 0;
  h2d_timing_count_[buf] = 0;
  pending_d2d_bytes_[buf] = 0;
  copy_split_recorded_[buf] = false;
  copy_batch_started_[buf] = true;
  d2d_slot_map_[buf].clear();
  packed_h2d_slot_map_[buf].clear();
  return true;
}

bool GgmlMoeRunner::mark_copy_d2d_done(int buf) {
  if (!layer_ready_ || buf < 0 || buf >= NBUF || !copy_batch_started_[buf])
    return false;
  if (copy_split_recorded_[buf]) return true;
  cudaSetDevice(device_);
  if (batch_d2d_unpack_ && use_packed_transfer() &&
      !d2d_slot_map_[buf].empty()) {
    if (!packed_base_ || !d_stage_slot_map_[buf]) return false;
    const int count = (int)d2d_slot_map_[buf].size();
    cudaError_t e = cudaMemcpyAsync(
        d_stage_slot_map_[buf], d2d_slot_map_[buf].data(),
        (size_t)count * sizeof(int32_t), cudaMemcpyHostToDevice,
        (cudaStream_t)copy_stream_);
    if (e == cudaSuccess) {
      e = launch_ggml_moe_unpack_scatter(
          packed_base_, w_gate_base_, w_up_base_, w_down_base_,
          (const int32_t *)d_stage_slot_map_[buf], count,
          weight_expert_stride_, gate_exp_bytes_, up_exp_bytes_,
          down_exp_bytes_, gate_exp_bytes_, up_exp_bytes_, down_exp_bytes_,
          (cudaStream_t)copy_stream_);
    }
    if (e != cudaSuccess) {
      LOG_ERROR("[GGML-MOE] batched D2D unpack failed: %s\n",
                cudaGetErrorString(e));
      return false;
    }
  }
  if (!backend_copy_split_event_[buf] ||
      cudaEventRecord((cudaEvent_t)backend_copy_split_event_[buf],
                      (cudaStream_t)copy_stream_) != cudaSuccess)
    return false;
  copy_split_recorded_[buf] = true;
  d2d_slot_map_[buf].clear();
  return true;
}

GgmlMoeStageSlot GgmlMoeRunner::stage_slot(int buf, int slot) const {
  GgmlMoeStageSlot s;
  if (!layer_ready_ || buf < 0 || buf >= NBUF || slot < 0 ||
      slot >= plan_.chunk_experts || !h_stage_[buf])
    return s;
  uint8_t *base = h_stage_[buf];
  const size_t stride = weight_expert_stride_;
  uint8_t *expert = base + (size_t)slot * stride;
  s.gate = expert;
  s.up = expert + gate_exp_bytes_;
  s.down = s.up + up_exp_bytes_;
  return s;
}

void GgmlMoeRunner::account_copy_buffer(int buf) {
  if (buf < 0 || buf >= NBUF || !pending_[buf]) return;

  cudaSetDevice(device_);
  float d2d_ms = 0.0f;
  float h2d_ms = 0.0f;
  bool timed = false;
  if (copy_split_recorded_[buf]) {
    if (backend_stream_copies_) {
      timed = cudaEventElapsedTime(
                  &d2d_ms, (cudaEvent_t)backend_copy_t0_event_[buf],
                  (cudaEvent_t)backend_copy_split_event_[buf]) == cudaSuccess &&
              cudaEventElapsedTime(
                  &h2d_ms, (cudaEvent_t)backend_copy_split_event_[buf],
                  (cudaEvent_t)backend_copy_t1_event_[buf]) == cudaSuccess;
    } else {
      timed = cudaEventElapsedTime(
                  &d2d_ms, (cudaEvent_t)ev_t0_[buf],
                  (cudaEvent_t)backend_copy_split_event_[buf]) == cudaSuccess &&
              cudaEventElapsedTime(
                  &h2d_ms, (cudaEvent_t)backend_copy_split_event_[buf],
                  (cudaEvent_t)ev_free_[buf]) == cudaSuccess;
    }
  }
  if (!timed) {
    float total_ms = 0.0f;
    if (backend_stream_copies_) {
      timed = cudaEventElapsedTime(
                  &total_ms, (cudaEvent_t)backend_copy_t0_event_[buf],
                  (cudaEvent_t)backend_copy_t1_event_[buf]) == cudaSuccess;
    } else {
      timed = cudaEventElapsedTime(&total_ms, (cudaEvent_t)ev_t0_[buf],
                                   (cudaEvent_t)ev_free_[buf]) == cudaSuccess;
    }
    // Without a valid split event, retain a conservative total in the H2D
    // bucket. This is a diagnostics fallback only; normal runs record the
    // split before the first RAM upload.
    h2d_ms = total_ms;
  }
  if (timed) {
    d2d_seconds_ += (double)d2d_ms / 1000.0;
    h2d_seconds_ += (double)h2d_ms / 1000.0;
  } else {
    h2d_seconds_ += std::chrono::duration<double>(
                         std::chrono::steady_clock::now() -
                         backend_copy_t0_[buf])
                         .count();
  }
  if (trace_pure_h2d()) {
    for (int i = 0; i < h2d_timing_count_[buf]; ++i) {
      float dma_ms = 0.0f;
      if (cudaEventElapsedTime(
              &dma_ms, (cudaEvent_t)h2d_t0_event_[buf][i],
              (cudaEvent_t)h2d_t1_event_[buf][i]) == cudaSuccess)
        h2d_dma_seconds_ += (double)dma_ms / 1000.0;
    }
  }
  d2d_bytes_ += pending_d2d_bytes_[buf];
  h2d_bytes_ += pending_bytes_[buf];
  pending_[buf] = false;
  pending_bytes_[buf] = 0;
  pending_d2d_bytes_[buf] = 0;
  copy_batch_started_[buf] = false;
  copy_split_recorded_[buf] = false;
  h2d_timing_count_[buf] = 0;
  direct_h2d_bytes_[buf] = 0;
  packed_h2d_slot_map_[buf].clear();
}

void GgmlMoeRunner::wait_stage_free(int buf) {
  if (buf < 0 || buf >= NBUF || !pending_[buf]) return;
  if (backend_stream_copies_) {
    cudaSetDevice(device_);
    ggml_backend_event_synchronize(
        (ggml_backend_event_t)backend_stage_event_[buf]);
    // This is the copy-stream service interval. It excludes the host's wait
    // for a reusable double-buffer slot and any time spent after the copy was
    // queued while the compute stream was running.
    account_copy_buffer(buf);
    return;
  }
  // MUST pin the device. A CUDA event belongs to the device that was current
  // when it was recorded, and waiting on it from a DIFFERENT current device
  // does not reliably complete -- it hangs. This is reachable in practice:
  // GpuTopology::finish_expert_hit() does cudaSetDevice(loc.gpu_id) and never
  // restores it (see its own comment -- the set is deliberate, the lack of a
  // restore is just how it is), so a single find_expert() hit on an expert
  // resident on the OTHER vault GPU silently leaves that GPU current, and the
  // next wait here blocks forever. MEASURED 2026-09-05: adding the VRAM-reuse
  // residency check introduced exactly that call and hung layer 0 at the
  // first chunk with a pending event, spinning in userspace at 0% SM
  // utilisation. Same class as PrefillExpertStream::pin_device().
  cudaSetDevice(device_);
  cudaEventSynchronize((cudaEvent_t)ev_free_[buf]);
  account_copy_buffer(buf);
}

bool GgmlMoeRunner::push_chunk(int buf, int first_expert, int count,
                               const bool *staged) {
  if (!layer_ready_ || buf < 0 || buf >= NBUF || count <= 0) return false;
  if (first_expert < 0 || first_expert + count > plan_.n_experts) return false;
  if (count > plan_.chunk_experts) return false;
  if (!begin_copy_batch(buf)) return false;
  if (!mark_copy_d2d_done(buf)) return false;
  cudaSetDevice(device_);

  size_t total = 0;
  cudaError_t e = cudaSuccess;
  if (use_packed_transfer()) {
    if (!packed_base_ || !d_stage_slot_map_[buf]) return false;

    // Cache hits leave holes in the chunk.  Compact the staged records in
    // place before the H2D: one chunk then becomes one contiguous DMA even
    // when its VRAM-resident experts are interleaved with RAM misses.  The
    // device map restores the original expert slot after the transfer, so
    // the GGML planes keep their normal global expert-major layout.
    auto &slot_map = stage_slot_map_[buf];
    slot_map.clear();
    for (int i = 0; i < count; ++i) {
      if (staged && !staged[i]) continue;
      const GgmlMoeStageSlot src_slot = stage_slot(buf, i);
      if (!src_slot.gate) return false;
      const size_t compact = slot_map.size();
      const GgmlMoeStageSlot dst_slot = stage_slot(buf, (int)compact);
      if (!dst_slot.gate) return false;
      if (compact != (size_t)i) {
        std::memmove(dst_slot.gate, src_slot.gate,
                     weight_expert_stride_);
      }
      slot_map.push_back(first_expert + i);
    }

    const int compact_count = (int)slot_map.size();
    const size_t direct_count =
        batch_d2d_unpack_ ? packed_h2d_slot_map_[buf].size() : 0;
    if (direct_count + (size_t)compact_count >
        (size_t)plan_.chunk_experts)
      return false;
    if (debug_check_device_output()) {
      int slot31_compact = -1;
      int slot63_compact = -1;
      for (int compact = 0; compact < compact_count; ++compact) {
        if (slot_map[(size_t)compact] == 31) {
          slot31_compact = compact;
        }
        if (slot_map[(size_t)compact] == 63) {
          slot63_compact = compact;
        }
      }
      LOG_INFO("[GGML-MOE-DEBUG] compact transfer buf=%d first=%d count=%d "
               "staged=%d slot31_compact=%d slot63_compact=%d\n",
               buf, first_expert, count, compact_count, slot31_compact,
               slot63_compact);
    }
    size_t staged_bytes = 0;
    if (compact_count > 0) {
      const size_t bytes = weight_expert_stride_ * (size_t)compact_count;
      staged_bytes = bytes;
      int h2d_timing_slot = -1;
      if (!begin_h2d_timing(buf, h2d_timing_slot)) return false;
      e = cudaMemcpyAsync(
          packed_base_ + direct_count * weight_expert_stride_, h_stage_[buf],
          bytes, cudaMemcpyHostToDevice,
          (cudaStream_t)copy_stream_);
      if (!end_h2d_timing(buf, h2d_timing_slot)) return false;
    }
    if (e == cudaSuccess && batch_d2d_unpack_ &&
        !debug_check_device_output()) {
      auto &combined_map = packed_h2d_slot_map_[buf];
      combined_map.insert(combined_map.end(), slot_map.begin(), slot_map.end());
    }
    const int map_count = batch_d2d_unpack_
                              ? (int)packed_h2d_slot_map_[buf].size()
                              : compact_count;
    const int32_t *map_host = batch_d2d_unpack_
                                  ? packed_h2d_slot_map_[buf].data()
                                  : slot_map.data();
    if (e == cudaSuccess && map_count > 0) {
      e = cudaMemcpyAsync(
          d_stage_slot_map_[buf], map_host,
          (size_t)map_count * sizeof(int32_t), cudaMemcpyHostToDevice,
          (cudaStream_t)copy_stream_);
      if (e == cudaSuccess) {
        e = launch_ggml_moe_unpack_scatter(
            packed_base_, w_gate_base_, w_up_base_, w_down_base_,
            (const int32_t *)d_stage_slot_map_[buf], map_count,
            weight_expert_stride_, gate_exp_bytes_, up_exp_bytes_,
            down_exp_bytes_, gate_exp_bytes_, up_exp_bytes_, down_exp_bytes_,
            (cudaStream_t)copy_stream_);
      }
    }
    if (e == cudaSuccess && compact_count > 0 &&
        debug_check_device_output() &&
        !batch_d2d_unpack_) {
        // Diagnostic only: compare the edge rows immediately after the
        // device-local unpack, before MUL_MAT_ID consumes them.
        e = cudaStreamSynchronize((cudaStream_t)copy_stream_);
        if (e == cudaSuccess) {
          const size_t gate_row = ggml_row_size(
              (ggml_type)plan_.gate_type, plan_.d_model);
          const size_t up_row = ggml_row_size(
              (ggml_type)plan_.up_type, plan_.d_model);
          const size_t down_row = ggml_row_size(
              (ggml_type)plan_.down_type, plan_.d_ff);
          std::vector<uint8_t> device_row;
          auto verify_row = [&](const char *name, const uint8_t *host,
                                const uint8_t *device, size_t row_bytes,
                                int slot, int compact) {
            if (row_bytes == 0) return;
            device_row.resize(row_bytes);
            const cudaError_t ce = cudaMemcpy(
                device_row.data(), device, row_bytes,
                cudaMemcpyDeviceToHost);
            const bool same = ce == cudaSuccess &&
                              std::memcmp(host, device_row.data(), row_bytes) ==
                                  0;
            if (!same) {
              LOG_INFO("[GGML-MOE-DEBUG] unpack mismatch chunk_buf=%d "
                       "compact=%d slot=%d plane=%s bytes=%zu copy=%s\n",
                       buf, compact, slot, name, row_bytes,
                       ce == cudaSuccess ? "ok" : cudaGetErrorString(ce));
            }
          };
          for (int compact = 0; compact < compact_count; ++compact) {
            const int slot = slot_map[(size_t)compact];
            const uint8_t *src_record =
                h_stage_[buf] + (size_t)compact * weight_expert_stride_;
            verify_row("gate-first", src_record,
                       w_gate_base_ + gate_exp_bytes_ * (size_t)slot,
                       gate_row, slot, compact);
            verify_row("gate-last",
                       src_record + gate_exp_bytes_ - gate_row,
                       w_gate_base_ + gate_exp_bytes_ * (size_t)slot +
                           gate_exp_bytes_ - gate_row,
                       gate_row, slot, compact);
            verify_row("up-first", src_record + gate_exp_bytes_,
                       w_up_base_ + up_exp_bytes_ * (size_t)slot, up_row, slot,
                       compact);
            verify_row("up-last",
                       src_record + gate_exp_bytes_ + up_exp_bytes_ - up_row,
                       w_up_base_ + up_exp_bytes_ * (size_t)slot +
                           up_exp_bytes_ - up_row,
                       up_row, slot, compact);
            verify_row("down-first", src_record + gate_exp_bytes_ +
                           up_exp_bytes_,
                       w_down_base_ + down_exp_bytes_ * (size_t)slot,
                       down_row, slot, compact);
            verify_row("down-last",
                       src_record + gate_exp_bytes_ + up_exp_bytes_ +
                           down_exp_bytes_ - down_row,
                       w_down_base_ + down_exp_bytes_ * (size_t)slot +
                           down_exp_bytes_ - down_row,
                       down_row, slot, compact);
          }
        }
      }
    // The route-map upload is small but is still a real H2D operation; keep
    // it in the source-specific accounting instead of silently dropping it.
    if (e == cudaSuccess)
      total += staged_bytes + (size_t)map_count * sizeof(int32_t);
  } else {
    // Walk maximal runs of consecutive staged experts. The reference path
    // keeps the established planar layout and deliberately does not use the
    // compact packed DMA.
    int i = 0;
    while (i < count && e == cudaSuccess) {
      if (staged && !staged[i]) {
        ++i;
        continue;
      }
      int j = i;
      while (j < count && (!staged || staged[j])) ++j;
      const GgmlMoeStageSlot s0 = stage_slot(buf, i);
      if (!s0.gate) return false;
      // Correctness reference: preserve the established planar GGML layout
      // and copy each projection through its own pitched host record. This
      // is intentionally slower, but removes the new device unpack kernel
      // from the experiment while keeping the graph and all routing identical.
      const size_t rows = (size_t)(j - i);
      e = cudaMemcpy2DAsync(
          w_gate_base_ + gate_exp_bytes_ * (size_t)(first_expert + i),
          gate_exp_bytes_, s0.gate, weight_expert_stride_, gate_exp_bytes_,
          rows, cudaMemcpyHostToDevice, (cudaStream_t)copy_stream_);
      if (e == cudaSuccess) {
        e = cudaMemcpy2DAsync(
            w_up_base_ + up_exp_bytes_ * (size_t)(first_expert + i),
            up_exp_bytes_, s0.up, weight_expert_stride_, up_exp_bytes_, rows,
            cudaMemcpyHostToDevice, (cudaStream_t)copy_stream_);
      }
      if (e == cudaSuccess) {
        e = cudaMemcpy2DAsync(
            w_down_base_ + down_exp_bytes_ * (size_t)(first_expert + i),
            down_exp_bytes_, s0.down, weight_expert_stride_, down_exp_bytes_,
            rows, cudaMemcpyHostToDevice, (cudaStream_t)copy_stream_);
      }
      total += (gate_exp_bytes_ + up_exp_bytes_ + down_exp_bytes_) * rows;
      i = j;
    }
  }
  if (e != cudaSuccess) {
    LOG_ERROR("[GGML-MOE] weight H2D failed: %s\n", cudaGetErrorString(e));
    return false;
  }
  // Recorded even for an all-resident chunk (total == 0): the event is what
  // wait_stage_free() polls before letting the caller refill this buffer, and
  // skipping it would leave a stale event from two chunks ago as the gate.
  if (backend_stream_copies_) {
    // backend_stage_event_ is a GGML event wrapper, but its CUDA event can be
    // recorded on the raw copy stream. The compute GGML stream waits on this
    // event immediately before graph submission.
    if (cudaEventRecord((cudaEvent_t)backend_copy_t1_event_[buf],
                        (cudaStream_t)copy_stream_) != cudaSuccess)
      return false;
    auto *event = (ggml_backend_event_t)backend_stage_event_[buf];
    if (cudaEventRecord((cudaEvent_t)event->context,
                        (cudaStream_t)copy_stream_) != cudaSuccess)
      return false;
  } else if (cudaEventRecord((cudaEvent_t)ev_free_[buf],
                             (cudaStream_t)copy_stream_) != cudaSuccess) {
    return false;
  }
  pending_[buf] = true;
  pending_bytes_[buf] = total + direct_h2d_bytes_[buf];
  direct_h2d_bytes_[buf] = 0;
  copy_batch_started_[buf] = false;
  return true;
}

bool GgmlMoeRunner::copy_expert_h2d(int buf, int expert_index,
                                    const void *src_slab) {
  if (!layer_ready_ || buf < 0 || buf >= NBUF || !src_slab ||
      expert_index < 0 || expert_index >= plan_.n_experts)
    return false;
  if (!begin_copy_batch(buf)) return false;

  cudaSetDevice(device_);
  const uint8_t *src = (const uint8_t *)src_slab;
  cudaStream_t st = (cudaStream_t)copy_stream_;
  if (!st) return false;
  cudaError_t e = cudaSuccess;
  int h2d_timing_slot = -1;
  if (!begin_h2d_timing(buf, h2d_timing_slot)) return false;
  if (!use_packed_transfer()) {
    uint8_t *gate_dst = w_gate_base_ + gate_exp_bytes_ * (size_t)expert_index;
    uint8_t *up_dst = w_up_base_ + up_exp_bytes_ * (size_t)expert_index;
    uint8_t *down_dst = w_down_base_ + down_exp_bytes_ * (size_t)expert_index;
    e = cudaMemcpyAsync(gate_dst, src, gate_exp_bytes_, cudaMemcpyHostToDevice,
                        st);
    if (e == cudaSuccess)
      e = cudaMemcpyAsync(up_dst, src + gate_exp_bytes_, up_exp_bytes_,
                          cudaMemcpyHostToDevice, st);
    if (e == cudaSuccess)
      e = cudaMemcpyAsync(down_dst, src + gate_exp_bytes_ + up_exp_bytes_,
                          down_exp_bytes_, cudaMemcpyHostToDevice, st);
  } else {
    if (!packed_base_) return false;
    if (batch_d2d_unpack_ && !debug_check_device_output()) {
      // Direct RAM sources are already persistently pinned. Give each one a
      // distinct packed record and defer the layout conversion until
      // push_chunk(), where all direct-H2D and staged-H2D records can share
      // one scatter-unpack launch.
      const size_t compact = packed_h2d_slot_map_[buf].size();
      if (compact >= (size_t)plan_.chunk_experts) return false;
      e = cudaMemcpyAsync(
          packed_base_ + compact * weight_expert_stride_, src,
          weight_raw_expert_stride_, cudaMemcpyHostToDevice, st);
      if (e == cudaSuccess)
        packed_h2d_slot_map_[buf].push_back(expert_index);
    } else {
      // Legacy packed path: direct calls are serialized on the same copy
      // stream, so one reusable record is sufficient.
      e = cudaMemcpyAsync(packed_base_, src, weight_raw_expert_stride_,
                          cudaMemcpyHostToDevice, st);
      if (e == cudaSuccess) {
        e = launch_ggml_moe_unpack(
            packed_base_,
            w_gate_base_ + gate_exp_bytes_ * (size_t)expert_index,
            w_up_base_ + up_exp_bytes_ * (size_t)expert_index,
            w_down_base_ + down_exp_bytes_ * (size_t)expert_index, 1,
            weight_expert_stride_, gate_exp_bytes_, up_exp_bytes_,
            down_exp_bytes_, gate_exp_bytes_, up_exp_bytes_, down_exp_bytes_,
            (cudaStream_t)copy_stream_);
      }
    }
  }
  if (!end_h2d_timing(buf, h2d_timing_slot)) return false;
  if (e != cudaSuccess) {
    LOG_ERROR("[GGML-MOE] direct RAM H2D failed for expert=%d: %s\n",
              expert_index, cudaGetErrorString(e));
    return false;
  }
  direct_h2d_bytes_[buf] += weight_raw_expert_stride_;
  return true;
}

bool GgmlMoeRunner::copy_expert_h2d_strided(int buf, int first_expert,
                                            int count, const void *src_slab,
                                            size_t src_stride) {
  if (!layer_ready_ || buf < 0 || buf >= NBUF || count <= 0 ||
      first_expert < 0 || first_expert + count > plan_.n_experts ||
      !src_slab || src_stride < weight_raw_expert_stride_)
    return false;
  if (use_packed_transfer()) {
    const uint8_t *src = static_cast<const uint8_t *>(src_slab);
    for (int i = 0; i < count; ++i) {
      if (!copy_expert_h2d(buf, first_expert + i,
                           src + (size_t)i * src_stride))
        return false;
    }
    return true;
  }
  if (!begin_copy_batch(buf)) return false;

  cudaSetDevice(device_);
  cudaStream_t st = (cudaStream_t)copy_stream_;
  if (!st) return false;
  const uint8_t *src = static_cast<const uint8_t *>(src_slab);
  const size_t rows = static_cast<size_t>(count);
  int h2d_timing_slot = -1;
  if (!begin_h2d_timing(buf, h2d_timing_slot)) return false;

  cudaError_t e = cudaMemcpy2DAsync(
      w_gate_base_ + gate_exp_bytes_ * (size_t)first_expert,
      gate_exp_bytes_, src, src_stride, gate_exp_bytes_, rows,
      cudaMemcpyHostToDevice, st);
  if (e == cudaSuccess) {
    e = cudaMemcpy2DAsync(
        w_up_base_ + up_exp_bytes_ * (size_t)first_expert, up_exp_bytes_,
        src + gate_exp_bytes_, src_stride, up_exp_bytes_, rows,
        cudaMemcpyHostToDevice, st);
  }
  if (e == cudaSuccess) {
    e = cudaMemcpy2DAsync(
        w_down_base_ + down_exp_bytes_ * (size_t)first_expert, down_exp_bytes_,
        src + gate_exp_bytes_ + up_exp_bytes_, src_stride, down_exp_bytes_,
        rows, cudaMemcpyHostToDevice, st);
  }
  if (!end_h2d_timing(buf, h2d_timing_slot)) return false;
  if (e != cudaSuccess) {
    LOG_ERROR("[GGML-MOE] strided RAM H2D failed at compact expert=%d "
              "count=%d: %s\n",
              first_expert, count, cudaGetErrorString(e));
    return false;
  }
  direct_h2d_bytes_[buf] += weight_raw_expert_stride_ * rows;
  return true;
}

bool GgmlMoeRunner::copy_expert_d2d(int buf, int expert_index,
                                    const void *src_slab) {
  if (!layer_ready_ || !src_slab || buf < 0 || buf >= NBUF ||
      !copy_batch_started_[buf])
    return false;
  if (expert_index < 0 || expert_index >= plan_.n_experts) return false;
  cudaSetDevice(device_);
  cudaStream_t st = (cudaStream_t)copy_stream_;
  if (!st) return false;
  const uint8_t *src = (const uint8_t *)src_slab;
  if (!use_packed_transfer()) {
    uint8_t *gate_dst = w_gate_base_ + gate_exp_bytes_ * (size_t)expert_index;
    uint8_t *up_dst = w_up_base_ + up_exp_bytes_ * (size_t)expert_index;
    uint8_t *down_dst = w_down_base_ + down_exp_bytes_ * (size_t)expert_index;
    cudaError_t e = cudaMemcpyAsync(gate_dst, src, gate_exp_bytes_,
                                    cudaMemcpyDeviceToDevice, st);
    if (e == cudaSuccess)
      e = cudaMemcpyAsync(up_dst, src + gate_exp_bytes_, up_exp_bytes_,
                          cudaMemcpyDeviceToDevice, st);
    if (e == cudaSuccess)
      e = cudaMemcpyAsync(down_dst, src + gate_exp_bytes_ + up_exp_bytes_,
                          down_exp_bytes_, cudaMemcpyDeviceToDevice, st);
    if (e != cudaSuccess) {
      LOG_ERROR("[GGML-MOE] planar weight D2D failed: %s\n",
                cudaGetErrorString(e));
      return false;
    }
    pending_d2d_bytes_[buf] += gate_exp_bytes_ + up_exp_bytes_ + down_exp_bytes_;
    raw_d2d_pending_ = true;
    return true;
  }
  // The VRAM cache stores an expert exactly as the RAM tier's slab does --
  // [gate|up|down] contiguous, no padding. Land it once in the packed device
  // buffer, then unpack into the three ordinary GGML planes.
  // As with direct H2D, all D2D calls are ordered on the copy stream.  The
  // default path reuses one record and unpacks immediately.  The Qwen-only
  // A/B path gives every queued D2D source its own record and defers one
  // scatter-unpack for the whole chunk to mark_copy_d2d_done(); otherwise a
  // cache hit costs one unpack kernel launch even though the copies are
  // already naturally ordered.
  if (batch_d2d_unpack_ && !debug_check_device_output()) {
    const size_t compact = d2d_slot_map_[buf].size();
    if (compact >= (size_t)plan_.chunk_experts) return false;
    cudaError_t e = cudaMemcpyAsync(
        packed_base_ + compact * weight_expert_stride_, src,
        weight_raw_expert_stride_, cudaMemcpyDeviceToDevice, st);
    if (e != cudaSuccess) {
      LOG_ERROR("[GGML-MOE] batched weight D2D failed: %s\n",
                cudaGetErrorString(e));
      return false;
    }
    d2d_slot_map_[buf].push_back(expert_index);
    pending_d2d_bytes_[buf] += weight_raw_expert_stride_;
    raw_d2d_pending_ = true;
    return true;
  }
  cudaError_t e = cudaMemcpyAsync(
      packed_base_, src,
      weight_raw_expert_stride_, cudaMemcpyDeviceToDevice, st);
  if (e == cudaSuccess) {
    e = launch_ggml_moe_unpack(
        packed_base_,
        w_gate_base_ + gate_exp_bytes_ * (size_t)expert_index,
        w_up_base_ + up_exp_bytes_ * (size_t)expert_index,
        w_down_base_ + down_exp_bytes_ * (size_t)expert_index, 1,
        weight_expert_stride_, gate_exp_bytes_, up_exp_bytes_, down_exp_bytes_,
        gate_exp_bytes_, up_exp_bytes_, down_exp_bytes_, st);
  }
  if (e != cudaSuccess) {
    LOG_ERROR("[GGML-MOE] weight D2D failed: %s\n", cudaGetErrorString(e));
    return false;
  }
  pending_d2d_bytes_[buf] += weight_raw_expert_stride_;
  if (debug_check_device_output()) {
    e = cudaStreamSynchronize(st);
    if (e != cudaSuccess) return false;
    const size_t gate_row =
        ggml_row_size((ggml_type)plan_.gate_type, plan_.d_model);
    const size_t up_row =
        ggml_row_size((ggml_type)plan_.up_type, plan_.d_model);
    const size_t down_row =
        ggml_row_size((ggml_type)plan_.down_type, plan_.d_ff);
    std::vector<uint8_t> source_row;
    std::vector<uint8_t> device_row;
    auto verify_d2d_row = [&](const char *name, const uint8_t *host,
                              const uint8_t *device, size_t row_bytes) {
      source_row.resize(row_bytes);
      device_row.resize(row_bytes);
      const cudaError_t src_e = cudaMemcpy(source_row.data(), host, row_bytes,
                                           cudaMemcpyDeviceToHost);
      const cudaError_t dst_e = cudaMemcpy(device_row.data(), device, row_bytes,
                                           cudaMemcpyDeviceToHost);
      if (src_e != cudaSuccess || dst_e != cudaSuccess ||
          std::memcmp(source_row.data(), device_row.data(), row_bytes) != 0) {
        LOG_INFO("[GGML-MOE-DEBUG] D2D unpack mismatch slot=%d plane=%s "
                 "bytes=%zu copy=%s\n",
                 expert_index, name, row_bytes,
                 src_e != cudaSuccess
                     ? cudaGetErrorString(src_e)
                     : (dst_e == cudaSuccess ? "ok" : cudaGetErrorString(dst_e)));
      }
    };
    const uint8_t *src_record = src;
    verify_d2d_row("gate-last", src_record + gate_exp_bytes_ - gate_row,
                   w_gate_base_ + gate_exp_bytes_ * (size_t)expert_index +
                       gate_exp_bytes_ - gate_row,
                   gate_row);
    verify_d2d_row("up-last", src_record + gate_exp_bytes_ + up_exp_bytes_ -
                                      up_row,
                   w_up_base_ + up_exp_bytes_ * (size_t)expert_index +
                       up_exp_bytes_ - up_row,
                   up_row);
    verify_d2d_row("down-last",
                   src_record + gate_exp_bytes_ + up_exp_bytes_ +
                       down_exp_bytes_ - down_row,
                   w_down_base_ + down_exp_bytes_ * (size_t)expert_index +
                       down_exp_bytes_ - down_row,
                   down_row);
  }
  raw_d2d_pending_ = true;
  return true;
}

void GgmlMoeRunner::sync_copies() {
  // Backend-stream copies are already ordered before the graph. Waiting here
  // would restore the old per-layer barrier. Completion events are drained
  // by collect_h2d() after the graph has been submitted.
  if (backend_stream_copies_) {
    if (raw_d2d_pending_ && copy_stream_ && !use_async_moe_graph_reuse()) {
      cudaSetDevice(device_);
      cudaStreamSynchronize((cudaStream_t)copy_stream_);
    }
    raw_d2d_pending_ = false;
    return;
  }
  if (!copy_stream_) return;
  cudaSetDevice(device_);
  cudaStreamSynchronize((cudaStream_t)copy_stream_);
  for (int i = 0; i < NBUF; ++i) {
    if (!pending_[i]) continue;
    account_copy_buffer(i);
  }
}

void GgmlMoeRunner::collect_h2d(double &seconds, size_t &bytes) {
  GgmlMoeTransferStats stats;
  collect_transfer_stats(stats);
  seconds = stats.h2d_seconds;
  bytes = stats.h2d_bytes;
}

void GgmlMoeRunner::collect_transfer_stats(GgmlMoeTransferStats &stats) {
  if (backend_stream_copies_ && !use_async_moe_graph_reuse()) {
    // Preserve the conservative backend-stream mode when queued graph reuse
    // is disabled.  The asynchronous mode leaves these events pending across
    // layers; finish_h2d() performs the single request-level drain.
    for (int i = 0; i < NBUF; ++i) wait_stage_free(i);
  }
  stats.d2d_seconds = d2d_seconds_;
  stats.h2d_seconds = h2d_seconds_;
  stats.h2d_dma_seconds = h2d_dma_seconds_;
  stats.d2d_bytes = d2d_bytes_;
  stats.h2d_bytes = h2d_bytes_;
  d2d_seconds_ = 0.0;
  d2d_bytes_ = 0;
  h2d_seconds_ = 0.0;
  h2d_dma_seconds_ = 0.0;
  h2d_bytes_ = 0;
}

void GgmlMoeRunner::finish_h2d() {
  if (backend_stream_copies_) {
    // wait_stage_free() waits for the backend event and accounts the
    // transfer. The deferred source holds are released only after this
    // drain, so an eviction cannot overwrite a VRAM source still read by the
    // copy stream.
    for (int i = 0; i < NBUF; ++i) wait_stage_free(i);
  }
  // External resident-layer weights are read directly by the shared GGML
  // compute stream rather than by copy_stream_.  Drain that stream before
  // releasing their temporary cache holds at the request boundary.
  if (backend_ && !deferred_vram_holds_.empty())
    ggml_backend_synchronize((ggml_backend_t)backend_);
  release_completed_vram_holds();
}

void GgmlMoeRunner::defer_vram_hold(void *topology, int layer, int expert,
                                     const uint8_t *device_ptr) {
  if (!topology || !device_ptr) return;
  deferred_vram_holds_.push_back({topology, layer, expert, device_ptr});
}

void GgmlMoeRunner::release_completed_vram_holds() {
  // This method is called at a runner reuse point or request end. Waiting here
  // is the safe lifetime boundary for source VRAM; callers no longer need to
  // impose that wait at the end of every layer.
  for (int i = 0; i < NBUF; ++i) wait_stage_free(i);
  for (const DeferredVramHold &hold : deferred_vram_holds_) {
    auto *topology = static_cast<GpuTopology *>(hold.topology);
    topology->release_hold(hold.layer, hold.expert, hold.device_ptr, false);
  }
  deferred_vram_holds_.clear();
  for (auto &release : deferred_host_releases_) {
    if (release)
      release();
  }
  deferred_host_releases_.clear();
}

void GgmlMoeRunner::defer_host_release(std::function<void()> release) {
  if (release)
    deferred_host_releases_.push_back(std::move(release));
}
#else  // !ZEROMOE_USE_GGML_CUDA
void moe::GgmlMoeTransferManager::reset_transfer_state(int) noexcept {}

bool GgmlMoeRunner::ensure_stage_buffers(size_t) { return false; }

GgmlMoeStageSlot GgmlMoeRunner::stage_slot(int, int) const {
  return GgmlMoeStageSlot{};
}

void GgmlMoeRunner::wait_stage_free(int) {}

bool GgmlMoeRunner::push_chunk(int, int, int, const bool *) { return false; }

bool GgmlMoeRunner::begin_copy_batch(int) { return false; }

bool GgmlMoeRunner::copy_expert_h2d(int, int, const void *) { return false; }

bool GgmlMoeRunner::copy_expert_h2d_strided(int, int, int, const void *,
                                            size_t) {
  return false;
}

bool GgmlMoeRunner::copy_expert_d2d(int, int, const void *) { return false; }

bool GgmlMoeRunner::mark_copy_d2d_done(int) { return false; }

void GgmlMoeRunner::sync_copies() {}

void GgmlMoeRunner::collect_h2d(double &s, size_t &b) { s = 0.0; b = 0; }

void GgmlMoeRunner::collect_transfer_stats(GgmlMoeTransferStats &stats) {
  stats = GgmlMoeTransferStats{};
}

void GgmlMoeRunner::finish_h2d() {}

void GgmlMoeRunner::defer_vram_hold(void *, int, int, const uint8_t *) {}

void GgmlMoeRunner::release_completed_vram_holds() {}
#endif  // ZEROMOE_USE_GGML_CUDA
