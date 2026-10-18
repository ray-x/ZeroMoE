#pragma once
// =====================================================================
//  Chunked, double-buffered expert-weight streaming for PREFILL.
//  EngineConfig::moe_prefill_stream -- default OFF.
// =====================================================================
//
// WHY THIS EXISTS (measured 2026-09-05, real 441-token qwen4exp prefill,
// warm page cache, moe_prefill_batched already ON):
//
//     prefill        25.60 s  (17.22 tok/s)
//       expert       14.18 s
//         get_expert  6.24 s   <- 31.07 GB pread+memcpy @ 5.08 GB/s
//         resolve_loc 2.97 s   \
//         active_up   1.24 s    |  VRAM LRU cache machinery
//         promote_up  0.86 s    |
//         vram_demote 0.34 s   /
//         gpu cmp     0.09 s   <- the ACTUAL expert matmul
//       attn          3.51 s
//       dense_gpu     2.10 s
//     cache = 0% hit
//
// Two independent structural problems, both invisible at decode:
//
//  1. THE CACHE CANNOT HELP DURING PREFILL, BY ARITHMETIC. A long prompt
//     activates essentially every expert in every layer (512 tokens x
//     top-10 over 512 experts = ~5120 draws, so coupon-collector puts all
//     512 in play), and 48 layers x 512 experts is ~68 GB of DISTINCT
//     slabs against a 29.5 GB VRAM cache. Every expert is touched EXACTLY
//     ONCE. That is why the run above reports `cache=0% hit` -- not a
//     tuning failure, and not fixable by resizing anything. So the whole
//     LRU/promotion/eviction/pinning/temp-pin apparatus (5.41 s above) is
//     pure overhead here. Decode is the opposite case (10 of 512 experts
//     per token, genuine reuse across tokens) and keeps using it.
//
//  2. DISK, PCIe AND COMPUTE WERE FULLY SERIALIZED. The batched loop did
//     get_expert (blocking read) -> resolve_gpu_loc (blocking H2D) ->
//     GEMM -> cudaStreamSynchronize, per expert, 24,576 times. The three
//     costs ADD (6.12 + ~5.4 + ...) where they should overlap to max().
//
// WHAT THIS DOES, mirroring what llama.cpp gets for free from mmap +
// ggml's op-offload:
//   * Weight bytes come straight from a whole-shard mmap (GGUFFile::
//     enable_mmap) -- the OS page cache is the cache, no private slab, no
//     pread, no RAM tier, no claim_or_wait.
//   * Experts are staged a CHUNK at a time into PINNED host memory with an
//     OMP-parallel memcpy (all cores, vs one thread doing 24,576 serial
//     preads), then pushed to the device as ONE large contiguous H2D.
//     73,728 tiny scattered copies become ~768 big ones, and a pinned
//     source is what lets the DMA actually run async instead of being
//     staged synchronously through a driver bounce buffer.
//   * Double-buffered: chunk c+1 is staged on the CPU while chunk c's
//     GEMMs run on the GPU, so read/copy/compute overlap.
//
// The device slab layout per expert slot is [gate][up][down] contiguous,
// which is exactly what populate_vram_shadow's slab-relative metas
// (gate=0, up=gb, down=gb+ub) already describe -- so the existing
// expert_ffn_gpu_issue / _finish_stage1 pointer math
// (`base + (up_meta.offset - gate_meta.offset)`) works unchanged against
// a slot in here. Nothing about the compute path had to be re-derived.
//
// SAFETY NOTE on stream choice: copy_stream is created NON-blocking on
// purpose. Every matmul launcher in cuda/ uses a bare <<<>>> (legacy
// default stream), and a *blocking* stream would implicitly serialize
// against it -- which is precisely the overlap we are trying to buy. The
// cost of non-blocking is that ordering becomes OUR job, so compute waits
// on h2d_done via an explicit cudaStreamWaitEvent, and buffer reuse waits
// on comp_done via cudaEventSynchronize. Both events are recorded once at
// construction so a first-iteration wait is a genuine no-op rather than a
// wait on a never-recorded event -- the exact trap ExpertLocation::
// copy_dispatched documents for active_layer_slots.

#include "logger.hpp"
#include "memory_guard.hpp"
#include <cstdint>
#include <limits>
#include <cuda_runtime.h>

struct PrefillExpertStream {
  static constexpr int NBUF = 2;

  uint8_t *h_stage[NBUF] = {nullptr, nullptr};
  uint8_t *d_slab[NBUF] = {nullptr, nullptr};
  cudaEvent_t h2d_done[NBUF] = {nullptr, nullptr};
  cudaEvent_t h2d_t0[NBUF] = {nullptr, nullptr};  // timing-enabled pair, see
  bool h2d_pending[NBUF] = {false, false};        // collect_h2d()
  cudaEvent_t comp_done[NBUF] = {nullptr, nullptr};
  cudaStream_t copy_stream = nullptr;

  size_t slab_stride = 0;    // ALLOCATED per-expert bytes (max over layers)
  // Bytes actually used per slot for the CURRENT layer. Allocation has to
  // cover the largest expert in the whole model, but a layer whose experts
  // are smaller should not pay to transfer the difference: at qwen4exp's
  // real shape the allocated stride is 3.07 MB while a typical layer's
  // expert is 2.51 MB, so a fixed stride shipped 38.33 GB to move 31.30 GB
  // of real weights -- 22% of the PCIe budget spent on padding.
  size_t active_stride = 0;
  int chunk_experts = 0;   // slots per chunk buffer
  int device_id = -1;
  bool ready = false;
  size_t host_reservation_bytes = 0;

  // Grow-only in both dimensions. Returns false on any allocation failure,
  // with everything released -- the caller falls back to the legacy path
  // rather than running half-configured (this file's own house rule for
  // every GPU fast path).
  bool ensure(int gpu_id, size_t max_slab_bytes, int experts_per_chunk) {
    if (ready && device_id == gpu_id && slab_stride >= max_slab_bytes &&
        chunk_experts >= experts_per_chunk)
      return true;
    if (gpu_id < 0 || max_slab_bytes == 0 || experts_per_chunk <= 0)
      return false;

    // Grow to the max of what we have and what is now asked for, so a
    // later layer with bigger experts (mixed-precision GGUFs are real --
    // see load_expert_slab's own mixed-layer comment) never shrinks us.
    const size_t want_stride =
        ((max_slab_bytes > slab_stride ? max_slab_bytes : slab_stride) + 255) &
        ~(size_t)255;
    const int want_experts =
        experts_per_chunk > chunk_experts ? experts_per_chunk : chunk_experts;

    destroy();

    if (cudaSetDevice(gpu_id) != cudaSuccess) return false;
    if (want_stride > std::numeric_limits<size_t>::max() /
                         (size_t)want_experts ||
        want_stride * (size_t)want_experts >
            std::numeric_limits<size_t>::max() / NBUF) {
      LOG_WARN("[PREFILL-STREAM] staging size overflow; falling back to the "
               "legacy cached path.\n");
      return false;
    }
    const size_t bytes = want_stride * (size_t)want_experts;
    const size_t host_bytes = bytes * NBUF;
    if (!MemoryGuard::try_reserve_host_bytes(host_bytes)) {
      LOG_WARN("[MEM-GUARD] prefill stream host staging refused: %.2f MiB "
               "(reserved=%.2f GiB, budget=%.2f GB); using the legacy path.\n",
               (double)host_bytes / (1024.0 * 1024.0),
               (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
               (double)MemoryGuard::host_budget_bytes() / 1.0e9);
      return false;
    }
    host_reservation_bytes = host_bytes;

    for (int b = 0; b < NBUF; ++b) {
      if (cudaMallocHost((void **)&h_stage[b], bytes) != cudaSuccess) {
        LOG_WARN("[PREFILL-STREAM] cudaMallocHost(%.1f MB) failed -- falling "
                 "back to the legacy cached path.\n", bytes / 1e6);
        destroy();
        return false;
      }
      if (cudaMalloc((void **)&d_slab[b], bytes) != cudaSuccess) {
        LOG_WARN("[PREFILL-STREAM] cudaMalloc(%.1f MB) on GPU %d failed -- "
                 "falling back to the legacy cached path.\n", bytes / 1e6, gpu_id);
        destroy();
        return false;
      }
      // h2d_done/h2d_t0 keep timing ENABLED (no cudaEventDisableTiming) so
      // the real on-GPU transfer duration can be read back once the chunk
      // is known complete -- a host-side timer around an async copy would
      // measure only the launch, which is exactly the mistake that makes
      // "the DMA is fast" claims unfalsifiable.
      if (cudaEventCreate(&h2d_done[b]) != cudaSuccess ||
          cudaEventCreate(&h2d_t0[b]) != cudaSuccess ||
          cudaEventCreateWithFlags(&comp_done[b], cudaEventDisableTiming) !=
              cudaSuccess) {
        destroy();
        return false;
      }
      // Pre-record so the first stage_begin()'s wait is a no-op instead of
      // a wait on an unrecorded event.
      cudaEventRecord(comp_done[b], 0);
    }
    if (cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking) !=
        cudaSuccess) {
      destroy();
      return false;
    }

    slab_stride = want_stride;
    active_stride = want_stride;
    chunk_experts = want_experts;
    device_id = gpu_id;
    ready = true;
    LOG_STREAM_INFO() << "[PREFILL-STREAM] GPU " << gpu_id << ": "
                      << want_experts << " expert slot(s)/chunk x "
                      << (want_stride / 1024) << " KB, " << NBUF
                      << " buffers (" << (2.0 * bytes / 1e6)
                      << " MB pinned host + VRAM)" << std::endl;
    return true;
  }

  // Clamped to what was actually allocated; a layer needing more than that
  // simply keeps the full stride (correct, just not optimal) rather than
  // indexing past the buffer.
  void set_active_stride(size_t bytes) {
    const size_t a = (bytes + 255) & ~(size_t)255;
    active_stride = (a == 0 || a > slab_stride) ? slab_stride : a;
  }
  size_t stride() const { return active_stride ? active_stride : slab_stride; }

  uint8_t *host_slot(int buf, int i) { return h_stage[buf] + (size_t)i * stride(); }
  uint8_t *dev_slot(int buf, int i) { return d_slab[buf] + (size_t)i * stride(); }

  // Block until buffer `buf`'s previous chunk has finished computing, so
  // its host and device memory can be refilled.
  void wait_free(int buf) { cudaEventSynchronize(comp_done[buf]); }

  // Every stream/event op below MUST run with our own device current.
  // Attention and the dense projections legitimately leave GPU 1 current
  // between layers in this dual-GPU config, and `cudaStreamWaitEvent(0,...)`
  // / `cudaEventRecord(..., 0)` name the CURRENT device's legacy default
  // stream -- so without this the compute gate was applied to the WRONG
  // GPU's stream 0 and the expert kernels could read weight bytes whose
  // H2D had not landed. MEASURED as diverged (but fluent, error-free)
  // output 2026-09-05 -- exactly the "finite but wrong" failure this
  // codebase keeps re-learning to distrust.
  void pin_device() { if (device_id >= 0) cudaSetDevice(device_id); }

  // Push `n_slots` worth of staged bytes and record h2d_done[buf].
  bool push(int buf, int n_slots) {
    pin_device();
    const size_t bytes = (size_t)n_slots * stride();
    cudaEventRecord(h2d_t0[buf], copy_stream);
    if (cudaMemcpyAsync(d_slab[buf], h_stage[buf], bytes,
                        cudaMemcpyHostToDevice, copy_stream) != cudaSuccess)
      return false;
    if (cudaEventRecord(h2d_done[buf], copy_stream) != cudaSuccess)
      return false;
    h2d_pending[buf] = true;
    last_push_bytes[buf] = bytes;
    return true;
  }

  // Only valid once buffer `buf` is known complete (i.e. right after
  // wait_free), which is why this is called from there and nowhere else.
  void collect_h2d(int buf, double &out_ms, size_t &out_bytes) {
    out_ms = 0.0;
    out_bytes = 0;
    if (!h2d_pending[buf]) return;
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, h2d_t0[buf], h2d_done[buf]) == cudaSuccess) {
      out_ms = (double)ms;
      out_bytes = last_push_bytes[buf];
    }
    h2d_pending[buf] = false;
  }

  size_t last_push_bytes[NBUF] = {0, 0};

  // Make the legacy default stream (where every cuda/ matmul launches)
  // wait for this chunk's transfer.
  void gate_compute(int buf) {
    pin_device();
    cudaStreamWaitEvent(0, h2d_done[buf], 0);
  }

  // Mark this chunk's compute complete so the buffer can be recycled.
  void mark_computed(int buf) {
    pin_device();
    cudaEventRecord(comp_done[buf], 0);
  }

  void destroy() {
    // ensure() calls this to GROW (a later layer with bigger experts), which
    // means freeing d_slab/h_stage that a previously-issued cudaMemcpyAsync
    // may still be reading from or writing to. Neither cudaFree nor
    // cudaFreeHost waits for that. Growth is rare, so a full device sync
    // here is free insurance against exactly the use-after-free class that
    // already produced fluent-but-wrong output once this session
    // (PinnedFloatBuffer::resize -- see its own comment).
    if (device_id >= 0) {
      cudaSetDevice(device_id);
      cudaDeviceSynchronize();
    }
    if (copy_stream) { cudaStreamDestroy(copy_stream); copy_stream = nullptr; }
    for (int b = 0; b < NBUF; ++b) {
      if (h2d_done[b]) { cudaEventDestroy(h2d_done[b]); h2d_done[b] = nullptr; }
      if (h2d_t0[b]) { cudaEventDestroy(h2d_t0[b]); h2d_t0[b] = nullptr; }
      h2d_pending[b] = false;
      if (comp_done[b]) { cudaEventDestroy(comp_done[b]); comp_done[b] = nullptr; }
      if (d_slab[b]) { cudaFree(d_slab[b]); d_slab[b] = nullptr; }
      if (h_stage[b]) { cudaFreeHost(h_stage[b]); h_stage[b] = nullptr; }
    }
    slab_stride = 0;
    active_stride = 0;
    chunk_experts = 0;
    device_id = -1;
    ready = false;
    MemoryGuard::release_host_bytes(host_reservation_bytes);
    host_reservation_bytes = 0;
  }

  ~PrefillExpertStream() { destroy(); }
};
