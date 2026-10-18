#pragma once
// Raw io_uring backend for LockFreePrefetcher (async_prefetcher.hpp): batches
// up to CAPACITY reads -- the prefetcher's per-expert gate/up/down
// projections -- into ONE io_uring submission, then waits for every
// completion in a single io_uring_enter call, instead of the equivalent
// number of sequential blocking pread()s (direct_io::pread_direct_aware).
// This is where io_uring's real win lives for this workload: pread()
// serializes -- the kernel can't even start the second read until the first
// one returns -- while io_uring hands the block layer/NVMe queue all of them
// at once, so their real disk latency overlaps instead of stacking.
//
// Raw syscalls (io_uring_setup/io_uring_enter via <linux/io_uring.h>), not
// liburing -- this project is zero-dependency (see CLAUDE.md), and these are
// stable kernel UAPI since 5.1 (IORING_OP_READ specifically needs 5.6+; see
// the opcode note below) with nothing to link.
//
// Linux-only: guarded so every other platform (this codebase is regularly
// iterated on from macOS -- see CLAUDE.md's Mac workflow note) gets a stub
// Ring whose init() always fails, so callers can use the identical
// `if (ring.available())` check everywhere without `#ifdef`s scattered
// through async_prefetcher.hpp.
//
// NOT thread-safe, and not meant to be: exactly one LockFreePrefetcher
// worker thread exists per engine instance, and a Ring is owned 1:1 by that
// thread -- no locking needed, same ownership model as direct_io.hpp's
// thread_local AlignedScratch.
//
// UNTESTED on real hardware from this development environment (macOS, no
// io_uring headers, can't even compile this file locally). The ring-index
// and completion-matching logic below follows the documented
// io_uring_setup(2)/io_uring_enter(2) ABI, but treat this the way this
// project treated direct_io.hpp and gguf_tool pack originally: verify it for
// real on a Linux box (io_uring=true in config, watch for [IO-URING] log
// lines, compare decode-time prefetch hit-rate/latency before and after)
// before trusting it in production. On ANY failure -- old kernel, io_uring
// disabled by seccomp/container policy, a bad read -- this always falls back
// to the existing sequential pread_direct_aware path, never a hard failure.
#include "direct_io.hpp"
#include "perf_timer.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__linux__)
#include <atomic>
#include <cerrno>
#include <fcntl.h> // posix_fadvise/POSIX_FADV_DONTNEED, for the drop_pages path below
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace io_uring_backend {

// One request in a batch: read `len` bytes from `fd` at `offset` into
// `dst`. If `is_direct`, `fd` was opened O_DIRECT and dst/offset/len are NOT
// assumed to already be block-aligned -- the Ring stages through an internal
// aligned scratch buffer and memcpy's out the exact range, mirroring
// direct_io::pread_direct_aware's contract exactly so callers don't need
// separate code paths for direct vs buffered requests.
struct ReadRequest {
  int fd = -1;
  bool is_direct = false;
  uint64_t alignment = 4096;
  void *dst = nullptr;
  size_t len = 0;
  uint64_t offset = 0;
};

#if defined(__linux__)

inline int raw_setup(unsigned entries, struct io_uring_params *p) {
  return (int)syscall(__NR_io_uring_setup, entries, p);
}
inline int raw_enter(int fd, unsigned to_submit, unsigned min_complete,
                     unsigned flags) {
  return (int)syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags,
                      nullptr, 0);
}
inline int raw_register(int fd, unsigned opcode, void *arg, unsigned nr_args) {
  return (int)syscall(__NR_io_uring_register, fd, opcode, arg, nr_args);
}

class Ring {
public:
  // Was 8 (>= a single expert's 3 projections: gate/up/down), sized only
  // for the prefetcher's per-hint batches. Bumped to cover
  // RAMSafetyManager's Phase B too, which batches an ENTIRE layer's
  // pending cache-miss set (up to topk experts, typically ~8, times 3
  // reads each) into one submission -- the same deep-queue-depth
  // concurrency io_uring backend uses for its own real miss-
  // servicing path (glm.c's pipe_dispatch/URING_REQ_MAX), rather than
  // this project's original per-hint-only usage. Cheap to size larger:
  // this is just the io_uring_setup() ring-buffer entry count (kernel-
  // side) plus a same-sized direct_io::AlignedScratch[CAPACITY] array
  // (a few bytes each, unallocated until actually used).
  static constexpr unsigned CAPACITY = 64;

  ~Ring() { destroy(); }

  // Sets up the ring. Returns false (never throws) on any failure -- old
  // kernel (<5.1), io_uring disabled by seccomp/container policy, fd/mmap
  // exhaustion -- so callers always have a safe fallback path.
  bool init() {
    struct io_uring_params params;
    std::memset(&params, 0, sizeof(params));
    int fd = raw_setup(CAPACITY, &params);
    if (fd < 0)
      return false;
    ring_fd = fd;

    // Kernels >= 5.4 (IORING_FEAT_SINGLE_MMAP) put the CQ ring inside
    // the SAME mmap as the SQ ring; older kernels (still >= 5.1) need a
    // second, separate mmap. #ifdef guards the header-version case
    // where this build's <linux/io_uring.h> predates the flag entirely.
#ifdef IORING_FEAT_SINGLE_MMAP
    bool single_mmap = (params.features & IORING_FEAT_SINGLE_MMAP) != 0;
#else
    bool single_mmap = false;
#endif

    size_t sq_ring_sz =
        params.sq_off.array + (size_t)params.sq_entries * sizeof(uint32_t);
    size_t cq_ring_sz = params.cq_off.cqes +
                        (size_t)params.cq_entries * sizeof(struct io_uring_cqe);
    size_t sq_map_sz =
        single_mmap ? std::max(sq_ring_sz, cq_ring_sz) : sq_ring_sz;

    sq_ring_ptr = mmap(nullptr, sq_map_sz, PROT_READ | PROT_WRITE, MAP_SHARED,
                       ring_fd, IORING_OFF_SQ_RING);
    if (sq_ring_ptr == MAP_FAILED) {
      sq_ring_ptr = nullptr;
      destroy();
      return false;
    }
    sq_ring_mmap_sz = sq_map_sz;

    if (single_mmap) {
      cq_ring_ptr = sq_ring_ptr;
    } else {
      cq_ring_ptr = mmap(nullptr, cq_ring_sz, PROT_READ | PROT_WRITE,
                         MAP_SHARED, ring_fd, IORING_OFF_CQ_RING);
      if (cq_ring_ptr == MAP_FAILED) {
        cq_ring_ptr = nullptr;
        destroy();
        return false;
      }
      cq_ring_mmap_sz = cq_ring_sz;
    }

    size_t sqes_sz = (size_t)params.sq_entries * sizeof(struct io_uring_sqe);
    sqes_ptr = mmap(nullptr, sqes_sz, PROT_READ | PROT_WRITE, MAP_SHARED,
                    ring_fd, IORING_OFF_SQES);
    if (sqes_ptr == MAP_FAILED) {
      sqes_ptr = nullptr;
      destroy();
      return false;
    }
    sqes_mmap_sz = sqes_sz;

    sq_tail = ptr_at<std::atomic<uint32_t>>(sq_ring_ptr, params.sq_off.tail);
    sq_mask = ptr_at<uint32_t>(sq_ring_ptr, params.sq_off.ring_mask);
    sq_array = ptr_at<uint32_t>(sq_ring_ptr, params.sq_off.array);
    sqes = reinterpret_cast<struct io_uring_sqe *>(sqes_ptr);

    cq_head = ptr_at<std::atomic<uint32_t>>(cq_ring_ptr, params.cq_off.head);
    cq_mask = ptr_at<uint32_t>(cq_ring_ptr, params.cq_off.ring_mask);
    cqes = ptr_at<struct io_uring_cqe>(cq_ring_ptr, params.cq_off.cqes);

    ready = true;
    return true;
  }

  bool available() const { return ready; }

  // Bounds this ring's own io-wq worker pool (IORING_REGISTER_IOWQ_MAX_
  // WORKERS: [0]=bounded workers for regular files/block devices, [1]=
  // unbounded workers for other fd types -- we only ever read plain
  // files, so both are set to the same `count`).
  // found by direct comparison with that reference implementation.
  // Meaningful ONLY once IOSQE_ASYNC is actually forcing reads onto
  // io-wq (see submit_and_wait): before that fix, this call would have
  // sized a worker pool that inline-completing reads never used. Safe
  // to call on kernels that don't support this opcode -- returns false,
  // caller treats it as "kernel default io-wq sizing applies," never a
  // hard failure (same fallback-never-fails discipline as init()).
  bool register_iowq_workers(unsigned count) {
#ifdef IORING_REGISTER_IOWQ_MAX_WORKERS
    if (!ready || count == 0)
      return false;
    unsigned limits[2] = {count, count};
    return raw_register(ring_fd, IORING_REGISTER_IOWQ_MAX_WORKERS, limits, 2) >=
           0;
#else
    // This build's <linux/io_uring.h> predates the opcode (kernel
    // headers older than the 5.15-era addition) -- no-op, same
    // "kernel default sizing applies" fallback as a runtime failure.
    (void)count;
    return false;
#endif
  }

  // Submit `n` (<= CAPACITY) reads and block until every one of them has
  // completed (successfully or not). `results[i]` receives the same
  // signal a pread() would for reqs[i]: >=0 = bytes actually read (after
  // direct_io alignment stripping, if is_direct), <0 = error. All-or-
  // nothing: either every request is submitted and every result[i] is
  // written, or this returns false and touches no result -- callers
  // (active_prefetch) already treat any failure as "fall back to
  // fadvise", so there's no reason to support a partial batch.
  //
  // `drop_pages` (default false, mirrors StreamingEngine::drop_expert_pages
  // exactly -- callers pass that field through, never invent their own
  // policy) issues a POSIX_FADV_DONTNEED for every successfully-completed
  // request's actual on-disk range immediately after reaping it. This is
  // the io_uring-side counterpart to load_expert_slab's own
  // drop_expert_pages-gated fadvise calls (engine.hpp) -- those only ever
  // covered the plain-pread fallback path, so with io_uring=true (this
  // project's default-recommended config) the vast majority of real expert
  // reads -- startup PIN preload (main.cpp's preload_batch_load), Phase B
  // decode-time misses (ram_manager.hpp), and background prefetch
  // (async_prefetcher.hpp) -- went through this Ring and never got their
  // pages dropped at all, regardless of drop_pages' setting. Real-hardware
  // fincore measurement confirmed the gap directly: flipping drop_pages=
  // true with this call site still unfixed left the primary (btrfs)
  // model's page-cache residency completely unchanged (~19-20GB) after a
  // full run, because the code path actually doing the reads never called
  // fadvise at all.
  bool submit_and_wait(const ReadRequest *reqs, int n, ssize_t *results,
                       bool drop_pages = false) {
    if (!ready || n <= 0 || (unsigned)n > CAPACITY)
      return false;

    // Stage direct-I/O requests through per-slot aligned scratch buffers
    // (index-matched to reqs[]) -- exactly direct_io::
    // pread_direct_aware's contract, just batched: dst may not itself be
    // alignment-safe (e.g. a pinned staging buffer's gate+up+down
    // sub-offsets), so direct requests always read into a block-aligned
    // scratch region and get memcpy'd out afterward.
    direct_io::AlignedWindow windows[CAPACITY];
    void *targets[CAPACITY];
    // Per-request: did this read actually need the scratch detour? A
    // request whose offset, length AND destination are all already
    // alignment-clean can be DMA'd straight into the caller's buffer,
    // skipping both the staging read-into-scratch and the memcpy-out
    // below. This is the common case for MoE expert reads by
    // construction -- per-expert GGUF offsets are exact multiples of the
    // 4096 alignment (verified against the real file: zero slack for
    // every expert), tensor byte sizes are exact multiples too, and
    // slab_buffer is posix_memalign(4096) with 4096-multiple
    // sub-offsets -- so the staging memcpy was pure waste on ~every
    // read. Measured scale before this fix: ~112 GB of avoidable memcpy
    // per session (preload + prefetch + phase_b), i.e. real CPU and
    // memory-bandwidth burn competing with the very compute this engine
    // is trying to overlap I/O against.
    bool staged[CAPACITY];
    for (int i = 0; i < n; ++i) {
      staged[i] = false;
      if (reqs[i].is_direct) {
        windows[i] = direct_io::compute_aligned_window(
            reqs[i].offset, reqs[i].len, reqs[i].alignment);
        const bool dst_aligned = (reinterpret_cast<uintptr_t>(reqs[i].dst) %
                                  (uintptr_t)reqs[i].alignment) == 0;
        if (windows[i].front_slack == 0 &&
            windows[i].aligned_len == reqs[i].len && dst_aligned) {
          targets[i] =
              reqs[i].dst; // O_DIRECT straight into the caller's buffer
          perf().n_uring_direct_unstaged.fetch_add(1,
                                                   std::memory_order_relaxed);
          perf().bytes_uring_stage_memcpy_saved.fetch_add(
              (int64_t)reqs[i].len, std::memory_order_relaxed);
        } else {
          scratch[i].ensure(windows[i].aligned_len, reqs[i].alignment);
          if (!scratch[i].buf)
            return false; // allocation failure -- caller falls back entirely
          targets[i] = scratch[i].buf;
          staged[i] = true;
          perf().n_uring_direct_staged.fetch_add(1, std::memory_order_relaxed);
        }
      } else {
        targets[i] = reqs[i].dst;
      }
    }

    // ---- submit ----
    uint32_t tail = sq_tail->load(std::memory_order_relaxed);
    for (int i = 0; i < n; ++i) {
      uint32_t idx = (tail + (uint32_t)i) & *sq_mask;
      struct io_uring_sqe *sqe = &sqes[idx];
      std::memset(sqe, 0, sizeof(*sqe));
      // IORING_OP_READ (needs kernel >= 5.6) reads directly into
      // `addr` for `len` bytes at file offset `off`, same semantics
      // as pread(). If this build's kernel predates 5.6, the
      // completion below simply reports -EINVAL/-EOPNOTSUPP per
      // request, which the existing failure handling already treats
      // as "this request failed" -- so an unsupported opcode fails
      // safe (falls back), it never corrupts anything.
      sqe->opcode = IORING_OP_READ;
      // without this, the kernel is allowed to execute a submitted read INLINE,
      // synchronously, inside the io_uring_enter() call below --
      // unless the file/filesystem natively supports true
      // nonblocking completion -- which would silently defeat the
      // entire point of batching N reads into one submission (the
      // block layer/NVMe queue never actually sees them
      // concurrently; the kernel just processes them one at a time
      // inside the syscall). IOSQE_ASYNC forces every read onto
      // io-wq instead, so N submitted reads genuinely overlap on
      // the device. See register_iowq_workers's own comment for
      // the other half of this fix -- sizing the pool this now
      // actually dispatches to.
      sqe->flags = IOSQE_ASYNC;
      sqe->fd = reqs[i].fd;
      sqe->off = reqs[i].is_direct ? windows[i].aligned_offset : reqs[i].offset;
      sqe->addr = reinterpret_cast<uint64_t>(targets[i]);
      sqe->len =
          (uint32_t)(reqs[i].is_direct ? windows[i].aligned_len : reqs[i].len);
      sqe->user_data = (uint64_t)i; // index back into reqs[]/results[]
      sq_array[idx] = idx;
    }
    sq_tail->store(tail + (uint32_t)n, std::memory_order_release);

    // Single combined submit+wait call. Per io_uring_enter(2): an EINTR
    // return means the call was interrupted BEFORE any submission queue
    // entry was processed, so it is documented-safe to reissue as-is --
    // no risk of double-submitting or losing track of which requests
    // went out, unlike retrying after a partial result would be.
    int r;
    int retries = 0;
    do {
      r = raw_enter(ring_fd, (unsigned)n, (unsigned)n, IORING_ENTER_GETEVENTS);
    } while (r < 0 && errno == EINTR && ++retries < 100);
    if (r < 0)
      return false;

    // ---- reap exactly n completions (the enter call above already
    // blocked until they were all available) ----
    uint32_t chead = cq_head->load(std::memory_order_relaxed);
    for (int reaped = 0; reaped < n; ++reaped) {
      uint32_t idx = (chead + (uint32_t)reaped) & *cq_mask;
      struct io_uring_cqe &cqe = cqes[idx];
      int i = (int)cqe.user_data;
      if (i < 0 || i >= n)
        continue; // defensive; should never happen
      ssize_t res = (ssize_t)cqe.res;
      if (reqs[i].is_direct && res >= 0 && staged[i]) {
        uint64_t have = (uint64_t)res > windows[i].front_slack
                            ? (uint64_t)res - windows[i].front_slack
                            : 0;
        uint64_t want = reqs[i].len;
        uint64_t got = have < want ? have : want;
        if (got > 0)
          std::memcpy(reqs[i].dst, scratch[i].buf + windows[i].front_slack,
                      got);
        results[i] = (ssize_t)got;
      } else if (reqs[i].is_direct && res >= 0) {
        // Unstaged: the DMA landed directly in reqs[i].dst. Clamp to
        // the requested length for an identical contract to the
        // staged path (aligned_len == len here, so this is a no-op
        // on success, but keeps a short read reporting the same way).
        results[i] = (ssize_t)((uint64_t)res < reqs[i].len ? (uint64_t)res
                                                           : reqs[i].len);
      } else {
        results[i] = res;
      }
      // Per-drive byte attribution (perf_timer.hpp's bytes_disk_read_
      // primary/_mirror) -- answers "is the striped read actually
      // balanced across both drives in practice" with real numbers,
      // not just design intent. is_mirror_fd() is a cheap linear scan
      // over a single-digit-length registry, negligible next to the
      // real disk I/O this reap loop is already waiting on.
      if (results[i] > 0) {
        if (is_mirror_fd(reqs[i].fd)) {
          perf().bytes_disk_read_mirror.fetch_add(results[i],
                                                  std::memory_order_relaxed);
        } else {
          perf().bytes_disk_read_primary.fetch_add(results[i],
                                                   std::memory_order_relaxed);
        }
      }
      // Evict the pages this read actually touched, using the SAME
      // [off, len) that was submitted in the sqe above (the aligned
      // window for a direct request, since that's the real range the
      // kernel read into -- fadvise'ing only the caller's narrower
      // `reqs[i].len` would leave the alignment-padding pages behind).
      // Best-effort: a failed fadvise() is not itself a read failure,
      // so its return value is deliberately ignored, matching every
      // other drop_expert_pages call site in this codebase.
      if (drop_pages && res >= 0) {
        uint64_t off =
            reqs[i].is_direct ? windows[i].aligned_offset : reqs[i].offset;
        uint64_t len = reqs[i].is_direct ? windows[i].aligned_len : reqs[i].len;
        posix_fadvise(reqs[i].fd, (off_t)off, (off_t)len, POSIX_FADV_DONTNEED);
      }
    }
    cq_head->store(chead + (uint32_t)n, std::memory_order_release);
    return true;
  }

private:
  template <typename T> static T *ptr_at(void *base, uint32_t offset) {
    return reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(base) + offset);
  }

  void destroy() {
    if (sqes_ptr) {
      munmap(sqes_ptr, sqes_mmap_sz);
      sqes_ptr = nullptr;
    }
    if (cq_ring_ptr && cq_ring_ptr != sq_ring_ptr)
      munmap(cq_ring_ptr, cq_ring_mmap_sz);
    cq_ring_ptr = nullptr;
    if (sq_ring_ptr) {
      munmap(sq_ring_ptr, sq_ring_mmap_sz);
      sq_ring_ptr = nullptr;
    }
    if (ring_fd >= 0) {
      close(ring_fd);
      ring_fd = -1;
    }
    ready = false;
  }

  int ring_fd = -1;
  void *sq_ring_ptr = nullptr;
  size_t sq_ring_mmap_sz = 0;
  void *cq_ring_ptr = nullptr;
  size_t cq_ring_mmap_sz = 0;
  void *sqes_ptr = nullptr;
  size_t sqes_mmap_sz = 0;

  // sq_head/cq_tail (the kernel-owned consumer-side indices) are
  // deliberately never read: submit_and_wait is fully synchronous (submit
  // N, block for N completions, return) with at most one batch in flight
  // at a time, so there's never a need to check remaining ring capacity
  // before writing SQEs or to poll for "how many completions are ready"
  // before reaping -- io_uring_enter's own min_complete blocking already
  // guarantees exactly N are ready when it returns.
  std::atomic<uint32_t> *sq_tail = nullptr;
  uint32_t *sq_mask = nullptr;
  uint32_t *sq_array = nullptr;
  struct io_uring_sqe *sqes = nullptr;

  std::atomic<uint32_t> *cq_head = nullptr;
  uint32_t *cq_mask = nullptr;
  struct io_uring_cqe *cqes = nullptr;

  direct_io::AlignedScratch
      scratch[CAPACITY]; // per-slot, index-matched to a batch's reqs[]
  bool ready = false;
};

#else // !__linux__

// Stub: init() always fails, so callers on macOS (or any non-Linux
// platform) unconditionally take the existing sequential pread_direct_aware
// fallback path via the same `if (ring.available())` check Linux uses --
// no #ifdefs needed in async_prefetcher.hpp.
class Ring {
public:
  // Mirrors the real Linux Ring::CAPACITY so callers (e.g. ram_manager.hpp's
  // disk_batch_padding) can reference it unconditionally without #ifdefs --
  // this stub is always unavailable, so the value itself is never load-bearing.
  static constexpr unsigned CAPACITY = 64;
  bool init() { return false; }
  bool available() const { return false; }
  bool register_iowq_workers(unsigned) { return false; }
  bool submit_and_wait(const ReadRequest *, int, ssize_t *, bool = false) {
    return false;
  }
};

#endif // __linux__

} // namespace io_uring_backend
