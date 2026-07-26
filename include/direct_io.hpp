#pragma once
// O_DIRECT-aware pread helper. Opt-in (EngineConfig::direct_io, default
// false) -- everything here is a no-op/pass-through to a plain pread()
// unless a caller explicitly asks for direct-I/O mode AND the underlying
// fd was actually opened with O_DIRECT (see GGUFFile::enable_direct_io in
// gguf_loader.hpp, which only flips a shard's fd to O_DIRECT after
// confirming both that the loaded model's own general.alignment KV value
// is large enough and that a real test read against that fd succeeds --
// this file never assumes either, it just respects whatever the caller
// tells it via `is_direct`).
//
// O_DIRECT (Linux) requires the read offset, the read length, and the
// destination buffer's address to all be multiples of the filesystem's
// logical block size (this project standardizes on 4096, matching
// tools/gguf_tool pack's own --align default). A GGUF file repacked by
// that tool satisfies the OFFSET requirement for every tensor by
// construction, but per-expert tensor LENGTHS are not generally multiples
// of 4096, and StreamingEngine's packed-slab destination buffers
// (MoEExpertBlock::slab_buffer) pack gate/up/down tightly adjacent with
// no rounding slack between them -- slab_capacity is read downstream as
// the exact real byte count to upload to VRAM (GpuTopology::
// upload_expert_async), so writing a rounded-up (padded) length directly
// into that buffer would leak alignment-padding garbage bytes into the
// VRAM upload. To avoid that entire class of risk, this helper ALWAYS
// stages O_DIRECT reads through an aligned scratch buffer and memcpy's
// out only the exact bytes requested -- it never writes more than `len`
// bytes to `dst`, regardless of alignment. The extra memcpy this costs
// (microseconds, for a few-MB expert slab) is negligible next to the
// disk read it's paired with, and worth it to keep every existing
// buffer's size/layout exactly as it was before this feature existed.
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace direct_io {

// Pure arithmetic, zero I/O, zero platform dependency -- deliberately
// extracted this way so it can be unit-tested on any machine (see
// tests/test_components.cpp::test_direct_io_aligned_window_logic),
// unlike the O_DIRECT syscall behavior itself, which this darwin dev
// machine cannot exercise at all. This is also the part most likely to
// have a subtle off-by-one bug (wrong rounding direction, an inclusive/
// exclusive mixup) -- exactly the kind of thing worth verifying in
// isolation before trusting it in the real read path.
struct AlignedWindow {
    uint64_t aligned_offset;  // <= real_offset, multiple of alignment
    uint64_t aligned_len;     // >= real_len, multiple of alignment, covers [real_offset, real_offset+real_len)
    uint64_t front_slack;     // real_offset - aligned_offset; real data starts at this byte within the aligned window
};

inline AlignedWindow compute_aligned_window(uint64_t real_offset, uint64_t real_len, uint64_t alignment) {
    uint64_t aligned_offset = real_offset & ~(alignment - 1);
    uint64_t front_slack = real_offset - aligned_offset;
    uint64_t real_end = real_offset + real_len;
    uint64_t aligned_end = (real_end + alignment - 1) & ~(alignment - 1);
    return AlignedWindow{aligned_offset, aligned_end - aligned_offset, front_slack};
}

// Thread-local, grow-only, aligned scratch buffer. Deliberately
// thread_local (not a shared member) -- RAMSafetyManager::
// get_experts_batch's Phase B calls StreamingEngine::load_expert_slab
// (and therefore this function) from multiple OMP worker threads
// concurrently, each servicing a different expert; a shared mutable
// scratch buffer would be a real data race, while a thread_local one
// needs no lock and no shared-state reasoning at all.
struct AlignedScratch {
    uint8_t* buf = nullptr;
    size_t cap = 0;
    ~AlignedScratch() { if (buf) free(buf); }
    void ensure(size_t need, size_t alignment) {
        if (cap >= need) return;
        if (buf) free(buf);
        buf = nullptr;
        if (posix_memalign((void**)&buf, alignment, need) != 0) buf = nullptr;
        cap = buf ? need : 0;
    }
};

// Emulates pread()'s contract exactly (returns (ssize_t)len on full
// success, or a value callers' existing `r != len` checks will flag on
// failure) so every existing call site's error handling/logging works
// unchanged after swapping a raw pread() call for this one. Never writes
// more than `len` bytes to `dst`, regardless of alignment or mode -- see
// this file's header comment for why that's a hard requirement here,
// not just a nice-to-have.
//
// is_direct == false: plain pass-through, byte-for-byte the same as the
// pread() call this replaces -- the default, zero-risk path for anyone
// who hasn't opted into direct_io.
//
// is_direct == true: reads the alignment-padded window that covers
// [offset, offset+len) into a thread_local aligned scratch buffer, then
// memcpy's out only the real [offset, offset+len) sub-range into `dst`.
inline ssize_t pread_direct_aware(int fd, bool is_direct, uint64_t alignment,
                                   void* dst, size_t len, uint64_t offset) {
    if (!is_direct || len == 0) {
        return pread(fd, dst, len, (off_t)offset);
    }

    AlignedWindow win = compute_aligned_window(offset, len, alignment);

    thread_local AlignedScratch scratch;
    scratch.ensure(win.aligned_len, alignment);
    if (!scratch.buf) return -1;  // allocation failure -- caller's short-read check will flag this

    ssize_t r = pread(fd, scratch.buf, win.aligned_len, (off_t)win.aligned_offset);
    if (r < 0) return r;

    if ((uint64_t)r < win.front_slack + len) {
        // Short of even the REAL bytes needed -- a genuine short/failed
        // read, not just the harmless "rounded up past true EOF" case
        // (which is expected and fine: aligned_len can run past a
        // shard's true end for whichever tensor happens to be last, and
        // the kernel simply returns fewer bytes there, same as a normal
        // pread() at EOF would). Report however much of the REAL range
        // we actually got -- the same short-read signal a plain pread()
        // would give, so existing `r != len` checks still work.
        uint64_t have = (uint64_t)r > win.front_slack ? (uint64_t)r - win.front_slack : 0;
        if (have > 0) memcpy(dst, scratch.buf + win.front_slack, have);
        return (ssize_t)have;
    }
    memcpy(dst, scratch.buf + win.front_slack, len);
    return (ssize_t)len;
}

} // namespace direct_io
