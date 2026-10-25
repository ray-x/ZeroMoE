#pragma once
// NVIDIA cuFile (GPUDirect Storage) backend: DMAs expert bytes straight from
// an O_DIRECT-opened NVMe file descriptor into a VRAM cache slot, bypassing
// the host-RAM hop entirely -- unlike this project's existing disk-read path
// (direct_io.hpp -> host buffer -> GpuTopology::upload_expert's cudaMemcpy
// H2D), there is no intermediate pageable/pinned host allocation at all for
// bytes that go through this path.
//
// Scoped narrowly and deliberately, per CLAUDE.md's `cuda_cufile` entry:
// SYNCHRONOUS cuFileRead only (no async/batch cuFile API -- that has its own
// completion/ordering model, not a cudaStream_t, and is not worth the added
// risk in a first cut on hardware this couldn't be verified against). Only
// ever called for a genuine "double miss" (not VRAM-resident AND not
// RAM-resident) -- see RAMSafetyManager::is_ram_resident and
// GpuTopology::upload_expert_cufile's own callers -- never a replacement for
// the RAM-mediated path when a RAM copy already exists.
//
// Zero-dependency by DEFAULT: this header is compiled two ways.
//   - ZEROMOE_HAVE_CUFILE defined (CMakeLists.txt found cufile.h/libcufile
//     at configure time -- see that file's own comment on why this can never
//     be a REQUIRED find_package): real implementation below, wraps the
//     documented cuFile API (cuFileDriverOpen/Close, cuFileHandleRegister/
//     Deregister, cuFileBufRegister/Deregister, cuFileRead).
//   - Undefined (the common case -- GDS is a separate install from the base
//     CUDA Toolkit, not guaranteed present): a stub namespace where every
//     function is a no-op returning failure/false, so call sites
//     (gguf_loader.hpp, gpu_topology.hpp, execution_layer.cpp) never need
//     their own #ifdef -- identical to io_uring_backend.hpp's Linux/non-Linux
//     stub split, just gated on library presence instead of platform.
//
// UNTESTED on real hardware from this development environment (macOS, no
// CUDA, no cufile.h at all -- this file could not even be lint-checked here,
// let alone compiled with ZEROMOE_HAVE_CUFILE defined). The function
// signatures/struct shapes below follow NVIDIA's documented, stable public
// cuFile API (GPUDirect Storage API Reference Guide) as accurately as
// available documentation allows, but treat this exactly the way this
// project treated direct_io.hpp/gguf_tool pack/io_uring_backend.hpp
// originally: verify it for real on a Linux box with GDS + nvidia-fs
// actually installed (cuda_cufile=true in config, watch for [CUFILE] log
// lines) before trusting it in production. On ANY failure at ANY stage --
// driver open, handle/buffer registration, or a single read -- this always
// falls back to the existing RAM-mediated path, never a hard failure.
#include <cstdint>
#include <cstddef>
#include <sys/types.h>

#ifdef ZEROMOE_HAVE_CUFILE
#include <cufile.h>
#endif

namespace cufile_io {

#ifdef ZEROMOE_HAVE_CUFILE

// Non-owning wrapper around a registered cuFile handle for one open (O_DIRECT)
// file descriptor. `valid` mirrors every other optional-backend's own
// "did this actually work" flag (Ring::available(), etc.) -- callers must
// check it before passing a handle to read().
struct FileHandle {
    CUfileHandle_t handle{};
    bool valid = false;
};

// Process-wide driver open/close -- call once at startup/shutdown (main.cpp),
// mirroring the single decode-thread io_uring Ring's own init() discipline.
// Never throws; returns false on any failure (old driver, nvidia-fs module
// not loaded, GDS not installed despite the library being link-time present)
// so callers always have a safe fallback.
inline bool driver_open() {
    CUfileError_t err = cuFileDriverOpen();
    return err.err == CU_FILE_SUCCESS;
}
inline void driver_close() {
    cuFileDriverClose();
}

// Registers one already-open file descriptor (MUST be O_DIRECT -- see
// GGUFFile::enable_cufile's own comment on why cuda_cufile requires
// direct_io) with the cuFile driver. Returns an invalid (valid=false) handle
// on any failure -- callers treat this exactly like a failed Ring::init(),
// falling back to the RAM-mediated path for that file/shard.
inline FileHandle register_file(int fd) {
    FileHandle h;
    CUfileDescr_t descr{};
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileError_t err = cuFileHandleRegister(&h.handle, &descr);
    h.valid = (err.err == CU_FILE_SUCCESS);
    return h;
}
inline void deregister_file(FileHandle& h) {
    if (h.valid) {
        cuFileHandleDeregister(h.handle);
        h.valid = false;
    }
}

// Registers a VRAM range (e.g. GpuTopology's whole expert_cache_pool, or
// active_layer_pool -- ONE call per cudaMalloc'd region, not per-expert-slot;
// see gpu_topology.hpp's own comments at each cudaMalloc site) so cuFileRead
// can DMA into any offset inside it. Returns false on failure (unsupported
// GPU/driver combination, VRAM not registerable for whatever reason) --
// caller treats the whole cuFile path as unavailable for that pool, same
// fallback-never-fails discipline as everything else here.
inline bool register_buffer(void* device_ptr, size_t bytes) {
    CUfileError_t err = cuFileBufRegister(device_ptr, bytes, 0);
    return err.err == CU_FILE_SUCCESS;
}
inline void deregister_buffer(void* device_ptr) {
    cuFileBufDeregister(device_ptr);
}

// Synchronous direct-to-VRAM read: `len` bytes from `h`'s file at
// `file_offset`, written to `device_ptr + device_offset` (device_ptr must lie
// inside a range previously passed to register_buffer -- typically
// GpuTopology's expert_cache_pool base, with device_offset selecting the
// destination slot + sub-projection, mirroring populate_packed_slab's
// gate@0/up@gb/down@gb+ub host-side layout exactly, just on the VRAM side).
// Returns bytes actually read (>=0) or a negative value on error, same
// pread()-shaped contract as direct_io::pread_direct_aware and
// io_uring_backend::Ring::submit_and_wait's own results[i].
inline ssize_t read(const FileHandle& h, void* device_ptr, size_t len,
                     off_t file_offset, off_t device_offset = 0) {
    if (!h.valid) return -1;
    return cuFileRead(h.handle, device_ptr, len, file_offset, device_offset);
}

inline bool available() { return true; }

#else // !ZEROMOE_HAVE_CUFILE

// Stub: every function fails/no-ops, so callers never need their own
// #ifdef ZEROMOE_HAVE_CUFILE -- identical in spirit to
// io_uring_backend::Ring's non-Linux stub.
struct FileHandle {
    bool valid = false;
};
inline bool driver_open() { return false; }
inline void driver_close() {}
inline FileHandle register_file(int) { return FileHandle{}; }
inline void deregister_file(FileHandle&) {}
inline bool register_buffer(void*, size_t) { return false; }
inline void deregister_buffer(void*) {}
inline ssize_t read(const FileHandle&, void*, size_t, off_t, off_t = 0) { return -1; }
inline bool available() { return false; }

#endif // ZEROMOE_HAVE_CUFILE

} // namespace cufile_io
