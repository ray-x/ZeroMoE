#pragma once

// POSIX_FADV_* constants and posix_fadvise() are Linux extensions. macOS
// needs both the values and a no-op so the shard reader compiles there.
//
// This used to live as two copies guarded by a shared #ifndef (engine.hpp and
// gguf_loader.hpp). Keeping one guarded header removes the convention that the
// copies have to stay textually identical, and the duplicate definition that
// broke the build the first time both headers landed in one translation unit.
#include <fcntl.h>
#include <sys/types.h>

#ifdef __APPLE__
#ifndef POSIX_FADV_WILLNEED
#define POSIX_FADV_WILLNEED 3
#define POSIX_FADV_DONTNEED 4
#define POSIX_FADV_RANDOM 1
static inline int posix_fadvise(int, off_t, off_t, int) { return 0; }
#endif
#endif
