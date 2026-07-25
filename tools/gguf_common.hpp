// tools/gguf_common.hpp
//
// Shared declarations + tiny inline helpers for the unified gguf_tool binary
// (pack / io / verify subcommands). Kept deliberately small: each subcommand
// lives in its own .cpp (gguf_pack.cpp / gguf_io.cpp / gguf_verify.cpp) and
// this header only carries what's genuinely shared across them — the common
// includes and the cmd_* entry points that gguf_tool.cpp's main() dispatches
// to. Subcommand-specific helpers (the raw KV reader, the bench loops, the
// completeness checker) stay file-local in their own .cpp.
//
// Build (no CMake / no CUDA -- runs on macOS AND Linux, same as the old
// standalone tools):
//   c++ -std=c++23 -O2 -Iinclude tools/gguf_tool.cpp tools/gguf_pack.cpp \
//       tools/gguf_io.cpp tools/gguf_verify.cpp -o build/gguf_tool
#pragma once

#include "gguf_loader.hpp"
#include "direct_io.hpp"
#include <cstdint>
#include <string>

namespace gguf_tool {

// Entry points implemented in the per-subcommand .cpp files. Each takes
// (argc, argv) where argv[0] is the SUBCOMMAND NAME (e.g. "pack"), so the
// subcommand parses its real arguments from argv[1..] exactly like a normal
// main() -- gguf_tool.cpp's dispatch forwards (argc-1, argv+1) accordingly.
// Returns the process exit code (0 = success, 1 = failure, 2 = usage error),
// matching the project's existing convention.
int cmd_pack(int argc, char **argv);
int cmd_io(int argc, char **argv);
int cmd_verify(int argc, char **argv);

// ---- inline helpers shared by more than one subcommand ----

// "blk.<L>.<rest>" -> L. Returns false if `name` doesn't start with "blk."
// followed by digits and a '.'. Used by io (layer discovery) and verify
// (layer scan). Kept here rather than duplicated in both .cpp files.
inline bool parse_blk_layer(const std::string &name, int &L) {
  if (name.rfind("blk.", 0) != 0)
    return false;
  size_t p = 4;
  int v = 0;
  bool any = false;
  while (p < name.size() && name[p] >= '0' && name[p] <= '9') {
    v = v * 10 + (name[p] - '0');
    any = true;
    ++p;
  }
  if (!any || p >= name.size() || name[p] != '.')
    return false;
  L = v;
  return true;
}

// Round `v` up to a multiple of `alignment` (alignment must be a power of two).
// pack uses this for tensor-offset layout; shared here so verify/io can't drift
// from pack's definition of alignment.
inline uint64_t align_up(uint64_t v, uint64_t alignment) {
  return (v + alignment - 1) & ~(alignment - 1);
}

} // namespace gguf_tool
