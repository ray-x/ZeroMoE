// Version and build provenance reported by `zeromoe --version`.
//
// The values are injected by CMake at configure time. A source archive built
// without .git still compiles: the commit simply reports "unknown".
#pragma once

#include <cstdio>

#ifndef ZEROMOE_VERSION
#define ZEROMOE_VERSION "dev"
#endif
#ifndef ZEROMOE_GIT_COMMIT
#define ZEROMOE_GIT_COMMIT "unknown"
#endif
#ifndef ZEROMOE_BUILD_TYPE
#define ZEROMOE_BUILD_TYPE "unknown"
#endif

inline void zeromoe_print_version() {
#if defined(ZEROMOE_ENGINE_CUDA)
  constexpr const char *kEngine = "CUDA";
#else
  constexpr const char *kEngine = "CPU-only";
#endif
  std::printf("ZeroMoE %s\n", ZEROMOE_VERSION);
  std::printf("  commit: %s\n", ZEROMOE_GIT_COMMIT);
  std::printf("  build:  %s, %s engine\n", ZEROMOE_BUILD_TYPE, kEngine);
  std::printf("  docs:   https://github.com/ray-x/ZeroMoE\n");
}
