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

inline void zeromoe_print_host_cpu() {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
  // Runtime CPUID, so `--version` on the target machine proves the binary can
  // actually execute here. Configure-time detection (CMakeLists.txt) already
  // refuses to build for a host that lacks the baseline; this covers binaries
  // copied between machines and stale build directories.
  const bool baseline = __builtin_cpu_supports("avx512f") &&
                        __builtin_cpu_supports("avx512cd") &&
                        __builtin_cpu_supports("avx512bw") &&
                        __builtin_cpu_supports("avx512dq") &&
                        __builtin_cpu_supports("avx512vl");
  const bool vnni = __builtin_cpu_supports("avx512vnni");
  if (baseline)
    std::printf("  cpu:    x86-64 AVX-512 F/CD/BW/DQ/VL%s\n", vnni ? " + VNNI" : " (no VNNI)");
  else
    std::printf("  cpu:    x86-64 WITHOUT the required AVX-512 baseline; this binary cannot run here\n");
#elif defined(__aarch64__) || defined(__arm64__)
  std::printf("  cpu:    arm64 NEON\n");
#endif
}

inline void zeromoe_print_version() {
#if defined(ZEROMOE_ENGINE_CUDA)
  constexpr const char *kEngine = "CUDA";
#else
  constexpr const char *kEngine = "CPU-only";
#endif
  std::printf("ZeroMoE %s\n", ZEROMOE_VERSION);
  std::printf("  commit: %s\n", ZEROMOE_GIT_COMMIT);
  std::printf("  build:  %s, %s engine\n", ZEROMOE_BUILD_TYPE, kEngine);
  zeromoe_print_host_cpu();
  std::printf("  docs:   https://github.com/ray-x/ZeroMoE\n");
}
