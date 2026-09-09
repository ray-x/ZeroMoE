#!/usr/bin/env bash
# Build + run the CPU-only component tests (tests/test_components.cpp) with the
# system C++ compiler -- deliberately NOT going through CMake/CUDA, so it runs
# on macOS as well as Linux. This is the "test a SUBSET on Mac" half of the
# workflow; the full engine suite (CUDA + AVX-512, main_test.cpp's [Test 1-10])
# is `make test` on Linux only.
#
# Usage:
#   ./tests/run_components.sh                # build + run ALL component tests
#   ./tests/run_components.sh rope           # only tests whose name contains 'rope'
#   ./tests/run_components.sh rope dequant   # rope OR any dequant test
#   CXX=g++ BIN=/tmp/tc ./tests/run_components.sh spsc   # override compiler/binary
#
# Recompiles every run (the TU is small and CPU-only, so this is seconds).
set -euo pipefail

# cd to repo root (one level up from this script's dir).
cd "$(dirname "$0")/.."

CXX="${CXX:-c++}"                 # c++ -> clang++ on macOS, g++ on most Linux
BIN="${BIN:-./build/test_components}"
SRC="tests/test_components.cpp"
SUPPORT_SRC="src/safetensors_adapter.cpp"

if [ ! -f "$SRC" ]; then
    echo "FATAL: $SRC not found (run from the MoEStream repo root)" >&2
    exit 2
fi

# Pick the CPU kernel backend matching this host.  On Apple Silicon we use the
# NEON implementation; everywhere else we fall back to the AVX-512 file (which
# also compiles with AVX2/scalar tails on older x86).
ARCH=$(uname -m)
EXTRA_SRCS=""
EXTRA_LDFLAGS=""
ARCH_FLAGS=""
INC_FLAGS=""
if [ "$ARCH" = "arm64" ] && [ "$(uname -s)" = "Darwin" ]; then
    KERNEL_SRC="src/kernels_cpu_neon.cpp"
    # kernels_cpu_neon.cpp calls into moestream_metal:: under #ifdef __APPLE__
    # (src/kernels_metal.mm, an Objective-C++ TU using MPSMatrixMultiplication)
    # -- link it in and pull the frameworks it needs, or every test binary on
    # Apple Silicon fails at link time with undefined moestream_metal symbols.
    EXTRA_SRCS="src/kernels_metal.mm"
    EXTRA_LDFLAGS="-framework Metal -framework MetalPerformanceShaders -framework Foundation"
else
    KERNEL_SRC="src/kernels_cpu.cpp"
    # kernels_cpu.cpp is the AVX-512 backend and uses F16C intrinsics
    # (_cvtsh_ss) unconditionally, so it does NOT build with plain -O2 on
    # x86: gcc rejects it with "inlining failed in call to always_inline
    # '_cvtsh_ss': target specific option mismatch". This target went
    # unnoticed because its documented workflow is the macOS one (CLAUDE.md:
    # "test a SUBSET here, run the full `make test` CUDA suite on Linux"),
    # and on Apple Silicon the branch above compiles kernels_cpu_neon.cpp
    # instead -- so `make test-components` had simply never been exercised
    # on an x86 Linux host. Found 2026-09-04 while trying to run the
    # qwen4exp CPU tests (GDN delta step / PLE hash / layer detection) on
    # the Linux box.
    #
    # Flag set mirrors CMakeLists.txt's MOESTREAM_AVX512_FLAGS exactly,
    # including its deliberate omission of -mavx512vnni (the source
    # compile-time-gates its own VNNI path with #if defined(__AVX512VNNI__);
    # forcing the flag risks SIGILL on an AVX-512 CPU without VNNI). Probed
    # rather than assumed, same as CMake's check_cxx_compiler_flag, so a
    # non-AVX-512 x86 host still gets the old behaviour (and the same
    # clearer compiler error) instead of an unknown-flag failure.
    # kernels_cpu.cpp also pulls two link-time dependencies the NEON file
    # does not: the IQ2/IQ3 grid tables (src/ggml_grids.cpp -- ms_grid_*) and
    # ggml's own quantize_row_q8_K_ref (used by the ggml-port AVX2 dot
    # prototypes). libggml is only present once the CUDA build has been
    # configured, so link it opportunistically and say so if it is missing,
    # rather than failing with a wall of undefined references.
    EXTRA_SRCS="src/ggml_grids.cpp"
    # ggml_grids.cpp includes ggml-common.h, which lives in the fetched
    # ggml source tree (not in include/), so add it to the search path.
    GGML_SRC_DIR="build/_deps/ggml-src/src"
    [ -d "$GGML_SRC_DIR" ] && INC_FLAGS="-I$GGML_SRC_DIR -Ibuild/_deps/ggml-src/include"
    GGML_LIB_DIR="build/ggml-build/src"
    if [ -f "$GGML_LIB_DIR/libggml-base.so" ]; then
        EXTRA_LDFLAGS="-L$GGML_LIB_DIR -lggml-base -Wl,-rpath,$PWD/$GGML_LIB_DIR"
    else
        echo "[run_components] WARNING: $GGML_LIB_DIR/libggml-base.so not found" \
             "(run 'make build' first); src/kernels_cpu.cpp will not link."
    fi
    X86_FLAGS="-mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mf16c"
    if "$CXX" -std=c++23 $X86_FLAGS -c -x c++ /dev/null -o /dev/null 2>/dev/null; then
        ARCH_FLAGS="$X86_FLAGS"
    else
        echo "[run_components] WARNING: compiler rejects AVX-512/F16C flags;" \
             "src/kernels_cpu.cpp likely will not build (see CLAUDE.md)."
    fi
fi

# OpenMP is optional.  If the compiler accepts -fopenmp we use it; otherwise
# the kernel sources compile single-threaded (the #pragma omp lines are
# ignored and <omp.h> is not included).
OMP_FLAGS=""
if "$CXX" -std=c++23 -fopenmp -c -x c++ /dev/null -o /dev/null 2>/dev/null; then
    OMP_FLAGS="-fopenmp"
fi

mkdir -p "$(dirname "$BIN")"
echo "[run_components] $CXX -std=c++23 -O2 -Iinclude $INC_FLAGS $ARCH_FLAGS $OMP_FLAGS $SRC $SUPPORT_SRC $KERNEL_SRC $EXTRA_SRCS $EXTRA_LDFLAGS -> $BIN"
"$CXX" -std=c++23 -O2 -Iinclude $INC_FLAGS $ARCH_FLAGS $OMP_FLAGS -o "$BIN" "$SRC" "$SUPPORT_SRC" "$KERNEL_SRC" $EXTRA_SRCS $EXTRA_LDFLAGS

echo "[run_components] running: $BIN $*"
exec "$BIN" "$@"
