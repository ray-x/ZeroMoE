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
SUPPORT_SRCS="backends/exl3/exl3_cpu.cpp backends/exl3/exl3_model_loader.cpp models/base/safetensors_adapter.cpp"

if [ ! -f "$SRC" ]; then
    echo "FATAL: $SRC not found (run from the ZeroMoE repo root)" >&2
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
    KERNEL_SRC="backends/cpu/kernels_cpu_neon.cpp"
    # kernels_cpu_neon.cpp calls into zeromoe_metal:: under #ifdef __APPLE__
    # (backends/cpu/kernels_metal.mm, an Objective-C++ TU using MPSMatrixMultiplication)
    # -- link it in and pull the frameworks it needs, or every test binary on
    # Apple Silicon fails at link time with undefined zeromoe_metal symbols.
    EXTRA_SRCS="backends/cpu/kernels_metal.mm"
    EXTRA_LDFLAGS="-framework Metal -framework MetalPerformanceShaders -framework Foundation"
else
    KERNEL_SRC="backends/cpu/kernels_cpu.cpp"
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
    # Flag set mirrors CMakeLists.txt's ZEROMOE_AVX512_FLAGS exactly,
    # including its deliberate omission of -mavx512vnni (the source
    # compile-time-gates its own VNNI path with #if defined(__AVX512VNNI__);
    # forcing the flag risks SIGILL on an AVX-512 CPU without VNNI). Probed
    # rather than assumed, same as CMake's check_cxx_compiler_flag, so a
    # non-AVX-512 x86 host still gets the old behaviour (and the same
    # clearer compiler error) instead of an unknown-flag failure.
    # kernels_cpu.cpp also pulls two link-time dependencies the NEON file
    # does not: the IQ2/IQ3 grid tables (backends/cpu/ggml_grids.cpp -- ms_grid_*) and
    # ggml's own quantize_row_q8_K_ref (used by the ggml-port AVX2 dot
    # prototypes). libggml is only present once the CUDA build has been
    # configured, so link it opportunistically and say so if it is missing,
    # rather than failing with a wall of undefined references.
    EXTRA_SRCS="backends/cpu/ggml_grids.cpp"
    # ggml_grids.cpp includes ggml-common.h, which lives in the fetched
    # ggml source tree (not in include/), so add it to the search path.
    GGML_SRC_DIR="${GGML_SRC_DIR:-build/_deps/ggml-src/src}"
    [ -d "$GGML_SRC_DIR" ] && INC_FLAGS="-I$GGML_SRC_DIR -I$GGML_SRC_DIR/../include"
    GGML_LIB_DIR="${GGML_LIB_DIR:-build/ggml-build/src}"
    if [ -f "$GGML_LIB_DIR/libggml-base.so" ]; then
        GGML_LIB_DIR="$(cd "$GGML_LIB_DIR" && pwd)"
        EXTRA_LDFLAGS="-L$GGML_LIB_DIR -lggml-cpu -lggml-base -Wl,-rpath,$GGML_LIB_DIR"
        # kernels_cpu.cpp's direct GGML adapters (including the Q2_0 x Q8_0
        # fallback used by the CPU components build) are guarded by this
        # feature define. The standalone test command links GGML manually,
        # so mirror the CMake target's compile definition when GGML is present.
        GGML_FLAGS="-DZEROMOE_USE_GGML=1"
    else
        echo "[run_components] WARNING: $GGML_LIB_DIR/libggml-base.so not found" \
             "(run 'make build' first); backends/cpu/kernels_cpu.cpp will not link."
    fi
    X86_FLAGS="-mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mf16c"
    if "$CXX" -std=c++23 $X86_FLAGS -c -x c++ /dev/null -o /dev/null 2>/dev/null; then
        ARCH_FLAGS="$X86_FLAGS"
    else
        echo "[run_components] WARNING: compiler rejects AVX-512/F16C flags;" \
             "backends/cpu/kernels_cpu.cpp likely will not build (see CLAUDE.md)."
    fi
fi

GGML_FLAGS="${GGML_FLAGS:-}"

# OpenMP is optional.  If the compiler accepts -fopenmp we use it; otherwise
# the kernel sources compile single-threaded (the #pragma omp lines are
# ignored and <omp.h> is not included).
OMP_FLAGS=""
if "$CXX" -std=c++23 -fopenmp -c -x c++ /dev/null -o /dev/null 2>/dev/null; then
    OMP_FLAGS="-fopenmp"
fi

mkdir -p "$(dirname "$BIN")"
echo "[run_components] $CXX -std=c++23 -O2 -Iinclude $INC_FLAGS $GGML_FLAGS $ARCH_FLAGS $OMP_FLAGS $SRC $SUPPORT_SRCS $KERNEL_SRC $EXTRA_SRCS $EXTRA_LDFLAGS -> $BIN"
"$CXX" -std=c++23 -O2 -Iinclude $INC_FLAGS $GGML_FLAGS $ARCH_FLAGS $OMP_FLAGS -o "$BIN" "$SRC" $SUPPORT_SRCS "$KERNEL_SRC" $EXTRA_SRCS $EXTRA_LDFLAGS

ROUTE_TEST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/zeromoe-route-test.XXXXXX")"
trap 'rm -rf "$ROUTE_TEST_DIR"' EXIT
ROUTE_TEST_BIN="$ROUTE_TEST_DIR/test_moe_partial_route"
echo "[run_components] compiling allocation-free route classification tests"
"$CXX" -std=c++20 -O2 -Icore -Ibackends/ggml -o "$ROUTE_TEST_BIN" tests/test_moe_partial_route.cpp
for MODE in capable stock; do
    echo "[run_components] route classification mode: $MODE"
    "$ROUTE_TEST_BIN" "$MODE"
done

# Dynamic DDR expert cache (residency bitmap, LRU, seeding, aligned O_DIRECT store reader, virtual arena) and its
# acceptance replay of the GLM-5.3-Flash router trace against the offline simulator's results. Linux only (O_DIRECT,
# preadv, mincore). Optional real-data check: ZEROMOE_TEST_STORE=<store.gguf> ZEROMOE_TEST_MODEL=<shard,...>.
if [ "$(uname -s)" = "Linux" ]; then
    CACHE_TEST_BIN="$ROUTE_TEST_DIR/test_expert_dram_cache"
    echo "[run_components] compiling dynamic expert cache tests"
    "$CXX" -std=c++20 -O2 -pthread -Iinclude \
        -o "$CACHE_TEST_BIN" tests/test_expert_dram_cache.cpp
    "$CACHE_TEST_BIN"
else
    echo "[run_components] skipping dynamic expert cache tests (Linux only)"
fi

# The shared shape-class registry itself is CPU-only, but its public header
# uses CUDA and GGML types. Run it where both SDK headers are present; retain
# the pure C++ storage/policy tests on CPU-only hosts.
CUDA_INCLUDE_DIR=""
for candidate in "${CUDA_HOME:-}/include" /usr/local/cuda/include \
                 /opt/cuda/include /opt/cuda/targets/x86_64-linux/include; do
    if [ -f "$candidate/cuda_runtime_api.h" ]; then
        CUDA_INCLUDE_DIR="$candidate"
        break
    fi
done
if [ -n "${GGML_SRC_DIR:-}" ] && \
   [ -f "$GGML_SRC_DIR/../include/ggml-backend.h" ] && \
   [ -n "$CUDA_INCLUDE_DIR" ]; then
    SHAPE_TEST_BIN="$ROUTE_TEST_DIR/test_qwen4exp_stage_shape_pool"
    echo "[run_components] compiling shared shape-class pool tests"
    "$CXX" -std=c++23 -O2 -pthread -Iinclude \
        -I"$GGML_SRC_DIR/../include" -I"$GGML_SRC_DIR" \
        -I"$CUDA_INCLUDE_DIR" \
        -o "$SHAPE_TEST_BIN" tests/test_qwen4exp_stage_shape_pool.cpp
    "$SHAPE_TEST_BIN"
else
    echo "[run_components] skipping shape-class pool test (CUDA/GGML headers unavailable)"
fi

echo "[run_components] running: $BIN $*"
"$BIN" "$@"
