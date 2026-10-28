#!/usr/bin/env bash
# setup.sh — ZeroMoE build helper
#
# Checks the toolchain, fetches the pinned ggml source (via CMake), applies the
# ggml patch series, and builds a Release binary.
#
#   ./setup.sh                  # Linux + CUDA (or macOS CPU-only)
#   ./setup.sh --cpu-only       # CPU-only build on Linux
#   ./setup.sh --build-dir DIR  # custom build directory
#   ./setup.sh --jobs N         # parallel build jobs
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./setup.sh [options]

  --cpu-only          Build CPU-only components (default on macOS)
  --build-dir DIR     Build directory (default: build, or build-macos on macOS)
  --jobs N            Parallel build jobs (default: all logical CPUs)
  -h, --help          Show this help

The CUDA engine fetches ggml from a pinned llama.cpp commit on first
configure. Model weights are not downloaded; point a profile at your own GGUF.
EOF
}

BUILD_DIR=""
CPU_ONLY=0
JOBS=""
UNAME_S="$(uname -s)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu-only) CPU_ONLY=1; shift ;;
        --build-dir) BUILD_DIR="${2:?--build-dir needs a path}"; shift 2 ;;
        --jobs) JOBS="${2:?--jobs needs a number}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$UNAME_S" == "Darwin" ]]; then
    CPU_ONLY=1
fi
if [[ -z "$BUILD_DIR" ]]; then
    if [[ "$UNAME_S" == "Darwin" ]]; then BUILD_DIR="build-macos"; else BUILD_DIR="build"; fi
fi
if [[ -z "$JOBS" ]]; then
    JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi

# Build relative to the repository, regardless of the caller's directory.
cd "$(dirname "$0")"

fail() { echo "  Error: $*" >&2; exit 1; }

echo "  ZeroMoE Engine — Setup and Compilation"
echo "----------------------------------------"
echo "[1/3] Checking build prerequisites..."

command -v cmake >/dev/null || fail "cmake is missing (need >= 3.25). Debian/Ubuntu: sudo apt install cmake, or pip install cmake."
command -v patch >/dev/null || fail "GNU patch is missing; the ggml patch series cannot be applied. Debian/Ubuntu: sudo apt install patch."
command -v python3 >/dev/null || fail "python3 is missing; build provenance and helper scripts need it. Debian/Ubuntu: sudo apt install python3."

CXX_BIN="${CXX:-c++}"
command -v "$CXX_BIN" >/dev/null || fail "C++ compiler '$CXX_BIN' is missing. Debian/Ubuntu: sudo apt install build-essential, or set CXX to your compiler."
command -v make >/dev/null || fail "make is missing. Debian/Ubuntu: sudo apt install build-essential."

cmake_version="$(cmake --version | head -n 1 | awk '{print $3}')"
awk -v v="$cmake_version" 'BEGIN {
    split(v, a, ".")
    if (a[1] < 3 || (a[1] == 3 && a[2] < 25)) exit 1
}' || fail "cmake $cmake_version is too old; version 3.25 or newer is required."

if [[ "$CPU_ONLY" -eq 0 ]]; then
    command -v nvcc >/dev/null || fail "nvcc is missing. Install the CUDA Toolkit (https://developer.nvidia.com/cuda-downloads) or use --cpu-only."
    command -v git >/dev/null || fail "git is missing. A git checkout is required for build provenance. Debian/Ubuntu: sudo apt install git."
    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        fail "CUDA builds require a git checkout (found no .git directory).
         GitHub ZIP archives cannot provide the pinned ggml source or build provenance.
         Clone the repository with git instead: git clone <repo-url>"
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "  Warning: nvidia-smi not found. Set -DCMAKE_CUDA_ARCHITECTURES explicitly."
    fi
    echo "  -> nvcc: $(nvcc --version | grep -o 'release [0-9.]*' | head -n 1)"
else
    if [[ ! -f third_party/llama.cpp/ggml/CMakeLists.txt ]]; then
        command -v git >/dev/null || fail "git is missing and the pinned ggml source is not present."
    fi
    echo "  -> CPU-only build (no CUDA required)"
fi

echo "  -> cmake: $cmake_version"
echo "  -> C++:   $("$CXX_BIN" --version | head -n 1)"
echo "  -> jobs:  $JOBS"
echo "  -> build: $BUILD_DIR"

echo "[2/3] Configuring..."
cmake_args=(-S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release)
if [[ "$CPU_ONLY" -eq 1 ]]; then
    cmake_args+=(-DZEROMOE_CPU_ONLY=ON)
fi
cmake "${cmake_args[@]}"

echo "[3/3] Compiling..."
cmake --build "$BUILD_DIR" --parallel "$JOBS"

if [[ "$CPU_ONLY" -eq 1 ]]; then
    BIN="$BUILD_DIR/zeromoe_cpu"
else
    BIN="$BUILD_DIR/zeromoe"
fi

echo "----------------------------------------"
echo "  ZeroMoE build complete."
if [[ ! -x "$BIN" ]]; then
    fail "the build finished but $BIN was not produced"
fi

# The build is only useful if the binary runs, and checking that must not need
# a model file. This also prints the commit the binary was built from.
echo "  -> Verifying the binary starts:"
"$BIN" --version | sed 's/^/     /'

# Pretty path for the next-steps block; --build-dir may be absolute.
BIN_DISPLAY="$BIN"
[[ "$BIN_DISPLAY" != /* ]] && BIN_DISPLAY="./$BIN_DISPLAY"

if [[ "$CPU_ONLY" -eq 1 ]]; then
    echo "  -> Component tests: ./$BUILD_DIR/test_components"
    echo "  -> GGUF tool:       ./$BUILD_DIR/gguf_tool"
    echo "  -> Note: zeromoe_cpu implements the Qwen families; the other"
    echo "     supported families need the CUDA engine."
else
    echo "  -> Test harness:    ./$BUILD_DIR/zeromoe_tests"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  -> GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
        | sed 's/^/     /' || true
fi

cat <<EOF

Next steps:
  1. Get a GGUF for a supported family (model weights are not bundled or
     downloaded by this script): docs/models/README.md
  2. Run a prompt with automatic memory sizing:
       $BIN_DISPLAY --model /path/to/model.gguf --prompt "Hello"
  3. For a repeatable setup, copy the starter profile and set model_path:
       cp configs/starter.toml my-model.toml
       $BIN_DISPLAY --config my-model.toml --prompt "Hello"
  4. Serve the OpenAI/Anthropic APIs:
       $BIN_DISPLAY --config my-model.toml --serve --port 8080
  5. Configuration: docs/configuration.md
     Validation:    docs/validation.md
EOF
