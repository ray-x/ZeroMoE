#!/usr/bin/env bash
# Configure and build ZeroMoE against a ZeroMoE-patched ggml. Recipe and rules: docs/build-ggml.md.
#
#   scripts/build_with_ggml.sh <build-dir> [cmake build args...]
#
# Environment:
#   GGML_ROOT      ggml source tree (headers; patched in place when GGML_PREBUILT is empty)
#                  default: <main-checkout>/build_ggmlcuda/ggml-src/ggml
#   GGML_PREBUILT  directory holding the already built libggml*.so (<dir>/src/lib*.so,
#                  <dir>/src/ggml-cuda/libggml-cuda.so). Default: the master-level build kept in the
#                  main checkout. Set to "" to compile ggml from GGML_ROOT and apply patches/ggml
#                  (cmake/ZeroMoEGgmlPatches.cmake) into it.
#   CUDA_ARCH      optional explicit architecture list (e.g. '86;120a'); defaults to
#                  all GPUs visible to nvidia-smi at build time.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
BUILD=${1:?usage: build_with_ggml.sh <build-dir> [cmake --build args]}
shift || true
MAIN=$(git rev-parse --path-format=absolute --git-common-dir | sed 's|/\.git$||')
GGML_ROOT=${GGML_ROOT:-$MAIN/build_ggmlcuda/ggml-src/ggml}
GGML_PREBUILT=${GGML_PREBUILT-$MAIN/build_ggmlcuda/ggml-build}
CUDA_ARCH=$(CUDA_ARCH="${CUDA_ARCH:-}" "$ROOT/scripts/detect_cuda_arch.sh")

if [[ -n $GGML_PREBUILT ]]; then
  [[ -f $GGML_PREBUILT/src/ggml-cuda/libggml-cuda.so ]] || { echo "no ggml build at $GGML_PREBUILT"; exit 1; }
  # The binary records this directory as its RUNPATH. One that lives inside another worktree
  # disappears when that worktree is cleaned up and the loader silently falls back to the system
  # libggml (sm_86 only, no PTX), which fails on the sm_120 GPU.
  case $GGML_PREBUILT in
    "$MAIN"/.worktrees/*) [[ $GGML_PREBUILT == "$ROOT"/* ]] || { echo "GGML_PREBUILT is inside another worktree: $GGML_PREBUILT"; exit 1; } ;;
  esac
fi

cmake -S . -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DZEROMOE_USE_GGML=ON -DZEROMOE_USE_GGML_CUDA=ON \
  -DZEROMOE_ENABLE_CUDA_GRAPHS=ON -DZEROMOE_ENABLE_AVX512_VNNI=ON \
  -DZEROMOE_ENABLE_GGML_HOST_NODE_PATCHES=ON \
  -DZEROMOE_GGML_ROOT="$GGML_ROOT" -DZEROMOE_GGML_PREBUILT_DIR="$GGML_PREBUILT"
cmake --build "$BUILD" -j"$(nproc)" --target zeromoe "$@"

echo "built $BUILD/zeromoe"
echo "RUNPATH: $(readelf -d "$BUILD/zeromoe" | grep -i runpath)"
ldd "$BUILD/zeromoe" | grep ggml-cuda
