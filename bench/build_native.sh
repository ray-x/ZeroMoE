#!/usr/bin/env bash
# Build the engine with ggml-cuda compiled to native SASS for this box's GPUs (RTX 3090 = sm_86, RTX 5060 Ti = sm_120) instead of
# JIT-compiling compute_75 PTX at run time. Measured +0.6-1.4% decode (benchmark 15.45 -> 15.67, held-out dockerfile 15.79 -> 15.89).
#   bench/build_native.sh [ggml-root] [build-dir]
# ggml-root must already carry the ZeroMoE host-node edits (the integration tree's working copy, or a private copy of it).
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
ggml=${1:-${GGML_ROOT:-$root/../llama.cpp/ggml}}
build=${2:-$root/build_native}
cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86;120" \
  -DZEROMOE_GGML_ROOT="$ggml" -DZEROMOE_GGML_PREBUILT_DIR= -DZEROMOE_USE_GGML=ON -DZEROMOE_USE_GGML_CUDA=ON \
  -DZEROMOE_ENABLE_GGML_HOST_NODE_PATCHES=ON -DZEROMOE_ENABLE_CUDA_GRAPHS=ON \
  -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_CUDA_GRAPHS=ON
cmake --build "$build" --target zeromoe -j"$(nproc)"
echo "built: $build/zeromoe"
