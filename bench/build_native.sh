#!/usr/bin/env bash
# Build the engine with ggml-cuda compiled to native SASS for this box's GPUs (RTX 3090 = sm_86, RTX 5060 Ti = sm_120) instead of
# JIT-compiling compute_75 PTX at run time. Measured +0.6-1.4% decode (benchmark 15.45 -> 15.67, held-out dockerfile 15.79 -> 15.89).
#   bench/build_native.sh [ggml-root] [build-dir]
# ggml-root must already carry the MoEStream host-node edits (the integration tree's working copy, or a private copy of it).
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
ggml=${1:-$HOME/Projects/llama.cpp-moestream-integration/ggml}
build=${2:-$root/build_native}
cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86;120" \
  -DMOESTREAM_GGML_ROOT="$ggml" -DMOESTREAM_GGML_PREBUILT_DIR= -DMOESTREAM_USE_GGML=ON -DMOESTREAM_USE_GGML_CUDA=ON \
  -DMOESTREAM_ENABLE_GGML_HOST_NODE_PATCHES=ON -DMOESTREAM_ENABLE_CUDA_GRAPHS=ON \
  -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_CUDA_GRAPHS=ON
cmake --build "$build" --target moestream -j"$(nproc)"
echo "built: $build/moestream"
