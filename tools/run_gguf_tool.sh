#!/usr/bin/env bash
# Build and run the standalone GGUF utility without CMake, CUDA, or the
# llama.cpp submodule. This is useful on macOS/Linux and while ggml is paused.
set -euo pipefail
cd "$(dirname "$0")/.."

CXX="${CXX:-c++}"
BIN="${BIN:-./build/gguf_tool}"
SOURCES=(
  tools/gguf_tool.cpp
  tools/gguf_pack.cpp
  tools/gguf_io.cpp
  tools/gguf_verify.cpp
  tools/gguf_plan.cpp
  tools/gguf_offload.cpp
  tools/gguf_dma.cpp
  tools/gguf_mix.cpp
  tools/gguf_mirror.cpp
)
INCLUDES=(
  -Iinclude
  -Icore
  -Imodels/base
  -Imemory
  -Imodels/graph
  -Ibackends/ggml
  -Ibackends/cpu
  -Icuda
)

if (($# == 0)); then
  echo "Usage: $0 <subcommand> [args...]" >&2
  exit 2
fi

mkdir -p "$(dirname "$BIN")"
"$CXX" -std=c++23 -O2 "${INCLUDES[@]}" -o "$BIN" "${SOURCES[@]}"
exec "$BIN" "$@"
