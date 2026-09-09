#!/usr/bin/env bash
# Build + run gguf_tool (tools/gguf_tool.cpp + gguf_pack/io/verify/plan/dma.cpp)
# with the system C++ compiler -- NOT through CMake/CUDA, so it runs on macOS
# as well as Linux. Single binary, subcommands: pack, io, verify, plan, dma.
#
# Usage:
#   ./tools/run_gguf_tool.sh <subcommand> [args...]
#   ./tools/run_gguf_tool.sh io <orig_packed.gguf|dir> <repacked.gguf|dir> [--reads N] [--seed S] [--direct]
#   ./tools/run_gguf_tool.sh verify <model_file_or_dir> [--against <other_model>]
#   ./tools/run_gguf_tool.sh pack <source_shard_file_or_dir> <output_dir> [options]
#   ./tools/run_gguf_tool.sh dma <model.gguf|dir> [<model2> ...] [--experts N[,N2,...]] [--submissions M]
#   CXX=g++ BIN=./build/gguf_tool ./tools/run_gguf_tool.sh io model.gguf ./repacked --reads 5000
#
# io's cache bypass is handled inside the tool (F_NOCACHE on macOS,
# posix_fadvise(FADV_RANDOM) on Linux). Use a LARGE real model so reads hit
# disk -- on a small fixture everything is cached and the A/B ratio is noise.
# dma requires O_DIRECT + a real io_uring ring (Linux only) -- it exits early
# with a clear message on macOS rather than reporting a meaningless result.
set -euo pipefail

cd "$(dirname "$0")/.."

CXX="${CXX:-c++}"
BIN="${BIN:-./build/gguf_tool}"
SRCS="tools/gguf_tool.cpp tools/gguf_pack.cpp tools/gguf_io.cpp tools/gguf_verify.cpp tools/gguf_plan.cpp tools/gguf_dma.cpp"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <subcommand> [args...]  (subcommands: pack, io, verify, plan, dma)" >&2
    echo "  (builds $SRCS with $CXX, then runs it with the given args)" >&2
    exit 2
fi

mkdir -p "$(dirname "$BIN")"
echo "[run_gguf_tool] $CXX -std=c++23 -O2 -Iinclude $SRCS -> $BIN"
"$CXX" -std=c++23 -O2 -Iinclude -o "$BIN" $SRCS

echo "[run_gguf_tool] running: $BIN $*"
exec "$BIN" "$@"
