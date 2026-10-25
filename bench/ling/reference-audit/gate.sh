#!/usr/bin/env bash
# The Ling regression gate recipe (short decode, 32 tokens, tokens compared within one tree).
#   bench/ling/reference-audit/gate.sh <zeromoe-binary> <tag>
# Logs: logs/ling_<tag>_short.{log,tokens,routes}, logs/ling_<tag>_run.out. The metric is the
# [REQUEST-DECODE-OUTPUT] rate (baseline 10.07 tok/s, runs of 9.7-10.4 are normal). Take numbers from a warm run:
# the first run after a fresh ggml build JIT-compiles for the sm_120 GPU (about 4.5 tok/s).
set -uo pipefail
cd "$(dirname "$0")/../../.."
LING_BIN=$1 LING_DDR_GIB=72 LING_CAP_GIB=95 LING_STOP_GIB=20 LING_GPU0_CACHE_GIB=3.0 LING_GPU1_CACHE_GIB=12.5 \
LING_PINNED_VRAM_CAPACITY=3700 LING_OFFLOAD_PINNED=0.5 LING_GPU_ASSIST=3 LING_ASSIST2_SLOTS=1 LING_CPU_THREADS=10 \
LING_USAGE_FILE="$PWD/bench/ling/reference-audit/usage-hi-there-robust4.pin" LING_LOG="$PWD/logs/ling_$2_short.log" \
  timeout 900 bench/ling/reference-audit/run.sh > "logs/ling_$2_run.out" 2>&1
echo "$2: $(grep -a 'REQUEST-DECODE-OUTPUT' "logs/ling_$2_short.log" | grep -o 'rate=[0-9.]*')"
