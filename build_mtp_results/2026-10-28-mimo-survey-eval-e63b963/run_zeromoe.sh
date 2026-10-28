#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ray/Projects/ZeroMoE/.worktrees/mimo-env-cleanup
RUN=/home/ray/Projects/ZeroMoE/.worktrees/mimo-env-cleanup/build_mtp_results/2026-10-28-mimo-survey-eval-e63b963
cd "$ROOT"
env CUDA_VISIBLE_DEVICES=0,1 ZEROMOE_DISABLE_ROUTER_BF16=1 \
  ZEROMOE_BENCH_IGNORE_STOP=1 ZEROMOE_BENCH_SILENT=1 \
  scripts/run_capped.sh --limit-gb 110 -- \
  build_mimo_baseline/zeromoe/zeromoe --config "$RUN/config.toml" \
  --graph-offload-ddr-gib 100 \
  --graph-offload-usage-file "$ROOT/bench/profiles/mimo26_combined.usage" \
  --predict 64 --temp 0 \
  < "$RUN/prompts.txt" > "$RUN/zeromoe.stdout.log" 2> "$RUN/zeromoe.engine.log"
