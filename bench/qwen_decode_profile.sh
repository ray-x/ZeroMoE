#!/usr/bin/env bash
# One Qwen3.8-Flash-Next (Q2_0, all-split early-ready) free-running decode, 3090 only.
#   bench/qwen_decode_profile.sh <engine> <tag> [VAR=val ...]
# Output: logs/perf/<tag>.{engine.log,stdout,tokens}; prints the engine's decode rate.
set -uo pipefail
bin=$(readlink -f "${1:?engine}"); tag=${2:?tag}; shift 2
root=$(cd "$(dirname "$0")/.." && pwd); out=$root/logs/perf; mkdir -p "$out"
toml_src=${QWEN_TOML:-$root/logs/graph_framework/merged-step2/correctness/early_1.toml}
prompt_ids=${QWEN_PROMPT_IDS:-$root/logs/graph_framework/inputs/prompt.tokens}
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done
sed "s#^log_path *=.*#log_path = \"$out/$tag.engine.log\"#" "$toml_src" > "$out/$tag.toml"
printf 'Explain briefly the key concepts of neural networks and gradient descent in plain English.\n:q\n' | \
  env CUDA_VISIBLE_DEVICES=0 GGML_CUDA_BF16_MMVF_CUBLAS=0 \
  MOESTREAM_QWEN4EXP_FULL_GRAPH=1 MOESTREAM_QWEN4EXP_FULL_GRAPH_DECODE=1 \
  MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_LAYERS=0 MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1 \
  MOESTREAM_QWEN_CPU_THREADS=10 MOESTREAM_BENCH_IGNORE_STOP=1 MOESTREAM_TOKEN_LOOP_DIAG=1 \
  MOESTREAM_PROMPT_TOKEN_INPUT_FILE="$prompt_ids" MOESTREAM_QWEN_EARLY_READY_ALL_SPLIT=1 \
  MOESTREAM_QWEN_EARLY_READY_LAYER13=0 MOESTREAM_GENERATED_TOKEN_FILE="$out/$tag.tokens" "$@" \
  "$bin" --config "$out/$tag.toml" --ngen "${NGEN:-64}" > "$out/$tag.stdout" 2>&1
echo "exit=$?"; grep -h 'Rates:' "$out/$tag.engine.log" | tail -1 | sed -E 's/.*(decode=[0-9.]+ tok\/s).*/\1/'
