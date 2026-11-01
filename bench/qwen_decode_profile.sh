#!/usr/bin/env bash
# One Qwen3.8-Flash-Next (Q2_0, all-split early-ready) free-running decode, 3090 only.
#   bench/qwen_decode_profile.sh <engine> <tag> [VAR=val ...]
# Output: logs/perf/<tag>.{engine.log,stdout,tokens}; prints the engine's decode rate.
set -uo pipefail
bin=$(readlink -f "${1:?engine}"); tag=${2:?tag}; shift 2
root=$(cd "$(dirname "$0")/.." && pwd); out=$root/logs/perf; mkdir -p "$out"
gpu_uuid=${QWEN_GPU_UUID:-GPU-fbddd113-32a1-56af-9c8e-7f2c877a9736}
toml_src=${QWEN_TOML:-$root/logs/graph_framework/merged-step2/correctness/early_1.toml}
prompt_ids=${QWEN_PROMPT_IDS:-$root/logs/graph_framework/inputs/prompt.tokens}
if [ ! -s "$toml_src" ]; then
  printf 'ERROR: QWEN_TOML is missing or empty: %s\n' "$toml_src" >&2
  printf 'Set QWEN_TOML to the exact measured Qwen profile before running.\n' >&2
  exit 2
fi
load1=$(awk '{print $1}' /proc/loadavg)
if awk -v avg="$load1" 'BEGIN { exit !(avg > 2.0) }'; then
  printf 'ERROR: host load average is %s (> 2.0); wait for an uncontended profile slot.\n' "$load1" >&2
  exit 75
fi
/usr/bin/nvidia-smi -i "$gpu_uuid" > "$out/$tag.host-before.txt" 2>&1
uptime >> "$out/$tag.host-before.txt"
if [ ! -s "$prompt_ids" ]; then
  printf 'ERROR: QWEN_PROMPT_IDS is missing or empty: %s\n' "$prompt_ids" >&2
  printf 'Set QWEN_PROMPT_IDS to the captured benchmark prompt token IDs before running.\n' >&2
  exit 2
fi
python3 - "$toml_src" <<'PYCFG'
import sys
import tomllib
from pathlib import Path
try:
    config = tomllib.loads(Path(sys.argv[1]).read_text())
except Exception as error:
    print(f"ERROR: cannot parse QWEN_TOML {sys.argv[1]}: {error}", file=sys.stderr)
    raise SystemExit(2)
model_path = config.get("model_path")
if not model_path or not Path(model_path).expanduser().exists():
    print(f"ERROR: QWEN_TOML model_path is missing: {model_path!r}", file=sys.stderr)
    raise SystemExit(2)
PYCFG
if [ $? -ne 0 ]; then exit 2; fi
while [ -n "$(nvidia-smi -i "$gpu_uuid" --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done
sed "s#^log_path *=.*#log_path = \"$out/$tag.engine.log\"#" "$toml_src" > "$out/$tag.toml"
if [ ! -s "$out/$tag.toml" ]; then
  printf 'ERROR: failed to generate non-empty benchmark TOML: %s\n' "$out/$tag.toml" >&2
  exit 2
fi
printf 'Explain briefly the key concepts of neural networks and gradient descent in plain English.\n:q\n' | \
  env CUDA_VISIBLE_DEVICES="$gpu_uuid" GGML_CUDA_BF16_MMVF_CUBLAS=0 \
  ZEROMOE_QWEN_CPU_THREADS=10 ZEROMOE_BENCH_IGNORE_STOP=1 \
  ZEROMOE_PROMPT_TOKEN_INPUT_FILE="$prompt_ids" ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT=1 \
  ZEROMOE_QWEN_EARLY_READY_LAYER13=0 ZEROMOE_GENERATED_TOKEN_FILE="$out/$tag.tokens" "$@" \
  "$bin" --verbosity 5 --graph-device-expert-layers 0 --graph-device-expert-trailing 1 --config "$out/$tag.toml" --predict "${NGEN:-64}" > "$out/$tag.stdout" 2>&1
/usr/bin/nvidia-smi -i "$gpu_uuid" > "$out/$tag.host-after.txt" 2>&1
uptime >> "$out/$tag.host-after.txt"
echo "exit=$?"; grep -h 'Rates:' "$out/$tag.engine.log" | tail -1 | sed -E 's/.*(decode=[0-9.]+ tok\/s).*/\1/'
