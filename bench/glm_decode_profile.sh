#!/usr/bin/env bash
# One GLM-5.3-Flash SSD-offload decode run with the graph's own timers on.
#   bench/glm_decode_profile.sh <engine> <tag> [extra VAR=val ...]
# Requires GLM_MODEL and GLM_STORE; GLM_USAGE and GLM_TOML can override the tracked defaults.
# Output: logs/perf/<tag>.{engine.log,stdout}. Extra VAR=val pairs are exported to the run.
set -uo pipefail
bin=$(readlink -f "${1:?engine}"); tag=${2:?tag}; shift 2
root=$(cd "$(dirname "$0")/.." && pwd); out=$root/logs/perf; mkdir -p "$out"
MODEL=${GLM_MODEL:?set GLM_MODEL to the model file or first shard}
STORE=${GLM_STORE:?set GLM_STORE to the expert-store file}
USAGE=${GLM_USAGE:-$root/docs/ssd-offload/data/glm53flash_usage_combined_old_plus_survey5.txt}
TOML_SRC=${GLM_TOML:-$root/configs/glm53flash_AJ_IQ2XXS_spread_offload.toml}
PROMPT=${PROMPT:-'Explain how a mixture-of-experts transformer routes tokens, in a few clear paragraphs.'}
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done
if [ "${NO_DROP:-0}" != 1 ]; then python3 -c '
import os, sys
for a in sys.argv[1:]:
    for f in a.split(":"):
        fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
' "$MODEL" "${MODEL/-00001-of-00002/-00002-of-00002}" "$STORE"; fi
python3 - "$TOML_SRC" "$out/$tag.toml" "$MODEL" "$out/$tag.engine.log" <<'PY'
import json
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
text, model_count = re.subn(r"^model_path = .*$",
                            f"model_path = {json.dumps(sys.argv[3])}",
                            text, count=1, flags=re.M)
text, log_count = re.subn(r"^log_path *=.*$",
                          f"log_path = {json.dumps(sys.argv[4])}",
                          text, count=1, flags=re.M)
if model_count != 1 or log_count != 1:
    raise SystemExit("config must contain one model_path and one log_path")
Path(sys.argv[2]).write_text(text)
PY
printf '%s\n:q\n' "$PROMPT" | env -u ZEROMOE_GLM5NEXT_FULL_GRAPH \
  ZEROMOE_GLM5NEXT_OFFLOAD_FIRST_LAYER=3 ZEROMOE_GLM5NEXT_OFFLOAD_DDR_GIB=32 \
  ZEROMOE_GLM5NEXT_OFFLOAD_PINNED=0.5 ZEROMOE_GLM5NEXT_OFFLOAD_POLICY=lru \
  ZEROMOE_GLM5NEXT_OFFLOAD_STORE="$STORE" ZEROMOE_GLM5NEXT_OFFLOAD_MODEL="$MODEL" \
  ZEROMOE_GLM5NEXT_OFFLOAD_USAGE="$USAGE" ZEROMOE_GLM5NEXT_OFFLOAD_ASYNC=1 \
  ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB=512 ZEROMOE_GLM5NEXT_CPU_THREADS=8 \
  CUDA_VISIBLE_DEVICES=0 ZEROMOE_BENCH_IGNORE_STOP=1 "$@" \
  systemd-run --user --scope -q -p MemoryMax=${GLM_MEMMAX:-80G} -p MemorySwapMax=0 \
  "$bin" --config "$out/$tag.toml" --ngen "${NGEN:-48}" > "$out/$tag.stdout" 2>&1
echo "exit=$?"; grep -h 'decode=' "$out/$tag.engine.log" | tail -1 | sed -E 's/.*(decode=[0-9.]+ tok\/s).*/\1/'
