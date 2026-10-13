#!/usr/bin/env bash
# One GLM-5.3-Flash SSD-offload decode run with the graph's own timers on.
#   bench/glm_decode_profile.sh <engine> <tag> [extra VAR=val ...]
# Output: logs/perf/<tag>.{engine.log,stdout}. Extra VAR=val pairs are exported to the run.
set -uo pipefail
bin=$(readlink -f "${1:?engine}"); tag=${2:?tag}; shift 2
root=$(cd "$(dirname "$0")/.." && pwd); out=$root/logs/perf; mkdir -p "$out"
MODEL=${GLM_MODEL:-/home/ray/vllm-workspace/GLM-5.3-Flash/GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf}
STORE=${GLM_STORE:-/data/moestream-offload/glm53flash-aj-iq2xxs-cold3173.gguf}
USAGE=${GLM_USAGE:-$HOME/work/moestream-prefetch/docs/ssd-offload/data/glm53flash_usage_combined_old_plus_survey5.txt}
TOML_SRC=${GLM_TOML:-$HOME/work/moestream-prefetch/glm_runs/spread_off.toml}
PROMPT=${PROMPT:-'Explain how a mixture-of-experts transformer routes tokens, in a few clear paragraphs.'}
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done
python3 -c '
import os, sys
for a in sys.argv[1:]:
    for f in a.split(":"):
        fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
' "$MODEL" "${MODEL/-00001-of-00002/-00002-of-00002}" "$STORE"
sed "s#^log_path *=.*#log_path = \"$out/$tag.engine.log\"#" "$TOML_SRC" > "$out/$tag.toml"
printf '%s\n:q\n' "$PROMPT" | env -u MOESTREAM_GLM5NEXT_FULL_GRAPH \
  MOESTREAM_GLM5NEXT_OFFLOAD_FIRST_LAYER=3 MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB=32 \
  MOESTREAM_GLM5NEXT_OFFLOAD_PINNED=0.5 MOESTREAM_GLM5NEXT_OFFLOAD_POLICY=lru \
  MOESTREAM_GLM5NEXT_OFFLOAD_STORE="$STORE" MOESTREAM_GLM5NEXT_OFFLOAD_MODEL="$MODEL" \
  MOESTREAM_GLM5NEXT_OFFLOAD_USAGE="$USAGE" MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC=1 \
  MOESTREAM_GLM5NEXT_ALLOC_FLOOR_MIB=512 MOESTREAM_GLM5NEXT_CPU_THREADS=8 \
  CUDA_VISIBLE_DEVICES=0 MOESTREAM_BENCH_IGNORE_STOP=1 "$@" \
  systemd-run --user --scope -q -p MemoryMax=80G -p MemorySwapMax=0 \
  "$bin" --config "$out/$tag.toml" --ngen "${NGEN:-48}" > "$out/$tag.stdout" 2>&1
echo "exit=$?"; grep -h 'decode=' "$out/$tag.engine.log" | tail -1 | sed -E 's/.*(decode=[0-9.]+ tok\/s).*/\1/'
