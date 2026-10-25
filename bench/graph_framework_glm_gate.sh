#!/usr/bin/env bash
# GLM-5.3-Flash SSD-offload gate for refactor steps that touch ggml_glm5next_graph.cpp
# or the expert store. RTX 3090 only, single prompt, greedy, 48 tokens.
#
#   bench/graph_framework_glm_gate.sh <step-name> <candidate-engine> [baseline-engine]
#
# Arms (each run once per binary, alternating order):
#   lru        offload, lru, no pilot            (miss node + store + async loads)
#   prefetch   pilot prefetch K=4, no VRAM copy  (stage worker + store.ensure(Prefetch))
#   stage      pilot staging K=1 + VRAM ring     (stage worker + reader_lock + ring)
#   lane       ZEROMOE_GLM5NEXT_EARLY_READY=1  (CPU lane, only the candidate binary has it)
# ARMS="lane" selects arms.
# For each arm the decode logits dump and generated tokens must be byte-identical
# between the two binaries. Decode tok/s is reported from the engine log.
#
# Refuses to start while any other process uses the GPU (waits up to 20 minutes).
set -uo pipefail
step=${1:?step name}
cand=$(readlink -f "${2:?candidate engine}")
base=$(readlink -f "${3:-$HOME/work/gf-baseline/moestream-master-b8cf7ff}")
root=$(cd "$(dirname "$0")/.." && pwd)
out=$root/logs/graph_framework/$step/glm
mkdir -p "$out"
cd "$root"

MODEL=${GLM_MODEL:-/home/ray/vllm-workspace/GLM-5.3-Flash/GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf}
STORE=${GLM_STORE:-/data/moestream-offload/glm53flash-aj-iq2xxs-cold3173.gguf}
USAGE=${GLM_USAGE:-$HOME/work/moestream-prefetch/docs/ssd-offload/data/glm53flash_usage_combined_old_plus_survey5.txt}
TOML_SRC=${GLM_TOML:-$HOME/work/moestream-prefetch/glm_runs/spread_off.toml}
PROMPT='Explain how a mixture-of-experts transformer routes tokens, in a few clear paragraphs.'

wait_idle() {
  for _ in $(seq 1 2160); do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] \
       && [ "$(pgrep -x 'cicc|ptxas|cc1plus|cc1|nvcc' 2>/dev/null | wc -l)" = 0 ]; then
      return 0
    fi
    sleep 5
  done
  echo "GPU/CPU never went idle (3 h)" >&2
  return 1
}

drop_cache() {
  python3 -c '
import os, sys
for f in sys.argv[1:]:
    fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
' "$MODEL" "${MODEL/-00001-of-00002/-00002-of-00002}" "$STORE"
}

run_one() {  # arm binary-label binary
  local arm=$1 label=$2 bin=$3 tag="$1_$2"
  wait_idle || return 1
  drop_cache
  sed "s#^log_path *=.*#log_path = \"$out/$tag.engine.log\"#" "$TOML_SRC" > "$out/$tag.toml"
  local env=(ZEROMOE_GLM5NEXT_OFFLOAD_FIRST_LAYER=3 ZEROMOE_GLM5NEXT_OFFLOAD_DDR_GIB=32
             ZEROMOE_GLM5NEXT_OFFLOAD_PINNED=0.5 ZEROMOE_GLM5NEXT_OFFLOAD_POLICY=lru
             ZEROMOE_GLM5NEXT_OFFLOAD_STORE="$STORE" ZEROMOE_GLM5NEXT_OFFLOAD_MODEL="$MODEL"
             ZEROMOE_GLM5NEXT_OFFLOAD_USAGE="$USAGE" ZEROMOE_GLM5NEXT_OFFLOAD_ASYNC=1
             ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB=512 ZEROMOE_GLM5NEXT_CPU_THREADS=8
             CUDA_VISIBLE_DEVICES=0 ZEROMOE_BENCH_IGNORE_STOP=1
             ZEROMOE_DECODE_LOGITS_DUMP="$out/$tag.logits"
             ZEROMOE_GENERATED_TOKEN_FILE="$out/$tag.tokens")
  case $arm in
    lru) ;;
    prefetch) env+=(ZEROMOE_GLM5NEXT_PILOT_STAGE=1 ZEROMOE_GLM5NEXT_PILOT_PREFETCH=4 ZEROMOE_GLM5NEXT_PILOT_COPY=0) ;;
    stage) env+=(ZEROMOE_GLM5NEXT_PILOT_STAGE=1) ;;
    lane) env+=(ZEROMOE_GLM5NEXT_EARLY_READY=1) ;;  # CPU lane (step 4); baseline binary ignores it
  esac
  printf '%s\n:q\n' "$PROMPT" | env -u ZEROMOE_GLM5NEXT_FULL_GRAPH "${env[@]}" \
      systemd-run --user --scope -q -p MemoryMax=80G -p MemorySwapMax=0 \
      "$bin" --config "$out/$tag.toml" --ngen 48 > "$out/$tag.stdout" 2>&1
  echo "$tag exit=$?"
}

# The prefetch session's queued arms (glm_runs/run_clean.sh) start whenever the box is
# quiet and yield to this file; hold it for the duration of the gate so the two
# do not race for the GPU. Only removed if this script created it.
flag=$HOME/work/.no-arms
if [ ! -e "$flag" ]; then
  echo "gf-glm-gate $step pid $$ $(date -Is)" > "$flag"
  trap 'rm -f "$flag"' EXIT
fi

status=0
i=0
for arm in ${ARMS:-lru prefetch stage}; do
  i=$((i + 1))
  if [ $((i % 2)) = 1 ]; then order=(base cand); else order=(cand base); fi
  for which in "${order[@]}"; do
    if [ "$which" = base ]; then run_one "$arm" base "$base"; else run_one "$arm" cand "$cand"; fi
  done
  if cmp -s "$out/${arm}_base.logits" "$out/${arm}_cand.logits" \
     && cmp -s "$out/${arm}_base.tokens" "$out/${arm}_cand.tokens" \
     && [ -s "$out/${arm}_base.logits" ]; then
    echo "$arm: logits and tokens IDENTICAL"
  else
    echo "$arm: DIFFER (or missing output)"; status=1
  fi
  for label in base cand; do
    r=$(grep -h 'decode=' "$out/${arm}_$label.engine.log" | tail -1 | sed -E 's/.*decode=([0-9.]+ tok\/s).*/\1/')
    echo "  $arm $label decode: ${r:-n/a}"
  done
done
exit $status
