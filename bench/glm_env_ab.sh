#!/usr/bin/env bash
# Order-balanced A/B of one GLM-5.3-Flash offload decode setup, logits + tokens compared.
#   bench/glm_env_ab.sh <engine> <tag> "<A: VAR=val ...>" "<B: VAR=val ...>" [rounds]
# Arm A and B are space-separated env assignments (use "" for none), e.g. a switch
# ("ZEROMOE_GLM5NEXT_SPLIT_LOAD=0" vs "ZEROMOE_GLM5NEXT_SPLIT_LOAD=1") or a library override
# ("LD_LIBRARY_PATH=/x/lib_ctrl" vs "LD_LIBRARY_PATH=/x/lib_new").
# Output: logs/perf/<tag>_{A,B}_<n>.{engine.log,logits,tokens}; prints decode tok/s per run.
set -uo pipefail
bin=$(readlink -f "$1"); tag=$2; a=$3; b=$4; rounds=${5:-2}
here=$(cd "$(dirname "$0")" && pwd); out=$here/../logs/perf; mkdir -p "$out"
touch ~/work/.no-arms; trap 'rm -f ~/work/.no-arms' EXIT
for r in $(seq 1 "$rounds"); do
  if [ $((r % 2)) = 1 ]; then order="A B"; else order="B A"; fi
  for arm in $order; do
    t=${tag}_${arm}_$r
    if [ "$arm" = A ]; then extra=($a); else extra=($b); fi
    extra+=(ZEROMOE_DECODE_LOGITS_DUMP="$out/$t.logits" ZEROMOE_GENERATED_TOKEN_FILE="$out/$t.tokens")
    echo "== $t: ${extra[*]}"; "$here/glm_decode_profile.sh" "$bin" "$t" "${extra[@]}" | tail -1
  done
done
for r in $(seq 1 "$rounds"); do
  cmp -s "$out/${tag}_A_$r.logits" "$out/${tag}_B_$r.logits" && cmp -s "$out/${tag}_A_$r.tokens" "$out/${tag}_B_$r.tokens" \
    && echo "round $r: logits+tokens IDENTICAL" || echo "round $r: DIFFER"
done
