#!/usr/bin/env bash
# A/B one GLM env switch on one binary, order-balanced, logits+tokens compared.
#   bench/glm_split_ab.sh <engine> <tag> <VAR> [rounds]
# Arm "on" runs with VAR=1, arm "off" with VAR=0. Output: logs/perf/<tag>_{off,on}_<n>.*
set -uo pipefail
bin=$(readlink -f "$1"); tag=$2; var=$3; rounds=${4:-2}
here=$(cd "$(dirname "$0")" && pwd); out=$here/../logs/perf
touch ~/work/.no-arms; trap 'rm -f ~/work/.no-arms' EXIT
for r in $(seq 1 "$rounds"); do
  if [ $((r % 2)) = 1 ]; then order="off on"; else order="on off"; fi
  for arm in $order; do
    t=${tag}_${arm}_$r
    extra=(MOESTREAM_DECODE_LOGITS_DUMP="$out/$t.logits" MOESTREAM_GENERATED_TOKEN_FILE="$out/$t.tokens")
    if [ "$arm" = on ]; then extra+=("$var=1"); else extra+=("$var=0"); fi
    echo "== $t"; "$here/glm_decode_profile.sh" "$bin" "$t" "${extra[@]}"
  done
done
for r in $(seq 1 "$rounds"); do
  cmp -s "$out/${tag}_off_$r.logits" "$out/${tag}_on_$r.logits" && cmp -s "$out/${tag}_off_$r.tokens" "$out/${tag}_on_$r.tokens" \
    && echo "round $r: logits+tokens IDENTICAL" || echo "round $r: DIFFER"
done
