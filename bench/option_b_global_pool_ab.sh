#!/bin/bash
# Four-way A/B for the Option-B global expert pool (Q2_0, one GPU, plain decode, chat prompt).
#
#   A  right-sized per-layer 0/44/4   (configs/testing/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml)  ~10.84 GiB
#   B  global pool, same VRAM as A    (configs/testing/qwen4exp_q2_0_option_b_global_topk48.toml, K=8421)      ~10.84 GiB
#   C  per-layer top-256 x 44 layers  (A with right_size_top_n=false)                                    ~14.5 GiB
#   D  global pool, same VRAM as C    (B with K=11264)                                                   ~14.5 GiB
#
# usage: bench/option_b_global_pool_ab.sh [gpu_index=0] [reps=5] [ngen=128]
# Needs the engine at ../moestream-build-upstream-master/moestream, an idle GPU and a quiet CPU.
# Writes logs/optb_global_ab_<cfg>_<rep>.{engine.log,stdout}; prints mean tok/s per config
# (TOKEN-LOOP-DIAG iter_total, steps >= 3). READ THE OUTPUT TEXT before trusting a number
# (grep -A6 '^Output:' logs/optb_global_ab_*.stdout): speed on a broken answer means nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${1:-0}; REPS=${2:-5}; NGEN=${3:-128}
[[ "$GPU" =~ ^[0-9]+$ && "$REPS" =~ ^[1-9][0-9]*$ && "$NGEN" =~ ^[1-9][0-9]*$ ]] || {
  echo "usage: $0 [gpu_index=0] [positive_reps=5] [positive_ngen=128]" >&2
  exit 2
}
[[ "$NGEN" -gt 3 ]] || { echo "ngen must exceed the 3 warmup steps" >&2; exit 2; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p logs
A=configs/testing/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml
B=configs/testing/qwen4exp_q2_0_option_b_global_topk48.toml
cp "$A" "$TMP/A.toml"; cp "$B" "$TMP/B.toml"
sed -E 's/^qwen4exp_option_b_right_size_top_n *=.*/qwen4exp_option_b_right_size_top_n = false/' "$A" > "$TMP/C.toml"
sed -E 's/^qwen4exp_option_b_global_top_k *=.*/qwen4exp_option_b_global_top_k = 11264/' "$B" > "$TMP/D.toml"
PROMPT="Explain briefly the key concepts of neural networks and gradient descent in plain English."

run() { # name rep
  local n=$1 r=$2 cfg="$TMP/$1.toml" tag="optb_global_ab_${1}_$2"
  sed -i -E "s#^log_path *=.*#log_path = \"logs/$tag.engine.log\"#" "$cfg"
  while :; do
    local competing
    competing=$(nvidia-smi --id="$GPU" --query-compute-apps=pid,process_name --format=csv,noheader)
    if ! pgrep -x zeromoe >/dev/null && [[ -z "$competing" ]]; then break; fi
    sleep 3
  done
  echo "$PROMPT" | env CUDA_VISIBLE_DEVICES=$GPU \
    ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS=0 \
    ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1 ZEROMOE_TOKEN_LOOP_DIAG=1 ZEROMOE_BENCH_IGNORE_STOP=1 \
    ../moestream-build-upstream-master/moestream --config "$cfg" --ngen "$NGEN" > "logs/$tag.stdout" 2>&1
  grep TOKEN-LOOP-DIAG "logs/$tag.engine.log" 2>/dev/null |
    sed -E 's/.*step=([0-9]+) iter_total=([0-9.]+)ms.*/\1 \2/' |
    awk -v n="$n" -v r="$r" -v expected="$((NGEN - 3))" '$1>=3{s+=$2;c++} END{ if(c==expected && s>0) printf "%s %s %.2f\n", n, r, 1000/(s/c); else {printf "%s %s incomplete: %d/%d timed steps\n", n, r, c, expected > "/dev/stderr"; exit 1} }'
}

echo "== warm-up (discarded)"; run A warm >/dev/null
RES="$TMP/results.txt"; : > "$RES"
for rep in $(seq 1 "$REPS"); do
  if [ $((rep % 2)) -eq 1 ]; then ord="A B C D"; else ord="D C B A"; fi
  for n in $ord; do run "$n" "$rep" | tee -a "$RES"; done
done
echo "== mean tok/s per config"
awk '$3!="NA"{s[$1]+=$3;c[$1]++} END{for(k in s) printf "%s n=%d mean %.2f\n", k, c[k], s[k]/c[k]}' "$RES" | sort
