#!/usr/bin/env bash
# 3090-ONLY Qwen resident-expert sweep (CUDA_VISIBLE_DEVICES=0). Never add GPU1.
set -u
W=/home/ray/work/moestream-graph; cd $W
OUT=$W/qwen_runs; ENG=$W/build_acc/moestream
BASE=$W/configs/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml
for p in nvtop htop; do pgrep -x $p >/dev/null && { echo "ABORT: $p running"; exit 2; }; done
[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { echo "ABORT: GPU busy"; exit 2; }
for N in ${@:-191 96 48 280 191}; do
  for rep in 1 2; do
    tag=n${N}_r${rep}; cfg=$OUT/$tag.toml
    sed -e "s/^qwen4exp_option_b_top_n *=.*/qwen4exp_option_b_top_n = $N/" \
        -e "s/^qwen4exp_option_b_min_coverage *=.*/qwen4exp_option_b_min_coverage = 0.0/" \
        -e "s/^qwen4exp_option_b_right_size_top_n *=.*/qwen4exp_option_b_right_size_top_n = false/" \
        -e "s#^log_path *=.*#log_path = \"$OUT/$tag.log\"#" $BASE > $cfg
    echo "== $tag"
    printf 'Explain briefly the key concepts of neural networks and gradient descent in plain English.\n:q\n' | \
    env -u MOESTREAM_GLM5NEXT_FULL_GRAPH CUDA_VISIBLE_DEVICES=0 GGML_CUDA_BF16_MMVF_CUBLAS=0 \
      MOESTREAM_QWEN4EXP_FULL_GRAPH=1 MOESTREAM_QWEN4EXP_FULL_GRAPH_DECODE=1 \
      MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_LAYERS=0 MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1 \
      MOESTREAM_QWEN_CPU_THREADS=10 MOESTREAM_TOKEN_LOOP_DIAG=1 MOESTREAM_BENCH_IGNORE_STOP=1 \
      MOESTREAM_GENERATED_TOKEN_FILE=$OUT/$tag.tokens \
      $ENG --config $cfg --ngen 64 > $OUT/$tag.stdout 2>&1
    python3 - "$OUT/$tag.log" "$tag" <<'PY'
import re,sys,statistics as s
c=open(sys.argv[1]).read()
t=[float(x) for st,x in re.findall(r'TOKEN-LOOP-DIAG.*step=(\d+) iter_total=([\d.]+)ms',c) if int(st)>=3]
r=re.findall(r'(\d[\d,]*) (?:device|resident)[^\n]{0,40}experts',c)[:1]
print(sys.argv[2],'steady_ms=%.2f tok/s=%.2f moe/s=%.0f n=%d'%(s.mean(t),1000/s.mean(t),480*1000/s.mean(t),len(t)) if t else 'NO TIMING', r)
PY
  done
done
