#!/usr/bin/env bash
# 3090-ONLY prefetch A/B on Q2_0 (accepted cov80 config, early-ready default). Usage: run_ab.sh <subdir> <arm:rep>...
set -u
W=/home/ray/work/moestream-graph; cd $W
OUT=$W/qwen_runs/$1; mkdir -p $OUT; shift
ENG=$W/build_acc/zeromoe
BASE=$W/configs/testing/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml
for p in nvtop htop; do pgrep -x $p >/dev/null && { echo "ABORT: $p running"; exit 2; }; done
for job in "$@"; do
  arm=${job%%:*}; rep=${job##*:}; tag=${arm}_r${rep}; cfg=$OUT/$tag.toml
  [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { echo "ABORT: GPU busy"; exit 2; }
  sed -e "s#^log_path *=.*#log_path = \"$OUT/$tag.log\"#" $BASE > $cfg
  case $arm in
    base) X=();;
    s1)  X=(ZEROMOE_QWEN_EARLY_READY_STAGE_LAYER=all ZEROMOE_QWEN_EARLY_READY_STAGE_MAX=1);;
    s3)  X=(ZEROMOE_QWEN_EARLY_READY_STAGE_LAYER=all ZEROMOE_QWEN_EARLY_READY_STAGE_MAX=3);;
    r1)  X=(ZEROMOE_QWEN_EARLY_READY_STAGE_LAYER=all ZEROMOE_QWEN_EARLY_READY_STAGE_MAX=1 ZEROMOE_QWEN_EARLY_READY_RIDGE=$W/qwen_runs/ridge/qwen_q2_0_ridge.bin);;
    r3)  X=(ZEROMOE_QWEN_EARLY_READY_STAGE_LAYER=all ZEROMOE_QWEN_EARLY_READY_STAGE_MAX=3 ZEROMOE_QWEN_EARLY_READY_RIDGE=$W/qwen_runs/ridge/qwen_q2_0_ridge.bin);;
    s10) X=(ZEROMOE_QWEN_EARLY_READY_STAGE_LAYER=all);;
    *) echo bad arm; exit 2;;
  esac
  echo "== $tag"
  printf 'Explain briefly the key concepts of neural networks and gradient descent in plain English.\n:q\n' | \
  env CUDA_VISIBLE_DEVICES=0 GGML_CUDA_BF16_MMVF_CUBLAS=0 \
    ZEROMOE_QWEN4EXP_FULL_GRAPH=1 ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE=1 \
    ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS=0 ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1 \
    ZEROMOE_QWEN_CPU_THREADS=10 ZEROMOE_TOKEN_LOOP_DIAG=1 ZEROMOE_BENCH_IGNORE_STOP=1 \
    ZEROMOE_GENERATED_TOKEN_FILE=$OUT/$tag.tokens "${X[@]}" \
    $ENG --config $cfg --ngen 64 > $OUT/$tag.stdout 2>&1
  python3 - "$OUT/$tag.log" "$tag" <<'PY'
import re,sys,statistics as s
c=open(sys.argv[1]).read()
t=[float(x) for st,x in re.findall(r'TOKEN-LOOP-DIAG.*step=(\d+) iter_total=([\d.]+)ms',c) if int(st)>=3]
print(sys.argv[2],'steady_ms=%.2f tok/s=%.2f n=%d'%(s.mean(t),1000/s.mean(t),len(t)) if t else 'NO TIMING')
PY
done
