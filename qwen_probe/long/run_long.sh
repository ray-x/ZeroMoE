#!/bin/bash
# usage: run_long.sh N   (long-decode probe: 640 generated tokens, routing/x taps on; 3090 only)
W=/home/ray/work/moestream-graph; N=$1; cd $W
rm -rf qwen_probe/long/d$N; mkdir -p qwen_probe/long/d$N
python3 - <<PY
import re
t=open('configs/testing/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml').read()
t=re.sub(r'(?m)^log_path\s*=.*$','log_path = "$W/qwen_probe/long/q$N.engine.log"',t) if re.search(r'(?m)^log_path',t) else 'log_path = "$W/qwen_probe/long/q$N.engine.log"\n'+t
t=re.sub(r'(?m)^max_new_tokens\s*=.*$','max_new_tokens = 640',t)
open('qwen_probe/long/q$N.toml','w').write(t)
PY
for w in $(seq 1 60); do pgrep -x zeromoe >/dev/null || break; sleep 5; done
{ cat qwen_probe/long/p$N.txt; echo; } | systemd-run --user --scope -q -p MemoryMax=72G -p MemorySwapMax=0 \
  env ZEROMOE_QWEN4EXP_PROBE_DUMP=$W/qwen_probe/long/d$N ZEROMOE_BENCH_IGNORE_STOP=1 CUDA_VISIBLE_DEVICES=0 \
  ZEROMOE_QWEN_EARLY_READY_LAYER13=0 ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT=1 ZEROMOE_QWEN_EARLY_READY_RESIDENT_TABLES=1 ZEROMOE_QWEN_EARLY_READY_ASYNC_OUTPUT=1 \
  ./build_acc/zeromoe --config qwen_probe/long/q$N.toml --ngen 640 > qwen_probe/long/q$N.stdout 2>&1
echo DONE > qwen_probe/long/done_$N
