#!/bin/bash
# usage: run_q.sh N   (Qwen4Exp early-ready profile, probe taps on, memory-capped)
W=/home/ray/work/moestream-graph; N=$1
cd $W
rm -rf qwen_probe/d$N; mkdir -p qwen_probe/d$N
python3 - <<PY
import re
t=open('configs/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml').read()
t=re.sub(r'(?m)^log_path\s*=.*$','log_path = "$W/qwen_probe/q$N.engine.log"',t) if re.search(r'(?m)^log_path',t) else 'log_path = "$W/qwen_probe/q$N.engine.log"\n'+t
t=re.sub(r'(?m)^max_new_tokens\s*=.*$','max_new_tokens = 64',t)
open('qwen_probe/q$N.toml','w').write(t)
PY
for w in $(seq 1 60); do pgrep -x zeromoe >/dev/null || break; sleep 5; done
{ cat glm_runs/probe/p$N.txt; echo; } | systemd-run --user --scope -q -p MemoryMax=72G -p MemorySwapMax=0 \
  env ZEROMOE_QWEN4EXP_PROBE_DUMP=$W/qwen_probe/d$N ZEROMOE_BENCH_IGNORE_STOP=1 CUDA_VISIBLE_DEVICES=0 \
  ZEROMOE_QWEN_EARLY_READY_LAYER13=0 ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT=1 ZEROMOE_QWEN_EARLY_READY_RESIDENT_TABLES=1 ZEROMOE_QWEN_EARLY_READY_ASYNC_OUTPUT=1 \
  ./build_acc/zeromoe --config qwen_probe/q$N.toml --ngen 64 > qwen_probe/q$N.stdout 2>&1
echo DONE > qwen_probe/done_$N
