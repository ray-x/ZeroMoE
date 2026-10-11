#!/bin/bash
# usage: [POLICY=lru|seg] [PREFETCH=0..8] [PINNED=0.5] [BYTECAP=0|1] [PROBATION=0.25] [DDR=32] run_seg_ab.sh NAME
# GLM-5.3-Flash SSD offload, 3090 only, 5 prompt/decode rounds with different topics in one session. Refuses to run
# while nvtop/htop or another moestream is up (they perturb timing).
B=/home/ray/work/moestream-graph/glm_runs; N=$1; cd /home/ray/work/moestream-graph
if pgrep -x nvtop >/dev/null || pgrep -x htop >/dev/null; then echo "refusing: nvtop/htop running" >&2; exit 3; fi
for w in $(seq 1 120); do pgrep -x moestream >/dev/null || break; sleep 5; done
python3 - <<'PY'
import os,glob
for f in glob.glob('/home/ray/vllm-workspace/GLM-5.3-Flash/GLM-5.3-Flash-AJ-IQ2_XXS-*.gguf')+['/data/moestream-offload/glm53flash-aj-iq2xxs-cold3173.gguf']:
    fd=os.open(f,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
PY
sed "s#baseline.engine.log#pinab/$N.engine.log#" $B/spread_off.toml > $B/pinab/$N.toml
export MOESTREAM_GLM5NEXT_OFFLOAD_FIRST_LAYER=3 MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB=${DDR:-32}
export MOESTREAM_GLM5NEXT_OFFLOAD_PINNED=${PINNED:-0.5} MOESTREAM_GLM5NEXT_OFFLOAD_POLICY=${POLICY:-lru}
export MOESTREAM_GLM5NEXT_OFFLOAD_PROBATION=${PROBATION:-0.25} MOESTREAM_GLM5NEXT_OFFLOAD_BYTE_CAP=${BYTECAP:-0}
export MOESTREAM_GLM5NEXT_OFFLOAD_STORE=/data/moestream-offload/glm53flash-aj-iq2xxs-cold3173.gguf
export MOESTREAM_GLM5NEXT_OFFLOAD_MODEL=/home/ray/vllm-workspace/GLM-5.3-Flash/GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf
export MOESTREAM_GLM5NEXT_OFFLOAD_USAGE=/home/ray/work/moestream-graph/docs/ssd-offload/data/glm53flash_usage_combined_old_plus_survey5.txt
export MOESTREAM_GLM5NEXT_OFFLOAD_THP=0
if [ "${PREFETCH:-0}" != "0" ]; then
  export MOESTREAM_GLM5NEXT_PILOT_STAGE=1 MOESTREAM_GLM5NEXT_PILOT_PREFETCH=$PREFETCH MOESTREAM_GLM5NEXT_PILOT_COPY=0
else unset MOESTREAM_GLM5NEXT_PILOT_STAGE MOESTREAM_GLM5NEXT_PILOT_PREFETCH MOESTREAM_GLM5NEXT_PILOT_COPY; fi
unset MOESTREAM_GLM5NEXT_FULL_GRAPH; export MOESTREAM_GLM5NEXT_ALLOC_FLOOR_MIB=512
export MOESTREAM_GLM5NEXT_CPU_THREADS=8 CUDA_VISIBLE_DEVICES=0 MOESTREAM_BENCH_IGNORE_STOP=1
printf '%s\n' \
 "Explain how a mixture-of-experts transformer routes tokens, in a few clear paragraphs." \
 "Write a Python function that parses a CSV file and returns the average of a numeric column, with error handling." \
 "Write a short story about a lighthouse keeper who finds a message in a bottle." \
 "Explain how TCP congestion control works, covering slow start and congestion avoidance." \
 "Write a bash script that finds the ten largest files under a directory and prints their sizes." \
 ":q" | systemd-run --user --scope -q -p MemoryMax=80G -p MemorySwapMax=0 \
   ${BIN:-./build_acc/moestream} --config $B/pinab/$N.toml --ngen 96 > $B/pinab/$N.stdout 2>&1
echo DONE > $B/pinab/$N.done
