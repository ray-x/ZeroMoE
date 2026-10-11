#!/bin/bash
# usage: [POLICY=lru|seg] [PREFETCH=0..8] [PINNED=0.5] [BYTECAP=0|1] [PROBATION=0.25] [DDR=32] run_seg_ab.sh NAME
# GLM-5.3-Flash SSD offload, 3090 only, 5 prompt/decode rounds with different topics in one session. Refuses to run
# while nvtop/htop or another moestream is up (they perturb timing).
B=$(cd "$(dirname "$0")" && pwd); N=$1; cd "$(dirname "$B")"
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
export MOESTREAM_GLM5NEXT_OFFLOAD_USAGE=$(dirname "$B")/docs/ssd-offload/data/glm53flash_usage_combined_old_plus_survey5.txt
export MOESTREAM_GLM5NEXT_OFFLOAD_THP=${THP:-0}
[ -n "$LAYER_TIMING" ] && export MOESTREAM_GLM5NEXT_GRAPH_LAYER_TIMING=1
if [ "${PREFETCH:-0}" != "0" ]; then
  export MOESTREAM_GLM5NEXT_PILOT_STAGE=1 MOESTREAM_GLM5NEXT_PILOT_PREFETCH=$PREFETCH MOESTREAM_GLM5NEXT_PILOT_COPY=0
else unset MOESTREAM_GLM5NEXT_PILOT_STAGE MOESTREAM_GLM5NEXT_PILOT_PREFETCH MOESTREAM_GLM5NEXT_PILOT_COPY; fi
unset MOESTREAM_GLM5NEXT_FULL_GRAPH; export MOESTREAM_GLM5NEXT_ALLOC_FLOOR_MIB=${FLOOR:-512}
export MOESTREAM_GLM5NEXT_CPU_THREADS=8 CUDA_VISIBLE_DEVICES=0 MOESTREAM_BENCH_IGNORE_STOP=1
# CPU sampler: process utime/stime ticks once a second (system time is where page zeroing and I/O syscalls show up)
( while true; do
    pid=$(pgrep -f "pinab/$N.toml" | while read p; do [ "$(readlink /proc/$p/exe 2>/dev/null | grep -c moestream)" = 1 ] && echo $p && break; done)
    if [ -n "$pid" ] && [ -r /proc/$pid/stat ]; then
      awk -v t="$(date +%s.%N)" '{print t, $14, $15}' /proc/$pid/stat >> $B/pinab/$N.cpu.csv 2>/dev/null
      # contamination watch: compile jobs or other GPU/engine processes running next to the arm
      nc=$(pgrep -c -x 'cicc|ptxas|cc1plus|cc1|nvcc|cmake|ninja|make|ld|as' 2>/dev/null); no=$(pgrep -x 'llama-bench|llama-cli|llama-server|ollama|vllm' 2>/dev/null | wc -l)
      echo "$(date +%s.%N) $(cut -d' ' -f1 /proc/loadavg) ${nc:-0} $no" >> $B/pinab/$N.load.csv 2>/dev/null
    fi
    sleep 1
  done ) &
SAMPLER=$!
printf '%s\n' \
 "Explain how a mixture-of-experts transformer routes tokens, in a few clear paragraphs." \
 "Write a Python function that parses a CSV file and returns the average of a numeric column, with error handling." \
 "Write a short story about a lighthouse keeper who finds a message in a bottle." \
 "Explain how TCP congestion control works, covering slow start and congestion avoidance." \
 "Write a bash script that finds the ten largest files under a directory and prints their sizes." \
 ":q" | systemd-run --user --scope -q -p MemoryMax=80G -p MemorySwapMax=0 \
   ${BIN:-./build_pf/moestream} --config $B/pinab/$N.toml --ngen 96 > $B/pinab/$N.stdout 2>&1
kill $SAMPLER 2>/dev/null
echo DONE > $B/pinab/$N.done
