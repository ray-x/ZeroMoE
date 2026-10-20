#!/bin/bash
# decode sweep on top of the working pilot-prefetch config
cd /home/ray/work/moestream-glm53
run() { # name, extra env...
  local N=$1; shift
  sed -i "s/glm53_dsa_graph2gpu_[a-z0-9]*\./glm53_dsa_graph2gpu_$N./g" logs/glm53_dsa_graph2gpu.toml logs/run_dsa_graph2gpu_426.sh
  env MOESTREAM_GLM5NEXT_PILOT_STAGE=8 MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC=1 MOESTREAM_GLM5NEXT_PILOT_PREFETCH=8 MOESTREAM_GLM5NEXT_PILOT_COPY=0 "$@" ./logs/run_dsa_graph2gpu_426.sh > logs/glm53_dsa_graph2gpu_$N.run.out 2>&1
  local L=logs/glm53_dsa_graph2gpu_$N.log
  echo "$N: $(grep -h 'Rates:' $L | cut -c26-190) | step $(grep 'FULL-DECODE-TIME' $L | awk -F'total=' '{split($2,a," "); if(NR>1){s+=a[1]; n++}} END {if(n) printf "%.0f ms", s/n}')"
}
run la3 MOESTREAM_GLM5NEXT_PILOT_LOOKAHEAD=3
run la2t8 MOESTREAM_GLM5NEXT_PILOT_LOOKAHEAD=2 MOESTREAM_GLM5NEXT_CPU_THREADS=8 MOESTREAM_GLM5NEXT_OFFLOAD_THP=1
