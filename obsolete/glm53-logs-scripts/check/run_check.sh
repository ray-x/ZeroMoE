#!/bin/bash
# usage: run_check.sh NAME layered|graph PROMPTFILE NGEN
cd /path/to/moestream-glm53
N=$1; MODE=$2; PF=$3; NG=$4
./bench/glm53/preflight.sh || exit 1
if [ "$MODE" = graph ]; then
  CFG=obsolete/glm53-logs-scripts/glm53_dsa_graph2gpu.toml
  sed -i "s/^log_path = .*/log_path = \"\/home\/ray\/work\/moestream-glm53\/logs\/check\/$N.log\"/" $CFG
  export MOESTREAM_DENSE_SCRATCH_RESERVE_MB=768 MOESTREAM_GLM_DSA_FULL_GRAPH=1 MOESTREAM_GLM_DSA_FULL_GRAPH_PREFILL=1 MOESTREAM_GLM5NEXT_PREFILL_OP_OFFLOAD=1 MOESTREAM_GLM5NEXT_PREFILL_CPU_THREADS=10 MOESTREAM_GLM5NEXT_CPU_THREADS=6 MOESTREAM_GLM5NEXT_OFFLOAD_SPLIT=1 MOESTREAM_GLM5NEXT_GPU1_HOT_GIB=${HOT:-11} MOESTREAM_GLM5NEXT_PILOT_STAGE=8 MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC=1 MOESTREAM_GLM5NEXT_PILOT_PREFETCH=8 MOESTREAM_GLM5NEXT_PILOT_COPY=0 MOESTREAM_GLM5NEXT_PILOT_LOOKAHEAD=2
  export MOESTREAM_GLM5NEXT_OFFLOAD_FIRST_LAYER=3 MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB=54 MOESTREAM_GLM5NEXT_OFFLOAD_STORE=/data/glm53_offload.gguf MOESTREAM_GLM5NEXT_OFFLOAD_MODEL=/path/to/models/GLM-5.3-GGUF-align MOESTREAM_GLM5NEXT_OFFLOAD_USAGE=/path/to/ZeroMoE/.zeromoe_usage_glm53_ud_iq2m MOESTREAM_GLM5NEXT_GRAPH_STRICT=1
  BIN=./build_g6/moestream
else
  CFG=logs/glm53_cold50.toml
  sed -i "s/^log_path = .*/log_path = \"\/home\/ray\/work\/moestream-glm53\/logs\/check\/$N.log\"/" $CFG
  export MOESTREAM_DENSE_SCRATCH_RESERVE_MB=768
  BIN=./build/moestream
fi
P=$(cat $PF)
(printf '%s\n:q\n' "$P" | timeout 1700 $BIN --config $CFG --ngen $NG --temp 0 > logs/check/$N.stdout 2>&1)
echo "$N: $(grep -h 'Rates:' logs/check/$N.log | cut -c26-190)"
echo "  OUT: $(grep -A1 '^Output' logs/check/$N.stdout | tr '\n' ' ' | cut -c1-260)"
