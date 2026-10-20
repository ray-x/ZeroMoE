#!/bin/bash
# usage: abrun.sh <tag> [extra env assignments...]  -- 22-token prompt default, per-run overrides applied AFTER defaults
cd "$(dirname "$0")/../.." && mkdir -p logs
tag=$1; shift
export MOESTREAM_GLM5NEXT_ALLOC_FLOOR_MIB=768 MOESTREAM_GLM_DSA_CPU_LANE=1 MOESTREAM_GLM5NEXT_ASSIST_DEVICE=0 MOESTREAM_GLM5NEXT_PILOT_STAGE=5 MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC=1 MOESTREAM_GLM5NEXT_PILOT_PREFETCH=5 MOESTREAM_GLM5NEXT_PILOT_COPY=0 MOESTREAM_GLM5NEXT_PILOT_LOOKAHEAD=2 MOESTREAM_GLM5NEXT_ASSIST2_DEVICE=1 MOESTREAM_GLM5NEXT_ASSIST2_SLOTS=3
for kv in "$@"; do export "$kv"; done
export MOESTREAM_BIN=${MOESTREAM_BIN:-./build/moestream} TOML=${TOML:-bench/glm53/glm53_dsa_core5060_retarget.toml} NGEN=${NGEN:-80} OFFLOAD_DDR_GIB=${OFFLOAD_DDR_GIB:-54}
export PROMPT_TEXT="${PROMPT_TEXT:-Explain in detail how a transformer language model generates text, step by step.}"
rm -f logs/glm53_dsa_graph2gpu_tke.log
./bench/glm53/run_dsa_graph2gpu.sh > logs/split_ab_$tag.out 2>&1
test -f logs/glm53_dsa_graph2gpu_tke.log && cp logs/glm53_dsa_graph2gpu_tke.log logs/split_ab_$tag.log
test -f logs/glm53_dsa_graph2gpu_tke.stdout && cp logs/glm53_dsa_graph2gpu_tke.stdout logs/split_ab_$tag.stdout
