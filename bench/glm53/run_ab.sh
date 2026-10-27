#!/bin/bash
# usage: abrun.sh <tag> [extra env assignments...]  -- 22-token prompt default, per-run overrides applied AFTER defaults
cd "$(dirname "$0")/../.." && mkdir -p logs
tag=$1; shift
export ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB=768 ZEROMOE_GLM_DSA_CPU_LANE=1 ZEROMOE_GLM5NEXT_ASSIST_DEVICE=0 ZEROMOE_GLM5NEXT_PILOT_STAGE=5 GLM_OFFLOAD_ASYNC=${GLM_OFFLOAD_ASYNC:-1} ZEROMOE_GLM5NEXT_PILOT_PREFETCH=5 GLM_PILOT_COPY=${GLM_PILOT_COPY:-0} ZEROMOE_GLM5NEXT_PILOT_LOOKAHEAD=2 ZEROMOE_GLM5NEXT_ASSIST2_DEVICE=1 GLM_ASSIST2_SLOTS=${GLM_ASSIST2_SLOTS:-3}
for kv in "$@"; do export "$kv"; done
export ZEROMOE_BIN=${ZEROMOE_BIN:-./build/zeromoe} TOML=${TOML:-configs/bench/glm53/glm53_dsa_core5060_retarget.toml} NGEN=${NGEN:-80} OFFLOAD_DDR_GIB=${OFFLOAD_DDR_GIB:-54}
export PROMPT_TEXT="${PROMPT_TEXT:-Explain in detail how a transformer language model generates text, step by step.}"
rm -f logs/glm53_dsa_graph2gpu_tke.log
rc=0
./bench/glm53/run_dsa_graph2gpu.sh > logs/split_ab_$tag.out 2>&1 || rc=$?
test -f logs/glm53_dsa_graph2gpu_tke.log && cp logs/glm53_dsa_graph2gpu_tke.log logs/split_ab_$tag.log
test -f logs/glm53_dsa_graph2gpu_tke.stdout && cp logs/glm53_dsa_graph2gpu_tke.stdout logs/split_ab_$tag.stdout
# Preserve the engine's status through this wrapper (the gate checks completion and
# termination; a grep/copy must never overwrite a nonzero engine exit with success).
exit "$rc"
