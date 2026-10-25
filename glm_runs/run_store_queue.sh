#!/bin/bash
# Offload store coverage: the 3,173-expert cold store (the other ~9k experts come from the main GGUF with buffered per-plane reads)
# vs the all-experts store (12,096 experts, one O_DIRECT read each). Same async/lru settings, 3090 only.
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
ALL=/data/moestream-offload/glm53flash-aj-iq2xxs-all12096.gguf
export ZEROMOE_GLM5NEXT_OFFLOAD_ASYNC=1
run() { n=$1; shift; env "$@" ./run_clean.sh "$n"; }
run st_cold_base POLICY=lru PREFETCH=0
run st_all_base  POLICY=lru PREFETCH=0 STORE=$ALL
run st_all_pf4   POLICY=lru PREFETCH=4 STORE=$ALL
echo STOREDONE > pinab/store.done
