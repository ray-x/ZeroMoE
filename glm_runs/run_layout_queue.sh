#!/bin/bash
# GLM-5.3-Flash on the interleaved (aligned + embedded expert blocks) model vs the original shards. Same async/lru; 3090 only.
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
IL=/data/glm53flash-il
export MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC=1
run() { n=$1; shift; env "$@" ./run_clean.sh "$n"; }
run ly_orig_cold  POLICY=lru PREFETCH=0
run ly_il_embed   POLICY=lru PREFETCH=0 MODEL_DIR=$IL STORE=$IL
run ly_il_cold    POLICY=lru PREFETCH=0 MODEL_DIR=$IL
run ly_il_embed4  POLICY=lru PREFETCH=4 MODEL_DIR=$IL STORE=$IL
echo LAYOUTDONE > pinab/layout.done
