#!/bin/bash
# Aligned (gguf_tool pack --align 4096) GLM-5.3-Flash vs the original shards, same async/lru, no prefetch; plus the full-store arm on top.
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
for i in $(seq 1 2400); do [ -f pinab/store.done ] && break; sleep 15; done
ALIGNED=/data/glm53flash-aligned
export MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC=1
run() { n=$1; shift; env "$@" ./run_clean.sh "$n"; }
run al_base      POLICY=lru PREFETCH=0 MODEL_DIR=$ALIGNED
run al_all_base  POLICY=lru PREFETCH=0 MODEL_DIR=$ALIGNED STORE=/data/moestream-offload/glm53flash-aj-iq2xxs-all12096.gguf
echo ALIGNEDDONE > pinab/aligned.done
