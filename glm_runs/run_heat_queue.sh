#!/bin/bash
# Dynamic VRAM promotion by decayed re-reference heat (host-LRU style score) vs the cumulative miss-count rule. Builds after the dynamic queue.
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
for i in $(seq 1 3000); do [ -f pinab/dynamic.done ] && break; sleep 15; done
(cd ../build_pf && make -j6 moestream > ../glm_runs/pinab/heat_build.log 2>&1)
STRIPE=/data/moestream-offload/glm53flash-stripe0.gguf:/home/ray/work/nvme1-store/glm53flash-stripe1.gguf
export MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC=1
run() { n=$1; shift; env "$@" ./run_clean.sh "$n"; }
run dh4_u8      POLICY=lru PREFETCH=0 STORE=$STRIPE MOESTREAM_GLM5NEXT_DYNAMIC_CACHE=1 MOESTREAM_GLM5NEXT_DYNAMIC_HEAT=4 MOESTREAM_GLM5NEXT_DYNAMIC_CACHE_UPLOADS_PER_TOKEN=8
run dh6_u8_h60  POLICY=lru PREFETCH=0 STORE=$STRIPE MOESTREAM_GLM5NEXT_DYNAMIC_CACHE=1 MOESTREAM_GLM5NEXT_DYNAMIC_HEAT=6 MOESTREAM_GLM5NEXT_DYNAMIC_HEAT_HALFLIFE=60 MOESTREAM_GLM5NEXT_DYNAMIC_CACHE_UPLOADS_PER_TOKEN=8
echo HEATDONE > pinab/heat.done
