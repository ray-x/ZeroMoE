#!/bin/bash
# MADV_FREE vs MADV_DONTNEED eviction (MOESTREAM_OFFLOAD_EVICT): same arms as the base queue, run after it.
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
for i in $(seq 1 2000); do [ -f pinab/queue.done ] && break; sleep 15; done
export MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC=1
run() { n=$1; shift; env "$@" ./run_clean.sh "$n"; }
run pq_base_free POLICY=lru PREFETCH=0 MOESTREAM_OFFLOAD_EVICT=free LAYER_TIMING=1
run pq_pf4_free  POLICY=lru PREFETCH=4 MOESTREAM_OFFLOAD_EVICT=free LAYER_TIMING=1
echo EVICTDONE > pinab/evict.done
