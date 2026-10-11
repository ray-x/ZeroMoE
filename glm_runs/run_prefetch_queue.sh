#!/bin/bash
# GLM-5.3-Flash SSD offload, 3090 only, DDR 32 GiB, 50% pinned, async loads: per-layer timing + CPU sampling, THP on/off, and the
# remaining confidence arms. Every arm goes through run_clean.sh (waits for a quiet box, retries when contaminated).
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
export MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC=1
run() { n=$1; shift; env "$@" ./run_clean.sh "$n"; }
run pq_base      POLICY=lru PREFETCH=0 LAYER_TIMING=1
run pq_pf4       POLICY=lru PREFETCH=4 LAYER_TIMING=1
run pq_pf4_thp   POLICY=lru PREFETCH=4 THP=1 LAYER_TIMING=1
run pq_base_thp  POLICY=lru PREFETCH=0 THP=1 LAYER_TIMING=1
run pq_k1_p1     POLICY=lru PREFETCH=1
run pq_k2_p2     POLICY=lru PREFETCH=2 MOESTREAM_GLM5NEXT_PILOT_LOOKAHEAD=2
echo QUEUEDONE > pinab/queue.done
