#!/bin/bash
# Pool 100 GiB (capped at the 74.2 GiB of routed experts) under a 110G cgroup, striped store, merged binary.
# Arm A: CPU lane off (ZEROMOE_GLM5NEXT_EARLY_READY=0) = comparable with the earlier pool table. Arm B: lane on (new default).
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
STRIPE=/data/moestream-offload/glm53flash-stripe0.gguf:/home/ray/work/nvme1-store/glm53flash-stripe1.gguf
export ZEROMOE_GLM5NEXT_OFFLOAD_ASYNC=1
env ZEROMOE_GLM5NEXT_EARLY_READY=0 POLICY=lru PREFETCH=0 STORE=$STRIPE DDR=100 MEMMAX=110G HOSTRAM=16 ./run_clean.sh m110_pool100_laneoff
env POLICY=lru PREFETCH=0 STORE=$STRIPE DDR=100 MEMMAX=110G HOSTRAM=16 ./run_clean.sh m110_pool100_laneon
echo M110DONE > pinab/m110b.done
