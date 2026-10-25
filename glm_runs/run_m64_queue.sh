#!/bin/bash
# "64 GiB machine": cgroup MemoryMax=64G, MoE pool 58 GiB (pinned 0.5 + LRU), the legacy RAM tier shrunk so the process fits. Striped store.
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
for i in $(seq 1 4000); do [ -f pinab/heat.done ] && break; sleep 15; done
STRIPE=/data/moestream-offload/glm53flash-stripe0.gguf:/home/ray/work/nvme1-store/glm53flash-stripe1.gguf
export ZEROMOE_GLM5NEXT_OFFLOAD_ASYNC=1
run() { n=$1; shift; env "$@" ./run_clean.sh "$n"; }
run m64_pool58 POLICY=lru PREFETCH=0 STORE=$STRIPE DDR=58 MEMMAX=64G HOSTRAM=2
echo M64DONE > pinab/m64.done
