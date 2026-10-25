#!/bin/bash
# Pool size vs machine size. host_ram_gb is the process-wide host cap (the dense resident tensors need ~10+ GiB); the pool itself is a virtual
# arena outside it. 64 GiB machine: cgroup 64G, pool 46 GiB (RSS = pool + ~17 GiB). Reference: pool 58 GiB with a 100G cap.
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
STRIPE=/data/moestream-offload/glm53flash-stripe0.gguf:/home/ray/work/nvme1-store/glm53flash-stripe1.gguf
export ZEROMOE_GLM5NEXT_OFFLOAD_ASYNC=1
run() { n=$1; shift; env "$@" ./run_clean.sh "$n"; }
run m64c_pool46 POLICY=lru PREFETCH=0 STORE=$STRIPE DDR=46 MEMMAX=64G HOSTRAM=16
run m100_pool58 POLICY=lru PREFETCH=0 STORE=$STRIPE DDR=58 MEMMAX=100G HOSTRAM=16
echo M64CDONE > pinab/m64c.done
