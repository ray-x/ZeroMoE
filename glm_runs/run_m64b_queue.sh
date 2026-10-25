#!/bin/bash
# 64 GiB machine, retry: host_ram_gb=2 was the process-wide host allocation cap and refused the dense tensors (empty outputs). Use 6 GiB.
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
STRIPE=/data/moestream-offload/glm53flash-stripe0.gguf:/home/ray/work/nvme1-store/glm53flash-stripe1.gguf
export ZEROMOE_GLM5NEXT_OFFLOAD_ASYNC=1
run() { n=$1; shift; env "$@" ./run_clean.sh "$n"; }
run m64b_pool58 POLICY=lru PREFETCH=0 STORE=$STRIPE DDR=58 MEMMAX=64G HOSTRAM=6
run dc_gates    POLICY=lru PREFETCH=0 STORE=$STRIPE ZEROMOE_GLM5NEXT_DYNAMIC_CACHE=1 ZEROMOE_GLM5NEXT_DYNAMIC_CACHE_UPLOADS_PER_TOKEN=4
echo M64BDONE > pinab/m64b.done
