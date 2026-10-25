#!/bin/bash
# Dynamic VRAM hot-set update (ZEROMOE_GLM5NEXT_DYNAMIC_CACHE: promote experts that keep missing, evict the least recently used swappable
# slot) on top of the best layout (store striped over two NVMes). Reference arm is ly_stripe (static hot set).
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
STRIPE=/data/moestream-offload/glm53flash-stripe0.gguf:/home/ray/work/nvme1-store/glm53flash-stripe1.gguf
export ZEROMOE_GLM5NEXT_OFFLOAD_ASYNC=1
run() { n=$1; shift; env "$@" ./run_clean.sh "$n"; }
run dc_auto     POLICY=lru PREFETCH=0 STORE=$STRIPE ZEROMOE_GLM5NEXT_DYNAMIC_CACHE=1
run dc_u4       POLICY=lru PREFETCH=0 STORE=$STRIPE ZEROMOE_GLM5NEXT_DYNAMIC_CACHE=1 ZEROMOE_GLM5NEXT_DYNAMIC_CACHE_UPLOADS_PER_TOKEN=4
run dc_u4_s50   POLICY=lru PREFETCH=0 STORE=$STRIPE ZEROMOE_GLM5NEXT_DYNAMIC_CACHE=1 ZEROMOE_GLM5NEXT_DYNAMIC_CACHE_UPLOADS_PER_TOKEN=4 ZEROMOE_GLM5NEXT_DYNAMIC_CACHE_SWAP_FRAC=0.5
echo DYNDONE > pinab/dynamic.done
