#!/bin/bash
# GLM-5.3-Flash read layout: original shards + cold store vs interleaved model (embedded blocks) vs the store striped over two NVMes.
# Same async/lru, no prefetch, 3090 only.
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
IL=/data/glm53flash-il
STRIPE=/data/moestream-offload/glm53flash-stripe0.gguf:/home/ray/work/nvme1-store/glm53flash-stripe1.gguf
export MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC=1
run() { n=$1; shift; env "$@" ./run_clean.sh "$n"; }
run ly_orig_cold  POLICY=lru PREFETCH=0
run ly_il_embed   POLICY=lru PREFETCH=0 MODEL_DIR=$IL STORE=$IL
run ly_stripe     POLICY=lru PREFETCH=0 STORE=$STRIPE
run ly_il_cold    POLICY=lru PREFETCH=0 MODEL_DIR=$IL
run ly_orig_cold2 POLICY=lru PREFETCH=0
echo LAYOUTDONE > pinab/layout.done
