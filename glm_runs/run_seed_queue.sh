#!/bin/bash
# Boot seeding: serial (SEED_PAR=1) vs 8 reads in flight on the striped store. Alternating, outputs and boot-seeding time compared.
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
STRIPE=/data/moestream-offload/glm53flash-stripe0.gguf:/home/ray/work/nvme1-store/glm53flash-stripe1.gguf
export MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC=1
run() { n=$1; shift; env "$@" ./run_clean.sh "$n"; }
run seed_par1_a MOESTREAM_OFFLOAD_SEED_PAR=1 POLICY=lru PREFETCH=0 STORE=$STRIPE DDR=32
run seed_par8_a MOESTREAM_OFFLOAD_SEED_PAR=8 POLICY=lru PREFETCH=0 STORE=$STRIPE DDR=32
run seed_par1_b MOESTREAM_OFFLOAD_SEED_PAR=1 POLICY=lru PREFETCH=0 STORE=$STRIPE DDR=32
run seed_par8_b MOESTREAM_OFFLOAD_SEED_PAR=8 POLICY=lru PREFETCH=0 STORE=$STRIPE DDR=32
echo SEEDDONE > pinab/seed.done
