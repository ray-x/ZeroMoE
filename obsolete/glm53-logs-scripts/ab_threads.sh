#!/bin/bash
# prefill-thread sweep: one run per value, each preceded by the preflight
cd /path/to/moestream-glm53
for T in 12 10; do
  sed -i "s/glm53_dsa_graph2gpu_[a-z0-9]*\./glm53_dsa_graph2gpu_t$T./g" logs/glm53_dsa_graph2gpu.toml logs/run_dsa_graph2gpu_426.sh
  MOESTREAM_GLM5NEXT_PREFILL_CPU_THREADS=$T MOESTREAM_GLM5NEXT_CPU_THREADS=6 MOESTREAM_GLM5NEXT_OFFLOAD_SPLIT=1 MOESTREAM_GLM5NEXT_GPU1_HOT_GIB=11 OFFLOAD_DDR_GIB=45 ./logs/run_dsa_graph2gpu_426.sh > logs/glm53_dsa_graph2gpu_t$T.run.out 2>&1
  echo "T=$T: $(grep -h 'Rates:' logs/glm53_dsa_graph2gpu_t$T.log | cut -c26-200)"
done
