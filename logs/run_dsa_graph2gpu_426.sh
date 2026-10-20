#!/bin/bash
cd /home/ray/work/moestream-glm53
./logs/preflight.sh || exit 1
P=${PROMPT_TEXT:-$(python3 -c "print(' '.join(['The history of computing spans mechanical calculators, vacuum tube machines, transistors, integrated circuits and modern GPUs; summarize the key turning points and why each mattered for software.']*12))")}
export MOESTREAM_GLM5NEXT_PREFILL_OP_OFFLOAD=${MOESTREAM_GLM5NEXT_PREFILL_OP_OFFLOAD:-1} MOESTREAM_GLM5NEXT_PREFILL_CPU_THREADS=${MOESTREAM_GLM5NEXT_PREFILL_CPU_THREADS:-10} MOESTREAM_GLM5NEXT_CPU_THREADS=${MOESTREAM_GLM5NEXT_CPU_THREADS:-6} MOESTREAM_GLM5NEXT_OFFLOAD_SPLIT=${MOESTREAM_GLM5NEXT_OFFLOAD_SPLIT:-1} MOESTREAM_GLM5NEXT_GPU1_HOT_GIB=${MOESTREAM_GLM5NEXT_GPU1_HOT_GIB:-11}
export MOESTREAM_BIN=${MOESTREAM_BIN:-./build_g6/moestream}
export MOESTREAM_DENSE_SCRATCH_RESERVE_MB=768 MOESTREAM_GLM_DSA_FULL_GRAPH=1 MOESTREAM_GLM_DSA_FULL_GRAPH_PREFILL=1
export MOESTREAM_GLM5NEXT_OFFLOAD_FIRST_LAYER=3 MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB=${OFFLOAD_DDR_GIB:-45}
export MOESTREAM_GLM5NEXT_OFFLOAD_STORE=/data/glm53_offload.gguf
export MOESTREAM_GLM5NEXT_OFFLOAD_MODEL=/home/ray/vllm-workspace/GLM-5.3-GGUF-align
export MOESTREAM_GLM5NEXT_OFFLOAD_USAGE=${MOESTREAM_GLM5NEXT_OFFLOAD_USAGE:-/home/ray/Projects/MoEStream/.moestream_usage_glm53_ud_iq2m}
export MOESTREAM_GLM5NEXT_GRAPH_STRICT=1
(printf "%s\n:q\n" "$P" | timeout 1500 ${MOESTREAM_BIN:-./build/moestream} --config ${TOML:-logs/glm53_dsa_graph2gpu.toml} --ngen ${NGEN:-48} --temp 0 > logs/glm53_dsa_graph2gpu_tke.stdout 2>&1)
grep -E "Rates:|OFFLOAD|FULL-DECODE|WARN|ERROR" logs/glm53_dsa_graph2gpu_tke.log | cut -c1-220 | head -40
