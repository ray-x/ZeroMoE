#!/bin/bash
cd "$(dirname "$0")/../.." && mkdir -p logs
./bench/glm53/preflight.sh || exit 1
P=${PROMPT_TEXT:-$(python3 -c "print(' '.join(['The history of computing spans mechanical calculators, vacuum tube machines, transistors, integrated circuits and modern GPUs; summarize the key turning points and why each mattered for software.']*12))")}
export ZEROMOE_GLM5NEXT_PREFILL_OP_OFFLOAD=${ZEROMOE_GLM5NEXT_PREFILL_OP_OFFLOAD:-1} ZEROMOE_GLM5NEXT_PREFILL_CPU_THREADS=${ZEROMOE_GLM5NEXT_PREFILL_CPU_THREADS:-10} ZEROMOE_GLM5NEXT_CPU_THREADS=${ZEROMOE_GLM5NEXT_CPU_THREADS:-6} ZEROMOE_GLM5NEXT_OFFLOAD_SPLIT=${ZEROMOE_GLM5NEXT_OFFLOAD_SPLIT:-1} ZEROMOE_GLM5NEXT_GPU1_HOT_GIB=${ZEROMOE_GLM5NEXT_GPU1_HOT_GIB:-11}
export ZEROMOE_BIN=${ZEROMOE_BIN:-./build/zeromoe}
export ZEROMOE_DENSE_SCRATCH_RESERVE_MB=768 ZEROMOE_GLM_DSA_FULL_GRAPH=1 ZEROMOE_GLM_DSA_FULL_GRAPH_PREFILL=1
export ZEROMOE_GLM5NEXT_OFFLOAD_FIRST_LAYER=3 ZEROMOE_GLM5NEXT_OFFLOAD_DDR_GIB=${OFFLOAD_DDR_GIB:-45}
export ZEROMOE_GLM5NEXT_OFFLOAD_STORE=/data/glm53_offload.gguf
export ZEROMOE_GLM5NEXT_OFFLOAD_MODEL=/home/ray/vllm-workspace/GLM-5.3-GGUF-align
export ZEROMOE_GLM5NEXT_OFFLOAD_USAGE=${ZEROMOE_GLM5NEXT_OFFLOAD_USAGE:-/home/ray/Projects/ZeroMoE/.zeromoe_usage_glm53_ud_iq2m}
export ZEROMOE_GLM5NEXT_GRAPH_STRICT=1
(printf "%s\n:q\n" "$P" | timeout 1500 ${ZEROMOE_BIN:-./build/zeromoe} --config ${TOML:-bench/glm53/glm53_dsa_core5060_retarget.toml} --ngen ${NGEN:-48} --temp 0 > logs/glm53_dsa_graph2gpu_tke.stdout 2>&1)
engine_rc=$?
grep -E "Rates:|OFFLOAD|FULL-DECODE|WARN|ERROR" logs/glm53_dsa_graph2gpu_tke.log | cut -c1-220 | head -40 || true
# Propagate the engine status: the gate must never treat a crashed candidate as a pass
# just because the wrapper's grep succeeded. (124 = timeout, 137 = SIGKILL, ...)
exit "$engine_rc"
