#!/bin/bash
cd /path/to/moestream-glm53
export MOESTREAM_DENSE_SCRATCH_RESERVE_MB=768 MOESTREAM_GLM_DSA_FULL_GRAPH=1 MOESTREAM_GLM_DSA_FULL_GRAPH_PREFILL=1
export MOESTREAM_GLM5NEXT_OFFLOAD_FIRST_LAYER=3 MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB=${OFFLOAD_DDR_GIB:-45}
export MOESTREAM_GLM5NEXT_OFFLOAD_STORE=/data/glm53_offload.gguf
export MOESTREAM_GLM5NEXT_OFFLOAD_MODEL=/path/to/models/GLM-5.3-GGUF-align
export MOESTREAM_GLM5NEXT_OFFLOAD_USAGE=/path/to/MoEStream/.moestream_usage_glm53_ud_iq2m
export MOESTREAM_GLM5NEXT_GRAPH_STRICT=1
(printf 'Explain in two sentences why the sky is blue.\n:q\n' | timeout 1500 ./build/moestream --config logs/glm53_dsa_graph.toml --ngen 48 --temp 0 > logs/glm53_dsa_graph_off4.stdout 2>&1)
grep -E "Rates:|OFFLOAD|FULL-DECODE|WARN|ERROR" logs/glm53_dsa_graph_off4.log | cut -c1-220 | head -40
