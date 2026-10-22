#!/usr/bin/env bash
# R2.1 conformance gate: family graph code composes the shared MoE execution
# strategy; it must not own expert execution. Fails when a family graph file
#   1. calls moe_forward_gguf directly,
#   2. instantiates ExpertStore or EarlyReadyLane, or
#   3. (MoE families) never declares/uses a MoEBlockSpec / ExecutionPolicy.
# Families that predate R2.1 are grandfathered below; the list may only shrink.
# New families (MiMo, Ling) are not on it.
set -euo pipefail
cd "$(dirname "$0")/.."

LEGACY_DIRECT=(
  models/deepseek/ggml_deepseek_v41_graph_decode.cpp   # DSV4.1 native cache path
)
LEGACY_OWNERS=(
  models/glm/ggml_glm5next_graph_offload.cpp           # GLM-5.x ExpertStore
  models/qwen/ggml_qwen4exp_graph_build_moe.cpp        # Qwen4Exp early lane
  models/qwen/ggml_qwen4exp_graph_host_node.cpp
)
# Strategy implementations may touch the primitives.
STRATEGY_DIR=models/graph

is_legacy() { local f=$1; shift; for l in "$@"; do [[ $f == "$l" ]] && return 0; done; return 1; }

fail=0
mapfile -t graph_files < <(find models -path "$STRATEGY_DIR" -prune -o \
  -type f -name 'ggml_*graph*.cpp' -print | sort)

for f in "${graph_files[@]}"; do
  if grep -nE '\bmoe_forward_gguf\s*\(' "$f" >/dev/null && ! is_legacy "$f" "${LEGACY_DIRECT[@]}"; then
    echo "FAIL: $f calls moe_forward_gguf directly (use the shared MoE strategy)"; fail=1
  fi
  if grep -nE 'ExpertStore::create|make_unique<\s*(::)?EarlyReadyLane\s*>' "$f" >/dev/null \
     && ! is_legacy "$f" "${LEGACY_OWNERS[@]}"; then
    echo "FAIL: $f instantiates ExpertStore/EarlyReadyLane (owned by the strategy layer)"; fail=1
  fi
done

# Every family that builds routed-MoE graphs must resolve through the policy layer.
for f in models/mimo/ggml_mimo2_graph.cpp models/glm/ggml_bailingmoe3_graph.cpp; do
  if ! grep -qE 'CacheBackedMoeRuntime|ExecutionPolicy|resolve_execution_policy|MoEBlockSpec' "$f"; then
    echo "FAIL: $f does not go through the shared MoE execution policy"; fail=1
  fi
done

[[ $fail == 0 ]] && echo "moe-conformance: OK (${#graph_files[@]} graph files checked)"
exit $fail
