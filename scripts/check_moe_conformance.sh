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

LEGACY_DIRECT=()
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
for f in models/mimo/ggml_mimo2_graph.cpp models/glm/ggml_bailingmoe3_graph.cpp \
         models/deepseek/ggml_deepseek_v41_graph_decode.cpp; do
  if ! grep -qE 'CacheBackedMoeRuntime|ExecutionPolicy|resolve_execution_policy|MoEBlockSpec' "$f"; then
    echo "FAIL: $f does not go through the shared MoE execution policy"; fail=1
  fi
done

# Tunables of migrated families come from the settings resolver, not direct env reads.
# (The INGRAPH_MOE / FULL_GRAPH feature gates are not tunables and stay as they are.)
for f in models/mimo/ggml_mimo2_graph.cpp models/graph/full_graph_models.cpp; do
  if grep -nE '(env|integer|flag_on|flag_unless_off|atoi_or|getenv)\(\s*"ZEROMOE_(MIMO2|LING)_' "$f" \
     | grep -vE 'ZEROMOE_MIMO2_(INGRAPH_MOE|FULL_GRAPH)"' | grep -q .; then
    echo "FAIL: $f reads a ZEROMOE_MIMO2_/LING_ tunable directly (add it to graph_settings.hpp)"; fail=1
  fi
done

# Family special cases in shared code go through capabilities (GraphEngineCaps, FullGraphModel
# virtuals), not ModelFamily comparisons. Counts of the families migrated so far may only shrink:
# what is left is per-family prompt templates (tokenizer format is genuinely family-specific) and
# other families' checks.
check_family_refs() { # file pattern max
  local n; n=$(grep -c "$2" "$1" || true)
  if (( n > $3 )); then
    echo "FAIL: $1 has $n reference(s) to $2 (allowed $3): use a FullGraphModel capability"; fail=1
  fi
}
check_family_refs core/engine_runner_cache.cpp 'ModelFamily::MiMo2' 0
check_family_refs core/engine_runner_cache.cpp 'ModelFamily::BailingMoE3' 0
check_family_refs core/engine_runner_session.cpp 'ModelFamily::MiMo2' 1
check_family_refs core/engine_runner_session.cpp 'ModelFamily::BailingMoE3' 1
check_family_refs core/engine_runner_session.cpp 'ModelFamily::GLM_DSA' 0
for f in models/graph/*.hpp models/graph/*.cpp; do
  n=$(grep -E 'ModelFamily::(MiMo2|BailingMoE3|GLM5NEXT|GLM_DSA)' "$f" | grep -vE 'ModelFamily family\(\) const|case ModelFamily' | wc -l || true)
  allowed=0
  [[ $f == models/graph/full_graph_model.hpp ]] && allowed=0
  if (( n > allowed )); then echo "FAIL: $f compares or names a migrated ModelFamily ($n)"; fail=1; fi
done

# Direct environment reads of GLM/Ling graph knobs may only shrink. What remains are experimental and
# diagnostic switches (dynamic cache heat, pilot ridge, GPU1 hot-set usage, dumps, traces); tunables of the
# supported configurations come from graph_settings.hpp, so a new one goes there.
GLM_ENV_READS=$(grep -rEh 'graphcfg::(env|flag_on|flag_unless_off|integer|atoi_or|atof_or)\("ZEROMOE_(GLM5NEXT|LING)_' models | wc -l)
GLM_ENV_READS_MAX=81
if (( GLM_ENV_READS > GLM_ENV_READS_MAX )); then
  echo "FAIL: $GLM_ENV_READS direct ZEROMOE_GLM5NEXT_/LING_ env reads (max $GLM_ENV_READS_MAX): add the knob to graph_settings.hpp"; fail=1
fi

# ModelFamily references in core/ may only shrink. What remains: Qwen4Exp and DeepSeek-V4/V4.1 special cases
# (those families' owners migrate them to capabilities), GLM-only paths, and per-family prompt templates.
CORE_FAMILY_REFS=$(grep -rE 'ModelFamily::(Qwen4Exp|DeepSeek_V41|DeepSeek_V4|GLM5NEXT|GLM_DSA|MiMo2|BailingMoE3)' core | wc -l)
CORE_FAMILY_REFS_MAX=112
if (( CORE_FAMILY_REFS > CORE_FAMILY_REFS_MAX )); then
  echo "FAIL: $CORE_FAMILY_REFS ModelFamily references in core/ (max $CORE_FAMILY_REFS_MAX): use a FullGraphModel capability"; fail=1
fi

[[ $fail == 0 ]] && echo "moe-conformance: OK (${#graph_files[@]} graph files checked)"
exit $fail
