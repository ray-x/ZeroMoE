#!/usr/bin/env bash
# Combined refactor gate: Qwen correctness (teacher-forced logits + free-running
# tokens vs the frozen baseline binary) followed by the GLM offload arms.
#
#   bench/graph_framework_gate_both.sh <candidate-engine> <step-name>
#
# Holds a shared no-arms flag while running. Needs logs/graph_framework/inputs/.
set -u
bin=$(readlink -f "${1:?candidate engine}"); name=${2:?step name}
root=$(cd "$(dirname "$0")/.." && pwd); cd "$root"
base=${BASELINE_ENGINE:?set BASELINE_ENGINE to the reference engine}
flag=${NO_ARMS_FLAG:-${TMPDIR:-/tmp}/zeromoe-no-arms}
touch "$flag"; trap 'rm -f "$flag"' EXIT
inputs=logs/graph_framework/inputs; out=logs/graph_framework/$name; mkdir -p "$out"
common=(--engine "$bin" --baseline-engine "$base" --prompt-ids $inputs/prompt.tokens
        --continuation $inputs/Q2.tokens --all-split --wait-idle "${WAIT_IDLE:-900}"
        --baseline-env ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT=1
        --candidate-env ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT=1)
python3 bench/option_b_early_ready_check.py "${common[@]}" --out-dir "$out/correctness" \
    --reps 1 --dump-logits > "$out/correctness.driver.log" 2>&1
python3 bench/option_b_early_ready_check.py "${common[@]}" --out-dir "$out/free" \
    --reps 1 --free-running > "$out/free.driver.log" 2>&1
if cmp -s "$out/free/baseline_1.tokens" "$out/free/early_1.tokens"; then
  echo "QWEN free-running tokens: IDENTICAL"; else echo "QWEN free-running tokens: DIFFER"; fi
python3 - "$out" <<'P'
import json, sys
c = json.load(open(sys.argv[1] + "/correctness/comparison.json"))["summary"]
print("QWEN teacher-forced logits: top1", c["top1_agreement"], "max KL", c["max_kl_ref_candidate"])
P
ARMS="${GLM_ARMS:-lru lane}" bench/graph_framework_glm_gate.sh "$name" "$bin" 2>&1 \
    | grep -E "IDENTICAL|DIFFER|decode:"
