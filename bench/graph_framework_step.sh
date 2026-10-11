#!/usr/bin/env bash
# Per-step gate for the graph-framework refactor: Qwen3.8-Flash-Next Q2_0
# split44 cov80, early-ready ALL_SPLIT (the accepted bundle), RTX 3090 only.
#
#   bench/graph_framework_step.sh <step-name> <candidate-engine> [baseline-engine]
#
# 1. correctness: teacher-forced decode logits, candidate vs baseline engine,
#    must be byte-identical; free-running generated tokens must match.
# 2. performance: six order-balanced pairs (baseline engine vs candidate engine).
#
# Output: logs/graph_framework/<step-name>/{correctness,free,perf}/ and a
# one-line verdict at the end. Runs nothing if another inference process holds
# the GPU (the harness waits up to 15 minutes, then aborts).
set -euo pipefail
step=${1:?step name}
cand=$(readlink -f "${2:?candidate engine}")
base=$(readlink -f "${3:-$HOME/work/gf-baseline/moestream-master-b8cf7ff}")
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
inputs=logs/graph_framework/inputs
out=logs/graph_framework/$step
mkdir -p "$out"
common=(--engine "$cand" --baseline-engine "$base"
        --prompt-ids "$inputs/prompt.tokens" --continuation "$inputs/Q2.tokens"
        --all-split --wait-idle "${WAIT_IDLE:-900}"
        --baseline-env MOESTREAM_QWEN_EARLY_READY_ALL_SPLIT=1
        --candidate-env MOESTREAM_QWEN_EARLY_READY_ALL_SPLIT=1)

echo "== $step: correctness (teacher-forced logits)"
python3 bench/option_b_early_ready_check.py "${common[@]}" \
    --out-dir "$out/correctness" --reps 1 --dump-logits \
    > "$out/correctness.driver.log" 2>&1
tail -3 "$out/correctness.driver.log"

echo "== $step: correctness (free-running tokens)"
python3 bench/option_b_early_ready_check.py "${common[@]}" \
    --out-dir "$out/free" --reps 1 --free-running \
    > "$out/free.driver.log" 2>&1
if cmp -s "$out/free/baseline_1.tokens" "$out/free/early_1.tokens"; then
    echo "free-running tokens identical"
else
    echo "FREE-RUNNING TOKENS DIFFER"
fi

echo "== $step: performance (6 pairs)"
python3 bench/option_b_early_ready_check.py "${common[@]}" \
    --out-dir "$out/perf" --reps 6 \
    > "$out/perf.driver.log" 2>&1
python3 - "$out" <<'EOF'
import json, sys
o = sys.argv[1]
s = json.load(open(f'{o}/perf/summary.json'))
c = json.load(open(f'{o}/correctness/comparison.json'))
print(f"perf: baseline {s['baseline']:.2f} tok/s, candidate {s['early']:.2f} tok/s, "
      f"gain {s['gain_pct']:+.2f}%, paired wins {s['paired_wins']}/6, "
      f"paired {[round(x, 1) for x in s['paired_gain_pct']]}")
print("logits comparison:", json.dumps(c)[:300])
EOF
