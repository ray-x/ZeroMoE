#!/bin/bash
# Set GLM_MODEL and GLM_STORE before running; optionally set GLM_MIRROR_GLOB.
cd "$(dirname "$0")/../.." && mkdir -p logs
root=$PWD
export GLM_MODEL=${GLM_MODEL:?set GLM_MODEL to the model file or shard directory}
export GLM_STORE=${GLM_STORE:?set GLM_STORE to the expert-store file}
export GLM_MIRROR_GLOB=${GLM_MIRROR_GLOB:-}
export ZEROMOE_GLM5NEXT_OFFLOAD_USAGE=${ZEROMOE_GLM5NEXT_OFFLOAD_USAGE:-$root/.zeromoe_usage_glm53_ud_iq2m}
./bench/glm53/preflight.sh || exit 1
P=${PROMPT_TEXT:-$(python3 -c "print(' '.join(['The history of computing spans mechanical calculators, vacuum tube machines, transistors, integrated circuits and modern GPUs; summarize the key turning points and why each mattered for software.']*12))")}
export ZEROMOE_GLM5NEXT_PREFILL_OP_OFFLOAD=${ZEROMOE_GLM5NEXT_PREFILL_OP_OFFLOAD:-1} ZEROMOE_GLM5NEXT_PREFILL_CPU_THREADS=${ZEROMOE_GLM5NEXT_PREFILL_CPU_THREADS:-10} ZEROMOE_GLM5NEXT_CPU_THREADS=${ZEROMOE_GLM5NEXT_CPU_THREADS:-6} ZEROMOE_GLM5NEXT_OFFLOAD_SPLIT=${ZEROMOE_GLM5NEXT_OFFLOAD_SPLIT:-1} ZEROMOE_GLM5NEXT_GPU1_HOT_GIB=${ZEROMOE_GLM5NEXT_GPU1_HOT_GIB:-11}
export ZEROMOE_BIN=${ZEROMOE_BIN:-./build/zeromoe}
export ZEROMOE_DENSE_SCRATCH_RESERVE_MB=768 ZEROMOE_GLM_DSA_FULL_GRAPH=1 ZEROMOE_GLM_DSA_FULL_GRAPH_PREFILL=1
export ZEROMOE_GLM5NEXT_OFFLOAD_FIRST_LAYER=3 ZEROMOE_GLM5NEXT_OFFLOAD_DDR_GIB=${OFFLOAD_DDR_GIB:-45}
export ZEROMOE_GLM5NEXT_OFFLOAD_STORE=$GLM_STORE
export ZEROMOE_GLM5NEXT_OFFLOAD_MODEL=$GLM_MODEL
export ZEROMOE_GLM5NEXT_GRAPH_STRICT=1
config=${TOML:-configs/bench/glm53/glm53_dsa_core5060_retarget.toml}
run_config=$(mktemp "${TMPDIR:-/tmp}/zeromoe-glm53.XXXXXX.toml")
trap 'rm -f "$run_config"' EXIT
python3 - "$config" "$run_config" "$GLM_MODEL" <<'PY'
import json
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
text, count = re.subn(r"^model_path = .*$",
                      f"model_path = {json.dumps(sys.argv[3])}",
                      text, count=1, flags=re.M)
if count != 1:
    raise SystemExit("config must contain exactly one model_path")
Path(sys.argv[2]).write_text(text)
PY
(printf "%s\n:q\n" "$P" | timeout 1500 "${ZEROMOE_BIN:-./build/zeromoe}" --config "$run_config" --ngen "${NGEN:-48}" --temp 0 > logs/glm53_dsa_graph2gpu_tke.stdout 2>&1)
engine_rc=$?
grep -E "Rates:|OFFLOAD|FULL-DECODE|WARN|ERROR" logs/glm53_dsa_graph2gpu_tke.log | cut -c1-220 | head -40 || true
# Propagate the engine status: the gate must never treat a crashed candidate as a pass
# just because the wrapper's grep succeeded. (124 = timeout, 137 = SIGKILL, ...)
exit "$engine_rc"
