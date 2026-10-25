#!/bin/bash
# Multi-request VRAM stress session for the Q2_0 Option-B global-pool config at expert budget K.
#
# usage: bench/option_b_vram_stress.sh K [ngen=48] [gpu=0]
#   env: ZEROMOE_* variables are passed through (e.g. ZEROMOE_QWEN4EXP_GRAPH_ALLOC_FLOOR_MIB=512 for K > ~14,729).
# One engine process receives five single-line requests on stdin: SHORT, LONG, SHORT, LONG, SHORT (each line is an
# independent request; the engine does not accumulate history between them). SHORT = the 28-token neural-networks question,
# LONG = a 402-token prompt (first 1,100 chars of docs/runtime-settings.md at commit b5e97c9 + a summarize question; built into
# logs/option_b_global_pool_check_prompt.txt if missing). The token counts are the engine's own PREFILL-DONE numbers.
# GPU memory is sampled every 250 ms for the whole process. Writes logs/optb_stress_K<K>.{csv,stdout,engine.log}.
# Reported per run: requests completed, layered-decode fallback (a FAILURE), error strings, arena release/rebuild events,
# min free VRAM (24 GiB minus the sampled peak; transient spikes between samples are NOT seen), prefill time per request and
# decode tok/s per request (TOKEN-LOOP-DIAG iter_total, steps >= 3, restarting at step 0 per request).
# Limits: two prompt sizes only, five requests, 48 generated tokens, context 2048, single run per K, answer text only eyeballed.
set -euo pipefail
cd "$(dirname "$0")/.."
K=${1:?usage: $0 K [ngen] [gpu]}; NGEN=${2:-48}; GPU=${3:-0}
[[ "$K" =~ ^[0-9]+$ && "$NGEN" =~ ^[0-9]+$ && "$GPU" =~ ^[0-9]+$ ]] || { echo "K, ngen and gpu must be integers" >&2; exit 2; }
TAG=optb_stress_K$K
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p logs
LONGFILE=logs/option_b_global_pool_check_prompt.txt
if [[ ! -s "$LONGFILE" ]]; then
  python3 - "$LONGFILE" <<'EOF'
import subprocess, sys
text = subprocess.check_output(["git", "show", "b5e97c9:docs/runtime-settings.md"], text=True).replace("\n", " ")
open(sys.argv[1], "w").write(text[:1100] + " Question: summarize the text above in two sentences.\n")
EOF
fi
sed -E "s/^qwen4exp_option_b_global_top_k *=.*/qwen4exp_option_b_global_top_k = $K/; s#^log_path *=.*#log_path = \"logs/$TAG.engine.log\"#" \
  configs/testing/qwen4exp_q2_0_option_b_global_topk48.toml > "$TMP/cfg.toml"
LONG=$(cat "$LONGFILE")
SHORT="Explain briefly the key concepts of neural networks and gradient descent in plain English."
while pgrep -x zeromoe >/dev/null || [[ -n "$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader)" ]]; do sleep 3; done
nvidia-smi --id="$GPU" --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits -lms 250 > "logs/$TAG.csv" &
SP=$!
printf '%s\n%s\n%s\n%s\n%s\n:q\n' "$SHORT" "$LONG" "$SHORT" "$LONG" "$SHORT" | env \
  CUDA_VISIBLE_DEVICES=$GPU ZEROMOE_QWEN4EXP_FULL_GRAPH=1 ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE=1 \
  ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS=0 ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1 ZEROMOE_TOKEN_LOOP_DIAG=1 \
  ../moestream-build-upstream-master/moestream --config "$TMP/cfg.toml" --ngen "$NGEN" > "logs/$TAG.stdout" 2>&1 || true
kill $SP 2>/dev/null || true
python3 - "$TAG" <<'EOF'
import re, statistics as st, sys
tag = sys.argv[1]
log = open(f"logs/{tag}.engine.log", errors="replace").read()
out = open(f"logs/{tag}.stdout", errors="replace").read()
rows = []
for line in open(f"logs/{tag}.csv"):
    p = [x.strip() for x in line.split(",")]
    if len(p) >= 3 and p[1].isdigit() and p[2].isdigit():
        rows.append((int(p[1]), int(p[2])))
total = rows[0][1]; peak = max(u for u, _ in rows)
reqs, cur = [], []
for m in re.finditer(r"step=(\d+) iter_total=([\d.]+)ms", log):
    s, v = int(m.group(1)), float(m.group(2))
    if s == 0 and cur:
        reqs.append(cur); cur = []
    cur.append((s, v))
if cur: reqs.append(cur)
rates = [1000 / st.mean(v for s, v in r if s >= 3) if sum(1 for s, _ in r if s >= 3) > 5 else None for r in reqs]
fell_back = "layered decode remains active" in log or "falling back to the legacy path" in log
errors = len(re.findall(r"CUDA error|out of memory|FATAL|Segmentation", out + log))
released = len(re.findall(r"released the per-expert split arena", log))
rebuilt = [round(float(x)) for x in re.findall(r"prepared the per-expert split arena[^\n]*? in ([\d.]+)ms", log)]
prefill = [(int(a), float(b)) for a, b in re.findall(r"PREFILL-DONE\] (\d+) tokens in ([\d.]+)s", log)]
answers = [m.group(1).strip()[:70] for m in re.finditer(r"Output:(.*?)(\[Generation Complete\]|$)", out, re.S)]
print(f"K session: requests completed={len(answers)}/5 | layered fallback={fell_back} | errors={errors} | arena releases={released} | rebuild ms={rebuilt}")
print(f"GPU used peak {peak/1024:.2f} GiB -> min free {(total-peak)/1024:.2f} GiB (250 ms samples)")
print(f"prefill (tokens, s): {prefill}")
print(f"decode tok/s per request: {[round(r, 1) if r else None for r in rates]}")
for i, a in enumerate(answers, 1): print(f"  answer {i}: {a!r}")
sys.exit(1 if fell_back or errors or len(answers) != 5 else 0)
EOF
