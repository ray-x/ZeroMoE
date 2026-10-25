#!/bin/bash
# Steady-state VRAM and decode speed of the Q2_0 Option-B global-pool config at a given expert budget K.
#
# usage: bench/option_b_vram_probe.sh K [ngen=200] [gpu=0]
#   K    number of experts kept in VRAM (qwen4exp_option_b_global_top_k). 8421 = the right-sized 0/44/4 count.
#   env  anything exported as ZEROMOE_* is passed through, e.g.
#        ZEROMOE_QWEN4EXP_GRAPH_ALLOC_FLOOR_MIB=512   (needed for K above ~14,729 on a 24 GiB card)
# Writes logs/optb_probe_K<K>.{csv,stdout,engine.log}. Needs an idle GPU and ../moestream-build-upstream-master/moestream.
# Reports: steady-state GPU memory used/free (nvidia-smi, 250 ms samples, "steady" = samples above 90% of the peak),
# decode tok/s (TOKEN-LOOP-DIAG iter_total, steps >= 3), whether the graph fell back to layered decode (that is a FAILURE:
# it runs all-CPU at ~0.8 s/token), and the first 120 characters of the answer. Read the text before trusting the speed.
set -euo pipefail
cd "$(dirname "$0")/.."
K=${1:?usage: $0 K [ngen] [gpu]}; NGEN=${2:-200}; GPU=${3:-0}
[[ "$K" =~ ^[0-9]+$ && "$NGEN" =~ ^[0-9]+$ && "$GPU" =~ ^[0-9]+$ ]] || { echo "K, ngen and gpu must be integers" >&2; exit 2; }
TAG=optb_probe_K$K
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
sed -E "s/^qwen4exp_option_b_global_top_k *=.*/qwen4exp_option_b_global_top_k = $K/; s#^log_path *=.*#log_path = \"logs/$TAG.engine.log\"#" \
  configs/qwen4exp_q2_0_option_b_global_topk48.toml > "$TMP/cfg.toml"
mkdir -p logs
while pgrep -x zeromoe >/dev/null || [[ -n "$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader)" ]]; do sleep 3; done
nvidia-smi --id="$GPU" --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits -lms 250 > "logs/$TAG.csv" &
SP=$!
echo "Explain briefly the key concepts of neural networks and gradient descent in plain English." | env \
  CUDA_VISIBLE_DEVICES=$GPU ZEROMOE_QWEN4EXP_FULL_GRAPH=1 ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE=1 \
  ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS=0 ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1 \
  ZEROMOE_TOKEN_LOOP_DIAG=1 ZEROMOE_BENCH_IGNORE_STOP=1 \
  ../moestream-build-upstream-master/moestream --config "$TMP/cfg.toml" --ngen "$NGEN" > "logs/$TAG.stdout" 2>&1 || true
kill $SP 2>/dev/null || true
python3 - "$TAG" <<'EOF'
import re, statistics as st, sys
tag = sys.argv[1]
rows = []
for line in open(f"logs/{tag}.csv"):
    p = [x.strip() for x in line.split(",")]
    if len(p) >= 3 and p[1].isdigit() and p[2].isdigit():
        rows.append((int(p[1]), int(p[2])))
used = [u for u, _ in rows]; total = rows[0][1]; peak = max(used)
steady = [u for u in used if u > peak * 0.9]
log = open(f"logs/{tag}.engine.log", errors="replace").read()
out = open(f"logs/{tag}.stdout", errors="replace").read()
fell_back = "layered decode remains active" in log or "falling back to the legacy path" in log
steps = [float(m.group(2)) for m in re.finditer(r"step=(\d+) iter_total=([\d.]+)ms", log) if int(m.group(1)) >= 3]
arena = re.findall(r"device expert arena=([\d.]+) GiB", log)[:1]
text = re.search(r"Output:(.*?)(\[Generation Complete\]|$)", out, re.S)
print(f"K arena GiB: {arena[0] if arena else '?'} | steady used {st.median(steady)/1024:.2f} GiB, free {(total-st.median(steady))/1024:.2f} GiB (min free {(total-peak)/1024:.2f})")
print(f"decode: {1000/st.mean(steps):.2f} tok/s over {len(steps)} steps" if steps else "decode: NO STEPS")
print(f"FELL BACK TO LAYERED DECODE: {fell_back}" + ("   <-- FAILURE (graph refused, e.g. admission floor)" if fell_back else ""))
print("text:", repr(text.group(1).strip()[:120]) if text else "NO OUTPUT")
sys.exit(1 if fell_back else 0)
EOF
