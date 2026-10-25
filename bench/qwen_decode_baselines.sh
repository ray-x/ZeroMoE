#!/bin/bash
# Qwen 3.8 Flash decode baselines on a single RTX 3090 (see docs/qwen-decode-baselines-2026-10-16.md).
#
# usage: bench/qwen_decode_baselines.sh split [threads=10] [ntok=400]   # 22 GPU layers + 26 CPU layers, UD-IQ1_M
#        bench/qwen_decode_baselines.sh hybrid [K=14000] [ntok=200] [threads=10]   # usage-ranked hot experts in VRAM, Q2_0
# Writes logs/qwen_decode_<mode>_<stamp>.log (stdout) and logs/qwen_decode_<mode>_<stamp>.engine.log (engine log).
# Waits for an idle engine/GPU first. Prints the rate line; read the text/log before trusting it.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:?usage: $0 split|hybrid ...}
# Unique (second-resolution) stamp, overridable so the gate can pass its own per-run stamp and
# never reuse a same-minute log from a prior invocation (stale-rate false PASS).
STAMP=${QWN_STAMP:-$(date +%Y%m%d_%H%M%S)}
mkdir -p logs
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
BIN=${ZEROMOE_BIN:-./build_native/zeromoe}

while pgrep -x zeromoe >/dev/null || pgrep -x llama-bench >/dev/null; do sleep 3; done
echo "loadavg before: $(cut -d' ' -f1-3 /proc/loadavg)  (a decode run is barrier-bound; >2 means something else is using cores)"

case "$MODE" in
  split)
    THREADS=${2:-10}; NTOK=${3:-400}
    LOG=logs/qwen_decode_split_$STAMP
    rm -f "$LOG.log" "$LOG.engine.log"
    # context 416 = 16-token prompt + 400 tokens; a longer prompt+gen pushes the last token onto the legacy path and fakes a gap
    sed -E "s/^omp_max_threads.*/omp_max_threads = $THREADS/; s/^max_new_tokens.*/max_new_tokens = $NTOK/; s#^log_path.*#log_path = \"$LOG.engine.log\"#" \
      configs/bench/qwen4exp_tg400_iq1m_22_26.toml > "$TMP/cfg.toml"
    # "Tell a story." is 4 tokens; with the chat template the prompt is 16 tokens ([PREFILL-DONE] must say 16)
    rc=0
    printf 'Tell a story.\n:q\n' | env CUDA_VISIBLE_DEVICES=0 \
      ZEROMOE_QWEN4EXP_FULL_GRAPH=1 ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE=1 \
      ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS=40 ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1 \
      ZEROMOE_BENCH_SILENT=1 ZEROMOE_BENCH_IGNORE_STOP=1 \
      timeout 900 "$BIN" --config "$TMP/cfg.toml" > "$LOG.log" 2>&1 || rc=$?
    grep -E "PREFILL-DONE\] 16|REQUEST-DECODE-WALL" "$LOG.engine.log" | cut -c1-170 || true
    # Emit teardown evidence so the gate can recognize a benign teardown crash (§3.7) rather
    # than treat any nonzero exit as benign.
    grep -m1 "driver shutting down" "$LOG.log" || true
    exit "$rc"
    ;;
  hybrid)
    # K=14000, not the 14554 of docs/qwen-decode-baselines-2026-10-16.md: with the pinned graph-RAM arenas (the Qwen default)
    # 14554 leaves 0.98 GiB free after a long prefill and the decode graph is refused (floor 1.00 GiB). See configs/bench/perf-baselines.md.
    K=${2:-14000}; NGEN=${3:-200}; THREADS=${4:-10}
    : "${ZEROMOE_MODEL_PATH:?set ZEROMOE_MODEL_PATH to the Qwen3.8-Flash-Next Q2_0 model directory (the configs ship a placeholder)}"
    LOG=logs/qwen_decode_hybrid_K${K}_$STAMP
    sed -E "s/^qwen4exp_option_b_global_top_k *=.*/qwen4exp_option_b_global_top_k = $K/; s/^omp_max_threads.*/omp_max_threads = $THREADS/; s#^log_path *=.*#log_path = \"$LOG.engine.log\"#" \
      configs/qwen38-flash.toml > "$TMP/cfg.toml"
    nvidia-smi --id=0 --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits -lms 250 > "$LOG.csv" &
    SP=$!
    echo "Explain briefly the key concepts of neural networks and gradient descent in plain English." | env CUDA_VISIBLE_DEVICES=0 \
      ZEROMOE_QWEN4EXP_FULL_GRAPH=1 ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE=1 \
      ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS=0 ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1 \
      ZEROMOE_TOKEN_LOOP_DIAG=1 ZEROMOE_BENCH_IGNORE_STOP=1 \
      "$BIN" --config "$TMP/cfg.toml" --ngen "$NGEN" > "$LOG.log" 2>&1 || true
    kill $SP 2>/dev/null || true
    python3 - "$LOG" <<'EOF'
import re, statistics as st, sys
log = open(sys.argv[1] + ".engine.log", errors="replace").read()
steps = [float(m.group(2)) for m in re.finditer(r"step=(\d+) iter_total=([\d.]+)ms", log) if int(m.group(1)) >= 3]
print(f"decode: {1000/st.mean(steps):.2f} tok/s over {len(steps)} steps" if steps else "decode: NO STEPS")
print("FELL BACK TO LAYERED DECODE:", "layered decode remains active" in log or "falling back to the legacy path" in log)
EOF
    ;;
  *) echo "mode must be split or hybrid" >&2; exit 2;;
esac
