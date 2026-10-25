#!/usr/bin/env bash
# regression_gate.sh — run each supported model's canonical decode recipe and check
# its throughput against a frozen baseline band.
#
# This is the "override guard": it exists so that tuning one model cannot silently
# regress another. Each model is run once with the recipe recorded in
# docs/models/<model>.md; the decode rate is compared to a frozen [min,max] band.
#
# WHEN TO RUN (tiered policy — a full run is ~15 min and is NOT needed for every change):
#   * Non-graph change (tokenizer, config parsing, API server, docs, CPU-only kernels
#     unrelated to the graph): no regression gate. Run `make test-components` instead.
#   * Graph change in ONE model (model A): before merge, run at least ONE OTHER model as
#     a witness — cross-model regression is the risk, and one witness catches shared-
#     machinery breakage cheaply. Fast witnesses: `--models qwen38` (single 3090) or
#     `--models ling31`.
#   * Major change / release candidate: run all five (the default).
#
# Usage:
#   bench/regression_gate.sh [BIN] [--models qwen38,glm53flash,ling31,dsv41,glm53,mimo26]
#   bench/regression_gate.sh [BIN] --help
#
#   BIN        defaults to ./build_master/zeromoe (the merged-master release build;
#              must link the prebuilt ggml with the host-node hook — see docs/models/README.md)
#   --models   comma-separated subset (test only these models); default is all five.
#              Every selected model is REQUIRED: a launch error or a missing decode
#              measurement is a FAIL, not a skip.
#
# Preconditions (enforced by the individual run scripts where they exist):
#   * no other zeromoe / llama.cpp process, GPUs idle
#   * enough free RAM (some recipes need host_ram_gb + 21.5 GiB margin; the gate uses
#     --force-start under scripts/run_capped.sh where the conservative margin would refuse)
#   * set MIMO26_MODEL when selecting the mimo26 recipe
#
# Coherence is NOT auto-checked: these models are non-deterministic under near-tie
# routing (a tiny numeric change flips an expert and the text diverges while staying
# coherent). The gate checks throughput only; read the logged output text yourself,
# or use the teacher-forced A/B harnesses (bench/graph_framework_step.sh,
# bench/graph_framework_glm_gate.sh) for a stronger correctness check.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bench/regression_gate.sh [BIN] [--models a,b,...] [--help]

Tiered policy (a full run is ~15 min; not needed for every change):
  * Non-graph change (tokenizer/config/docs/CPU-only kernels): no gate — run `make test-components`.
  * Graph change in one model: before merge, run ONE OTHER model as a witness.
    Fast witnesses: --models qwen38 (single 3090) or --models ling31.
  * Major change / release candidate: run all five (default).

Models: qwen38, glm53flash, ling31, dsv41, glm53, mimo26
EOF
}

cd "$(dirname "$0")/.."   # repository root

BIN="${GATE_BIN:-./build_master/zeromoe}"
MODELS="${MODELS-qwen38,glm53flash,ling31,dsv41,glm53,mimo26}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --bin) BIN="${2:?--bin needs a path}"; shift 2 ;;
    --models) MODELS="${2:?--models needs a comma list}"; shift 2 ;;
    -*) echo "unknown flag: $1 (use --help)" >&2; exit 2 ;;
    *) BIN="$1"; shift ;;   # first non-flag positional is the candidate binary
  esac
done
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"

LOG_DIR="${GATE_LOG_DIR:-logs/regression_gate}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"

# Frozen baseline bands: [min, max] decode tok/s.
#
# Baseline policy: each floor comes from the median of the current best run(s),
# not from the latest passing run. Never lower a floor to admit a regression;
# re-baseline deliberately with a recorded measurement (date, commit, config,
# N runs). Models without an explicit metric below keep the historical
# whole-request output rate; glm53flash uses the steady-state median so its
# floor is directly comparable to docs/GLM5.3-Flash-baseline.md.
declare -A MIN MAX METRIC BASELINE TOL
MIN[ling31]=9.0 ; MAX[ling31]=11.5        # request output; 10.23 measured; doc 10.15
# dsv41 re-baselined 2026-10-22 after 0c0b422 (async demand H2D), on the
# canonical +10 GB mirror config: 64-token DNS gate prompt, 1 async + 1 sync
# run = 4.04 / 3.96 tok/s; 150-token prompt 061, 3 interleaved pairs =
# 4.31 / 4.24. Floor is ~13% below the slower 64-token measurement; the old
# 3.2 ceiling was below the documented 4.0-4.5 steady range.
MIN[dsv41]=3.5 ; MAX[dsv41]=4.6           # request output; 4.04 async / 3.96 sync measured; doc 4.0-4.5 steady
MIN[glm53]=1.8 ; MAX[glm53]=2.4           # request output; 2.14 measured; doc 2.00-2.09
# mimo26 (2026-10-23): mean over the held-out prompts 13.4-16.7 tok/s decode, 0 SSD reads. Floor 12.0 is ~10% under the
# slowest measured prompt; ceiling is loose. Extra gates in run_mimo26: 0 decode store misses/loads, no coverage
# preflight finding, prefill >= 20 tok/s on the short prompt.
MIN[mimo26]=12.0 ; MAX[mimo26]=19.0

# qwen38: request output, split 22/26. Long-tested 27-28 tok/s; idle ABBA
# (2026-10-21) 26.47-26.81. Baseline 27.0 with a 3% noise allowance -> 26.19.
# The 25.83 run occurred at loadavg 2.19 and must fail, not silently pass.
METRIC[qwen38]=request
BASELINE[qwen38]=27.0
TOL[qwen38]=0.03
MIN[qwen38]=$(awk -v b="${BASELINE[qwen38]}" -v t="${TOL[qwen38]}" \
  'BEGIN{printf "%.2f", b*(1-t)}')
MAX[qwen38]=30.0

# glm53flash: steady median, steps >= 20. Current best runs are 7.29 and 7.26
# tok/s (lane-off/automatic normal graph); baseline 7.275, 4% noise allowance.
# The fixed automatic 300-token run measured 7.19; the lane-on regression was
# 6.49-6.51, which this floor must reject.
METRIC[glm53flash]=steady
BASELINE[glm53flash]=7.275
TOL[glm53flash]=0.04
MIN[glm53flash]=$(awk -v b="${BASELINE[glm53flash]}" -v t="${TOL[glm53flash]}" \
  'BEGIN{printf "%.2f", b*(1-t)}')
MAX[glm53flash]=8.5

# Known model names (must match the run_* functions below). A typo in --models must fail
# the gate loudly, never silently select nothing and report a successful empty run.
KNOWN_MODELS=(qwen38 glm53flash ling31 dsv41 glm53 mimo26)
if [[ -z "$MODELS" ]]; then
  echo "FAIL  no models selected (--models is empty)" >&2
  exit 2
fi
# `read` drops a trailing empty field, so check the raw list shape explicitly; otherwise
# `--models qwen38,` would silently run only qwen38 instead of rejecting the empty element.
if [[ "$MODELS" == ,* || "$MODELS" == *, || "$MODELS" == *,,* ]]; then
  echo "FAIL  empty model name in --models '$MODELS' (comma list has an empty element)" >&2
  exit 2
fi
IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
SELECTED=0
for m in "${MODEL_LIST[@]}"; do
  if [[ -z "$m" ]]; then
    echo "FAIL  empty model name in --models '$MODELS' (comma list has an empty element)" >&2
    exit 2
  fi
  ok=0
  for k in "${KNOWN_MODELS[@]}"; do
    [[ "$m" == "$k" ]] && { ok=1; break; }
  done
  if [[ "$ok" != "1" ]]; then
    echo "FAIL  unknown model '$m' (known: ${KNOWN_MODELS[*]})" >&2
    exit 2
  fi
  SELECTED=$((SELECTED+1))
done
if [[ "$SELECTED" -eq 0 ]]; then
  echo "FAIL  no models selected by --models '$MODELS'" >&2
  exit 2
fi

PASS=0; FAIL=0

# Grab the steady-state decode rate (tok/s): median of [TOKEN-LOOP-DIAG]
# iter_total over steps >= start. This is the metric used in the model baselines
# (e.g. docs/GLM5.3-Flash-baseline.md); it excludes the first-token/setup ramp
# that makes a short whole-request average look slower and noisier.
# Prints "0" when nothing matches.
grab_steady_rate() {
  local f="$1" start="${2:-20}"
  awk -v start="$start" '
    {
      line=$0
      if (match(line, /\[TOKEN-LOOP-DIAG\] step=[0-9]+ iter_total=[0-9.]+ms/)) {
        s=line; sub(/.*\[TOKEN-LOOP-DIAG\] step=/, "", s); sub(/ .*/, "", s); step=s+0
        t=line; sub(/.*iter_total=/, "", t); sub(/ms.*/, "", t); ms=t+0
        if (step >= start) a[++n]=ms
      }
    }
    END {
      if (n==0) { print "0"; exit }
      for (i=1;i<=n;i++) for (j=i+1;j<=n;j++) if (a[j]<a[i]) { tmp=a[i]; a[i]=a[j]; a[j]=tmp }
      med=(n%2) ? a[(n+1)/2] : (a[n/2]+a[n/2+1])/2
      printf "%.2f\n", 1000/med
    }' "$f"
}

# Grab the decode rate (tok/s) from a log. Tries, in order:
#   [REQUEST-DECODE-OUTPUT] ... rate=N   (first output to last output; excludes one-time setup)
#   [REQUEST-DECODE-WALL]   ... rate=N   (includes one-time decode-graph setup)
#   Rates: ... decode=N
# Prints "0" when nothing matches.
grab_rate() {
  local f="$1" r=""
  r="$(sed -nE 's/.*\[REQUEST-DECODE-OUTPUT\].*rate=([0-9.]+).*/\1/p' "$f" 2>/dev/null | tail -1)"
  [[ -z "$r" ]] && r="$(sed -nE 's/.*\[REQUEST-DECODE-WALL\].*rate=([0-9.]+).*/\1/p' "$f" 2>/dev/null | tail -1)"
  [[ -z "$r" ]] && r="$(sed -nE 's/.*Rates:.*decode=([0-9.]+).*/\1/p' "$f" 2>/dev/null | tail -1)"
  echo "${r:-0}"
}

# Completion evidence: whole-request generated-token count. Prints "0" when nothing matches.
grab_tokens() {
  local f="$1" t=""
  t="$(sed -nE 's/.*\[REQUEST-DECODE-WALL\].*tokens=([0-9]+).*/\1/p' "$f" 2>/dev/null | tail -1)"
  [[ -z "$t" ]] && t="$(sed -nE 's/.*\[REQUEST-DECODE-OUTPUT\].*tokens=([0-9]+).*/\1/p' "$f" 2>/dev/null | tail -1)"
  echo "${t:-0}"
}

# Select the metric declared for the model. Models without an explicit metric
# keep the historical whole-request output rate; glm53flash uses the steady
# median so the documented 7.26 tok/s baseline is directly comparable.
grab_metric() {
  local name="$1" f="$2"
  if [[ "${METRIC[$name]:-request}" == "steady" ]]; then
    grab_steady_rate "$f" 20
  else
    grab_rate "$f"
  fi
}

# name rate rc teardown-log tokens expected_tokens
report() {
  local name="$1" rate="$2" rc="$3" log="${4:-}" tokens="${5:-0}" expected="${6:-0}"
  if [[ "$rate" == "0" ]]; then
    printf "FAIL  %-12s no decode rate measured (launch/parse error, rc=%s)\n" "$name" "$rc"
    FAIL=$((FAIL+1)); return
  fi
  local lo="${MIN[$name]}" hi="${MAX[$name]}" note=""
  if [[ "$rc" != "0" ]]; then
    # 124 = timeout(1); >=128 = killed by a signal. 134 (SIGABRT) is the one exception:
    # ggml's CUDA teardown calls abort(), so it may be admitted only by the completion
    # evidence below. 137 (SIGKILL) and every other signal status remain failures.
    if [[ "$rc" == "124" || ( "$rc" -ge 128 && "$rc" != "134" ) ]]; then
      printf "FAIL  %-12s %6.2f tok/s  (rc=%s: timeout/signal termination)\n" "$name" "$rate" "$rc"
      FAIL=$((FAIL+1)); return
    fi
    # A nonzero exit is tolerable ONLY when it is the recognized teardown
    # abort/crash (§3.7 "driver shutting down") AFTER generation completed: the log must
    # carry that signature AND the request summary must show the expected token count.
    # A partial result plus the teardown string is still a FAIL.
    if [[ -n "$log" ]] && grep -q "driver shutting down" "$log" 2>/dev/null \
       && (( expected > 0 && tokens >= expected )); then
      note="  [rc=$rc: recognized teardown abort after $tokens/$expected tokens]"
    else
      printf "FAIL  %-12s %6.2f tok/s  (rc=%s, not completed generation + teardown evidence; %s/%s tokens)\n" \
        "$name" "$rate" "$rc" "$tokens" "$expected"
      FAIL=$((FAIL+1)); return
    fi
  else
    # A zero exit is not by itself evidence of completion either: reject a rate measured
    # over a partial generation (the failure mode behind the two-token false PASS).
    if (( expected > 0 && tokens < expected )); then
      printf "FAIL  %-12s %6.2f tok/s  (incomplete generation: %s/%s tokens)\n" \
        "$name" "$rate" "$tokens" "$expected"
      FAIL=$((FAIL+1)); return
    fi
  fi
  if awk -v r="$rate" -v a="$lo" -v b="$hi" 'BEGIN{exit !(r>=a && r<=b)}'; then
    printf "PASS  %-12s %6.2f tok/s  (band %.1f-%.1f)%s\n" "$name" "$rate" "$lo" "$hi" "$note"
    PASS=$((PASS+1))
  else
    printf "FAIL  %-12s %6.2f tok/s  (band %.1f-%.1f)%s  <- INVESTIGATE\n" "$name" "$rate" "$lo" "$hi" "$note"
    FAIL=$((FAIL+1))
  fi
}

wants() { [[ ",$MODELS," == *",$1,"* ]]; }

# A neutral-named symlink to BIN: bench/glm53/preflight.sh greps process command
# lines for the literal string "zeromoe", so a wrapper that spells the path
# ./build_master/zeromoe trips its own guard. Run through a symlink instead.
# Keep it in an invocation-private directory: a shared /tmp name lets a concurrent
# gate replace the link mid-run, so gate A would execute candidate B while reporting
# candidate A's hash.
GATE_TMP="$(mktemp -d "${TMPDIR:-/tmp}/zeromoe-gate.XXXXXX")"
trap 'rm -rf "$GATE_TMP"' EXIT
NBIN="$GATE_TMP/zeromoe-gate-bin"
ln -sfn "$BIN" "$NBIN"

echo "== ZeroMoE regression gate ($STAMP) =="
echo "   binary: $BIN  sha256=$(sha256sum "$BIN" | awk '{print $1}')"
echo "   git:    $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "   models: $MODELS"
echo "   (each model takes 1-4 min; a full gate is ~15 min. Read the logged output for"
echo "    coherence — this gate checks throughput only.)"
echo

# ---------------------------------------------------------------------------
# qwen38 — Qwen 3.8 Flash (Qwen4Exp), split 22 GPU / 26 CPU layers, single 3090.
# ---------------------------------------------------------------------------
run_qwen38() {
  local log="$LOG_DIR/qwen38.$STAMP.log" eng="logs/qwen_decode_split_$STAMP.engine.log" rc=0
  echo "--- qwen38: split 22/26, 400 tokens (single 3090) ---"
  ZEROMOE_BIN="$BIN" QWN_STAMP="$STAMP" bench/qwen_decode_baselines.sh split 10 400 > "$log" 2>&1 || rc=$?
  report qwen38 "$(grab_rate "$eng")" "$rc" "$log" "$(grab_tokens "$eng")" 400
}

# ---------------------------------------------------------------------------
# glm53flash — GLM 5.3 Flash (GLM5NEXT), whole-layer placement, single 3090.
# ---------------------------------------------------------------------------
run_glm53flash() {
  local log="$LOG_DIR/glm53flash.$STAMP.log" eng="$LOG_DIR/glm53flash.$STAMP.engine.log" rc=0
  echo "--- glm53flash: wholelayer8, 150 tokens (single 3090) ---"
  echo "    metric: steady median (TOKEN-LOOP-DIAG, steps >= 20); baseline 7.275, floor 6.98"
  echo "Explain the difference between CPU and GPU architecture in terms of parallelism, memory hierarchy and scheduling, in detail." \
    | env CUDA_VISIBLE_DEVICES=0 ZEROMOE_GLM5NEXT_FULL_GRAPH=1 \
        ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB=512 ZEROMOE_GLM5NEXT_CPU_THREADS=6 \
        ZEROMOE_TOKEN_LOOP_DIAG=1 \
        scripts/run_capped.sh -- "$BIN" \
        --config configs/bench/glm53flash_AJ_IQ2XXS_wholelayer8_cpu.toml \
        --log-path "$eng" \
        --temp 0 --ngen 150 --force-start > "$log" 2>&1 || rc=$?
  report glm53flash "$(grab_metric glm53flash "$eng")" "$rc" "$log" "$(grab_tokens "$eng")" 150
}

# ---------------------------------------------------------------------------
# ling31 — Ling 3.1 Flash (BailingMoE3), no-copy offload graph, dual GPU.
# ---------------------------------------------------------------------------
run_ling31() {
  local log="$LOG_DIR/ling31.$STAMP.log" eng="/tmp/ling-gate.$STAMP.log" rc=0
  echo "--- ling31: short decode, 32 tokens (dual GPU) ---"
  LING_DDR_GIB=72 LING_CAP_GIB=95 LING_STOP_GIB=20 \
  LING_GPU0_CACHE_GIB=3.0 LING_GPU1_CACHE_GIB=12.5 \
  LING_PINNED_VRAM_CAPACITY=3700 LING_OFFLOAD_PINNED=0.5 \
  LING_GPU_ASSIST=3 LING_ASSIST2_SLOTS=1 LING_CPU_THREADS=10 \
  LING_USAGE_FILE="$PWD/bench/ling/reference-audit/usage-hi-there-robust4.pin" \
  LING_LOG="$eng" LING_BIN="$BIN" \
  bench/ling/reference-audit/run.sh > "$log" 2>&1 || rc=$?
  report ling31 "$(grab_rate "$eng")" "$rc" "$log" "$(grab_tokens "$eng")" 32
}

# ---------------------------------------------------------------------------
# mimo26 — MiMo V2.6 Flash, in-graph hot/cold MoE, dual GPU, 100 GiB DDR tier (needs the model drive mounted).
# One fixed prompt, 64 greedy tokens. Beyond the decode band it fails on: any expert load during decode, a coverage
# or idle-VRAM preflight finding, or prefill under 20 tok/s.
# ---------------------------------------------------------------------------
run_mimo26() {
  local log="$LOG_DIR/mimo26.$STAMP.log" eng="$LOG_DIR/mimo26.$STAMP.engine.log" rc=0
  local model=${MIMO26_MODEL:?set MIMO26_MODEL to the MiMo model file or shard directory}
  local config="$LOG_DIR/mimo26.$STAMP.toml"
  echo "--- mimo26: DNS prompt, 64 tokens (dual GPU, 100 GiB DDR) ---"
  if [[ ! -r "$model" ]]; then
    printf "FAIL  %-12s model file not readable: %s\n" mimo26 "$model"; FAIL=$((FAIL+1)); return
  fi
  python3 - "configs/bench/mimo26_flash_gsq_rco_3bit_ingraph_dual.toml" "$config" "$model" <<'PY'
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
  echo "Explain what DNS is and how a domain name gets resolved to an IP address." \
    | env CUDA_VISIBLE_DEVICES=0,1 ZEROMOE_DISABLE_ROUTER_BF16=1 ZEROMOE_MIMO2_OFFLOAD_DDR_GIB=100 \
        ZEROMOE_MIMO2_OFFLOAD_MODEL="$model" \
        ZEROMOE_MIMO2_OFFLOAD_USAGE="$PWD/bench/profiles/mimo26_combined.usage" \
        scripts/run_capped.sh --limit-gb 110 -- "$BIN" \
        --config "$config" \
        --log-path "$eng" --temp 0 --ngen 64 > "$log" 2>&1 || rc=$?
  local before=$FAIL
  report mimo26 "$(grab_rate "$eng")" "$rc" "$log" "$(grab_tokens "$eng")" 64
  (( FAIL > before )) && return
  local loads prefill
  loads="$(sed -nE 's/.*\[MIMO2-LANE\].*loads=([0-9]+).*/\1/p' "$eng" | tail -1)"
  if [[ -z "$loads" || "$loads" != "0" ]]; then
    printf "FAIL  %-12s decode-time expert loads = %s (expected 0: the DDR tier must cover the working set)\n" mimo26 "${loads:-unknown}"
    PASS=$((PASS-1)); FAIL=$((FAIL+1)); return
  fi
  if grep -q '\[PREFLIGHT\] mimo2:' "$eng"; then
    printf "FAIL  %-12s preflight finding: %s\n" mimo26 "$(grep -m1 '\[PREFLIGHT\] mimo2:' "$eng" | sed 's/.*mimo2: //')"
    PASS=$((PASS-1)); FAIL=$((FAIL+1)); return
  fi
  prefill="$(sed -nE 's/.*\[PREFILL-DONE\] [0-9]+ tokens in [0-9.]+s \(([0-9.]+) tok\/s\).*/\1/p' "$eng" | head -1)"
  if [[ -z "$prefill" ]] || ! awk -v p="$prefill" 'BEGIN{exit !(p>=20.0)}'; then
    printf "FAIL  %-12s prefill %s tok/s (floor 20)\n" mimo26 "${prefill:-unknown}"
    PASS=$((PASS-1)); FAIL=$((FAIL+1)); return
  fi
  echo "      mimo26: 0 decode loads, no preflight findings, prefill ${prefill} tok/s"
}

# ---------------------------------------------------------------------------
# dsv41 — DeepSeek V4.1 Flash (DeepSeek_V41), cache-backed graph, dual GPU.
# ---------------------------------------------------------------------------
run_dsv41() {
  local log="$LOG_DIR/dsv41.$STAMP.log" eng="$LOG_DIR/dsv41.$STAMP.engine.log" rc=0
  echo "--- dsv41: DNS prompt, 64 tokens (dual GPU) ---"
  echo "Explain what DNS is and how a domain name gets resolved to an IP address." \
    | env CUDA_VISIBLE_DEVICES=0,1 ZEROMOE_DEEPSEEK_V41_FULL_GRAPH=1 \
        scripts/run_capped.sh -- "$BIN" \
        --config configs/bench/deepseek_v41_100gib_ddr_17gib_vram_dual_gpu.toml \
        --log-path "$eng" \
        --temp 0 --ngen 64 --force-start > "$log" 2>&1 || rc=$?
  report dsv41 "$(grab_rate "$eng")" "$rc" "$log" "$(grab_tokens "$eng")" 64
}

# ---------------------------------------------------------------------------
# glm53 — GLM 5.3 (GLM_DSA), SSD-offloaded experts, dual GPU.
# ---------------------------------------------------------------------------
run_glm53() {
  local tag="gate-$STAMP" log="$LOG_DIR/glm53.$STAMP.log" rc=0
  local eng="logs/split_ab_$tag.log" teardown="logs/split_ab_$tag.stdout"
  echo "--- glm53: SSD offload, 80 tokens (dual GPU) ---"
  rm -f "logs/split_ab_$tag.log" "$teardown"
  systemd-run --user --scope -p MemorySwapMax=0 \
    env ZEROMOE_BIN="$NBIN" \
    bench/glm53/run_ab.sh "$tag" \
    ZEROMOE_GLM5NEXT_GPU1_HOT_GIB=22 OFFLOAD_DDR_GIB=64 NGEN=80 \
    ZEROMOE_GLM5NEXT_PILOT_PREFETCH=3 ZEROMOE_GLM5NEXT_PILOT_STAGE=3 \
    ZEROMOE_GLM5NEXT_MODEL_DIRECT=1 > "$log" 2>&1 || rc=$?
  # The engine log carries the request summary; the .stdout capture carries the
  # ggml/CUDA teardown text (the wrapper itself only greps selected lines).
  report glm53 "$(grab_rate "$eng")" "$rc" "$teardown" "$(grab_tokens "$eng")" 80
}

wants qwen38      && run_qwen38
wants glm53flash  && run_glm53flash
wants ling31      && run_ling31
wants dsv41       && run_dsv41
wants mimo26      && run_mimo26
wants glm53       && run_glm53

echo
echo "== summary: $PASS passed, $FAIL failed =="
# Every selected model must produce a result: an empty or short run is a FAIL, not a pass.
if [[ $((PASS + FAIL)) -ne "$SELECTED" ]]; then
  echo "FAIL  selected $SELECTED model(s) but only $((PASS + FAIL)) produced a result" >&2
  exit 1
fi
[[ "$FAIL" -eq 0 ]] || exit 1
