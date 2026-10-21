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
#   bench/regression_gate.sh [BIN] [--models qwen38,glm53flash,ling31,dsv41,glm53]
#   bench/regression_gate.sh [BIN] --help
#
#   BIN        defaults to ./build_master/moestream (the merged-master release build;
#              must link the prebuilt ggml with the host-node hook — see docs/models/README.md)
#   --models   comma-separated subset (test only these models); default is all five.
#              Every selected model is REQUIRED: a launch error or a missing decode
#              measurement is a FAIL, not a skip.
#
# Preconditions (enforced by the individual run scripts where they exist):
#   * no other moestream / llama.cpp process, GPUs idle
#   * enough free RAM (some recipes need host_ram_gb + 21.5 GiB margin; the gate uses
#     --force-start under scripts/run_capped.sh where the conservative margin would refuse)
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

Models: qwen38, glm53flash, ling31, dsv41, glm53
EOF
}

cd "$(dirname "$0")/.."   # repository root

BIN="${GATE_BIN:-./build_master/moestream}"
MODELS="${MODELS:-qwen38,glm53flash,ling31,dsv41,glm53}"
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
STAMP="$(date +%Y%m%d_%H%M%S)"

# Frozen baseline bands: [min, max] decode tok/s, measured on the merged master
# (see docs/models/README.md for the full provenance and how to re-baseline).
declare -A MIN MAX
MIN[qwen38]=22.0 ; MAX[qwen38]=30.0       # 26.03 measured; doc 28.66, spread 24.6-28.7
MIN[glm53flash]=6.5 ; MAX[glm53flash]=8.5 # 7.50 measured; doc 7.26
MIN[ling31]=9.0 ; MAX[ling31]=11.5        # 10.23 measured; doc 10.15
MIN[dsv41]=1.8 ; MAX[dsv41]=3.2           # 2.18 measured; doc 2.90 cold / 4.0-4.5 steady
MIN[glm53]=1.8 ; MAX[glm53]=2.4           # 2.14 measured; doc 2.00-2.09

# Known model names (must match the run_* functions below). A typo in --models must fail
# the gate loudly, never silently select nothing and report a successful empty run.
KNOWN_MODELS=(qwen38 glm53flash ling31 dsv41 glm53)
SELECTED=0
for m in ${MODELS//,/ }; do
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

PASS=0; FAIL=0

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

# name rate exit_code [log-with-stderr]
report() {
  local name="$1" rate="$2" rc="$3" log="${4:-}"
  if [[ "$rate" == "0" ]]; then
    printf "FAIL  %-12s no decode rate measured (launch/parse error, rc=%s)\n" "$name" "$rc"
    FAIL=$((FAIL+1)); return
  fi
  local lo="${MIN[$name]}" hi="${MAX[$name]}" note=""
  if [[ "$rc" != "0" ]]; then
    # A nonzero exit is tolerable ONLY when it is the specifically recognized teardown
    # crash (§3.7 "CUDA error: driver shutting down" in destructors AFTER generation
    # completed). Anything else — SIGKILL, an abort mid-generation, OOM — is a real
    # failure even when a rate line happened to print.
    if [[ -n "$log" ]] && grep -q "driver shutting down" "$log" 2>/dev/null; then
      note="  [rc=$rc: benign teardown crash]"
    else
      printf "FAIL  %-12s %6.2f tok/s  (rc=%s, not a recognized teardown crash)\n" "$name" "$rate" "$rc"
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
# lines for the literal string "moestream", so a wrapper that spells the path
# ./build_master/moestream trips its own guard. Run through a symlink instead.
NBIN="/tmp/moestream-gate-bin"
ln -sfn "$BIN" "$NBIN"

echo "== MoEStream regression gate ($STAMP) =="
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
  local log="$LOG_DIR/qwen38.$STAMP.log" rc=0
  echo "--- qwen38: split 22/26, 400 tokens (single 3090) ---"
  MOESTREAM_BIN="$BIN" QWN_STAMP="$STAMP" bench/qwen_decode_baselines.sh split 10 400 > "$log" 2>&1 || rc=$?
  report qwen38 "$(grab_rate "$log")" "$rc" "$log"
}

# ---------------------------------------------------------------------------
# glm53flash — GLM 5.3 Flash (GLM5NEXT), whole-layer placement, single 3090.
# ---------------------------------------------------------------------------
run_glm53flash() {
  local log="$LOG_DIR/glm53flash.$STAMP.log" rc=0
  echo "--- glm53flash: wholelayer8, 150 tokens (single 3090) ---"
  echo "Explain the difference between CPU and GPU architecture in terms of parallelism, memory hierarchy and scheduling, in detail." \
    | env CUDA_VISIBLE_DEVICES=0 MOESTREAM_GLM5NEXT_FULL_GRAPH=1 \
        MOESTREAM_GLM5NEXT_ALLOC_FLOOR_MIB=512 MOESTREAM_GLM5NEXT_CPU_THREADS=6 \
        scripts/run_capped.sh -- "$BIN" \
        --config configs/glm53flash_AJ_IQ2XXS_wholelayer8_cpu.toml \
        --temp 0 --ngen 150 --force-start > "$log" 2>&1 || rc=$?
  report glm53flash "$(grab_rate "$log")" "$rc" "$log"
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
  report ling31 "$(grab_rate "$eng")" "$rc" "$log"
}

# ---------------------------------------------------------------------------
# dsv41 — DeepSeek V4.1 Flash (DeepSeek_V41), cache-backed graph, dual GPU.
# ---------------------------------------------------------------------------
run_dsv41() {
  local log="$LOG_DIR/dsv41.$STAMP.log" rc=0
  echo "--- dsv41: DNS prompt, 64 tokens (dual GPU) ---"
  echo "Explain what DNS is and how a domain name gets resolved to an IP address." \
    | env CUDA_VISIBLE_DEVICES=0,1 MOESTREAM_DEEPSEEK_V41_FULL_GRAPH=1 \
        scripts/run_capped.sh -- "$BIN" \
        --config configs/deepseek_v41_100gib_ddr_17gib_vram_dual_gpu.toml \
        --temp 0 --ngen 64 --force-start > "$log" 2>&1 || rc=$?
  report dsv41 "$(grab_rate "$log")" "$rc" "$log"
}

# ---------------------------------------------------------------------------
# glm53 — GLM 5.3 (GLM_DSA), SSD-offloaded experts, dual GPU.
# ---------------------------------------------------------------------------
run_glm53() {
  local tag="gate-$STAMP" log="$LOG_DIR/glm53.$STAMP.log" rc=0
  echo "--- glm53: SSD offload, 80 tokens (dual GPU) ---"
  rm -f "logs/split_ab_$tag.log"
  systemd-run --user --scope -p MemorySwapMax=0 \
    env MOESTREAM_BIN="$NBIN" \
    bench/glm53/run_ab.sh "$tag" \
    MOESTREAM_GLM5NEXT_GPU1_HOT_GIB=22 OFFLOAD_DDR_GIB=64 NGEN=80 \
    MOESTREAM_GLM5NEXT_PILOT_PREFETCH=3 MOESTREAM_GLM5NEXT_PILOT_STAGE=3 \
    MOESTREAM_GLM5NEXT_MODEL_DIRECT=1 > "$log" 2>&1 || rc=$?
  report glm53 "$(grab_rate "logs/split_ab_$tag.log")" "$rc" "$log"
}

wants qwen38      && run_qwen38
wants glm53flash  && run_glm53flash
wants ling31      && run_ling31
wants dsv41       && run_dsv41
wants glm53       && run_glm53

echo
echo "== summary: $PASS passed, $FAIL failed =="
# Every selected model must produce a result: an empty or short run is a FAIL, not a pass.
if [[ $((PASS + FAIL)) -ne "$SELECTED" ]]; then
  echo "FAIL  selected $SELECTED model(s) but only $((PASS + FAIL)) produced a result" >&2
  exit 1
fi
[[ "$FAIL" -eq 0 ]] || exit 1
