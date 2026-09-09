#!/usr/bin/env bash
# Run colibri (CPU reference engine) on the same fixed prompt.
#
# colibri's residual-norm trace is UNCONDITIONALLY ON and hardcodes its output
# to /tmp/colibri_trace.log (glm.c trace_log_file(), no env var to redirect it).
# We run, then copy that file to the requested output path.
#
# colibri's `glm` binary takes the prompt from the PROMPT env var (NOT argv,
# NOT stdin). argv is purely numeric: <expert_cache_cap> <expert_bits> <dense_bits>.
# NGEN=0 = prefill only (no decode-step trace lines) — the cleanest diff.
# TEMP=0 = greedy.
set -euo pipefail

# <-- SET THESE TWO for your machine:
SNAP="${COLIBRI_SNAP:-/path/to/GLM-5.2-colibri-int4}"            # HF: jlnsrk/GLM-5.2-colibri-int4
GLM_BIN="${COLIBRI_GLM:-/Users/rayxu/project/colibri/c/glm}"

PROMPT="${1:-Hello}"
OUT="${2:-colibri_trace.log}"

if [ ! -x "$GLM_BIN" ]; then
  echo "Building colibri (CPU)..." >&2
  (cd "$(dirname "$GLM_BIN")" && make glm)
fi

rm -f /tmp/colibri_trace.log
SNAP="$SNAP" PROMPT="$PROMPT" NGEN=0 TEMP=0 "$GLM_BIN" 64 4 4 >&2 || true
# colibri may exit non-zero on some SIGPIPE paths during NGEN=0; the trace file
# is fflushed after every line, so it's complete regardless.

if [ -f /tmp/colibri_trace.log ]; then
  cp /tmp/colibri_trace.log "$OUT"
  n=$(grep -c 'TRACE l=' "$OUT" || true)
  echo "colibri trace -> $OUT ($n trace lines)" >&2
else
  echo "ERROR: /tmp/colibri_trace.log not produced — check SNAP path & build." >&2
  exit 1
fi
echo "$OUT"
