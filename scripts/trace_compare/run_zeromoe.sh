#!/usr/bin/env bash
# Run ZeroMoE on a fixed prompt, emitting the per-layer residual-norm trace.
#
# Produces lines like:
#   [VERBOSE] [TRACE l=N] in       |hidden|=X
#   [VERBOSE] [TRACE l=N] post-attn|hidden|=X
#   [VERBOSE] [TRACE l=N] post-ffn |hidden|=X
# (one triplet per layer, for the LAST token of the prefill batch).
#
# The [TRACE] lines are gated on log_level VERBOSE (the norm is only even
# computed at that level), so --verbosity 4 is required. If your config
# file forces a different log_level, this CLI flag must win — if no TRACE lines
# appear in the output, check the config isn't pinning log_level below VERBOSE.
set -euo pipefail

CONFIG="${ZEROMOE_CONFIG:-configs/testing/glm52_dual.toml}"   # <-- the GLM-5.2 config you run with
BINARY="${ZEROMOE_BIN:-./build/zeromoe}"
PROMPT="${1:-Hello}"
OUT="${2:-ms_trace.log}"

# --predict 1 : generate one token, then stdin EOF exits the REPL cleanly.
# --temp 0 : greedy / argmax — deterministic, no sampling noise.
# The diff script takes the FIRST occurrence of each (layer, stage), i.e. the
# prefill pass, so the single decode step's trace lines are ignored.
printf '%s\n' "$PROMPT" | "$BINARY" --config "$CONFIG" \
    --predict 1 --temp 0 --verbosity 4 --log-file "$OUT" >&2

n=$(grep -c 'TRACE l=' "$OUT" || true)
echo "ZeroMoE trace -> $OUT ($n trace lines)" >&2
echo "$OUT"
