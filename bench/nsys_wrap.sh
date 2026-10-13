#!/usr/bin/env bash
# Engine wrapper for glm_decode_profile.sh: runs the engine under nsys for a window of the decode phase.
#   NSYS_DELAY (s, default 100) NSYS_DURATION (s, default 8) NSYS_OUT (report base name)
exec nsys profile -t cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true \
  --delay="${NSYS_DELAY:-100}" --duration="${NSYS_DURATION:-8}" \
  -o "${NSYS_OUT:-/tmp/glm_nsys}" "${NSYS_REAL_BIN:?}" "$@"
