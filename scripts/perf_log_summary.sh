#!/usr/bin/env bash
# Pull just the handful of log lines this round's disk-I/O / prefetch
# investigation (direct_io, io_uring, multi-threaded prefetcher, pilot
# gate+bias fix) has been tracking, instead of pasting a full debug.log.
#
# Usage:
#   scripts/perf_summary.sh debug.log
#   ./build/moestream --config configs/foo.toml ... 2>&1 | scripts/perf_summary.sh
set -euo pipefail

if [ "$#" -gt 1 ]; then
    echo "usage: $0 [log_path]" >&2
    echo "  (reads the given log file; omit to read from stdin instead)" >&2
    exit 2
fi

LOG="${1:-/dev/stdin}"
if [ "$LOG" != "/dev/stdin" ] && [ ! -r "$LOG" ]; then
    echo "$0: cannot read '$LOG' (no such file, or not readable)" >&2
    exit 1
fi

echo "== one-time config confirmation (only worth checking after a config change) =="
grep -E '\[CONFIG\] (Direct I/O|io_uring|Prefetch worker threads)' "$LOG" || true
grep -E '\[DIRECT-IO\] Enabled|\[IO-URING\] Enabled|\[I/O\] Prefetcher:' "$LOG" || true

echo
echo "== per-request =="
grep '\[REQUEST\]' "$LOG" || true

echo
echo "== performance report: disk / cache / PIPE =="
grep -E \
  'Disk read \(expert\)|MoE expert FFN|Expert cache:|Prefetch \(PIPE\)|PIPE by phase|no-match detail|Prefetch pickup|io_uring batches|Pilot recall|Staged-unused evict|Throughput:' \
  "$LOG" || true
