#!/usr/bin/env bash
# run_capped.sh -- launch moestream under a REAL, kernel-enforced memory cap.
#
# WHY THIS EXISTS: a 2026-09-09 incident. A 480-token full-graph prefill on
# this box crashed the WHOLE MACHINE hard enough to need a reboot (no
# surviving kernel/OOM log -- this box has no persistent journald storage,
# so the exact mechanism was never confirmed). The application-level
# defenses that already existed (MemoryGuard's admission ledger,
# startup_preflight.hpp's one-time check) are SOFTWARE checks: they only
# protect an allocation that remembers to call them. This script adds a
# REAL, kernel-enforced ceiling on top, via a cgroup, that catches
# EVERYTHING the process (or any of its threads) does, with no cooperation
# required from the code at all.
#
# THE PART THAT ACTUALLY MATTERS, confirmed by direct testing on this box
# before this script was written (not assumed): `systemd-run -p
# MemoryMax=N` ALONE does NOT reliably kill a runaway process here. This
# machine's only swap is zram, and cgroup v2 will happily page a cgroup's
# anonymous memory out to swap once it hits MemoryMax instead of invoking
# the OOM killer -- `memory.max` events fire, `memory.current` stays
# capped, but `oom_kill` stays at 0 and the process keeps right on
# allocating (verified: it grew to 9.8GB against a 300MB cap and was never
# killed). Since this project's own CLAUDE.md already documents zram
# compressing already-quantized weights at ~1.0x (real CPU cost, ZERO real
# capacity gained), that "success" is actually the exact pathological-
# thrashing failure mode this script exists to prevent, just hidden behind
# swap instead of an immediate, clean kill. Adding `MemorySwapMax=0`
# alongside `MemoryMax` closes this: verified on this box, the same test
# was killed (SIGKILL) within about a second of crossing the real limit.
# Never use MemoryMax here without MemorySwapMax=0 -- one without the
# other is worse than neither (it hides the failure instead of catching it).
#
# USAGE:
#   scripts/run_capped.sh [--limit-gb N] -- <moestream args...>
#
# Defaults to 10 GB above whatever host_ram_gb the target config declares
# (parsed with a plain grep -- good enough for this project's own simple
# TOML files, not a general parser), or 110 GB if it can't find one. This
# mirrors this project's own "leave 10 GB of real headroom" policy
# (MemoryGuard's own watermarks, startup_preflight.hpp's margin) applied at
# the kernel level instead of the application level.
set -euo pipefail

LIMIT_GB=""
ARGS=()
CONFIG_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit-gb)
      LIMIT_GB="$2"
      shift 2
      ;;
    --)
      shift
      ARGS+=("$@")
      break
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

# Best-effort: find --config <path> among the passed-through args so we can
# read that config's own host_ram_gb and size the cap relative to it.
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--config" && $((i + 1)) -lt ${#ARGS[@]} ]]; then
    CONFIG_PATH="${ARGS[$((i + 1))]}"
    break
  fi
done

if [[ -z "$LIMIT_GB" ]]; then
  HOST_RAM_GB=""
  if [[ -n "$CONFIG_PATH" && -f "$CONFIG_PATH" ]]; then
    HOST_RAM_GB=$(grep -E '^\s*host_ram_gb\s*=' "$CONFIG_PATH" | head -1 | sed -E 's/^[^=]*=\s*([0-9.]+).*/\1/')
  fi
  if [[ -n "$HOST_RAM_GB" ]]; then
    # Round up host_ram_gb + 10 GB headroom to the nearest whole GB.
    LIMIT_GB=$(python3 -c "import math; print(math.ceil(float(\"$HOST_RAM_GB\") + 10))")
    echo "[run_capped] $CONFIG_PATH declares host_ram_gb=$HOST_RAM_GB -- capping at ${LIMIT_GB} GB (+10 GB headroom)" >&2
  else
    LIMIT_GB=110
    echo "[run_capped] could not read host_ram_gb from a --config argument -- defaulting to ${LIMIT_GB} GB. Pass --limit-gb N to override." >&2
  fi
fi

TOTAL_GB=$(free -g | awk '/^Mem:/{print $2}')
# python3, not bash $(( )), since LIMIT_GB can be fractional (an explicit
# --limit-gb, or a config's own host_ram_gb + 10, e.g. "0.3" or "110.5") --
# bash arithmetic expansion only handles integers and would error out here.
if [[ -n "$TOTAL_GB" && "$TOTAL_GB" -gt 0 ]]; then
  REMAIN=$(python3 -c "print(round(${TOTAL_GB} - ${LIMIT_GB}, 1))" 2>/dev/null || echo "")
  if [[ -n "$REMAIN" ]] && python3 -c "import sys; sys.exit(0 if ${REMAIN} < 10 else 1)" 2>/dev/null; then
    echo "[run_capped] WARNING: this box has ${TOTAL_GB} GB total RAM; a ${LIMIT_GB} GB cap leaves only ${REMAIN} GB for the OS and everything else. Proceeding anyway (explicit --limit-gb, or this config's own host_ram_gb, was honored), but consider lowering it." >&2
  fi
fi

echo "[run_capped] launching under systemd --user scope: MemoryMax=${LIMIT_GB}G, MemorySwapMax=0" >&2
exec systemd-run --user --scope --same-dir \
  -p "MemoryMax=${LIMIT_GB}G" \
  -p "MemorySwapMax=0" \
  -- "${ARGS[@]}"
