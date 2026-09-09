#!/usr/bin/env bash
# tools/repack_all.sh
#
# DEPRECATED convenience wrapper. repack_gguf itself now detects a
# directory as its first argument and walks every .gguf file in it,
# one at a time, automatically -- see repack_gguf.cpp's own directory-mode
# handling in main(). For a plain "repack everything in this directory"
# run, just call repack_gguf directly:
#
#   ./build/repack_gguf <source_dir> <output_dir> [--align N] [--max-shard-gb N] \
#       [--verify-samples N] [-d|--delete-source]
#
# That single-binary path is now the primary, tested way to do this (see
# CLAUDE.md's "repack tool" section) -- it has the exact same one-shard-at-
# a-time disk footprint, same per-shard verify-then-optionally-delete
# behavior, and same stop-on-first-failure semantics as this script, just
# without needing bash or a separate process per shard.
#
# This script is kept only because it lists files with `find`/`sort`
# itself instead of relying on repack_gguf's own std::filesystem walk --
# there's no remaining functional reason to prefer it over calling
# repack_gguf directly on a directory. Kept for now in case a difference
# in file-discovery semantics (recursion, hidden files, sort locale)
# between bash's `find`/`sort` and repack_gguf's own C++
# `fs::directory_iterator` walk ever matters to a specific setup; delete
# it once that's confirmed to never be needed.
#
# Each shard is fully independent (repack_gguf doesn't require sibling
# projections to be found in the same file -- see its own header
# comment), so shards can be processed in any order and this script's own
# "list, then loop" behavior is naturally resumable: if interrupted, just
# re-run it -- already-deleted source shards (via -d/--delete-source)
# simply won't be in the listing anymore, and re-processing a
# not-yet-deleted shard safely overwrites its own prior output
# (repack_gguf opens output files in truncate mode).
#
# Usage:
#   tools/repack_all.sh <source_dir> <output_dir> \
#       [--bin PATH] [--align N] [--max-shard-gb N] [--verify-samples N] \
#       [-d | --delete-source]
#
# -d/--delete-source is DESTRUCTIVE and forwarded straight through to
# repack_gguf -- deletion happens INSIDE that tool, immediately after its
# own internal verify pass confirms success for that specific shard, not
# via a separate check in this script. Deliberately not reimplemented
# here: one source of truth for "did this actually verify" avoids a
# wrapper-script exit-code check ever drifting from what repack_gguf
# itself considers success. OFF by default -- see repack_gguf's own -d
# guard (refuses to combine with --verify-samples 0). No undo.

set -euo pipefail

usage() {
    echo "Usage: $0 <source_dir> <output_dir> [--bin PATH] [--align N] [--max-shard-gb N] [--verify-samples N] [-d|--delete-source]" >&2
    exit 2
}

[ $# -ge 2 ] || usage
SRC_DIR="$1"; shift
OUT_DIR="$1"; shift

BIN="./build/repack_gguf"
ALIGN=4096
MAX_SHARD_GB=40
VERIFY_SAMPLES=500
DELETE_SOURCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN="$2"; shift 2 ;;
        --align) ALIGN="$2"; shift 2 ;;
        --max-shard-gb) MAX_SHARD_GB="$2"; shift 2 ;;
        --verify-samples) VERIFY_SAMPLES="$2"; shift 2 ;;
        -d|--delete-source) DELETE_SOURCE=1; shift ;;
        -h|--help) usage ;;
        *) echo "[repack_all] Unknown argument: $1" >&2; usage ;;
    esac
done

[ -d "$SRC_DIR" ] || { echo "[repack_all] FATAL: source dir '$SRC_DIR' does not exist" >&2; exit 1; }
[ -x "$BIN" ] || {
    echo "[repack_all] FATAL: repack_gguf binary not found/executable at '$BIN'." >&2
    echo "  Build it first: cmake --build build --target repack_gguf   (or pass --bin PATH)" >&2
    exit 1
}
mkdir -p "$OUT_DIR"

# List every .gguf file directly in SRC_DIR (not recursive -- a model
# directory is expected to be flat), sorted. Sort order matches GGUF's
# own "-00001-of-NNNNN.gguf" shard naming convention, so progress output
# reads in a predictable order -- not required for correctness (each
# shard is independent), just easier to follow and to resume after an
# interruption.
mapfile -t FILES < <(find "$SRC_DIR" -maxdepth 1 -type f -name '*.gguf' | sort)
if [ "${#FILES[@]}" -eq 0 ]; then
    echo "[repack_all] FATAL: no .gguf files found in $SRC_DIR" >&2
    exit 1
fi

echo "[repack_all] Found ${#FILES[@]} shard(s) in $SRC_DIR:"
for f in "${FILES[@]}"; do echo "  $f"; done
echo "[repack_all] Output directory: $OUT_DIR"
echo "[repack_all] align=$ALIGN max-shard-gb=$MAX_SHARD_GB verify-samples=$VERIFY_SAMPLES delete-source=$DELETE_SOURCE"
if [ "$DELETE_SOURCE" -eq 1 ]; then
    echo "[repack_all] WARNING: -d/--delete-source is ON. Each source shard is permanently deleted by"
    echo "  repack_gguf itself, immediately after ITS OWN verify pass confirms success. No undo."
fi
echo

EXTRA_ARGS=()
[ "$DELETE_SOURCE" -eq 1 ] && EXTRA_ARGS+=(-d)

DONE=0
for f in "${FILES[@]}"; do
    DONE=$((DONE + 1))
    echo "[repack_all] (${DONE}/${#FILES[@]}) Processing: $f"
    if "$BIN" "$f" "$OUT_DIR" --align "$ALIGN" --max-shard-gb "$MAX_SHARD_GB" --verify-samples "$VERIFY_SAMPLES" "${EXTRA_ARGS[@]}"; then
        echo "[repack_all] (${DONE}/${#FILES[@]}) done: $f"
    else
        echo "[repack_all] (${DONE}/${#FILES[@]}) FAILED: $f -- stopping here, source file left in place." >&2
        echo "[repack_all] Fix the issue and re-run this script -- shards already verified above are" >&2
        echo "  unaffected (re-running safely overwrites only the failed shard's own output)." >&2
        exit 1
    fi
    echo
done

echo "[repack_all] All ${#FILES[@]} shard(s) processed and verified successfully."
if [ "$DELETE_SOURCE" -eq 0 ]; then
    echo "[repack_all] Source shards were left in place (pass -d/--delete-source to remove each"
    echo "  one automatically as soon as its own verification passes)."
fi
