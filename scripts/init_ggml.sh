#!/usr/bin/env bash
# Check out the pinned ggml sources into third_party/llama.cpp, fetching only what the build uses.
#
#   scripts/init_ggml.sh
#
# ZeroMoE builds ggml from the ggml/ directory of one pinned llama.cpp commit (the gitlink recorded for
# third_party/llama.cpp). A plain `git submodule update --init` fetches the whole llama.cpp tree at that commit:
# about 390 MB of objects and a 180 MB working tree, of which the build reads 23 MB. This script fetches a
# shallow, blob-less, sparse copy instead: about 5 MB in .git and 26 MB checked out (ggml/ only), in a few seconds.
# The checked-out ggml/ is byte-identical to the full checkout (same git tree hash).
#
# Safe to rerun. It never touches an existing non-empty third_party/llama.cpp: if you already initialised the
# submodule the long way, remove it yourself first (git submodule deinit -f third_party/llama.cpp and
# rm -rf .git/modules/third_party/llama.cpp), after making sure it has no local changes you want to keep.
set -euo pipefail
cd "$(dirname "$0")/.."
SUB=third_party/llama.cpp
URL=$(git config -f .gitmodules submodule.$SUB.url)
PIN=$(git ls-tree HEAD "$SUB" | awk '{print $3}')
[[ -n $PIN ]] || { echo "no gitlink recorded for $SUB at HEAD" >&2; exit 1; }

if [[ -e $SUB/ggml/CMakeLists.txt ]]; then
  have=$(git -C "$SUB" rev-parse HEAD 2>/dev/null || true)
  if [[ $have == "$PIN" ]]; then echo "$SUB is already at the pinned commit ${PIN:0:12}"; exit 0; fi
  echo "$SUB exists but is at ${have:0:12}, not the pinned ${PIN:0:12}; not touching it" >&2; exit 1
fi
if [[ -n $(ls -A "$SUB" 2>/dev/null) ]]; then
  echo "$SUB is not empty and has no ggml/CMakeLists.txt; not touching it" >&2; exit 1
fi

mkdir -p "$SUB"
git submodule init -- "$SUB" >/dev/null   # register it, so "git submodule status" reports it normally
git -C "$SUB" init -q
git -C "$SUB" remote add origin "$URL"
git -C "$SUB" config extensions.partialClone origin
git -C "$SUB" config remote.origin.promisor true
git -C "$SUB" config remote.origin.partialclonefilter blob:none
git -C "$SUB" sparse-checkout init --no-cone
git -C "$SUB" sparse-checkout set '/ggml/'
git -C "$SUB" fetch -q --depth 1 --filter=blob:none origin "$PIN"
git -C "$SUB" checkout -q --detach "$PIN"
echo "ggml ${PIN:0:12}: .git $(du -sh "$SUB/.git" | cut -f1), checked out $(du -sh --exclude=.git "$SUB" | cut -f1) (ggml/ only)"
