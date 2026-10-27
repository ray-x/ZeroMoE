#!/usr/bin/env bash
# Build the ZeroMoE-patched ggml from source into a self-contained directory. Rules and the reason for
# every step: docs/build-ggml.md.
#
#   scripts/build_ggml.sh <out-dir>
#
# Produces <out-dir>/ggml-src/ggml    the patched ggml sources (headers ZeroMoE compiles against)
#          <out-dir>/ggml-build       libggml*.so (src/lib*.so, src/ggml-cuda/libggml-cuda.so)
#
# Environment:
#   GGML_REPO     git repo holding upstream llama.cpp   (required)
#   GGML_COMMIT   upstream commit to build              (default 8baabd53b, ggml 0.27.0)
#   CUDA_ARCH     optional explicit CMake architecture list (e.g. '86;120a'). By default,
#                 detect all GPUs visible to nvidia-smi at build time.
#   EXTRA_PATCH_DIRS  more patch directories to apply after the series, space separated
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=$(realpath -m "${1:?usage: build_ggml.sh <out-dir>}")
# Default: the llama.cpp submodule (third_party/llama.cpp, commit 8baabd53b), whose ggml/ directory is built.
# A standalone ggml repository also works (its root is the ggml root).
GGML_REPO=${GGML_REPO:-$ROOT/third_party/llama.cpp}
[[ -d "$GGML_REPO/.git" || -f "$GGML_REPO/.git" ]] || { echo "GGML_REPO is not a git checkout: $GGML_REPO" >&2; exit 1; }
GGML_COMMIT=${GGML_COMMIT:-HEAD}
GGML_COMMIT_RESOLVED=$(git -C "$GGML_REPO" rev-parse --verify "${GGML_COMMIT}^{commit}")
CUDA_ARCH=$(CUDA_ARCH="${CUDA_ARCH:-}" "$ROOT/scripts/detect_cuda_arch.sh")
SRC=$OUT/ggml-src

if [[ -e $SRC ]]; then echo "$SRC exists; remove it first (the patches are applied once)"; exit 1; fi
mkdir -p "$SRC"
if git -C "$GGML_REPO" cat-file -e "$GGML_COMMIT_RESOLVED:ggml/CMakeLists.txt" 2>/dev/null; then
  # llama.cpp checkout: ggml lives in its ggml/ subdirectory.
  git -C "$GGML_REPO" archive "$GGML_COMMIT_RESOLVED" ggml | tar -x -C "$SRC"
else
  # Standalone ggml repository: the ggml root is the repository root.
  mkdir -p "$SRC/ggml"
  git -C "$GGML_REPO" archive "$GGML_COMMIT_RESOLVED" | tar -x -C "$SRC/ggml"
fi
echo "ggml source: $GGML_REPO @ $GGML_COMMIT_RESOLVED" > "$OUT/ggml-src.txt"

# The series (patches/ggml, then patches/ggml-host-node) is written for the upstream commit above. GNU patch
# applies it with fuzz 0: hunks may move (offset) but their context must match exactly, so a hunk that
# no longer fits fails the build instead of landing in the wrong function. (Not `git apply`: inside a
# checkout of another repository, which this script's output directory may be, it silently skips files
# outside the current directory.) Patches for an older upstream live in patches/obsolete and are never applied.
apply_dir() {
  local dir=$1
  for p in $(ls "$ROOT/$dir"/0*.patch 2>/dev/null | sort); do
    local n; n=$(basename "$p")
    if (cd "$SRC/ggml" && patch -p1 -F0 --dry-run -s < "$p" >/dev/null 2>&1); then
      (cd "$SRC/ggml" && patch -p1 -F0 --no-backup-if-mismatch -s < "$p") && echo "applied $dir/$n"
    else
      echo "FAILED  $dir/$n does not apply to $GGML_COMMIT"; exit 1
    fi
  done
}
apply_dir patches/ggml
apply_dir patches/ggml-host-node
for d in ${EXTRA_PATCH_DIRS:-}; do apply_dir "$d"; done

# Guard against a silent no-op: the series must have left its marks.
grep -q zeromoe_mmid_negative_skip "$SRC/ggml/src/ggml-cuda/mmvq.cu" || { echo "patches were not applied"; exit 1; }
grep -q zeromoe_ggml_backend_cuda_get_stream "$SRC/ggml/src/ggml-cuda/ggml-cuda.cu" || { echo "bridge patch missing"; exit 1; }

# ggml's CMakeLists configures ggml.pc.in, which a plain `git archive` of llama.cpp's ggml/ does not contain.
[[ -f $SRC/ggml/ggml.pc.in ]] || printf 'Name: ggml\nDescription: ggml\nVersion: @GGML_VERSION@\n' > "$SRC/ggml/ggml.pc.in"

cmake -S "$SRC/ggml" -B "$OUT/ggml-build" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_OPENMP=ON \
-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
-DGGML_BUILD_COMMIT="$GGML_COMMIT_RESOLVED" \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
cmake --build "$OUT/ggml-build" -j"$(nproc)"

if ! grep -Fq "#define GGML_COMMIT  \"$GGML_COMMIT_RESOLVED\"" "$OUT/ggml-build/src/ggml-version.h"; then
    echo "generated GGML_COMMIT does not match upstream revision $GGML_COMMIT_RESOLVED" >&2
    exit 1
fi

echo "ggml built: $OUT/ggml-build"
echo "PTX images in libggml-cuda: $(cuobjdump --list-ptx "$OUT/ggml-build/src/ggml-cuda/libggml-cuda.so" 2>/dev/null | grep -c 'PTX file')"
