#!/usr/bin/env bash
# Print the CUDA architectures for all NVIDIA GPUs visible at build time.
# An explicit CUDA_ARCH environment variable always takes precedence.
set -euo pipefail

if [[ -n ${CUDA_ARCH:-} ]]; then
  printf '%s\n' "$CUDA_ARCH"
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; set CUDA_ARCH explicitly (for example CUDA_ARCH='86;120a')" >&2
  exit 2
fi

compute_caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader) || {
  echo "nvidia-smi could not query CUDA compute capabilities; set CUDA_ARCH explicitly" >&2
  exit 2
}

architectures=$(awk -F. '
  NF == 2 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ {
    arch = $1 $2
    # CUDA toolchains expose the current architecture-specific feature sets
    # for compute capability 12.x under the 12xa names (for example 120a).
    if ($1 >= 12) arch = arch "a"
    if (!seen[arch]++) {
      if (count++) result = result ";"
      result = result arch
    }
  }
  END { if (count) print result }
' <<< "$compute_caps")

if [[ -z $architectures ]]; then
  echo "nvidia-smi returned no parseable compute capabilities; set CUDA_ARCH explicitly" >&2
  exit 2
fi

printf '%s\n' "$architectures"
