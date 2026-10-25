#!/usr/bin/env bash
# Kernel-level numerics check, free of model noise: for Ling (IQ1_S/IQ2_XXS/IQ2_S) and MiMo (Q2_K/Q3_K) shapes,
#  * aliased expert views at the engine's slot strides are bitwise identical to a contiguous tensor (CUDA);
#  * with activations that 8-bit rounding represents exactly (EXACT_INPUTS=1), CPU and CUDA match float64 to 1e-4.
# Ordinary inputs differ 0.5-0.9% per op between backends (Q8_K vs Q8_1 activation rounding); that is not a bug.
#   scripts/check_kernel_numerics.sh [build-dir]    (default build_mimo; needs a CUDA GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
B=${1:-build_mimo}
cmake --build "$B" --target phase_alias_numerics_test -j8 >/dev/null
EXACT_INPUTS=1 "$B/phase_alias_numerics_test" | tee "${TMPDIR:-/tmp}/kernel_numerics.out" | tail -n 3
grep -q "KERNEL ERROR\|LAYOUT MISMATCH" "${TMPDIR:-/tmp}/kernel_numerics.out" && { echo FAIL; exit 1; }
echo PASS
