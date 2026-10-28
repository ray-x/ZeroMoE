#!/usr/bin/env bash
set -euo pipefail

# Sweeps ZEROMOE_QWEN4EXP_RING_SPIN_US against
# qwen_cpu_offload_ring_spin_bench to measure its effect on
# QwenCpuOffloadRing's wait-detection latency in isolation, with no model
# load and no decode-loop noise. The env var is cached on first read inside
# models/qwen/qwen_cpu_offload_ring.cpp, so this script launches one process per
# value -- that caching is why the values can't be swept within one run.
#
# Usage: bench/run_ring_spin_sweep.sh [iterations] [d_model] [slots] [cpu_compute_us]

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin="${repo_dir}/build/qwen_cpu_offload_ring_spin_bench"
iterations="${1:-500}"
d_model="${2:-2560}"
slots="${3:-2}"
cpu_compute_us="${4:-1000}"

if [[ ! -x "${bin}" ]]; then
    echo "missing ${bin}; build with:" >&2
    echo "  cmake --build build --target qwen_cpu_offload_ring_spin_bench" >&2
    exit 1
fi

spin_values=(0 25 50 100 200 400)

printf "%-10s %-8s %-10s %-10s %-10s %-10s\n" \
    "spin_us" "n" "mean_us" "p50_us" "p90_us" "max_us"

for spin in "${spin_values[@]}"; do
    out="$(ZEROMOE_QWEN4EXP_RING_SPIN_US="${spin}" "${bin}" \
        "${iterations}" "${d_model}" "${slots}" "${cpu_compute_us}" \
        2>/dev/null)"
    line="$(printf '%s\n' "${out}" | grep 'detect_latency')"
    # Field-aware parse (not substring grep): "mean=26.51us" contains the
    # literal text "n=", so a naive `grep -oE 'n=[0-9]+'` over the whole
    # line matches inside "mea{n=2}6.51us" too -- caught by the sanity run
    # before trusting this script's numbers.
    read -r n mean p50 p90 maxv <<<"$(printf '%s\n' "${line}" | awk '{
        for (i = 1; i <= NF; i++) {
            split($i, kv, "=")
            gsub(/us$/, "", kv[2])
            if (kv[1] == "n") n = kv[2]
            else if (kv[1] == "mean") mean = kv[2]
            else if (kv[1] == "p50") p50 = kv[2]
            else if (kv[1] == "p90") p90 = kv[2]
            else if (kv[1] == "max") maxv = kv[2]
        }
        print n, mean, p50, p90, maxv
    }')"
    printf "%-10s %-8s %-10s %-10s %-10s %-10s\n" \
        "${spin}" "${n}" "${mean}" "${p50}" "${p90}" "${maxv}"
done
