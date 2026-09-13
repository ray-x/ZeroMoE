#!/usr/bin/env bash
set -euo pipefail

# Build and run the same GGML-only Qwen CPU MoE graph benchmark against the
# current MoEStream and BeeLlama GGML shared libraries.  No model is loaded.

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bee_dir="${BEE_DIR:-/home/ray/Projects/beellama.cpp}"
moe_build="${MOESTREAM_GGML_BUILD:-${repo_dir}/build_ggmlcuda/ggml-build}"
bee_build="${BEELLAMA_GGML_BUILD:-${bee_dir}/build-qwen38-rtx3090/bin}"
bee_include="${bee_dir}/ggml/include"
source_file="${repo_dir}/bench/qwen_cpu_moe_ggml_bench.cpp"
out_dir="${repo_dir}/build_ggmlcuda/bench"

mkdir -p "${out_dir}"

build_one() {
    local include_dir="$1"
    local link_dir="$2"
    local output="${out_dir}/qwen_cpu_moe_ggml_bench_compare"
    c++ -std=c++17 -O2 -march=native -fopenmp \
        -I"${include_dir}" "${source_file}" \
        -L"${link_dir}" \
        -lggml -lggml-cpu -lggml-base -o "${output}"
    echo "${output}"
}

compare_bin="$(build_one "${bee_include}" "${moe_build}/src")"

run_with_libs() {
    local lib_dir="$1"
    shift
    LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        "${compare_bin}" "$@"
}

echo "[MoEStream ggml]"
run_with_libs "${moe_build}/src"
echo "[BeeLlama ggml]"
run_with_libs "${bee_build}"

if [[ "${MOESTREAM_QWEN_CPU_BENCH_THREAD_SWEEP:-0}" == "1" ]]; then
    echo "[Thread scaling: same graph, both ggml builds]"
    for label_and_dir in \
        "MoEStream:${moe_build}/src" \
        "BeeLlama:${bee_build}"; do
        label="${label_and_dir%%:*}"
        lib_dir="${label_and_dir#*:}"
        echo "-- ${label}"
        for threads in 1 2 4 8 12; do
            echo "threads=${threads}"
            MOESTREAM_QWEN_CPU_BENCH_THREADS="${threads}" \
                MOESTREAM_QWEN_CPU_BENCH_STAGE=full \
                run_with_libs "${lib_dir}"
        done
    done
fi
