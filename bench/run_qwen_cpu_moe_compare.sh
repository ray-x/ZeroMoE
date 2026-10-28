#!/usr/bin/env bash
set -euo pipefail

# Build and run the same GGML-only Qwen CPU MoE graph benchmark against the
# current ZeroMoE and llama.cpp GGML shared libraries.  No model is loaded.

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_cpp_dir="${LLAMA_CPP_DIR:-${repo_dir}/../llama.cpp}"
moe_build="${ZEROMOE_GGML_BUILD:-${repo_dir}/build/ggml-build}"
llama_cpp_build="${LLAMA_CPP_GGML_BUILD:-${llama_cpp_dir}/build-qwen38-rtx3090/bin}"
llama_cpp_include="${llama_cpp_dir}/ggml/include"
source_file="${repo_dir}/bench/qwen_cpu_moe_ggml_bench.cpp"
out_dir="${repo_dir}/build/bench"

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

compare_bin="$(build_one "${llama_cpp_include}" "${moe_build}/src")"

run_with_libs() {
    local lib_dir="$1"
    shift
    LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        "${compare_bin}" "$@"
}

echo "[ZeroMoE ggml]"
run_with_libs "${moe_build}/src"
echo "[llama.cpp ggml]"
run_with_libs "${llama_cpp_build}"

if [[ "${ZEROMOE_QWEN_CPU_BENCH_THREAD_SWEEP:-0}" == "1" ]]; then
    echo "[Thread scaling: same graph, both ggml builds]"
    for label_and_dir in \
        "ZeroMoE:${moe_build}/src" \
        "llama.cpp:${llama_cpp_build}"; do
        label="${label_and_dir%%:*}"
        lib_dir="${label_and_dir#*:}"
        echo "-- ${label}"
        for threads in 1 2 4 8 12; do
            echo "threads=${threads}"
            ZEROMOE_QWEN_CPU_BENCH_THREADS="${threads}" \
                ZEROMOE_QWEN_CPU_BENCH_STAGE=full \
                run_with_libs "${lib_dir}"
        done
    done
fi
