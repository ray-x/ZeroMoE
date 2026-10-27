#!/usr/bin/env bash
# Run the survey batches (bench/make_survey_batches.py) through the dual-GPU engine with the route trace on.
#   bench/glm_route_survey.sh <engine> [batches...]    output: logs/perf/survey/b<i>.trace
set -uo pipefail
bin=$(readlink -f "${1:?engine}"); shift
root=$(cd "$(dirname "$0")/.." && pwd); cd "$root"
root_libs=${GGML_BASE_LIBS:-$root/build_ggmlcuda/ggml-build/src}
kernel_libs=${GGML_KERNEL_LIBS:?set GGML_KERNEL_LIBS to the required ggml kernel library directories}
LP=${GGML_CUDA_LIBS:-$kernel_libs}:$kernel_libs:$root_libs:$root_libs/ggml-cuda
for b in "${@:-0 1 2 3}"; do
  head -n -1 logs/perf/survey/batch$b.txt > logs/perf/survey/p$b.txt
  rm -f logs/perf/survey/b$b.trace
  PROMPT="$(cat logs/perf/survey/p$b.txt)" GLM_TOML=$root/logs/perf/dual_final.toml NGEN=96 \
    bench/glm_decode_profile.sh "$bin" survey_b$b CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$LP \
    ZEROMOE_GLM5NEXT_FULL_GRAPH_PREFILL=1 ZEROMOE_GLM5NEXT_GPU1_HOT_GIB=14.2 --graph-offload-ddr-gib 58 \
    ZEROMOE_GLM5NEXT_ROUTE_TRACE=$root/logs/perf/survey/b$b.trace > /dev/null 2>&1
done
touch logs/perf/survey/all.done
