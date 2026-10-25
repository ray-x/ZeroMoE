#!/usr/bin/env bash
# Old vs refit ranking on identical tokens for one corpus prompt.
#   bench/ranking_ab_prompt.sh <engine> <tag> <prompt-file>
# 1) free run (old ranking) writes the generated tokens, 2) teacher-forced old ranking, 3) teacher-forced refit ranking.
set -uo pipefail
bin=$(readlink -f "${1:?engine}"); tag=${2:?}; pf=${3:?}
root=$(cd "$(dirname "$0")/.." && pwd); cd "$root"
P="$(tr '\n' ' ' < "$pf" | tr -s ' ' | cut -c1-900)"
common=(CUDA_VISIBLE_DEVICES=0,1 ZEROMOE_GLM5NEXT_GPU1_HOT_GIB=15.0 ZEROMOE_GLM5NEXT_CPU_THREADS=10 ZEROMOE_GLM5NEXT_OFFLOAD_DDR_GIB=58)
new=$root/docs/ssd-offload/data/glm53flash_usage_base_plus_decode40.txt
tok=$root/logs/perf/${tag}.tokens
PROMPT="$P" GLM_TOML=$root/logs/perf/dual_final.toml NGEN=64 bench/glm_decode_profile.sh "$bin" ${tag}_gen "${common[@]}" ZEROMOE_GENERATED_TOKEN_FILE=$tok > /dev/null 2>&1
PROMPT="$P" GLM_TOML=$root/logs/perf/dual_final.toml NGEN=64 bench/glm_decode_profile.sh "$bin" ${tag}_old "${common[@]}" ZEROMOE_TEACHER_FORCE_TOKEN_FILE=$tok > /dev/null 2>&1
PROMPT="$P" GLM_TOML=$root/configs/testing/glm_dual_ranking40.toml GLM_USAGE=$new NGEN=64 bench/glm_decode_profile.sh "$bin" ${tag}_new "${common[@]}" ZEROMOE_GLM5NEXT_GPU1_HOT_USAGE=$new ZEROMOE_TEACHER_FORCE_TOKEN_FILE=$tok > /dev/null 2>&1
touch logs/perf/${tag}.done
