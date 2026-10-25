#!/usr/bin/env bash
# Survey MiMo's expert routes and (optionally) evaluate the surveyed profile. Reproducible: prompts are
# bench/mimo_survey_prompts.txt (training) and bench/mimo_eval_prompts.txt (held out), config is the dual-GPU one.
#
#   bench/mimo_survey.sh survey <out-profile> [binary]   # writes <out-profile> (decode counts) and <out-profile>.prefill
#   bench/mimo_survey.sh eval <profile> [binary] [tag]   # 4 held-out prompts, 64 greedy tokens, prints decode/prefill rates
#   bench/mimo_survey.sh long <profile> [binary] [tag]   # 2 long prompts (about 600 and 1000 tokens), 32 tokens
#
# The survey uses ZEROMOE_MIMO2_USAGE_RECORD (graph_route_survey.hpp): a side tap on each layer's route ids, decode
# and prefill counted separately. Needs the box idle (about 10 min per run).
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:?survey|eval|long} PROFILE=${2:?profile path}
BIN=${3:-./build_master/zeromoe}
CONFIG=configs/bench/mimo26_flash_gsq_rco_3bit_ingraph_dual.toml
BASE_PROFILE=.zeromoe_usage_mimo26_gsq_rco3
COMMON=(ZEROMOE_DISABLE_ROUTER_BF16=1 CUDA_VISIBLE_DEVICES=0,1 ZEROMOE_MIMO2_OFFLOAD_DDR_GIB=100)
mkdir -p logs
case $MODE in
  survey)
    # The DDR seed order comes from the existing profile; the survey itself only records.
    env "${COMMON[@]}" ZEROMOE_MIMO2_OFFLOAD_USAGE="$PWD/$BASE_PROFILE" ZEROMOE_MIMO2_USAGE_RECORD="$PWD/$PROFILE" \
      scripts/run_capped.sh --limit-gb 110 -- "$BIN" --config "$CONFIG" --ngen 96 --temp 0.7 \
      < bench/mimo_survey_prompts.txt > "logs/mimo_survey_$(date +%H%M%S).log" 2>&1
    wc -l "$PROFILE" "$PROFILE.prefill" ;;
  eval|long)
    TAG=${4:-$MODE}
    PROMPTS=bench/mimo_eval_prompts.txt NGEN=64
    if [[ $MODE == long ]]; then PROMPTS=bench/mimo_long_prompts.txt NGEN=32; fi
    cfg=$(mktemp /tmp/mimo-eval-config.XXXXXX.toml)
    sed -e "s|^pin_file = .*|pin_file = \"$PWD/$PROFILE\"|" ${EMBED:+-e "s|^freeze_usage_stats = .*|&\nembedding_device = \"$EMBED\"|"} "$CONFIG" > "$cfg"
    env "${COMMON[@]}" ZEROMOE_MIMO2_OFFLOAD_USAGE="$PWD/$PROFILE" \
      scripts/run_capped.sh --limit-gb 110 -- "$BIN" --config "$cfg" --ngen $NGEN --temp 0 \
      < $PROMPTS > "logs/mimo_${TAG}.log" 2>&1 || true
    rm -f "$cfg"
    echo "$TAG: decode $(grep -a REQUEST-DECODE-OUTPUT logs/mimo_${TAG}.log | grep -o 'rate=[0-9.]*' | tr '\n' ' ') prefill $(grep -a 'PREFILL-DONE\] [0-9]' logs/mimo_${TAG}.log | grep -o '([0-9.]* tok/s)' | tr '\n' ' ')" ;;
esac
