#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
binary="${1:-./build_fork_diag/moestream}"
config="${2:-configs/qwen4exp_gsq_rco_iq2xs_rco50_3090_gpu_only.toml}"
output_dir="${3:-logs/qwen4exp_state_handoff_ab}"
prompt_tokens="${MOESTREAM_HANDOFF_PROMPT_TOKENS:-tests/qwen4exp_state_handoff_prompt_tokens.txt}"
prompt_length="$(wc -w < "$prompt_tokens")"
mkdir -p "$output_dir"

for phase in prefill decode; do
    for mode in 0 1; do
        prefix="$output_dir/${phase}_${mode}"
        logits_dump=""
        hidden_dump=""
        if [[ "${MOESTREAM_HANDOFF_CHECK_LOGITS:-0}" == 1 ]]; then
            logits_dump="$prefix.logits"
            hidden_dump="$prefix.hidden"
        fi
        context="$prompt_length"
        generated=0
        if [[ "$phase" == decode ]]; then
            generated=129
            context="$((prompt_length + generated - 1))"
        fi
        prompts=()
        for request in 1 2 3 4 5 6; do
            prompts+=("What is 2 plus 2? Explain briefly.")
        done
        printf '%s\n' "${prompts[@]}" |
            env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
                MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_LAYERS=48 \
                MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1 \
                MOESTREAM_QWEN4EXP_GRAPH_PARALLEL=1 \
                MOESTREAM_QWEN4EXP_PREFILL_STATE_HANDOFF="$mode" \
                MOESTREAM_PROMPT_TOKEN_INPUT_FILE="$prompt_tokens" \
                MOESTREAM_BENCH_IGNORE_STOP=1 \
                MOESTREAM_BENCH_SILENT=1 \
                MOESTREAM_DECODE_LOGITS_DUMP="$logits_dump" \
                MOESTREAM_PREFILL_FINAL_HIDDEN_DUMP="$hidden_dump" \
                "$binary" --config "$config" --raw-prompt \
                    --ctx "$context" --ngen "$generated" \
                    --log-path "$prefix.log" > "$prefix.stdout" 2>&1
        echo "Completed $phase state_handoff=$mode: $prefix.log"
    done
done

if [[ "${MOESTREAM_HANDOFF_CHECK_LOGITS:-0}" == 1 ]]; then
    cmp "$output_dir/decode_0.hidden" "$output_dir/decode_1.hidden"
    cmp "$output_dir/decode_0.logits" "$output_dir/decode_1.logits"
    echo "Prefill hidden state and all decode logits are bit-identical."
fi

python3 - "$output_dir" <<'PY'
import pathlib
import re
import statistics
import sys

output_dir = pathlib.Path(sys.argv[1])
for phase in ("prefill", "decode"):
    for mode in (0, 1):
        contents = (output_dir / f"{phase}_{mode}.log").read_text()
        prefills = re.findall(r"\[PREFILL-DONE\] (\d+) tokens in ([\d.]+)s", contents)
        if len(prefills) != 6:
            raise RuntimeError(f"{phase}/{mode}: expected six completed prefills")
        warm_prefills = [(int(tokens), float(seconds)) for tokens, seconds in prefills[1:]]
        prefill_rate = sum(tokens for tokens, seconds in warm_prefills) / sum(seconds for tokens, seconds in warm_prefills)
        readback = [float(value) for value in re.findall(r"state_cache_get=([\d.]+) ms", contents)][1:]
        print(f"{phase} handoff={mode}: warm prefill={prefill_rate:.2f} tok/s, state readback={statistics.mean(readback):.3f} ms")
        if phase == "decode":
            decode_wall = [float(value) for value in re.findall(r"\[REQUEST-DECODE-OUTPUT\] tokens=129 intervals=128 wall=([\d.]+)s", contents)][1:]
            print(f"  decode={128 / statistics.mean(decode_wall):.2f} tok/s, first-to-last output, 128 intervals; graph setup excluded")
PY
