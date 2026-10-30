#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
binary="${1:-./build_fork_diag/zeromoe}"
config="${2:-configs/testing/qwen4exp_gsq_rco_iq2xs_rco50_3090_gpu_only.toml}"
output_dir="${3:-logs/qwen4exp_state_handoff_ab}"
prompt_tokens="${ZEROMOE_HANDOFF_PROMPT_TOKENS:-tests/qwen4exp_state_handoff_prompt_tokens.txt}"
prompt_length="$(wc -w < "$prompt_tokens")"
mkdir -p "$output_dir"
temporary_config=""
cleanup_temporary_config() {
    if [[ -n "$temporary_config" ]]; then
        rm -f -- "$temporary_config"
    fi
}
trap cleanup_temporary_config EXIT

for phase in prefill decode; do
    for mode in 0 1; do
        prefix="$output_dir/${phase}_${mode}"
temporary_config="$(mktemp "${config}.handoff.XXXXXX")"
python3 - "$config" "$temporary_config" "$mode" <<'PY_CONFIG'
from pathlib import Path
import re
import sys

source = Path(sys.argv[1]).read_text()
value = "true" if sys.argv[3] == "1" else "false"
pattern = re.compile(
    r"(?m)^([ \t]*qwen4exp_prefill_state_handoff[ \t]*=[ \t]*)(true|false)([^\r\n]*)$"
)
matches = list(pattern.finditer(source))
if len(matches) > 1:
    raise SystemExit("config has multiple qwen4exp_prefill_state_handoff keys")
if matches:
    match = matches[0]
    source = source[:match.start(2)] + value + source[match.end(2):]
else:
    table = re.search(r"(?m)^[ \t]*\[", source)
    insertion = table.start() if table else len(source)
    before, after = source[:insertion], source[insertion:]
    separator = "" if not before or before.endswith("\n") else "\n"
    source = before + separator + f"qwen4exp_prefill_state_handoff = {value}\n\n" + after
Path(sys.argv[2]).write_text(source)
PY_CONFIG
        logits_dump=""
        hidden_dump=""
        if [[ "${ZEROMOE_HANDOFF_CHECK_LOGITS:-0}" == 1 ]]; then
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
                                ZEROMOE_PROMPT_TOKEN_INPUT_FILE="$prompt_tokens" \
                                ZEROMOE_BENCH_SILENT=1 \
                ZEROMOE_DECODE_LOGITS_DUMP="$logits_dump" \
                ZEROMOE_PREFILL_FINAL_HIDDEN_DUMP="$hidden_dump" \
                "$binary" --ignore-eos --config "$temporary_config" --raw-prompt \
                    --ctx-size "$context" --predict "$generated" \
                    --log-file "$prefix.log" > "$prefix.stdout" 2>&1
rm -f -- "$temporary_config"
temporary_config=""
        echo "Completed $phase state_handoff=$mode: $prefix.log"
    done
done

if [[ "${ZEROMOE_HANDOFF_CHECK_LOGITS:-0}" == 1 ]]; then
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
