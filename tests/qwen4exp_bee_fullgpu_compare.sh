#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
engine="${1:-./build/zeromoe}"
bee="${2:-logs/qwen4exp_perf_gap/bee_forced/llama-bench}"
output_dir="${3:-logs/qwen4exp_bee_fullgpu_compare}"
config=configs/testing/qwen4exp_gsq_rco_iq2xs_rco50_3090_gpu_only.toml
model=/data/sda/models/qwen3.8-flash-next/GSQ-RCO-IQ2_XS_RCO50/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-RCO50-IQ1M-boundary-00001-of-00002.gguf
prompt_tokens="$PWD/tests/qwen4exp_state_handoff_prompt_tokens.txt"
bee_libraries="$PWD/build-bee-instrumented/bin"
mkdir -p "$output_dir"
output_dir="$(realpath "$output_dir")"
if [[ ! -x "$engine" || ! -x "$bee" ]]; then
    echo 'Both the ZeroMoE engine and the exact-decode Bee benchmark must be built.' >&2
    exit 2
fi
while IFS= read -r variable; do
    unset "$variable"
done < <(compgen -v | rg '^ZEROMOE_' || true)

wait_for_gpu() {
    while true; do
        local gpu_processes engine_processes
        gpu_processes="$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader)"
        engine_processes="$(pgrep -x zeromoe || true)"
        if [[ -z "$gpu_processes" && -z "$engine_processes" ]]; then
            return
        fi
        echo 'Waiting for competing inference to finish; leaving it untouched.' >&2
        sleep 2
    done
}

./build_fork_diag/gguf_tool verify "$(dirname "$model")" > "$output_dir/verify.log" 2>&1
runtime=(CUDA_VISIBLE_DEVICES=0 ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS=48
         ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1
         ZEROMOE_QWEN4EXP_GRAPH_PARALLEL=1
         ZEROMOE_PROMPT_TOKEN_INPUT_FILE="$prompt_tokens"
         ZEROMOE_BENCH_IGNORE_STOP=1 ZEROMOE_BENCH_SILENT=1)
wait_for_gpu
printf '%s\n' 'What is 2 plus 2? Explain briefly.' |
    env "${runtime[@]}" ZEROMOE_GENERATED_TOKEN_FILE="$output_dir/decode.tokens" \
        "$engine" --config "$config" --raw-prompt --ctx-size 151 --predict 129 \
        --log-file "$output_dir/capture.log" > "$output_dir/capture.stdout" 2>&1
wait_for_gpu
env CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$bee_libraries" \
    BEE_LLAMA_BENCH_TOKEN_FILE="$prompt_tokens" \
    BEE_LLAMA_BENCH_DECODE_TOKEN_FILE="$output_dir/decode.tokens" \
    BEE_LLAMA_BENCH_ISOLATED_PHASE_TIMING=1 \
    "$bee" -m "$model" -p 0 -n 0 -pg 23,128 -ngl 999 -ncmoe 0 \
        -dev CUDA0 -fa on -lm mmap -t 12 -r 5 > "$output_dir/bee.stdout" 2>&1
prompts=()
for request in 1 2 3 4 5 6; do
    prompts+=('What is 2 plus 2? Explain briefly.')
done
wait_for_gpu
printf '%s\n' "${prompts[@]}" |
    env "${runtime[@]}" ZEROMOE_TEACHER_FORCE_TOKEN_FILE="$output_dir/decode.tokens" \
        ZEROMOE_TEACHER_FORCE_SKIP_SAMPLING=1 \
        "$engine" --config "$config" --raw-prompt --ctx-size 151 --predict 129 \
        --log-file "$output_dir/zeromoe.log" > "$output_dir/zeromoe.stdout" 2>&1

python3 - "$output_dir" <<'PY'
from pathlib import Path
import re
import statistics
import sys

output_dir = Path(sys.argv[1])
bee = (output_dir / 'bee.stdout').read_text()
engine = (output_dir / 'zeromoe.log').read_text()
phases = [tuple(map(float, row)) for row in re.findall(
    r'prompt_tokens=23 decode_tokens=128 prefill_ms=([\d.]+) decode_ms=([\d.]+)', bee)]
prefill = [float(value) for value in re.findall(
    r'\[PREFILL-DONE\] 23 tokens in ([\d.]+)s', engine)][1:]
decode = [float(value) for value in re.findall(
    r'\[REQUEST-DECODE-OUTPUT\] tokens=129 intervals=128 wall=([\d.]+)s', engine)][1:]
if len(phases) != 5 or len(prefill) != 5 or len(decode) != 5:
    raise RuntimeError('Expected five warm measurements for each engine and phase')
if '48 complete layer(s)' not in engine or 'scheduler nodes cuda=6512 cpu=0' not in engine:
    raise RuntimeError('ZeroMoE did not run the required fully GPU-resident decode path')
bee_prefill = 23000 / statistics.mean(row[0] for row in phases)
bee_decode = 128000 / statistics.mean(row[1] for row in phases)
engine_prefill = 23 / statistics.mean(prefill)
engine_decode = 128 / statistics.mean(decode)
difference = abs(engine_decode / bee_decode - 1) * 100
print(f'Warm pp23: Bee {bee_prefill:.2f}, ZeroMoE {engine_prefill:.2f} tok/s')
print(f'Matched tg128: Bee {bee_decode:.2f}, ZeroMoE {engine_decode:.2f} tok/s')
print(f'Decode difference: {difference:.2f}% (target < 3%)')
if difference >= 3:
    raise RuntimeError('Decode parity target is not met')
if engine_prefill < bee_prefill * 0.97:
    raise RuntimeError('Prefill is still more than 3% below Bee')
PY
