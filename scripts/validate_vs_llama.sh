#!/usr/bin/env bash
# Compare ZeroMoE's decode logits with llama.cpp's for the same model, prompt and steps.
# Docs: README.md "Validating a new model or an FFN change".
#
#   scripts/validate_vs_llama.sh <model.gguf> <zeromoe.toml> ["prompt"] [steps] [zeromoe-binary]
#
# Env: LLAMA_SRC   llama.cpp source tree        (default ../llama.cpp)
#      LLAMA_BUILD llama.cpp build with the libs (default $LLAMA_SRC/build-avx512-stock, an UNPATCHED build)
#      DEVS        CUDA_VISIBLE_DEVICES for ZeroMoE (default 0,1)
#      MS_ENV      extra environment-only controls for ZeroMoE; use CLI flags for graph settings
# Needs the machine idle: llama.cpp loads the whole model on the CPU, ZeroMoE then runs it.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:?model.gguf} CONFIG=${2:?zeromoe.toml} PROMPT=${3:-The capital of France is} STEPS=${4:-8}
BIN=${5:-./build_master/zeromoe}
LLAMA_SRC=${LLAMA_SRC:-../llama.cpp}
LLAMA_BUILD=${LLAMA_BUILD:-$LLAMA_SRC/build-avx512-stock}
OUT=${OUT:-logs/validate_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$(dirname "$OUT")"

# 1. The reference harness (one file; build once).
REF=build_logits_ref
if [[ ! -x $REF || tools/ling_llama_logits.cpp -nt $REF ]]; then
  g++ -std=c++17 -O1 tools/ling_llama_logits.cpp -o "$REF" \
    -I"$LLAMA_SRC/include" -I"$LLAMA_SRC/ggml/include" \
    -L"$LLAMA_BUILD/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$LLAMA_BUILD/bin"
fi

# 2. llama.cpp: greedy, CPU, raw prompt (BOS added by the tokenizer), one logits row per step.
"./$REF" "$MODEL" "$PROMPT" "$STEPS" "$OUT.llama.bin" 2> "$OUT.llama.log"
echo "llama.cpp prompt ids: $(grep -a 'prompt ids' "$OUT.llama.log" | head -1)"

# 3. ZeroMoE: --raw-prompt so no chat template is applied; temp 0; the same row dump.
printf '%s\n' "$PROMPT" | env CUDA_VISIBLE_DEVICES="${DEVS:-0,1}" ${MS_ENV:-} \
  ZEROMOE_DECODE_LOGITS_DUMP="$PWD/$OUT.ms.bin" \
  scripts/run_capped.sh -- "$BIN" --config "$CONFIG" --ngen "$STEPS" --temp 0 --raw-prompt > "$OUT.ms.log" 2>&1 || true

# 4. Compare row by row.
python3 tools/compare_logits.py "$OUT.ms.bin" "$OUT.llama.bin" | tee "$OUT.compare.txt"
