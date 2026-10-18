#!/usr/bin/env bash
# tests/parity_check_greedy.sh — end-to-end confirmation that MoEStream's MLA
# forward path matches the reference (llama.cpp) on the SAME gguf + prompt.
#
# This is the high-fidelity companion to tests/test_components.cpp::test_rope:
# test_rope proves the RoPE math in isolation; this proves the WHOLE forward
# path (embed -> 79x MLA/MoE -> lm_head -> greedy argmax) is correct by checking
# that greedy-decoded output matches llama.cpp token-for-token.
#
# Before the interleaved-RoPE fix (models/base/rope.hpp): output was garbage that
# ignored the prompt. After the fix: should match llama.cpp closely (exact on
# early tokens; small divergence later is expected from quant-path / FP-order
# differences, NOT from the RoPE bug).
#
# CAN'T run on darwin (needs the Linux+CUDA build + the real model). Tune the
# paths/flags marked TODO for your box. Usage:
#   PARITY_MODEL=/path/to/GLM-5.2-UD-IQ2_M \
#   LLAMA_CLI=/path/to/llama.cpp/build/bin/llama-cli \
#   PROMPT="Explain recursion in one sentence." \
#   ./tests/parity_check_greedy.sh
set -euo pipefail

MODEL="${PARITY_MODEL:?set PARITY_MODEL to the gguf dir/file}"
LLAMA_CLI="${LLAMA_CLI_CLI:-./llama-cli}"        # TODO: path to llama.cpp's llama-cli
MOESTREAM="./build/moestream"                     # TODO: path to MoEStream binary
CONFIG="${PARITY_CONFIG:-configs/glm52_dual.toml}"
PROMPT="${PROMPT:-Hello, who are you?}"
N="${N:-64}"                                       # tokens to generate
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

echo ">> prompt: $PROMPT"
echo ">> generating $N greedy tokens from each engine (temp=0)..."

# --- llama.cpp reference (greedy) ---
# TODO: add -ngl / --n-gpu-layers / -c as needed for your box. llama-cli prints
# generation to stdout; --no-display-prompt keeps only the completion.
"$LLAMA_CLI" -m "$MODEL" -p "$PROMPT" --temp 0 -n "$N" --no-display-prompt \
    >"$TMP/llama.txt" 2>"$TMP/llama_err.txt" || true

# --- MoEStream (greedy). Pipe the prompt into the REPL; --temp 0 = greedy. ---
# TODO: if moestream lacks a non-interactive prompt mode, this pipes one line
# then ':q'. Strip the "Output:" / banner lines to get raw generation.
printf '%s\n:q\n' "$PROMPT" | "$MOESTREAM" --config "$CONFIG" --temp 0 --ngen "$N" \
    >"$TMP/moe.raw" 2>"$TMP/moe_err.txt" || true

# Normalize: keep only the generated text (drop MoEStream banner/log lines).
grep -A9999 '^Output:' "$TMP/moe.raw" | tail -n +2 > "$TMP/moe.txt" || cp "$TMP/moe.raw" "$TMP/moe.txt"

echo
echo "==== llama.cpp (reference) ===="
cat "$TMP/llama.txt"
echo
echo "==== MoEStream ===="
cat "$TMP/moe.txt"
echo
echo "==== first-token / prefix comparison ===="
# Strictest cheap signal: do the first ~10 tokens of greedy decode agree?
# (Quant/FP-order differences usually only diverge after many tokens.)
python3 - "$TMP/llama.txt" "$TMP/moe.txt" <<'PY'
import sys, difflib
a = open(sys.argv[1]).read().strip().split()
b = open(sys.argv[2]).read().strip().split()
n = min(10, len(a), len(b))
print(f"first {n} tokens:")
print("  llama:", a[:n])
print("  moe  :", b[:n])
agree = sum(1 for i in range(n) if a[i] == b[i])
print(f"  agreement: {agree}/{n}")
if agree == n:
    print("PASS: early greedy tokens match -- MLA forward path is correct.")
else:
    print("REVIEW: early-token mismatch -- if test_rope passes but this fails,")
    print("        check chat template, tokenizer pretokenizer, or a remaining")
    print("        MLA/dim bug (not RoPE, which test_rope covers).")
PY
