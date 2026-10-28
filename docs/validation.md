# Validating a new model, or a change to a model's FFN flow

Throughput numbers say nothing about correctness: a broken graph can look faster. Check correctness first, then
speed. Use the layers below from cheapest to most expensive; a change to attention, routing, the MoE/FFN block,
quant handling or expert placement should pass all that apply.

## 0. CUDA MoE graph-selection gate

Before implementing CUDA kernels for a new MoE, trace its exact GGUF
architecture through `resolve_family_ops()` to the `FullGraphModel` registry.
The adapter's `plan_decode_memory()` must reject unsupported architectures or
tensor topologies before large allocations. Only an `Insufficient` result with
required/available byte counts and a reason permits layered decode; `Unknown`
requires graph decode, and `Unsupported` fails startup. Graph build/run failure
is not a memory plan.

For a full-fit hardware run, require graph decode to be enabled and exercised,
with zero legacy one-token forwards or per-expert dispatches:

```bash
python3 tools/check_full_graph_moe_run.py <run.log> --full-fit
```

For Qwen3.5/Qwen3.6 full-fit runs, also pass
`--require-all-expert-layers-on-device` to require the graph's placement proof
to report all expert layers resident on the device.

Coherent output is a smoke check, not a performance pass. For a same-GGUF,
same-GPU combined prompt/decode reference, save `llama-bench` JSON and enforce
the default 90% steady-decode bar with:

```bash
llama-bench -m /path/to/model.gguf -pg 16,400 -ngl 99 -dev CUDA1 \
    -mg 0 -sm none -t 10 -r 3 -o json > llama-bench.json
python3 tools/check_full_graph_moe_perf.py <run.log> llama-bench.json \
    --n-prompt 16 --n-gen 400 --device CUDA1
```

The ZeroMoE gate uses time spent in graph decode replays and reports graph
construction/upload separately. A result below the bar keeps onboarding in
progress; it is not accepted because generated text happens to look coherent.

Use `--allow-memory-fallback` only for a run whose log records an explicit
insufficient-memory plan and its reason.

## 1. Unit and structure checks (seconds, no model)

```bash
cmake --build build --target quant_table_test graph_settings_test layer_plan_test
./build/quant_table_test && ./build/graph_settings_test tests/graph_settings_defaults.snapshot && ./build/layer_plan_test
make moe_conformance                                         # shared-code rules (no family checks, no direct env reads)
```

## 2. Logits against llama.cpp (the oracle for a new model)

llama.cpp runs the same GGUF on the CPU; ZeroMoE dumps the raw decode logits (before repetition penalty and
sampling) with `ZEROMOE_DECODE_LOGITS_DUMP`. `tools/ling_llama_logits.cpp` is the reference harness (greedy,
one logits row per step, same binary row format) and `tools/compare_logits.py` compares the two files. Build the
reference from an **unpatched** llama.cpp checkout at the pinned commit `8baabd53b77d1c48c6ef8d3b620436292f3f6231`
(the harness uses that revision's `llama_batch_ext` API).

```bash
scripts/validate_vs_llama.sh <model.gguf> <zeromoe.toml> "The capital of France is" 8 ./build/zeromoe
# env: LLAMA_SRC, LLAMA_BUILD (an UNPATCHED llama.cpp build), DEVS, MS_ENV="ZEROMOE_<NAME>=..." for model-specific knobs
```

It builds the harness once (`build_logits_ref`), runs llama.cpp, then ZeroMoE with `--raw-prompt --temp 0`, and
prints one line per step:

```
step=3 vocab=152576 max_abs=0.41 rmse=0.07 cosine=0.99981 top1=374/374 top1_match=True
```

How to read it:
- **Step 0 is the prompt's last-token logits**: it isolates prefill + the whole stack, with no accumulated error.
  Steps 1+ follow each engine's own greedy choices, so once the two pick different tokens the rows are no longer
  comparable. Stop reading at the first `top1_match=False`.
- **Pass:** `top1_match=True` for every step before any divergence and `cosine` above about 0.999. `max_abs` and
  `rmse` are not zero: different kernels, different summation order, and (for quantized experts) different
  CPU/GPU quant paths. A cosine near 0.9 or a wrong first token is a real bug (wrong RoPE, router, scale, shared
  expert, or a quant type with the wrong block size).
- A near-tie can flip top-1 legitimately. Compare the logit gap of the two candidates in the dump before calling
  it a bug, and check a second prompt.
- Keep both prompts short at first (5-20 tokens) so a mismatch points at the stack, not at chunking.
- The machine must be idle: llama.cpp loads the whole model into RAM on the CPU.

## 3. Bisecting a mismatch

- Same graph, different placement: run once with everything on one GPU and once with the hot/cold split. If only
  one differs, the bug is in placement or the cold path, not the math (the MiMo NaN from a repeated placeholder id
  was found this way).
- `--graph-node-trace 2` (MiMo) reports the first non-finite node. Use `--verbosity 4` for tensor traces;
  capture to a file and filter the resulting log after the run.
- Check tensor sizes first: `./quant_table_test` and the `[PREFLIGHT]` lines at startup flag a tensor type that is
  missing from `memory/quant_table.hpp`.

## 4. Before/after a change to an existing model (the regression witness)

For a change inside one model's FFN or MoE flow, compare your tree against `master` on the **same prompt, greedy,
same hardware**, and require the generated tokens to match (`ZEROMOE_GENERATED_TOKEN_FILE=<file>` writes them):

```bash
bench/ling/reference-audit/gate.sh <binary> <tag>     # Ling: 32 tokens, [REQUEST-DECODE-OUTPUT] rate (baseline 10.07 tok/s)
bench/regression_gate.sh <binary> --models dsv41      # DeepSeek-V4.1, 64 tokens (band 3.5-4.6 tok/s)
bench/regression_gate.sh <binary> --models qwen38,glm53flash,ling31,dsv41,glm53   # all five, ~15 min
```

- Tokens must be identical when only structure changed (refactor). When the numerics changed on purpose (a new
  kernel, a different hot set), tokens may diverge after some steps; then read the text and run step 2.
- Speed: the first run after a fresh ggml build JIT-compiles for the sm_120 GPU and is 2-10x slow; use a warm run
  and an ABBA order (A, B, B, A), because the box's load moves results by several percent.
- Compare tokens only between binaries built from the same tree and ggml (`docs/build-ggml.md`).

## 5. Recording the result

Put the oracle comparison (prompt, steps, cosine range, first divergence) and the gate numbers in
`docs/models/<model>.md`, and update `configs/bench/perf-baselines.md` when a number improves or a band moves.
