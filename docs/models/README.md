# Per-model playbook

This directory is the **knowledge store** for per-model tuning. One file per model records the
canonical recipe, the frozen baseline, the tricks applied *and why*, and the gotchas — so that
tuning knowledge survives when work moves to the next model, and so the regression gate
(`bench/regression_gate.sh`) has a frozen baseline to compare against.

The profile fields follow the design review: key by **hardware, model dimensions, quantization,
memory budgets and execution phase**, and record **prefill throughput, time-to-first-token,
decode throughput, peak memory, and the selected strategy**.

| Model | Family (GGUF arch) | Quant | Decode band (tok/s) | File |
|---|---|---|---|---|
| Qwen 3.8 Flash | `Qwen4Exp` (`qwen4exp`) | UD-IQ1_M | 22.0 – 30.0 | [`qwen38-flash.md`](qwen38-flash.md) |
| GLM 5.3 Flash | `GLM5NEXT` (`glm5next`) | AJ-IQ2_XXS | 6.5 – 8.5 | [`glm53-flash.md`](glm53-flash.md) |
| Ling 3.1 Flash | `BailingMoE3` (`bailingmoe3`) | AD-IQ1_S | 9.0 – 11.5 | [`ling31-flash.md`](ling31-flash.md) |
| DeepSeek 4.1 Flash | `DeepSeek_V41` (`deepseek41`) | Q2_K | 1.8 – 3.2 | [`deepseek41-flash.md`](deepseek41-flash.md) |
| GLM 5.3 | `GLM_DSA` (`glm-dsa`) | UD-IQ2_M | 1.8 – 2.4 | [`glm53.md`](glm53.md) |

## Baseline provenance

The decode bands above were measured once on the merged master (`1f17a99`), 2026-10-21, on the
box in `docs/GLM5.3-baseline.md` §1: Xeon W-2235 (6C/12T, AVX-512 VNNI), 125 GiB RAM, RTX 3090
(24 GiB, PCIe 4) + RTX 5060 Ti (16 GiB, PCIe 3), two NVMe. The binary is `build_master/moestream`,
built Release with `MOESTREAM_GGML_ROOT=/home/ray/Projects/llama.cpp/ggml` and
`MOESTREAM_GGML_PREBUILT_DIR=.worktrees/ling-3.1-flash/build-ling/ggml-build` (the prebuilt ggml
that carries the early-CPU scheduler host-node hook — **an in-tree ggml build against the
manually-edited fork silently drops this hook and disables the CPU lane**, see the "gotchas").

The bands are deliberately wide (≈±15 %) because the models are **not bit-deterministic** under
near-tie routing and single-run decode spread is real (e.g. Qwen 24.6–28.7 tok/s). A regression is
a *sustained* drop below the band, not a single low run.

## When to run the gate (tiered policy)

A full run is ~15 min, so it is not needed for every change:

| Change | Gate |
|---|---|
| Non-graph (tokenizer, config parsing, API server, docs, CPU-only kernels unrelated to the graph) | none — run `make test-components` |
| Graph change in one model (model A) | before merge, run **one other model** as a witness (catches shared-machinery breakage cheaply) |
| Major change / release candidate | run **all five** |

Fast witnesses: `--models qwen38` (single 3090) or `--models ling31`.

## How to re-baseline / run the gate

```bash
bench/regression_gate.sh ./build_master/moestream                  # all five (~15 min)
bench/regression_gate.sh ./build_master/moestream --models qwen38,ling31
bench/regression_gate.sh ./build_master/moestream --models ling31   # one specific model
bench/regression_gate.sh --help
```

The gate prints the binary's sha256 and the checkout revision, runs each model's recipe, extracts
`[REQUEST-DECODE-OUTPUT]` (falling back to `Rates:` `decode=`), and FAILs any selected model with a
missing measurement or an out-of-band rate. Coherence is *not* auto-checked — read the logged text.

## Common gotchas (all models)

- **Build must link the host-node hook.** See the note above; verify with
  `nm -D build_master/moestream | grep sched_early_cpu`.
- **Preflight RAM margin.** `startup_preflight` needs `host_ram_gb + 21.5 GiB` of `MemAvailable`;
  several recipes exceed it on a 125 GiB box, so the gate runs them under `scripts/run_capped.sh`
  with `--force-start` (kernel `MemoryMax` + `MemorySwapMax=0` bounds the risk).
- **Benign teardown crash.** DeepSeek and GLM 5.3 abort at `exit()` with `CUDA error: driver
  shutting down` (a `ggml` CUDA buffer freed in a static/`unique_ptr` destructor during teardown).
  Decode completes and all metrics print first; it is pre-existing, not a merge regression.
- **Non-determinism.** Judge by long-run coherence, not exact tokens.
