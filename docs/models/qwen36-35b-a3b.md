# Qwen3.6-35B-A3B UD-IQ2_M

## Status

CUDA support is accepted for text-generation decode of the user-provided Qwen3.6-35B-A3B-UD-IQ2_M GGUF (`qwen35moe`) on the RTX 5060 Ti. Other quantizations and multimodal inputs are outside this validation. The model uses the shared full-graph runtime. The full-fit graph, all-expert placement, llama.cpp parity checks, and the project performance gates pass.

The measured 1-token-prefix / 400-token decode is 115.21 tok/s median in ZeroMoE (115.12–115.38) and 120.67 tok/s in llama.cpp. ZeroMoE is 95.9% of the llama.cpp graph-gate rate and above the requested 110 tok/s minimum. The detailed comparison and build artifacts are recorded below.

Qwen3.6 MTP is disabled and was not tested, as requested.

## GGUF architecture and adapter

The GGUF reports `general.architecture = qwen35moe`, 40 transformer blocks, 256 routed experts per MoE block, top-8 routing, a shared expert, and a mixture of Gated DeltaNet and full-attention blocks. The adapter explicitly accepts this architecture and validates its tensor topology before allocating the full graph.

The adapter reuses the shared full-graph runtime, Qwen GDN and full-attention builders, routed-MoE builder, placement planner, weight arena, and recurrent-state cache. Model-specific equations remain in the Qwen adapter. If the memory planner says the full graph fits, all 40 expert layers are graph-owned on the selected CUDA device. A graph build or execution failure is fatal; it cannot route into the legacy per-expert executor.

## Full-fit placement on the RTX 5060 Ti

The final 4096-context run reported a required memory plan of 14,007,948,866 bytes and 16,365,256,704 available bytes. The planner returned `Fits`. The graph owns the full device block `[0, 40)`, including all 40 expert layers; the 9.0957 GiB of routed expert weights are part of the graph-owned device storage alongside dense weights, KV/GDN state, and graph workspace.

The full-fit run passed `check_full_graph_moe_run.py`. Its graph gate recorded one graph prefill, one graph decode, zero one-token layered forwards, and zero legacy per-expert dispatches. The 400-token performance run also passed the same gate with four graph-prefill runs, 1,204 graph-decode replays, and zero layered or legacy dispatches.

For the matched performance workload at context 401, the planner required 13,856,601,666 bytes and had 16,365,256,704 available bytes. This also returned `Fits` and kept all expert layers on CUDA1.

## llama.cpp operation parity

The earlier mismatch came from changing the normalization operation sequence. llama.cpp's Q/K path applies RMS normalization with epsilon divided by the head dimension, then scales by the reciprocal square root of the head dimension. ZeroMoE had used `ggml_l2_norm`. The expressions are mathematically equivalent, but the floating-point operation order differs; the small Q/K error accumulated through later blocks and caused near-boundary route changes.

ZeroMoE now follows the llama.cpp RMSNorm-then-scale sequence in the same order. On a teacher-forced decode with prompt token IDs `39113, 264, 3255, 13` and forced token `146377`, the final build matches llama.cpp bitwise at all 40 layers for:

- The pre-MoE hidden state, router logits, router probabilities, routed expert IDs, and normalized route weights.
- The routed expert output, shared expert output, and combined MoE output.
- The detailed gate, up, SwiGLU, and down tensors at layers 0, 1, 14, 17, and 37.
- The GDN Q/K normalized tensors and GDN output projection at those same sampled layers.

This includes the three layers (14, 17, and 37) where the prior trace selected a different near-boundary expert. Their routes now match exactly. The compact comparison data and per-field results are persisted in `build_qwen36_sm120/results/qwen36_teacher_forced_final_20261027/`; `parity_summary.json` records the 40-layer exactness counts.

The trace used context 416, 10 CPU threads, the same GGUF and CUDA1 device in both engines, and no MTP.

## Performance on the RTX 5060 Ti

The final matched decode benchmark uses the same GGUF, RTX 5060 Ti (`CUDA1`), one-token prefix, 400 generated tokens, context 401, and 10 CPU threads. llama.cpp's effective context is 401 (`1 + 400`). Both engines place all model layers on CUDA1 and use no CPU MoE experts.

| Measurement | ZeroMoE | llama.cpp |
|---|---:|---:|
| Steady decode | 115.21 tok/s median (115.12–115.38) | 120.67 tok/s average |
| Full-graph gate decode | 115.73 tok/s | 120.67 tok/s |
| End-to-end, 401 tokens | 112.14 tok/s median | Not used for the acceptance ratio |
| TTFT | 107.6 ms median | Not used for the acceptance ratio |
| Prefill, 16-token prompt | 167.8 tok/s median | 595.4 tok/s average (418–703 across three samples) |

The `--minimum-tps 110` and default 90% llama.cpp checks both pass. The short-prompt prefill measurement remains much slower than llama.cpp and is reported separately; the onboarding decode threshold is based on steady-state decode.

The first full-graph build in the same benchmark took 3.162 s for node construction. Loading/binding the 9.10 GiB expert weights took 2.932 s, including 1.476 s of VRAM file reads. H2D copies took 1.417 s for expert weights and 0.223 s for dense weights. These are one-time model/graph setup costs, separate from steady-state decode.

## Build provenance and persistent reference binary

- ZeroMoE source commit: `de4883fdba93d9b7c1752fb964ae6d7b107921ea`, with the current working-tree changes included in the built binary.
- GGML: 0.27.0 from upstream base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`. The generated ZeroMoE `ggml-version.h` was checked and reports this exact commit.
- ZeroMoE applies the tracked patch series in [`patches/README.md`](../../patches/README.md), including its project-specific CUDA and CPU patches. llama.cpp uses the upstream GGML source, so this is a product-level comparison, not a kernel-identical A/B.
- CUDA toolkit 13.3.73, driver 610.57.04, GNU 16.2.1. The reference binary contains native `sm_120a` code (and `sm_86` compatibility code).
- The persistent llama.cpp runtime bundle is `build_llama_qwen36_sm120a_ref/bin/llama-bench` plus its adjacent shared libraries. `BUILD-INFO.md` records the build. Its ELF RUNPATH is `$ORIGIN`, and the copied binary resolves its libraries from this persistent folder rather than `/tmp`.
- The final ZeroMoE binary and result artifacts are under the ignored, persistent `build_qwen36_sm120/` directory. They are not recreated on machine reboot.

The reference llama.cpp source commit is `8baabd53b77d1c48c6ef8d3b620436292f3f6231`. The 5060 Ti is `CUDA1`; both GPUs were visible during the final benchmarks.

## Reproduction

The matched ZeroMoE workload uses the dedicated context-401 profile. Create the ignored result folders before running:

```sh
mkdir -p build_qwen36_sm120/results build_llama_qwen36_sm120a_ref/results
```

```sh
CUDA_VISIBLE_DEVICES=0,1 \
ZEROMOE_QWEN4EXP_FULL_GRAPH=1 \
ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE=1 \
ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS=0 \
ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1 \
ZEROMOE_QWEN_CPU_THREADS=10 \
ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT=1 \
build_qwen36_sm120/zeromoe \
  --config configs/bench/qwen36_35b_a3b_iq2m_5060_tg400.toml --bench \
  --bench-random-pp 16 --bench-random-tg 400 --bench-reps 3 \
  --bench-warmup 1 --bench-seed 1 \
  --bench-out build_qwen36_sm120/results/qwen36_fullfit_normfix_context401.json
```

Run the matching llama.cpp decode benchmark and both gates:

```sh
CUDA_VISIBLE_DEVICES=0,1 build_llama_qwen36_sm120a_ref/bin/llama-bench \
  --model /home/ray/vllm-workspace/Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
  --n-prompt 0 --n-gen 0 -pg 1,400 --repetitions 3 --threads 10 \
  --n-gpu-layers 99 --device CUDA1 --output json \
  > build_llama_qwen36_sm120a_ref/results/qwen36_pg1_tg400.json
python3 tools/check_full_graph_moe_run.py \
  build_qwen36_sm120/results/qwen36_fullfit_normfix_context401.log \
  --full-fit --require-all-expert-layers-on-device --require-full-graph-prefill
python3 tools/check_full_graph_moe_perf.py \
  build_qwen36_sm120/results/qwen36_fullfit_normfix_context401.log \
  build_llama_qwen36_sm120a_ref/results/qwen36_pg1_tg400.json \
  --n-prompt 1 --n-gen 400 --device CUDA1 --minimum-tps 110
```
