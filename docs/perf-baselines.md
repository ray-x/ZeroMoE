# Performance baselines and update rule

Rule (set by the owner, 2026-10-16): update this file every time a measured number improves, and record the config that
produced it. Any measurement more than 5% below its baseline is reported to the owner at once, with the suspected cause or
"not investigated".

Hardware: RTX 3090 (24 GB, PCIe Gen3 x16) + RTX 5060 Ti (16 GB, PCIe Gen3 x8), Xeon W-2235, 125 GB RAM, zram swap only.
Tokens are the engine's own tokenizer counts; prefill figures are the engine's `[PREFILL-DONE]` line.

## GLM-5.3-Flash (AJ IQ2_XXS), dual GPU

Config: `configs/glm53flash_AJ_IQ2XXS_spread_offload.toml` (`moe_cache_gb = 15.0`, `context_size = 8192`), recipe env
`bench/glm_recipe_env.sh` (`CUDA_VISIBLE_DEVICES=0,1`, `GPU1_HOT_GIB=15.0`, `GPU1_MIN_LAYER=31`, `CPU_THREADS=10`, ranking
`docs/ssd-offload/data/glm53flash_usage_base_plus_decode100.txt`, `FULL_GRAPH_PREFILL=1`, `PREFILL_OP_OFFLOAD=1`).
Prefetch ring on (2 sets, drain, chunk 2), DDR pool 60-76 GiB, batch default 2048.

| metric | baseline | date | notes |
|---|---|---|---|
| prefill, 805 tokens | **251 tok/s** (3.20 s) | 2026-10-15 | `OFFLOAD_PINNED=1.0` |
| prefill, 814 tokens | 243 tok/s (3.35 s) | 2026-10-16 | recipe default `OFFLOAD_PINNED=0.5` (-3%) |
| prefill, 1689 tokens | 264 tok/s | 2026-10-16 | one graph |
| prefill, 3867 tokens | 263 tok/s | 2026-10-16 | 2048 + 1819 |
| prefill, 5117 tokens | 232 tok/s | 2026-10-16 | 2048 + 2048 + 1021 |
| prefill, 4090 tokens (peak) | **272.7 tok/s** | 2026-10-16 | full record: `docs/perf-record-2026-10-16.md` |
| prefill, 2083 tokens | 191 tok/s (was 146: tail stepped) | 2026-10-16 | fixed: first batch shortened so the tail >= 64 |
| decode, tokens 16-32 | 70.1 ms/token | 2026-10-15 | `moe_cache_gb` 15.8 -> 15.0: 70.7 (+0.9%) |
| prefill, before this work | 159 tok/s (5.05 s) | 2026-10-14 | for the record |

## Qwen 3.8 Flash (UD-IQ1_M), single 3090

Config: `configs/qwen4exp_3090_only.toml` (`moe_cache_gb = 11.5`, `context_size = 8192`), env as in the Qwen decode
harness: `MOESTREAM_QWEN4EXP_FULL_GRAPH=1 ..._FULL_GRAPH_DECODE=1 ..._GRAPH_PRELOAD_LAYERS=0 ..._GRAPH_PRELOAD_TRAILING=1
MOESTREAM_QWEN_CPU_THREADS=10 MOESTREAM_QWEN_EARLY_READY_ALL_SPLIT=1`. Since 2026-10-16 the prefill scheduler runs the
host-resident experts on the GPU (`MOESTREAM_QWEN4EXP_PREFILL_OP_OFFLOAD`, default 1; one pipeline copy; ggml patch 0007).

| metric | baseline | date | notes |
|---|---|---|---|
| prefill, 333 tokens | **102 tok/s** (3.27 s) | 2026-10-16 | GPU experts |
| prefill, 1611 tokens | **312 tok/s** (5.17 s) | 2026-10-16 | was 61-63 with CPU experts |
| prefill, 2033 tokens | **355 tok/s** (5.72 s) | 2026-10-16 | |
| prefill, 3989 tokens | **507 tok/s** (7.87 s) | 2026-10-16 | one graph, no batching needed |
| prefill, 5895 tokens (peak) | **548 tok/s** | 2026-10-16 | full record: `docs/perf-record-2026-10-16.md` |
| decode, short context | 19.5 tok/s (51 ms/token) | 2026-10-16 | 16.5 tok/s at 4-6k context |
| prefill, 1611 tokens, CPU experts (`OP_OFFLOAD=0`) | 61-63 tok/s | 2026-10-16 | previous behaviour |
| prefill, 2824 tokens (recorded) | 138 tok/s | 2026-09 | older "ggml MUL_MAT_ID prefill" path |

BeeLlama (`~/Projects/beellama.cpp/build-qwen38-rtx3090/bin/llama-bench`), same model file, same card,
`CUDA_VISIBLE_DEVICES=0 -ngl 99 -fa 1 -t 10 -r 2`:

| `-ncmoe` | `-ub` | pp512 | pp2048 |
|---|---|---|---|
| 25 | 512 | 199.6 | 204.4 |
| 25 | 2048 | 205.4 | - |
| 48 | 512 | 209.6 | 214.9 |
| 48 | 2048 | 207.4 | 257.4 |
| 48, `-nopo 1` (no op-offload) | 512 | 30.6 | 30.3 (pp1536) |

The Qwen graph's prefill had been ~3x below BeeLlama because the scheduler was created with `op_offload=false` (cold experts
on the CPU, the AVX2-width IQ1_M kernel; BeeLlama's CPU-only number is 30.6). With op-offload on and one pipeline copy the
reserve is 0.6 GiB + ~2.6 MiB/token (4.7 GiB at 1611 tokens; 44 GiB with the four pipeline copies decode uses).
Last-row hidden is within relL2 0.19 of the CPU-expert run, same tokens.

## Accepted scope for long-prompt prefill (owner decision, 2026-10-16)

Every chunk of a long prompt runs through the graph for GLM (1152-5117 tokens tested, `context_size = 8192`) and Qwen
(333-3989 tokens tested, `context_size = 8192`; one graph up to ~7k tokens with GPU op-offload). Accepted bounds:
prompts past `context_size` are refused with the value to raise; Qwen has no non-graph prefill under the graph-only expert
placement; GLM's selection beyond top-k has consistency evidence only (no independent reference in the engine).
