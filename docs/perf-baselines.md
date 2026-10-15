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
| decode, tokens 16-32 | 70.1 ms/token | 2026-10-15 | `moe_cache_gb` 15.8 -> 15.0: 70.7 (+0.9%) |
| prefill, before this work | 159 tok/s (5.05 s) | 2026-10-14 | for the record |

## Qwen 3.8 Flash (UD-IQ1_M), single 3090

Config: `configs/qwen4exp_3090_only.toml` (`moe_cache_gb = 11.5`, `context_size = 8192`), env as in the Qwen decode
harness: `MOESTREAM_QWEN4EXP_FULL_GRAPH=1 ..._FULL_GRAPH_DECODE=1 ..._GRAPH_PRELOAD_LAYERS=0 ..._GRAPH_PRELOAD_TRAILING=1
MOESTREAM_QWEN_CPU_THREADS=10 MOESTREAM_QWEN_EARLY_READY_ALL_SPLIT=1`. Cold experts run on the CPU (`op_offload=false`).

| metric | value | date | notes |
|---|---|---|---|
| prefill, 1611 tokens | 61-63 tok/s | 2026-10-16 | one graph, 25.5 s |
| prefill, 3989 tokens | 61 tok/s | 2026-10-16 | 1448 + 1448 + 1093 |
| prefill, 2824 tokens (recorded) | **138 tok/s** | 2026-09 | older config/path, not reproduced |

**Open regression flag:** the current Qwen prefill is ~56% below the recorded 138 tok/s. Cause not established: the older
number came from a different path ("ggml MUL_MAT_ID prefill" era). My changes do not touch the single-batch path.
