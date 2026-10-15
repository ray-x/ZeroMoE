# GLM-5.3-Flash and Qwen 3.8 Flash: prefill and decode performance record (2026-10-16)

All numbers are measured on this box with the build in this repository (`graph-framework`, the commit that added this
file), two runs per point. Baselines and the update rule are in `docs/perf-baselines.md`.

## Hardware and software

| | |
|---|---|
| GPUs | RTX 3090 24 GB (PCIe Gen3 x16), RTX 5060 Ti 16 GB (PCIe Gen3 x8; used by GLM only) |
| CPU / RAM | Xeon W-2235 (6C/12T), 125 GB, zram swap only, `vm.swappiness=150` |
| Storage | NVMe (`/data/sdb` models) |
| Engine | `build_native/moestream` (CUDA, native SASS 86;120), ggml fork `~/work/ggml-iq2xxs` with patches 0006-0008 |
| Isolation | one engine at a time (startup lock), nothing else on the GPU, MemAvailable >= 90 GB before each run |

## How it was tested

Each point is a fresh engine process fed one prompt on stdin (`printf '%s\n:q\n'`), generating 96 tokens (128 for the short
prompt), greedy (temperature 0). Prompts are slices of project docs (markdown) cut by bytes, so token counts differ by model.
Prefill is the engine's own `[PREFILL-DONE] N tokens in S s (R tok/s)` line (excludes startup, expert preload and graph
build is inside it for batches after the first). Decode is steady-state, excluding the first token (graph setup):
- **GLM**: wall-clock between the `[GLM5NEXT-LANE] tokens=16/32/...` log lines, i.e. ms/token per 16-token window. *mean* =
  average of the windows of both runs, *best* = fastest window.
- **Qwen**: per-token `[GGML-QWEN-FULL-DECODE-TIME] pos=.. total=.. ms` (needs `MOESTREAM_QWEN4EXP_GRAPH_REPLAY_TIMING=1`,
  which adds no per-node callback); *mean* over 94-126 tokens, first token dropped; *best* = fastest token.
Free-running decode text differs between runs (GLM assist picks experts by readiness), so decode figures vary ~3-5%.

### GLM-5.3-Flash (AJ IQ2_XXS, dual GPU)

Model `/home/ray/vllm-workspace/GLM-5.3-Flash/GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf`. Config
`configs/glm53flash_AJ_IQ2XXS_spread_offload.toml` (`moe_cache_gb = 15.0`, `context_size = 8192`, `prefill_batch_tokens`
unset = 2048). Command shape (`bench/glm_decode_profile.sh` runs it):
```
. bench/glm_recipe_env.sh        # CUDA_VISIBLE_DEVICES=0,1, GPU1_HOT_GIB=15.0, GPU1_MIN_LAYER=31, CPU_THREADS=10,
                                 # ranking glm53flash_usage_base_plus_decode100.txt, FULL_GRAPH_PREFILL=1, PREFILL_OP_OFFLOAD=1
MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB=60 MOESTREAM_GLM5NEXT_GPU1_HOT_USAGE=$GLM_RANKING MOESTREAM_GLM5NEXT_OFFLOAD_USAGE=$GLM_RANKING
```
Prefill: graph with GPU expert compute, prefetch ring (when it fits), drain, one copy thread per GPU, batch 2048;
`OFFLOAD_PINNED` at its default 0.5.

| prompt tokens | prefill s (run 1 / 2) | prefill tok/s (mean) | decode ms/token mean (best) | decode tok/s |
|---|---|---|---|---|
| 24 (short) | 1.28 / 1.27 | 18.8 | 72.8 (69.7) | 13.7 (14.3) |
| 510 | 3.29 / 3.28 | 155.5 | 77.5 (75.2) | 12.9 (13.3) |
| 1036 | 5.71 / 5.72 | 181.3 | 83.8 (80.2) | 11.9 (12.5) |
| 2083 | 14.31 / 14.25 (before fix) -> 10.90 | 145.8 -> 191.2 | 85.3 (78.2) | 11.7 (12.8) |
| 4090 | 15.01 / 15.00 | 272.6 | 84.8 (79.9) | 11.8 (12.5) |

**Prefill peak 272.7 tok/s (4090 tokens); average over the four prompt sizes 200.2 tok/s** (after the fix below). The cost per
batch is roughly constant (all cold experts are restreamed, ~43 GB at ~19 GB/s over the two links), so throughput rises with
batch size. **Decode peak 14.3 tok/s (best 16-token window, short prompt); average 13.7 tok/s short, 11.7-12.9 at 0.5-4k
tokens of context** (attention and KV grow). Earlier record (benchmark prompt, teacher-forced): 15.3 tok/s average, peak
17.1 (`docs/ssd-offload/13-...md`); this record is free-running on different text.

**Regression found and fixed in this sweep:** a prompt of 2049-2111 tokens left a tail of fewer than 64 tokens after the
2048-token batch, which was stepped one token at a time (2083 tokens: 14.3 s, 146 tok/s). The first batch is now shortened so
the tail is a real batch (2016 + 67: 10.9 s, 191 tok/s).

### Qwen 3.8 Flash (UD-IQ1_M, single 3090)

Model `/data/sdb/models/qwen3.8-flash-next/UD-IQ1_M/` (3 shards, 69.4 GiB, 1.75 bpw). Config
`configs/qwen4exp_3090_only.toml` (`moe_cache_gb = 11.5`, `context_size = 8192`, graph batch = whole prompt). Environment
(as `bench/qwen_decode_profile.sh`):
```
CUDA_VISIBLE_DEVICES=0 GGML_CUDA_BF16_MMVF_CUBLAS=0 MOESTREAM_QWEN4EXP_FULL_GRAPH=1 MOESTREAM_QWEN4EXP_FULL_GRAPH_DECODE=1
MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_LAYERS=0 MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1 MOESTREAM_QWEN_CPU_THREADS=10
MOESTREAM_BENCH_IGNORE_STOP=1 MOESTREAM_QWEN_EARLY_READY_ALL_SPLIT=1 MOESTREAM_QWEN_EARLY_READY_LAYER13=0
MOESTREAM_QWEN4EXP_GRAPH_REPLAY_TIMING=1
```
Prefill: host experts run on the GPU (`MOESTREAM_QWEN4EXP_PREFILL_OP_OFFLOAD` default 1, one scheduler copy). Decode: full
graph with the experts on the CPU (no expert layers resident: `PRELOAD_LAYERS=0`).

| prompt tokens | prefill s (run 1 / 2) | prefill tok/s (mean) | decode ms/token mean (best) | decode tok/s |
|---|---|---|---|---|
| 28 (short) | 1.01 / 1.00 | 27.9 | 51.5 (49.3) | 19.4 (20.3) |
| 497 | 3.57 / 3.71 | 136.6 | 51.0 (49.0) | 19.6 (20.4) |
| 992 | 4.55 / 4.40 | 221.7 | 50.9 (49.5) | 19.6 (20.2) |
| 2033 | 5.68 / 5.66 | 358.7 | 58.9 (49.4) | 17.0 (20.2) |
| 3947 | 7.86 / 7.78 | 504.9 | 60.2 (57.3) | 16.6 (17.4) |
| 5895 | 10.76 / 11.31 | 534.8 (peak run 548.1) | 60.5 (57.1) | 16.5 (17.5) |

**Prefill peak 548 tok/s (5895 tokens); average over the five prompts >= 497 tokens 351 tok/s.** One graph per prompt in
all of these (workspace 0.6 GiB + ~2.6 MiB/token). **Decode peak 20.4 tok/s (best token, <= 1k context); average 19.4-19.6
tok/s up to ~1k tokens of context, 16.5-17.0 at 2-6k.** The single-3090 config keeps no expert layers on the GPU for decode;
Qwen with 40 layers resident reached 26-27 tok/s in September (different config), not re-measured here.

### BeeLlama reference (same model file, same card)

`CUDA_VISIBLE_DEVICES=0 llama-bench -m <shard 1> -ngl 99 -ncmoe 25|48 -fa 1 -t 10 -p 512,2048 -n 0 -ub 512,2048 -r 2`
(`~/Projects/beellama.cpp/build-qwen38-rtx3090`): pp512 200-210, pp2048 204-257 tok/s; with `-nopo 1` (no op-offload) 30.6.
MoEStream prefill is at or above BeeLlama from ~1k tokens (222 vs ~205 at 992; 359 vs 204-257 at 2033).

## Reproduce

Per point: build `build_native` at this commit, then run the command shapes above with the prompt sliced from
`docs/ssd-offload/{13,11,10,12}-*.md` (flattened to one line, first N bytes: GLM 1600/3200/6400/12800 B, Qwen
1300/2600/5200/10400/15300 B). Sweep driver and parser were in the session scratchpad; the logs are under
`logs/perf/recg_*` and `logs/perf/recq_*` in the working tree (untracked).
