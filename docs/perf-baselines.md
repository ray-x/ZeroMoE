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

### Qwen decode baselines by placement (measured 2026-10-16, `build_native`, 3090 only, 10 CPU threads)

Env for all rows: `CUDA_VISIBLE_DEVICES=0 MOESTREAM_QWEN4EXP_FULL_GRAPH=1 ..._FULL_GRAPH_DECODE=1 ..._GRAPH_PRELOAD_TRAILING=1`
(`GRAPH_PRELOAD_LAYERS=40` for the split, `0` for the hybrid), `MOESTREAM_BENCH_SILENT=1 MOESTREAM_BENCH_IGNORE_STOP=1`.

| placement | config | metric | baseline | notes |
|---|---|---|---|---|
| split: 22 layers fully on GPU, 26 fully on CPU (UD-IQ1_M) | `configs/qwen4exp_tg400_iq1m_22_26.toml` with `omp_max_threads = 10` | `[REQUEST-DECODE-WALL]`, 16-token prompt "Tell a story.", 400 tokens | **28.66 tok/s** (24.6 / 26.0 with an indexer using ~1.5 cores) | reference 26.2-26.9 (2026-10-02); how it was achieved: `docs/qwen-decode-baselines-2026-10-16.md`, run with `bench/qwen_decode_baselines.sh split 10 400` |
| hybrid: usage-ranked hot experts in VRAM, K = 14,554 of 24,576 (Q2_0) | `configs/qwen4exp_q2_0_option_b_global_topk48.toml`, `omp_max_threads = 10`, K via `bench/option_b_vram_probe.sh 14554` | mean `iter_total`, steps >= 3, 200 tokens | **43.12 tok/s** (41.3 in the first run) | run with `bench/qwen_decode_baselines.sh hybrid 14554 200 10`; arena 18.74 GiB, steady free VRAM 1.62 GiB, coherent text, no fallback; reference 39-42 (2026-10-05) |
| partial residency + RAM tier (UD-IQ1_M) | `configs/qwen4exp_3090_only.toml` | decode | 19.5 tok/s | the config the 19.5 row above was measured on; not the fast path |

Thread count matters as much as any code change here. The CPU part of the graph is OpenMP-barrier-bound, so one thread that
shares a core with another process stalls all of them: the same split config measured **3.05 tok/s with 12 threads** while a
background `clangd --background-index` and a niced ggml compile were running, and 24.6-26.0 tok/s with 10 threads. Run with
`omp_max_threads` at (cores - 2) unless the box is known quiet, and check `/proc/loadavg` before a decode measurement.

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

## GLM-5.3 (UD-IQ2_M, glm-dsa), layered path, 3090 + dense on 5060 Ti (2026-10-18)

426-token prompt, 48 new tokens, temperature 0, `MOESTREAM_DENSE_SCRATCH_RESERVE_MB=768`. Single samples.

| Setup | Decode tok/s | Prefill tok/s | Disk GB/s | Hit |
|---|---|---|---|---|
| one NVMe | 0.88 | 1.82 | 2.60 | 85.8% |
| + sparse second-drive mirror (`gguf_tool mirror-subset`, 47.7 GiB, 19.6% of reads from drive 2) | 0.96 | 1.84 | 2.93 | 86.7% |

The mirror gives about +9% decode and +1% prefill (single sample, identical output). Prefill still reads ~150 GB per 426-token prompt on
this path; only a batched graph prefill, where each cold expert is read once per batch, can approach 10 tok/s.

GLM_DSA full graph (opt-in: `MOESTREAM_GLM_DSA_FULL_GRAPH=1`, `..._PREFILL=1`; branch `glm53-layered-speed`), same 426-token prompt,
3090 only as the dense+moe device (`logs/glm53_dsa_graph.toml`, 4 GiB VRAM expert cache), SSD offload store of the 12,480 coldest experts
(`gguf_tool offload`, 132.7 GiB on /data, byte-verified) and a 45 GiB DDR pool (`MOESTREAM_GLM5NEXT_OFFLOAD_*`, see `logs/run_dsa_graph_426.sh`):

| Phase | tok/s | Note |
|---|---|---|
| Prefill (graph, one batch of 426) | 4.99 | 2.7x the layered 1.82; 85.4 s; disk-bound (~1.9 GB/s of cold reads) |
| Decode (graph) | 0.69 overall, ~0.7-0.75 steady (1.3-1.5 s/token) | BELOW the layered 0.88-0.96: 32% of routes miss the 45 GiB pool |

Output is coherent and the 17-token sky prompt reproduces the layered answer ("Rayleigh scattering"); a token-for-token comparison against the layered
path is still to do. A 64 GiB pool exhausts host RAM (MemAvailable < 20 GB, decode graph declines); ~45-50 GiB is the ceiling with 14 GiB of live host data.
The graph needs the dense weights on the same GPU as its workspace: with the 5060 Ti as the core device it had 0.14 GiB free and declined.

GLM_DSA full graph on BOTH GPUs (3090 = dense + graph device with a 4 GiB expert cache, 5060 Ti = 11 GiB of hot experts via `MOESTREAM_GLM5NEXT_GPU1_HOT_GIB=11`,
45 GiB DDR pool; `logs/glm53_dsa_graph2gpu.toml`, `logs/run_dsa_graph2gpu_426.sh`), same 426-token prompt:

| Phase | tok/s | Note |
|---|---|---|
| Prefill (graph) | 5.12 | 83.2 s; same as single GPU (4.99): disk-bound, the 5060 Ti hot set does not help |
| Decode (graph) | 0.68 overall; 0.86 steady (mean 1160 ms/step) | the first token carries ~20 s of graph build + state seeding |

Setup notes: the graph device is the one holding the dense weights (13 GiB), so with the 5060 Ti as the core device the workspace has 0.02-0.14 GiB free and every prefill batch
declines; roles must be 3090 = dense+moe, 5060 Ti = moe. After a two-GPU build the current CUDA device is the second GPU; `decode_prepare` must `cudaSetDevice(g->device)` before it seeds
the KV state, otherwise the seed fails and the decode graph is declined.

### GLM_DSA graph: what limits it (2026-10-19, same 426-token prompt, both GPUs, 45 GiB pool)

| Variant | Prefill tok/s | Decode mean step | Note |
|---|---|---|---|
| baseline (2 GPU, 6 CPU threads) | 5.12 | 1160 ms | |
| + sparse mid-tier mirror on KIOXIA (1,246 experts, 13.3 GiB) | 5.00 | 1377 ms | WORSE: KIOXIA already served ~78% of reads (store 17.9k + mirror 2.0k vs Samsung ~5.6k) |
| + split reads (`MOESTREAM_GLM5NEXT_OFFLOAD_SPLIT=1`: store experts read gate/up-down across both drives, 3.5 ms vs 7.5 ms per read in isolation, bytes identical) | 4.89 | 1164 ms | no change: the SSDs are NOT the bottleneck |
| + 14 CPU threads (`MOESTREAM_GLM5NEXT_CPU_THREADS=14`) | 6.18 | 1311 ms | prefill +21%, decode worse: cold-expert MUL_MAT_ID (225 nodes) runs on the CPU |
| + `PREFILL_OP_OFFLOAD=1` | crash | | `CUDA error: invalid argument` in ggml_backend_cuda_set_tensor_async during prefill |

Conclusion: drive bandwidth is not what limits the graph (halving read time changed nothing). Prefill is limited by the CPU matmuls of experts outside VRAM,
decode by the per-token miss handling. The earlier "disk-bound" statement in this file was wrong. Levers left: a separate (higher) CPU thread count for
prefill only, getting op-offload to work with the offload arena, and a larger VRAM-resident expert set.

### GLM_DSA graph prefill: 12.2 tok/s with GPU op-offload (2026-10-19)

Per-op profile of the 6.2 tok/s prefill: 61 of 69 s were `MUL_MAT_ID` on the CPU (iq2_xxs 41.2 s, iq3_xxs 18.8 s); everything on the GPUs was under 1 s. Letting the
scheduler run the host-resident expert matmuls on the GPU (`MOESTREAM_GLM5NEXT_PREFILL_OP_OFFLOAD=1`) removes that:

| Prefill, 426 tokens, both GPUs, 45 GiB pool | tok/s | Wall |
|---|---|---|
| CPU expert matmuls, 6 threads | 5.12 | 82 s |
| CPU expert matmuls, 10-14 threads | 6.2 | 69 s |
| GPU op-offload (this) | **12.16** | **35 s** |

Output text identical to the CPU-matmul run (181 chars compared). Decode is unchanged (0.66 tok/s overall, mean step 1131 ms, 6 threads).

It needs a ggml that has (a) `patches/ggml/0006` (batched MMQ treats negative ids as inactive; the prebuilt `moestream-build-upstream-master` ggml-cuda dates from Oct 4, before it, and
crashes with `illegal memory access`) and (b) `patches/ggml/0007` (arena-safe expert copy; without it: `CUDA error: invalid argument`, or `GGML_ASSERT(id >= 0 ...)` on the -1 route ids). The in-tree build
did not apply 0007 by itself; it was applied by hand to a private copy of the committed integration source (`~/work/ggml-0006-src`), built as `build_g6`
(`cmake -S . -B build_g6 -DMOESTREAM_GGML_ROOT=<that ggml> -DCMAKE_CUDA_ARCHITECTURES="86;120" ...` with no PREBUILT dir). Run with `logs/run_dsa_graph2gpu_426.sh`.

### GLM_DSA graph decode: pilot prefetch (2026-10-19, 426-token prompt, 48 new tokens, both GPUs, 45 GiB pool, single runs)

Decode profile before prefetch (per token): SSD-miss callback 649 ms (169 misses/token, 1.9 GB read at ~3.4 GB/s = both drives' bandwidth), CPU expert matmuls ~350 ms
(`iq2_xxs` 190 + `iq3_xxs` 152), everything on the GPUs ~75 ms. Decode is miss-bandwidth-bound; the earlier "disk is not the bottleneck" holds for prefill only.

| Decode config | Mean step | Misses/token | Decode tok/s (48 tok, incl. ~20 s first-token graph build) |
|---|---|---|---|
| baseline | 1131 ms | 169 | 0.66 |
| `OFFLOAD_ASYNC=1 PILOT_PREFETCH=8 PILOT_COPY=0 PILOT_LOOKAHEAD=2` only (pilot stage NOT active: needs `PILOT_STAGE`>0) | 1059 ms | 168 | 0.71 |
| + `PILOT_STAGE=8` (**best**) | **934 ms** | **67** | **0.75** (steady ~1.07 tok/s) |
| same, `PILOT_LOOKAHEAD=3` | 1019 ms | | 0.69 |
| same, lookahead 2, 8 CPU threads, `OFFLOAD_THP=1` | 1049 ms | | 0.66 |

Best config: `logs/ab_pilot.sh` env minus the swept knobs. Target 1.7 tok/s decode is not reached; 590 ms/token would be needed. Remaining per-token cost at the best config: ~417 ms miss
callback, ~350 ms CPU expert matmuls, ~75 ms GPU. Known issue: `CUDA error: driver shutting down` at process exit after the pilot stage runs (output complete; teardown order).

### GLM_DSA graph: DDR pool size and GPU1 hot set (2026-10-19, best pilot config, 426-token prompt, single runs)

| Pool GiB | GPU1 hot GiB | Mean decode step | Decode tok/s (overall) | Prefill tok/s | MEM-GUARD blocks |
|---|---|---|---|---|---|
| 45 | 11 | 934 ms | 0.75 | 12.2 | 0 |
| 54 | 11 | 888 ms | 0.75 | 12.5 | 0 |
| 45 | 14 | 964 ms | 0.68 | 12.1 | 0 |
| 60 | 11 | 848 ms | 0.77 | 12.8 | 4 |
| 64 | 11 | 824 ms | 0.76 | 9.4 | 5 |

A bigger pool buys ~10% on the decode step up to 60 GiB; at 64 GiB host memory hits the 20 GB stop threshold and prefill drops 26%. More GPU1 hot experts do not help. 54 GiB is the safe setting
(no guard events); 60 GiB is the edge. Steady-state decode at 848 ms is ~1.18 tok/s; the 1.7 tok/s target (590 ms/token) is not reached by pool size alone.

### GLM_DSA graph: correctness check against layered, corrected numbers, GPU utilisation (2026-10-19)

**Correctness.** Greedy (temp 0) comparison with the layered path (`logs/check/`): "What is the capital of France?" -> `Paris`; a 399-token prompt with a planted code -> `7391`
(recalled from the middle of the context). The graph with 11 GiB of hot experts on the 5060 Ti prepends an extra `</think>` token on both prompts and in every earlier 2-GPU run; the graph
with `GPU1_HOT_GIB=0`, or with `MOESTREAM_GLM5NEXT_PREFILL_GPU1=0` (hot experts computed through the cold branch during prefill), matches layered exactly. So the in-place GPU1 hot-expert
compute in the batched prefill graph is wrong for this model (cause not yet found; decode with GPU1 hot experts is fine). **All earlier 2-GPU numbers in this file that did not set
`PREFILL_GPU1=0` ran with that defect** (prefill 12.2-12.8 tok/s; they are speed-only, the text diverged from layered).

**Correct configuration** (`PREFILL_GPU1=0 GPU1_HOT_GIB=11 PILOT_STAGE=8 OFFLOAD_ASYNC=1 PILOT_PREFETCH=8 PILOT_COPY=0 PILOT_LOOKAHEAD=2`, 54 GiB pool, 426-token prompt):

| Run | Prefill tok/s | Decode mean step | Decode tok/s overall |
|---|---|---|---|
| final1 | 12.02 | 973 ms | 0.69 |
| util | 11.85 | n/a | 0.71 |
| GPU1 hot = 0 | 11.93 | 1025 ms | 0.85 |

(the hot=0 run's higher overall decode figure is a shorter first-token build, not a faster steady state.) Steady decode is ~1.0 tok/s; layered is 0.88-0.96.

**GPU utilisation** (`nvidia-smi dmon`, 1 s samples, correct config): prefill 3090 46% avg (max 68%), 5060 Ti 0%; first-token graph build 14% / 10%; **decode 3% (max 6%) on the 3090 and 1% (max 2%) on the 5060 Ti**.
Decode is >95% waiting on CPU expert matmuls and the SSD-miss callback, which is why GPU count, hot-set size and thread tuning barely moved it.

## GLM-5.3 DSA full graph: decode vs layered profile (2026-10-19)

Config: 3090 dense+moe, 5060 Ti moe, PREFILL_GPU1=0, GPU1_HOT_GIB=11, pilot stage 8 / prefetch 8 / lookahead 2, DDR pool 54 GiB, 426-token prompt.

- Decode graph build (first token only): layer loop 17.0 s = attn 1 ms + pilot 128 ms + FFN incl. expert/hot uploads 14.8 s; sched alloc 11 ms. Prefill graph build: 2.7 s.
- Steady decode step: 875-1270 ms (mean about 970 ms, about 1.0 tok/s), the same as the layered path (0.88-0.96 tok/s over 48 tokens).
- The lower "overall" decode figure (0.69-0.71 tok/s) is this one-time 17 s hot-expert upload amortised over a short run, not a slower steady state.
- GPU utilisation 3%/1% is expected: per step about 400 ms is the SSD-miss callback, about 350 ms CPU expert matmuls, about 75 ms GPU compute.
