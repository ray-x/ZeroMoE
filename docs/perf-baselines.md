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

Config: `configs/qwen4exp_3090_only.toml` (`moe_cache_gb = 11.5`, `context_size = 8192`). The 2026-10-16 harness used retired full-graph/preload environment names; full graph is now the default, and placement uses `graph_device_expert_layers` / `graph_device_expert_trailing` in TOML or `--graph-device-expert-layers` / `--graph-device-expert-trailing` on CLI. Other recorded settings include `ZEROMOE_QWEN_CPU_THREADS=10` and `ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT=1`. Since 2026-10-16 the prefill scheduler runs the
host-resident experts on the GPU (`ZEROMOE_QWEN4EXP_PREFILL_OP_OFFLOAD`, default 1; one pipeline copy; ggml patch 0007).

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

Historical 2026-10-16 runs used the retired full-graph/preload env aliases. Current equivalent: full graph default plus `graph_device_expert_layers` / `graph_device_expert_trailing` in TOML or generic `--graph-*` CLI options; the recorded placement used 40 layers for split and 0 for hybrid. Other env settings were `CUDA_VISIBLE_DEVICES=0`, `ZEROMOE_BENCH_SILENT=1`, and `ZEROMOE_BENCH_IGNORE_STOP=1`.

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

426-token prompt, 48 new tokens, temperature 0, `ZEROMOE_DENSE_SCRATCH_RESERVE_MB=768`. Single samples.

| Setup | Decode tok/s | Prefill tok/s | Disk GB/s | Hit |
|---|---|---|---|---|
| one NVMe | 0.88 | 1.82 | 2.60 | 85.8% |
| + sparse second-drive mirror (`gguf_tool mirror-subset`, 47.7 GiB, 19.6% of reads from drive 2) | 0.96 | 1.84 | 2.93 | 86.7% |

The mirror gives about +9% decode and +1% prefill (single sample, identical output). Prefill still reads ~150 GB per 426-token prompt on
this path; only a batched graph prefill, where each cold expert is read once per batch, can approach 10 tok/s.

GLM_DSA full graph (opt-in: `ZEROMOE_GLM_DSA_FULL_GRAPH=1`, `..._PREFILL=1`; branch `glm53-layered-speed`), same 426-token prompt,
3090 only as the dense+moe device (`obsolete/glm53-logs-scripts/glm53_dsa_graph.toml`, 4 GiB VRAM expert cache), SSD offload store of the 12,480 coldest experts
(`gguf_tool offload`, 132.7 GiB on /data, byte-verified) and a 45 GiB DDR pool (`ZEROMOE_GLM5NEXT_OFFLOAD_*`, see `obsolete/glm53-logs-scripts/run_dsa_graph_426.sh`):

| Phase | tok/s | Note |
|---|---|---|
| Prefill (graph, one batch of 426) | 4.99 | 2.7x the layered 1.82; 85.4 s; disk-bound (~1.9 GB/s of cold reads) |
| Decode (graph) | 0.69 overall, ~0.7-0.75 steady (1.3-1.5 s/token) | BELOW the layered 0.88-0.96: 32% of routes miss the 45 GiB pool |

Output is coherent and the 17-token sky prompt reproduces the layered answer ("Rayleigh scattering"); a token-for-token comparison against the layered
path is still to do. A 64 GiB pool exhausts host RAM (MemAvailable < 20 GB, decode graph declines); ~45-50 GiB is the ceiling with 14 GiB of live host data.
The graph needs the dense weights on the same GPU as its workspace: with the 5060 Ti as the core device it had 0.14 GiB free and declined.

GLM_DSA full graph on BOTH GPUs (3090 = dense + graph device with a 4 GiB expert cache, 5060 Ti = 11 GiB of hot experts via `ZEROMOE_GLM5NEXT_GPU1_HOT_GIB=11`,
45 GiB DDR pool; `obsolete/glm53-logs-scripts/glm53_dsa_graph2gpu.toml`, `obsolete/glm53-logs-scripts/run_dsa_graph2gpu_426.sh`), same 426-token prompt:

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
| + split reads (`ZEROMOE_GLM5NEXT_OFFLOAD_SPLIT=1`: store experts read gate/up-down across both drives, 3.5 ms vs 7.5 ms per read in isolation, bytes identical) | 4.89 | 1164 ms | no change: the SSDs are NOT the bottleneck |
| + 14 CPU threads (`ZEROMOE_GLM5NEXT_CPU_THREADS=14`) | 6.18 | 1311 ms | prefill +21%, decode worse: cold-expert MUL_MAT_ID (225 nodes) runs on the CPU |
| + `PREFILL_OP_OFFLOAD=1` | crash | | `CUDA error: invalid argument` in ggml_backend_cuda_set_tensor_async during prefill |

Conclusion: drive bandwidth is not what limits the graph (halving read time changed nothing). Prefill is limited by the CPU matmuls of experts outside VRAM,
decode by the per-token miss handling. The earlier "disk-bound" statement in this file was wrong. Levers left: a separate (higher) CPU thread count for
prefill only, getting op-offload to work with the offload arena, and a larger VRAM-resident expert set.

### GLM_DSA graph prefill: 12.2 tok/s with GPU op-offload (2026-10-19)

Per-op profile of the 6.2 tok/s prefill: 61 of 69 s were `MUL_MAT_ID` on the CPU (iq2_xxs 41.2 s, iq3_xxs 18.8 s); everything on the GPUs was under 1 s. Letting the
scheduler run the host-resident expert matmuls on the GPU (`ZEROMOE_GLM5NEXT_PREFILL_OP_OFFLOAD=1`) removes that:

| Prefill, 426 tokens, both GPUs, 45 GiB pool | tok/s | Wall |
|---|---|---|
| CPU expert matmuls, 6 threads | 5.12 | 82 s |
| CPU expert matmuls, 10-14 threads | 6.2 | 69 s |
| GPU op-offload (this) | **12.16** | **35 s** |

Output text identical to the CPU-matmul run (181 chars compared). Decode is unchanged (0.66 tok/s overall, mean step 1131 ms, 6 threads).

It needs a ggml that has (a) `patches/ggml/0006` (batched MMQ treats negative ids as inactive; the prebuilt `moestream-build-upstream-master` ggml-cuda dates from Oct 4, before it, and
crashes with `illegal memory access`) and (b) `patches/ggml/0007` (arena-safe expert copy; without it: `CUDA error: invalid argument`, or `GGML_ASSERT(id >= 0 ...)` on the -1 route ids). The in-tree build
did not apply 0007 by itself; it was applied by hand to a private copy of the committed integration source (`~/work/ggml-0006-src`), built as `build_g6`
(`cmake -S . -B build_g6 -DZEROMOE_GGML_ROOT=<that ggml> -DCMAKE_CUDA_ARCHITECTURES="86;120" ...` with no PREBUILT dir). Run with `obsolete/glm53-logs-scripts/run_dsa_graph2gpu_426.sh`.

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

Best config: `obsolete/glm53-logs-scripts/ab_pilot.sh` env minus the swept knobs. Target 1.7 tok/s decode is not reached; 590 ms/token would be needed. Remaining per-token cost at the best config: ~417 ms miss
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

**Correctness.** Greedy (temp 0) comparison with the layered path (`obsolete/glm53-logs-scripts/check/`): "What is the capital of France?" -> `Paris`; a 399-token prompt with a planted code -> `7391`
(recalled from the middle of the context). The graph with 11 GiB of hot experts on the 5060 Ti prepends an extra `</think>` token on both prompts and in every earlier 2-GPU run; the graph
with `GPU1_HOT_GIB=0`, or with `ZEROMOE_GLM5NEXT_PREFILL_GPU1=0` (hot experts computed through the cold branch during prefill), matches layered exactly. So the in-place GPU1 hot-expert
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

### GLM-5.3 DSA full graph: pilot depth sweep and I/O budget (2026-10-19)

| pilot stage/prefetch K | mean decode step (35 steps) | tok/s |
|---|---|---|
| 4 | 998 ms | 1.00 |
| 6 | 1009 ms | 0.99 |
| 8 | about 970 ms | about 1.0 |

- Pilot precision by rank (K=8): r1 96%, r2 87%, r3 76%, r4 64%, r5 52%, r6 41%, r7 32%, r8 26%.
- Per decoded token: about 250 pilot-prefetch loads (prefetch_loads/jobs, about 30 tokens) plus about 90 demand loads (rough, from cumulative counters), about 340 loads. At 10.9 MB per expert (store size / expert count) that is roughly 3.7 GB per token over about 1 s, i.e. about 3-4 GB/s, which is the SSD ceiling measured earlier (2.7-4.8 GB/s).
- Conclusion: decode is SSD-bandwidth-bound. Prefetch depth only moves loads around. Reaching 590 ms per step needs about 40% fewer bytes read per token, i.e. a lower cold miss rate (about 50% now), from a bigger DDR pool or a better eviction policy, not deeper prefetch.

### GLM-5.3 DSA full graph: segmented LRU vs LRU (2026-10-19)

Same config as above (K=8, 54 GiB pool, 426-token prompt, 35 steady steps), run alone on the box.

| policy | mean decode step | tok/s | demand misses at the last progress line |
|---|---|---|---|
| lru (default) | about 970 ms | about 1.0 | 3892 (different run length, not directly comparable) |
| seg, probation 0.25 | 1055 ms | 0.95 | 3200 |
| seg, probation 0.10 | 1069 ms | 0.94 | 3207 |

- Segmented LRU is about 8-10% slower even though it records fewer demand misses, so the demand-miss count alone does not predict step time. Likely it displaces prefetched experts, which are inserted as Prefetch kind. Not diagnosed further.
- The first attempt at this A/B was invalid: another session's engine (ling-3.1-flash worktree) was running on the GPU and the script read a stale log. Those numbers were discarded.
- Keep the default LRU.

## GLM-5.3 (glm-dsa) decode route survey: ranking refit does not help (2026-10-19)

Hypothesis: the DSA usage file (`.zeromoe_usage_glm53_ud_iq2m`, 35,400 counts) is thin, so a decode-only route survey
(`ZEROMOE_GLM5NEXT_ROUTE_TRACE`, 4 prompts x 47 decode tokens x 75 MoE layers, `obsolete/glm53-logs-scripts/survey_dsa_all.sh`) could raise the
resident-set hit rate and cut SSD bytes/token (needed: ~-40% for 1.7 tok/s).

Leave-one-prompt-out, fraction of routed experts inside the top-N ranked set (base = existing file; refit = base + 3 training prompts):

| N resident | base | refit (w=1) |
|-----------:|-----:|------------:|
| 2000 | 28.6% | 30.2% |
| 4000 | 41.3% | 43.7% |
| 5300 | 47.6% | 50.8% |
| 7000 | 56.2% | 59.4% |

Gain is ~+3 points (~6% fewer misses at best), far from -40%. Each prompt touches 8-9k of 19.2k experts; routing is prompt-specific.
Not A/B'd end to end (the DDR pool is LRU-managed; seeding only affects the initial fill). Closed negative.
Decode stays ~1.0 tok/s (~970 ms/step, SSD-bound); 1.7 tok/s is not reached.

### GLM-5.3 (glm-dsa) decode: O_DIRECT model-side reads do not help (2026-10-19)

Standalone random reads of 3.6 MB on the two drives (concurrent): O_DIRECT 3.3 GB/s (model drive, ~0.15 cores) + 2.85 GB/s (store drive);
buffered+fadvise reaches the same 3.35 GB/s but burns ~1.44 cores. The engine only achieves ~3-3.7 GB/s combined, so `ModelReader` was made to
read aligned planes with O_DIRECT (`ZEROMOE_GLM5NEXT_MODEL_DIRECT=1`; counter `model planes: direct=54533 buffered=0` confirms it is taken).

Same config (54 GiB pool, pilot stage 8, UD-IQ2_M, 426-token prompt, 48 new tokens, preflight-gated, cold cache):

| run | model reads | steady decode step |
|---|---|---|
| A | buffered | 887 ms |
| B/D/E | direct | 924 / 925 / 862 ms (D and E each showed one 57 s / 37 s stall mid-run; cause not established, buffered runs' full logs were overwritten so not comparable) |
| C | buffered | 940 ms (first 13 steps) |

Differences are inside run-to-run noise; the 1.4 cores freed from buffered reads are not what limits decode. Left opt-in (default off). Decode stays ~0.9-1.0 tok/s;
1.7 tok/s (590 ms/step) is not reached. Note a pre-existing exit-time CUDA abort in `~StageRuntime` after the run (harmless to results).

### Provenance of the "1.7 tok/s last recorded peak" (2026-10-19)

Searching every log in `logs/` for decode >= 1.6 tok/s finds exactly one line: `logs/check/short_layered.log`, "decode=1.74 tok/s (2 tokens / 1.15s)", the old layered
path with an 18-token prompt (the answer "Paris" plus end-of-generation). That is a 2-token sample, not a steady rate. The same layered path on the 399-token prompt
measured 0.98 tok/s (4 tokens / 4.07 s), and every multi-token run on any path is 0.77-1.18 tok/s. The full-graph path's best steady decode (848 ms/step, 1.18 tok/s)
is therefore at or above the layered path's real rate; 1.7 tok/s has never been measured over a meaningful number of tokens.

### GLM-5.3 (glm-dsa) decode: posting the next-layer pilot job before the demand load does not help (2026-10-19)

Tried moving `stage_post_job()` for the next layer ahead of the blocking `ensure(Demand)` in `stage_miss_cb`, so the prefetch worker's reads overlap the demand reads
(drives are at ~3-3.7 of ~6.4 GB/s). ABAB, same config (54 GiB pool, pilot stage 8, 426-token prompt, 48 new tokens, preflight-gated), steady step:
off 913 ms, on 907 ms, off 910 ms, on 910 ms. No effect; change reverted. Decode stays ~1.1 tok/s (~910 ms/step); 1.7 tok/s (590 ms/step) not reached.
Prefetch is already issued one layer ahead (lookahead 2); the demand misses that remain are experts the pilot did not predict, so there is nothing earlier to overlap them with.

### GLM-5.3 (glm-dsa) full graph: dense weights on the 5060 Ti (core = device 1), hot experts on the 3090 (2026-10-19)

Config `logs/glm53_dsa_core5060.toml` (5060 Ti `dense+moe`, 3090 `moe`, MLA qabsorb/score GPU caches off, 5060 `moe_cache_gb = 0`), 45 GiB DDR pool, 426-token prompt, `PREFILL_GPU1=0`, `GPU1_HOT_GIB=19`.

**OOM cause and fix.** Dense arena (~13 GiB) + CUDA context + the legacy MLA qabsorb/WUV/mirror GPU caches (~2.4 GB) left 0.02-0.13 GiB free on the 5060 Ti, so every prefill batch declined; later the 2 GiB graph-allocator floor (`graph_alloc_vram_headroom_ok`) refused the build at 1.33 GiB free. Fixed by turning the legacy MLA GPU caches off in the config and running with `ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB=768` (0.40 GiB left on the 5060 Ti after the build, so there is no margin for a bigger batch or longer KV).
The graph's hardcoded "second GPU = device 1" is now `second_gpu_device(core)` (backend, hot-expert buffers, offload ring, assist device). Hot experts are now preloaded onto the non-core GPU at boot (`ZEROMOE_GLM5NEXT_GPU1_HOT_PRELOAD=0` turns it off): 19 GiB in 17.5 s, previously uploaded during the first decode graph build.

| Layout | Prefill tok/s | Steady decode | Decode tok/s (40 tok, incl. build) |
|---|---|---|---|
| dense 3090, hot 11 GiB on 5060 (control, 45 GiB pool) | 11.82 | ~1124 ms | 0.65 (48 tok) |
| dense 5060 Ti, hot 19 GiB on 3090 | 11.71 | ~1157 ms | 0.78 |

No gain from the swap (+3% on step time is within noise). Output text repeats the prompt question rather than the control's outline: a different hot set changes numerics slightly; not verified with teacher forcing.

**GPU utilisation during decode (`nvidia-smi -l 1`, 44 s window): 3090 1% (max 2%, 117 W), 5060 Ti 4% (max 7%, 33 W).** Decode is waiting on the SSD-miss callback and CPU cold-expert matmuls, as in the earlier utilisation note. The user remembers 20-30% on both GPUs during the 1.4 tok/s run, so that run computed far more expert work on the GPUs than this graph path does (no log of it survives).

#### Trials 5-9: pilot prefetch, CPU lane and GPU-assist ring with dense on the 5060 Ti (2026-10-19)

Same config (`glm53_dsa_core5060.toml`, 426-token prompt, NGEN 40, `PREFILL_GPU1=0`). Steady = mean of the last 30 decode steps.

| Trial | Build / settings | Prefill tok/s | Steady ms/step | Decode tok/s |
|---|---|---|---|---|
| 5 | no pilot, hot 11 GiB, 45 GiB pool | 11.40 | 1095 | 0.82 |
| 6 | pilot stage 8 / prefetch 8 / copy 0 / lookahead 2, hot 16, 54 GiB | 11.76 | 1083 | 0.78 |
| 7 | + `ZEROMOE_GLM_DSA_CPU_LANE=1`, build_g6 (ggml without the early-CPU hook, lane could not build), hot 11 | 11.39 | 927 | 0.88 |
| 8 | same as 7 on build_g7 (ggml-0006 + early-CPU hook): lane enabled, GPU assist engaged | 8.44 | 872 | 0.93 |
| 9 | build_g7, lane off (CPU_LANE unset) | 11.64 | 900 | 0.91 |

Findings:
- `GlmDsaGraphModel::supports_cpu_lane()` was missing (the lane never existed for GLM_DSA); it is now gated by `ZEROMOE_GLM_DSA_CPU_LANE=1`. The lane also needs a ggml with the `zeromoe_ggml_sched_early_cpu` scheduler hook (`/path/to/ggml-0006-early`, `build_g7`).
- With the lane on, the GPU-assist ring runs only ~1.6 experts/token (14 layer-runs per 16 tokens), so the gain over lane-off is ~3% (872 vs 900 ms), within noise. Trial 8's prefill (8.44) is 27% below trials 7/9 (11.4-11.6); not diagnosed (lane on only in 8; variance or a lane side effect). Lane stays off by default.
- Prediction (pilot l+1 precision by rank): r1 96-97%, r2 89-90%, r3 81-82%, r4 66-69%, r5 56-57%, r6 41-45%, r7 33-34%, r8 25-27%. Misses are already under 1 per layer (~62 SSD loads/token at ~5.2 ms, ~306 ms/token of reads); the rest of the ~900 ms step is CPU cold-expert MUL_MAT_ID plus ~75 ms GPU. GPU utilisation 1-6%. Better prediction is not the lever at this point.
- The 3090 holds only the 11 GiB hot set (preloaded at boot, 11-12 s); `[VRAM-SLOTS] 0 / 349` in graph mode, i.e. the old usage-ranked pinned/LRU VRAM fill at boot is not replicated.
- `perf-tuning.md` in the repo root is a DeepSeek-V4 log (1.0-1.2 tok/s), not a GLM-5.3 1.4+ tok/s run, so it does not identify the earlier configuration.
- Decode target (1.7 tok/s) is NOT met: best 0.93 tok/s (872 ms/step, 590 ms/step needed). No lossy lever was used.

### Trials 10-11: CPU lane across all layers, with and without pilot prefetch (2026-10-19)

Same setup as trials 5-9 (dense on the 5060 Ti, `build_g7`, 426-token prompt, 40 tokens, no expert is dropped).

| trial | config | prefill tok/s | decode tok/s | notes |
|---|---|---|---|---|
| 10 | lane + GPU assist on 75 layers, no pilot | 11.58 | 0.93 | ~157 misses/token, lane wait 628 ms/token, compute 299 ms/token, read 516 ms/token |
| 11 | lane + assist + pilot prefetch-only (l+2) | 11.61 | 0.97 | ~63 misses/token, lane wait 71 ms/token, CPU compute 714 ms/token, split stall 301 ms/token, ~92 experts/token on GPU assist |

Finding: with `PILOT_STAGE>0` the old `!stage` condition built the lane for only one layer, so the pilot and the CPU lane/assist ring were mutually exclusive. With `PILOT_COPY=0` the lane now coexists with the prefetch pilot (`stage_post_node` posts the l+2 prefetch job). Pilot precision by rank: r1 97%, r2 92%, r3 83%, r4 71%, r5 59%, r6 48%, r7 35%, r8 27%.
After the change SSD waiting is no longer dominant; CPU expert compute is. Decode target (1.7 tok/s) is still NOT met.

Trial 12 (trial 11 + `GPU_ASSIST=6` on the 3090, `ASSIST2_DEVICE=1 ASSIST2_SLOTS=2` on the 5060 Ti): prefill 11.52, decode 0.94 tok/s. About 105 experts/token went to GPU assist (92 in trial 11), yet lane compute stayed at ~720 ms/token and split stall rose to ~341 ms/token. A larger assist ring does not help; the bottleneck is not the number of experts the GPUs take.

### Trials 13-14 (GLM-5.3 glm-dsa, dense on 5060 Ti, lane + pilot coexisting, PILOT_COPY=0, LOOKAHEAD=2)

| Trial | PILOT_PREFETCH/STAGE | Prefill tok/s | Decode tok/s | Lane wait / compute / split stall (ms/token) | Misses | Read ms/token |
|---|---|---|---|---|---|---|
| 11 | 8 / 8 | 11.61 | 0.97 | 71 / ~715 / ~300 | ~63/token | ~365 |
| 13 | 5 / 5 | 11.62 | 1.04 | 89 / 614 / 267 | 2789 (32 tok) | 436 |
| 14 | 3 / 3 | 11.20 | 0.57 | 94 / 630 / 287 | 3632 (32 tok) | 486 |

Prefetch 5 is the best setting so far (+7% over 8). Prefetch 3 loses too many predicted experts (misses +30%) and decode drops 45%. Decode is still bound by CPU cold-expert compute (~610 ms/token) plus split stall (~270 ms/token); target is 1.7 tok/s (~590 ms/step).

Trial 15 (PREFETCH=5, GPU1_HOT_GIB 11 -> 15): prefill 11.75, decode 0.86 tok/s (-17% vs trial 13). Lane compute 700 ms/token, split stall 341, SSD read 522 ms/token (misses 106/token). Boot preload took 122 s (vs 12.5 s), so host memory was under pressure; a larger hot set did not pay. Keep GPU1_HOT_GIB=11. Best config remains trial 13 (1.04 decode / 11.62 prefill).

Trial 16 (trial 13 config, ZEROMOE_GLM5NEXT_CPU_THREADS 6 -> 10): prefill 11.35, decode 0.92 tok/s (-12% vs trial 13). Lane compute rose 614 -> 742 ms/token and SSD read per load 5.2 -> 6.0 ms: extra decode threads contend with the prefetch/IO workers rather than speeding the expert kernels. Keep CPU_THREADS=6.

Summary of lossless levers tried on the dense-on-5060 Ti config (426-token prompt, NGEN=40): best is trial 13 (PILOT_PREFETCH=PILOT_STAGE=5, LOOKAHEAD=2, PILOT_COPY=0, lane on, GPU1_HOT_GIB=11, CPU_THREADS=6) = 1.04 decode / 11.62 prefill. Target 1.7 decode (~590 ms/step) is not reached; remaining cost is CPU cold-expert compute (~610 ms/token) + split stall (~270 ms/token) + SSD reads (~436 ms/token, overlapped).

Trial 17 (trial 13 config, CPU_THREADS=4): prefill 11.59, decode 0.96 tok/s (-8% vs trial 13). Lane compute 660 ms/token, split stall 225-243. Threads 4/6/10 = 0.96/1.04/0.92, so 6 is the optimum and the thread count is not the lever.

Trial 18 (trial 13 config + second GPU assist on the 5060 Ti, ASSIST2_DEVICE=1, 3 slots): prefill 11.56, decode 1.07 tok/s (+3% vs trial 13, inside run-to-run noise). Assisted experts rose 3004 -> 3455 per 32 tokens (~94 -> 108/token) and lane compute fell 614 -> 602 ms/token, split stall 267 -> 253, but the SSD read (427 ms/token, 5.1 ms per load) and the CPU compute that remains still set the step. Best decode so far 1.04-1.07 tok/s; target 1.7 not reached.

Trial 19 (trial 18 config, same shape as the old layered 1.18 run: 22-token prompt, NGEN=80): decode 1.11 tok/s by the Rates line (80 tokens / 71.9 s), 1.15 tok/s first-to-last output, prefill 1.76 tok/s (22 tokens). Old layered engine (ZeroMoE_v18/v19 `CLAUDE.v4.md`, `perf-tuning-prefetch.md`): 1.10-1.12 on the same 80-token shape, 1.16-1.18 on the shorter 40-token A/B. So on an identical shape the full graph is at parity with the layered engine, not better. Lane compute 570-620 ms/token, split stall ~300, and the cold-start SSD read falls from 770 to 534 ms/token over the run (10.9 -> 6.4 ms per load as the pool warms), so short runs under-state steady state.

Trial 19 follow-up (SSD split A/B, 22-token prompt, NGEN=80, run under `systemd-run --user --scope -p MemorySwapMax=0`, swap 0 B at the end; opt-in knobs in `memory/offload_store.hpp`, default off): default plane alternation decode 1.08-1.09 tok/s (clean control: compute 633, split stall 311, read 543 ms/token, 6.57 ms per load). `SPLIT_MODE=balanced` byte-share split at 0.55: 1.05 (clean) / 1.01 (earlier); at 0.40: 1.02. Both are -3% to -7% vs the default: read rises to 603-630 ms/token and lane compute to ~690 ms/token (more IO contention on the CPU lane). `MODEL_CHUNK_KB=512` parallel chunked model-side preads: 0.90 (measured with swap in use, suspect, but not promising). The mirrored two-NVMe read (store drive + model drive per missed expert) is already what `SplitSource` does; the per-miss latency (~6.5 ms) rather than drive bandwidth (~1.7 of ~6.7 GB/s used) limits it, so re-balancing bytes does not help. Regression >5% for balanced@0.55 noted for the owner; defaults unchanged. Runs with swap in use drifted slower arm over arm (compute 615 -> 785 ms/token), so swap must stay off for benchmarks.

Trial 20 (hot-set size on the non-core GPU, 22-token prompt, NGEN=80, swap-free scope, preflight-gated, trial-19 config otherwise; `logs/run_split_queue4..7.sh`). The 3090 (`moe` role) held only the 11 GiB hot set plus a 4 GB legacy expert cache the graph never uses (`[VRAM-SLOTS] 0`), so `logs/glm53_dsa_core5060_c05.toml` cuts `moe_cache_gb` 4.0 -> 0.5 and the hot set grows into the freed VRAM:

| Hot set (`GPU1_HOT_GIB`) | Decode tok/s (Rates) | First-to-last | Lane compute ms/tok | Split stall | SSD read ms/tok |
|---|---|---|---|---|---|
| 11 (control, 4 GB cache) | 1.13 | 1.16 | 616 | 306 | 522 |
| 18 (4 GB cache) | 1.21 / 1.20 | 1.25 / 1.25 | 570 | 269-278 | 468-471 |
| 21 (0.5 GB cache) | 1.25 / 1.23 | 1.30 / 1.27 | 532-543 | 260-265 | 413-420 |
| 22 (0.5 GB cache) | 1.29 | 1.34 | 518 | 249 | 393 |
| 23, 24 | graph build declined (strict mode fails the request): no VRAM headroom for the decode graph | | | | |

So the hot set is the lever: 11 -> 22 GiB is +14% decode (1.13 -> 1.29), and 22 GiB is the ceiling with the 3090 VRAM left. Boot preload is 17-21 s. Output text is coherent. This earlier-negative result (trial 15, 15 GiB, 0.86) was taken with swap in use and a 122 s preload, so it was a memory-pressure artefact, not a hot-set effect. Best decode so far 1.29 Rates / 1.34 first-to-last; 1.7 not reached. Note: the zram swap was re-activated at 20:32 (systemd), swappiness is 150 again; runs stay swap-free through `MemorySwapMax=0`.

Trial 21 (22 GiB hot set, `glm53_dsa_core5060_c05.toml`, swap-free scope; 22-token prompt, NGEN=80). GPU assist slots do not matter: 2 slots 1.27, 3 slots 1.29, 5 slots 1.26 tok/s (compute 518-530, read 392-401 ms/token). DDR pool size does: `OFFLOAD_DDR_GIB` 54 -> 64 gives decode 1.41 tok/s (Rates; 1.47 first-to-last), a +9% step over 1.29: lane compute 452 ms/token, split stall 191, SSD read 327 ms/token (3169 loads vs 3790). Seeding 6014 experts took 29.5 s; no swap was used by the run (MemorySwapMax=0), `free` after the run shows 107 GiB available. Best decode so far 1.41 Rates / 1.47 first-to-last; 1.7 not yet reached. A larger pool (74, 84 GiB) is the next test; it needs host-RAM headroom (125 GB box, ~10 GiB reserve is enforced by the host admission check).

### Trial 21b: why the 3090 hot set stops at 22 GiB
hot23/24 fail at the boot preload (`boot preload failed`), then the layer-76 build_ffn failure is the consequence. The 3090 has 24.48 GB (~22.8 GiB) free after setup; a 22 GiB hot set plus the 768 MiB allocation floor consumes it. This is a physical VRAM cap, not a software bug. Remaining headroom: dropping the 0.5 GB legacy cache is worth <=~2%. The only lever left of size is the DDR pool (74/84 GiB; 54->64 GiB gave +9%), which still needs a permitted launch of `logs/run_split_queue11.sh`.

### Trial 21c: DDR pool 74 GiB does not fit
`OFFLOAD_DDR_GIB=74` (hot 22 GiB, c05 toml, swap-free) seeds fine (6954 experts, 31 s) but RSS reaches 94 GB, MemAvailable drops below the 20480 MB MEM-GUARD stop threshold, and the request is refused by the 82 GB host envelope (reserved 16.9 GB + 74 GiB pool). 0 tokens generated, no decode number. The usable pool ceiling on this 125 GB box is therefore roughly 64-68 GiB; 64 GiB (1.41 tok/s) is the best verified point. A 54->64 GiB gain of +9% extrapolates to only ~+3-5% for the extra 4 GiB that might fit, i.e. about 1.45-1.5, still short of 1.7.

### Trial 22: refit usage ranking (survey counts, weight 3) did not improve decode
A/B on one config (22 GiB hot set, 64 GiB pool, c05 toml, swap-free scope, 22-token prompt, NGEN=80, refit arm first). Refit file: `docs/ssd-offload/data/glm53_usage_refit_w3.txt` (base usage + 18-prompt route-survey decode counts x3).

| arm | decode tok/s (Rates) | first-to-last | compute ms/tok | split stall ms/tok | read ms/tok | misses | loads |
|---|---|---|---|---|---|---|---|
| refit_w3 | 1.39 | 1.45 | 457.8 | 219.1 | 368.5 | 5324 | 3250 |
| current ranking | 1.43 | 1.50 | 440.8 | 188.5 | 320.6 | 5056 | 3207 |

The offline held-out cold-route fraction dropped (resident set 42.1% -> 32.0%), but the real decode did not follow: -3% tok/s, within run-to-run noise but certainly not a gain. The 22-token benchmark prompt is not in the survey set, and the survey is 18 topical prompts, so the weight-3 counts may over-fit survey topics. Not adopted; the current usage file stays the default. The base arm text is coherent (the refit arm's text was overwritten: both runs share `logs/glm53_dsa_graph2gpu_tke.stdout`).

### Trial 23: IQ3_XXS AVX-512VL dot (microbench only, not applied) and split-stall analysis
`bench/iq3k/iq3_dot_bench.c` (standalone, `gcc -O3 -march=native`) compares the stock AVX2 `ggml_vec_dot_iq3_xxs_q8_K` with the same mask-expanded restructuring already used for IQ2_XXS (block sign words and scales expanded once, masked subtract instead of per-group sign vectors). Result on 4096 rows x 4096 K: 0 mismatching rows (bit-identical), 1.26-1.33x faster pinned to one core (about 1.9 ms -> 1.5 ms per matrix). IQ3_XXS cold-expert dots are about 152 of the roughly 340 ms/token of CPU MUL_MAT_ID, so the expected decode gain is +5-7% (about 1.5 tok/s). Not yet applied to the ggml tree (`/path/to/ggml-0006-early`) or measured end to end.

Split stall (about 190 ms/token) is the time `finish_load()` still blocks after phase 1 (`models/graph/early_ready_lane.hpp:662`). Reads run 6.4 ms per load, about 50 loads/token, on two drives; phase 1 only overlaps the resident experts of the same layer. Earlier trials already cover the obvious levers: lookahead 3 and prefetch 3/8 lose, CPU threads 4/10 lose, balanced/chunk split lost, pool above 64 GiB does not fit. What remains is fewer loads (better prediction or more resident experts) rather than more overlap.

### Trial 24: 426-token prompt, full graph, prefill met, decode 0.98
Config: 22 GiB hot set, `glm53_dsa_core5060_c05.toml`, swap-free scope (swap stayed 14 GiB, unchanged), NGEN=48, `logs/run_split_queue15.sh`. Logs: `logs/split_ab_p426_hot22_ddr56.{log,out}`.

| DDR pool | prefill tok/s | decode tok/s (Rates) | first-to-last | result |
|---|---|---|---|---|
| 64 GiB | 12.95 (426 tok / 32.9 s) | none | none | decode graph refused: MemAvailable 20.4 GB < 20.48 GB MEM-GUARD stop, 192 MiB host graph context declined, strict mode failed the request |
| 56 GiB | 12.42 (426 tok / 34.3 s) | 0.98 | 1.02 | completes; 32-token lane stats: compute 638 ms/tok, split stall 307 ms/tok, read 452 ms/tok, 2926 misses / 2881 loads (about 90 loads/token) |

Prefill meets the 10 tok/s target on the 426-token prompt. Decode does not meet 1.7. The 64 GiB pool that gives 1.43 on the 22-token prompt does not fit after a 426-token prefill, and the 56 GiB pool roughly doubles loads per token. The prompt also differs, so this is not a clean pool A/B. Output text echoes the repeated prompt sentence (weak coherence evidence).

Exit: after "Generation Complete" the process aborts with `CUDA error: driver shutting down` from `ggml_backend_cuda_buffer_free_buffer` in the static `StageRuntime` destructor at `exit()`. Not diagnosed; does not affect measurements. Same class as the earlier Phase 2 shutdown crash (CUDA globals need early reset).

### Trial 25: IQ3_XXS AVX-512VL dot applied, no measurable end-to-end gain
Kernel: `patches/experimental/0011-ggml-cpu-iq3_xxs-avx512vl-dot.patch` (applied to `/path/to/ggml-0006-early`, Release rebuild of `build_g7`; compiled kernel contains the masked `vpsubb`). Config: 22 GiB hot set, 64 GiB pool, c05 toml, swap-free scope, 22-token prompt, NGEN=80, `logs/run_split_queue16.sh`.

| run | decode tok/s (Rates) | first-to-last | compute ms/tok | split stall ms/tok | read ms/tok | loads |
|---|---|---|---|---|---|---|
| stock kernel (trial 22 base) | 1.43 | 1.50 | 440.8 | 188.5 | 320.6 | 3207 |
| IQ3 AVX-512VL, run b | 1.44 | 1.51 | 435.6 | 200.0 | 323.4 | 3212 |
| IQ3 AVX-512VL, run a | invalid | | | | | decode graph refused: MemAvailable 20338 MB < 20480 MB MEM-GUARD stop; request failed in strict mode |

Compute fell 1.2% (440.8 -> 435.6), against about 6% predicted from the 1.23x microbenchmark. One valid run, so the difference is inside run-to-run noise. Output text is coherent (the answer starts "How a Transformer Language Model Generates Text ... Let me walk through the complete process"). Correction of the first explanation: I guessed decode was bound by memory traffic. `bench/iq3k/iq3_cold_bench.c` (2 GB of weights, 6 or 12 threads, each row read once) shows the dot product, not DRAM, is the limit on this CPU: stock 18.5 GB/s at 6 threads and 19.9 at 12 against a 58-64 GB/s raw read, and the AVX-512VL kernel is 1.15x (6 threads) to 1.25x (12 threads) faster, so the kernel gain survives cold memory. So the 1.2% end-to-end compute change is not explained by memory traffic. Cause unknown: the IQ3_XXS share of the lane compute may be smaller than the 152 ms I attributed, the 6 lane threads may not be running the new branch (verified afterwards: the loaded libggml-cpu.so contains the masked vpsubb, so the branch is in the binary), or the run was within noise. The 64 GiB pool is marginal against the MEM-GUARD stop threshold: it failed 1 of 2 runs here and 1 of 1 at 426 tokens.

### Trial 26: decode CPU profile and lane thread count

Config as trial 25: 22-token prompt, NGEN=80, 22 GiB 3090 hot set, 64 GiB pool, swap-free, new IQ3_XXS kernel. `perf record -F 199` over the whole run, `logs/perf/perf1.data` (decode run 1.45 tok/s).

| Component | Share of samples |
|---|---|
| libgomp | 43.1% |
| IQ2_XXS dot | 22.6% |
| IQ3_XXS dot | 15.8% |
| dequantize_row_q4_K | 7.5% |
| dot_f32_avx512 | 3.2% |
| libcuda | 2.9% |

Five worker threads each carry about 7.5% libgomp, 6.3% ggml-cpu and 0.7% ggml-base. The main thread has 6.0% ggml-cpu and 3.6% libcuda. The libgomp samples are one `pause` / counter / compare loop at 0x2c7b0 (27.5% of all samples, objdump of the system libgomp), which is OpenMP barrier spin-wait. IQ3_XXS is 15.8% of samples, so a 1.23x kernel saves at most about 3% of CPU time; that matches the 1.2% observed in trial 25.

Lane thread count (`ZEROMOE_GLM5NEXT_CPU_THREADS`), one run each:

| Threads | Decode tok/s | compute ms/token | split stall | read ms/token |
|---|---|---|---|---|
| 5 | 1.43 | 438.6 | 188.4 | 318.2 |
| 6 (baseline) | 1.44 | 435.6 | 200.0 | 323.4 |
| 8 | 1.45 | 432.7 | 214.2 | 324.8 |
| 12 | 1.10 | 653.1 | 136.0 | 382.8 |

Thread count is not a lever: 5 to 8 threads is flat, and 12 threads (more than the 6 physical cores) is 24% slower. Barrier spin is therefore not fixed by thread count. Not yet tested: `OMP_WAIT_POLICY`, `GOMP_SPINCOUNT` and `OMP_PROC_BIND`. The spin may also be time spent waiting for split loads, not imbalance. Regression note: 12 threads is a >5% drop against 6.

### Trial 27: OpenMP binding and wait policy

Same config as trial 26 (6 lane threads, 64 GiB pool, 22-token prompt, NGEN=80, swap-free), one run each, new IQ3_XXS kernel.

| Setting | Decode tok/s | compute ms/token | split stall | read ms/token |
|---|---|---|---|---|
| baseline (trial 25 run b) | 1.44 | 435.6 | 200.0 | 323.4 |
| OMP_PROC_BIND=close OMP_PLACES=cores | 1.15 | 584.4 | 326.8 | 459.7 |
| OMP_WAIT_POLICY=passive | 1.38 | 458.1 | 177.5 | 323.8 |
| GOMP_SPINCOUNT=1000000 | 1.45 | 431.0 | 202.5 | 321.1 |

None of these is a lever. Binding to cores is 20% slower (a >5% regression; prefill also fell 2.24 -> 1.81 tok/s), probably because it pins lane threads onto cores that the CUDA, prefetch and I/O threads also use. Passive wait costs 4% (wake-up latency). A longer spin count is flat, so the default already spins as long as it needs to. Together with trial 26 this closes the OpenMP and thread-count levers: the barrier spin is waiting for the slowest thread's work, not a scheduling defect.

Correction to trials 25-27 (code read, `models/graph/early_ready_lane.hpp:643-720`): the logged `compute` is `done - copied`, and that span contains the `finish_load()` wait that is logged separately as `split stall`. So `compute` is not pure CPU work. At 64 GiB: compute 436 ms/token = about 236 ms of real expert math + about 200 ms blocked waiting for the SSD loads of the experts that had to be read (phase 2 of the split). The libgomp barrier spin (43% of samples) is the lane threads idle during that stall (200/436 = 46%), not load imbalance and not a scheduling defect. Per-token time: about 694 ms = compute 436 (236 math + 200 stall) + lane wait 62 + about 196 outside the lane. The stall is the single largest recoverable item: removing it entirely would give about 494 ms/token (2.0 tok/s); halving it gives about 594 ms (1.68 tok/s). A faster IQ2_XXS or IQ3_XXS kernel only shrinks the 236 ms of math.

### Trial 28: pilot depth sweep (corrected runner) and raw drive throughput

`/tmp/abrun.sh` applied per-run overrides BEFORE its own pilot defaults, so queue20 (lookahead 3 / prefetch 8 / both) silently ran the default config three times. Fixed (overrides now apply after defaults) and rerun as queue21, 22-token prompt, NGEN=80, 64 GiB pool, swap-free:

| Setting | Decode tok/s | compute ms/token | split stall | loads/64 tok |
|---|---|---|---|---|
| default (stage 5 / prefetch 5 / lookahead 2) | 1.44 | 435.6 | 200.0 | 3212 |
| PILOT_LOOKAHEAD=3 | 1.42 | 441.7 | 206.0 | 3590 |
| PILOT_PREFETCH=8 PILOT_STAGE=8 | 1.20 | 573.0 | 314.8 | 3473 |
| lookahead 3 + prefetch/stage 8 | 1.20 | 577.9 | 320.1 | 3884 |

Deeper prediction adds loads and does not remove stall; prefetch/stage 8 is a >5% regression (-17%). Prefetch depth is closed as a lever.

Raw drive throughput (`bench/iobench/qd_bench.c`, O_DIRECT random reads, nothing else running):

| Chunk | Store drive (KIOXIA, /data) qd1 / qd2 / qd4 | Model drive (Samsung 980 PRO) qd1 / qd2 / qd4 |
|---|---|---|
| 4 MB | 2.48 / 2.85 / 2.89 GB/s | 2.67 / 3.22 / 3.52 |
| 6.5 MB | 2.67 / 2.97 / 2.86 | 2.85 / 3.35 / 3.57 |
| 10 MB | 2.83 / 3.06 / 3.07 | 2.90 / 3.46 / 3.57 |

Both drives together (qd2 each): about 3.05 + 3.47 = 6.5 GB/s. The Samsung's PCIe link trained at Gen3 x4 (max Gen4); Gen3 x4 tops out near 3.9 GB/s and the drive delivers 3.5, so the Gen3 link costs little here.

Correction: earlier in this session I estimated 1.7 GB/s per load using 10.9 MB per expert, a figure from an older note. The `[GGML-GLM5NEXT-STAGE] ring ready` log line gives plane sizes 4030464/4030464/6684672 bytes (14.7 MB per expert, assuming those are one expert's gate/up/down). At 14.7 MB and 6.2-6.5 ms per load that is about 2.3-2.4 GB/s per load, the same as one drive at queue depth 1. About 56 loads per token is about 825 MB per token.

### Trial 29: MADV_FREE eviction (ZEROMOE_OFFLOAD_EVICT=free), interleaved A/B x2

Standalone, reading 6.5 MB O_DIRECT into freshly faulted pages is 25-40% slower than into warm pages (store drive 1.87-1.92 vs 2.64-2.66 GB/s; model drive 1.63-2.24 vs 2.75), which suggested the engine's MADV_DONTNEED eviction costs read bandwidth. End to end it does not: 22-token prompt, NGEN=80, 64 GiB pool, swap-free.

| Arm | Decode tok/s | compute ms/token | split stall | ms per load |
|---|---|---|---|---|
| evict=free, run a | 1.40 | 456.5 | 206.9 | 6.66 |
| baseline, run a | 1.46 | 427.9 | 196.4 | 6.51 |
| evict=free, run b | 1.40 | 452.1 | 203.5 | 6.62 |
| baseline, run b | 1.43 | 435.5 | 199.8 | 6.39 |

evict=free is 2-4% slower (below the 5% regression threshold), with no gain in ms per load. Not a lever; left off. The page-fault cost seen standalone is not what limits an engine load.

### Trial 30: decode-window drive utilisation (`/proc/diskstats`, 1 s samples, run `dstat`, 1.43 tok/s)

- Decode window (80 tokens, 55.9 s): nvme0n1 (store) mean 1.32 GB/s, 49% busy; nvme1n1 (model dir) mean 1.48 GB/s, 47% busy; combined 2.79 GB/s, 148 GB read in the window. Peak per drive about 1.9 GB/s. Standalone qd2 ceiling is about 6.5 GB/s combined.
- Reconciliation: 148 GB / 80 tokens = 1.85 GB/token, not the 0.83 GB/token implied by demand loads alone (about 51-56 demand loads/token x 14.7 MB). The remainder is speculative pilot prefetch: `prefetch_loads` is about 100/token (6560 over 64 tokens) and pilot precision falls from 91% at rank 1 to 37% at rank 5, so roughly 40-60% of prefetch reads are never used.
- Meaning: the drives are not idle. About half the bytes they serve are speculative, and the demand read that `finish_load()` blocks on (about 200 ms/token) queues behind them on the same two devices. Total throughput is about 2.8 GB/s on both drives together, which is below the 6.5 GB/s standalone figure, so the stall is a mix of queueing behind prefetch and per-read latency (6.4-7 ms per 14.7 MB load = 2.1-2.3 GB/s per read).
- Not yet tested: demand-priority issue (prefetch only ranks 1-2, or pause prefetch while a demand load is in flight). The earlier pilot sweep varied stage/prefetch depth but kept all depths at or above 5 reads/layer, so it did not isolate this.

### Trial 31: shallow pilot prefetch (queue23), 22-token prompt, NGEN=80, 64 GiB pool, swap-free

| Setting (prefetch=stage) | Decode tok/s | compute ms/token | split stall | loads/64 tok | ms per load |
|---|---|---|---|---|---|
| default (5/5, lookahead 2), Trials 28/30 | 1.43-1.44 | 435 | 200 | 3212 | 6.4 |
| 3 / 3 | 1.50 | 404 | 183 | 3923 | 5.6 |
| 2 / 2 | 1.46 | 420 | 199 | 4674 | 5.1 |
| 1 / 1, lookahead 1 | 1.42 | 439 | 220 | 5424 | 4.6 |

Cutting speculative reads shortens each demand read (6.4 -> 4.6 ms) but converts misses into more demand loads (3212 -> 5424), so the stall does not fall. The best point (3/3, +4%) is within run-to-run spread (1.43-1.46 for the same config), so it is not a confirmed win. Prefetch depth in either direction is closed as a lever; the remaining stall is serialised demand-read latency, not prefetch interference. Next lever: overlap the demand reads with phase-1 compute / compute each loaded expert as its read lands.

### Trial 32: prefetch 3/3 repeat and O_DIRECT model-drive reads (queue24)

| Arm | Decode tok/s | compute ms/token | split stall | ms per load |
|---|---|---|---|---|
| prefetch/stage 3 (Trial 31) | 1.50 | 404 | 183 | 5.6 |
| prefetch/stage 3 repeat | 1.50 | 404 | 183 | 5.6 |
| prefetch/stage 3 + ZEROMOE_GLM5NEXT_MODEL_DIRECT=1 | 1.53 | 390 | 168 | 5.4 |

The 3/3 result reproduces exactly (1.50 twice vs 1.43-1.44 for the default 5/5 in three runs), so about +4% is probably real, not noise. O_DIRECT on the model drive adds about +2% on top (single run). Both are small; together about 1.53 tok/s, still 10% short of 1.7. Neither is made default yet.

Coherence check (last run's stdout, prefetch 3/3): the 22-token prompt continues as a coherent, well-formed transformer-generation explanation ("# How a Transformer Language Model Generates Text ..."). Prefetch depth and O_DIRECT only change when bytes are read, not which experts are computed, so the other arms are lossless by construction; only the final arm's text was inspected because the shared stdout file is overwritten each run.

### Trial 33: per-side split read timing (ZEROMOE_SPLIT_TIMING=1, default off), prefetch 3/3, decode 1.49 tok/s

Cumulative mean over the first 11000 split reads (includes prefill and prefetch reads, so it is a contention-inflated figure, not the demand-only latency): store drive side 6.0 ms, model drive side 7.5 ms, whole read (join of both) 8.3 ms. The model side is the slower one (buffered reads by default; O_DIRECT was +2% in Trial 32), and the join pays for the slower side on every read.

Why pipelining per-expert compute is not the next lever: most layers have 0-1 pending experts (about 0.65 on average), so there is nothing to overlap; the stall is the latency of one miss, about 2.7 ms/layer. An ideal 14.7 MB read split across two drives at about 3 GB/s each would take about 2.5 ms, but a demand load measures 5.4-5.6 ms. Cutting that per-miss latency in half would save about 90 ms/token (1.50 -> about 1.75 tok/s), so per-read latency, not bandwidth or overlap, is the lever.

### Trial 34: per-miss latency knobs on top of prefetch 3/3 + MODEL_DIRECT=1 (queue26), 22-token prompt, NGEN=80, 64 GiB pool, swap-free

| Arm | Decode tok/s | compute ms/token | split stall | ms per load |
|---|---|---|---|---|
| 3/3 + direct (Trial 32) | 1.53 | 390 | 168 | 5.4 |
| + MODEL_CHUNK_KB=2048 | 1.53 | 389 | 166 | 5.3 |
| + SPLIT_MODE=balanced | 1.34 | 478 | 249 | 6.9 |
| + OFFLOAD_EVICT=free | 1.54 | 387 | 155 | 5.2 |

Chunking and MADV_FREE are within run-to-run spread (about +-0.03). Balanced split is a -12% regression (>5%): it puts more bytes on the slower model drive. All three knobs are closed; the best recipe is prefetch/stage 3 + MODEL_DIRECT=1, about 1.53 tok/s (+7% over the 1.43-1.44 default).

Note (queue27): `ZEROMOE_GLM5NEXT_GRAPH_LAYER_TIMING=1` aborts this graph at `early_ready_lane.hpp:728` ("early-ready marker requires the scheduler bridge"), so it cannot be used to profile the full-graph lane path. Known out-of-lane cost: LM head about 50 ms/token (3.98 s / 80 tokens).

### Trial 35: recipe (prefetch/stage 3 + MODEL_DIRECT=1) vs default, 56 GiB pool, NGEN=48, swap-free

| Prompt | Arm | Decode tok/s | Prefill tok/s | split stall ms/token | loads/32 tok |
|---|---|---|---|---|---|
| 426 tokens (queue29) | default | 0.98 | 12.11 | 325 | 2889 |
| 426 tokens (queue29) | recipe | 1.02 (+4%) | 12.46 | 287 | 3661 |
| 22 tokens (queue28, prompt left at abrun default by mistake) | default | 1.24 | 2.09 | 259 | 1794 |
| 22 tokens (queue28) | recipe | 1.17 (-6%) | 1.99 | 254 | 2118 |

The recipe is not a consistent win: +4% on the long prompt, -6% on the short prompt at 56 GiB / NGEN=48 (where the 64 GiB / NGEN=80 runs showed +4-7%). The 22-token 56 GiB pair is a >5% drop for the recipe, so it is NOT made default. Run-to-run spread is about +-0.03 tok/s, and the NGEN=48 windows are short, so these pairs are single runs each. Treat the recipe as at most a few percent. The 426-token prompt still decodes at about 1.0 tok/s, well under 1.7.

### Trial 36: I/O floor check, and why an io_uring batched read is not built

Recipe run (prefetch/stage 3, 22-token prompt, 64 tokens): 43.2 split calls/token, 61.3 demand loads/token, so 1.42 loads (about 21 MB) per split call; stall 183 ms/token = 4.25 ms per split call. Two drives at 2.9-3.3 GB/s each move that 21 MB in 3.2-3.6 ms, so the floor is 137-155 ms/token of stall. The measured stall is within 15-30% of that floor.

This corrects Trial 33, which compared one 14.7 MB load against an ideal 2.5 ms and concluded halving per-miss latency would give about 1.75 tok/s. A split call carries 1.42 loads, so the ideal is 3.2-3.6 ms, not 2.5 ms. Perfect batching of the reads would save at most about 30-45 ms/token: 1.53 -> about 1.6 tok/s, still below 1.7. An io_uring batch path is therefore not built.

What is left is bytes per token (about 0.9 GB demand-loaded per token) and the 236 ms/token of expert math plus 50 ms/token LM head. Fewer bytes requires a larger resident hot set or fewer experts routed; the former is bounded by VRAM/DDR, the latter is ruled out (no pruning).

### Trial 37 (2026-10-20): larger 3090 hot set is not available

Recipe (PREFETCH=3, STAGE=3, MODEL_DIRECT=1), 22-token prompt, NGEN=80, 64 GiB pool, swap-free, `logs/run_split_queue30.sh`.

| GPU1_HOT_GIB | result |
|---|---|
| 22 (reference) | decode 1.53 tok/s (80 tok / 52.25 s), prefill 2.27 tok/s (22-token prompt); coherent |
| 23 | decode graph build fails: `build_ffn failed at layer=76`; request failed under GRAPH_STRICT |
| 23.5 | same failure at layer=75 |

22 GiB is the VRAM ceiling on the 3090 for the full graph, so the hot-set lever is exhausted. Earlier `hot23c05` and `hot24c05` logs show the same failure.

Target status: prefill >= 10 tok/s is met (12.42 tok/s on the 426-token prompt, `logs/split_ab_p426_hot22_ddr56.log`; 12.1-12.5 repeatable). Decode >= 1.7 tok/s is not met: 1.50-1.54 on the 22-token prompt, 0.98-1.02 on the 426-token prompt. The "1.7 peak" is the 2-token layered-path sample `logs/check/short_layered.log` (1.74 tok/s), not a sustained rate.

### Trial 38: long-prompt decode gap is mostly topic, not length (and not pool size)

Config: recipe (prefetch 3, stage 3, direct model reads), 22 GiB 3090 hot set, c05 toml, swap-free, NGEN=80. Logs: `logs/split_ab_p426_ddr60_recipe`, `p426_ddr62_recipe`, `p426_transformer_topic`, `s_computing_topic`.

| Prompt | Pool GiB | Rates decode tok/s | loads per 64 tok | mean step ms |
|---|---|---|---|---|
| 22-token transformer topic (queue32) | 64 | 1.64 | 3908 | 533 |
| 25-token computing-history topic | 60 | 1.32 | 6239 | - |
| 413-token transformer topic | 60 | 1.24 | 6418 | - |
| 426-token computing-history topic | 56 / 60 / 62 | 1.08 / 1.11 / 1.12 | 8143 (60) | 828 |

Pool size 56 -> 62 GiB changes nothing measurable (+3-4%, within noise); 60 and 62 GiB both fit on the 426-token prompt, so the MEM-GUARD miss at 64 GiB is only a 64 GiB problem. Topic costs about 20% on short prompts and length about 10%. Loads per token roughly double off-topic, so the usage ranking (and every short-prompt benchmark) is tuned to the transformer-explanation prompt. The 1.64 headline is an in-distribution number; off-topic decode is 1.1-1.3.

Trial 38b: refit usage ranking (`glm53_usage_refit_w3.txt`, 18-prompt survey) on the off-topic computing-history prompts, recipe, 60 GiB pool, 22 GiB hot set. 25-token: 1.35 vs 1.32 baseline; 426-token: 1.04 vs 1.11 baseline. Both inside run-to-run spread (about +-0.03-0.07), so the ranking is not the off-topic lever. Loads per 64 tokens stay 5.3k (short) and 8.2k (long). Not adopted. Logs: `logs/split_ab_s_computing_refit`, `p426_computing_refit`.

Correction to Trials 30-37: the Samsung 980 PRO training at Gen3 is the platform limit (Skylake-E C422 root ports are Gen3 only; /sys/class/nvme/nvme1/device max_link_speed reports 16 GT/s but the CPU cannot train it), not a BIOS/slot issue. Both NVMe drives and both GPUs are at the platform maximum.

### Trial 39: demand-priority gate for SSD reads (negative)

Prefetch reads waited (bounded 30 ms) while a demand read was pending (opt-in env, since removed). 22-token prompt, recipe, 64 GiB pool, 22 GiB hot set, swap-free, interleaved off/on/on/off. Decode (Rates): off 1.71 (and 1.62 in an earlier clean run), on 1.68 and 1.64. Ms per load unchanged (5.21 off, 5.07 / 5.21 on), stall 161 vs 159 / 169 ms/token. Prefetch queueing is not what makes a demand load take 5.2 ms: a 14.7 MB O_DIRECT read is already at the single-drive ceiling (2.8 GB/s). Same-config spread is 1.62-1.71, so differences under about 0.07 are noise.

### Trial 40: DDR pool 66 / 68 GiB (host_ram_gb 92, pinned budget 70, c06 toml)

Short prompt, recipe: 66 GiB 1.64, 68 GiB 1.73 (loads/64 tok 3880 / 3747 vs 3904 at 64 GiB). The load reduction is about 4% and the throughput change is inside the noise band. Both fit the 20 GiB MEM-GUARD on the short prompt. The 20 GiB floor is `MemoryGuard::kMinimumHostReserveMiB` (unit-tested); not lowered. Pool size beyond 64 GiB is worth at most about +2-3%.

Note: `/tmp/abrun.sh` is the A/B wrapper; a copy is at `logs/abrun.sh`, but run it from /tmp because `bench/glm53/preflight.sh` matches "zeromoe" in process command lines and flags a wrapper under `zeromoe-glm53/`.

### Trial 41: bigger DDR pool on the 426-token prompt, MemoryGuard floor 16 GiB (ZEROMOE_MEM_GUARD_FLOOR_MB=16384, experiment-only env)

Recipe, 22 GiB hot set, c06/c07 toml (host_ram_gb 92/100, pinned budget 70/80), swap-free, NGEN=80, computing-history 426-token prompt. Default floor stays 20 GiB.

| Pool GiB | Decode tok/s (Rates) | loads / 64 tok | split stall ms/tok | MemAvailable low-water |
|---|---|---|---|---|
| 56 (20 GiB floor) | 1.08 | 8143 | 307 | - |
| 64 | 1.15 | 7752 | 305 | 31.6 GB at graph source |
| 68 | 1.22 | 7246 | 266 | 27.4 GB |
| 74 | 1.31 | 6503 | 235 | 22.5 GB |
| 78 | 1.36 (1.41 first-to-last) | - | - | 15.0 GB (9 guard block events, 0 refusals) |

About +0.012 tok/s per GiB on long prompts (+26% from 56 to 78). 78 GiB touches the 16 GiB floor, so 74 GiB is the practical edge on this 125 GB box. Short prompts gain nothing measurable beyond 64 GiB (Trial 40).

### Trial 42: file-backed (mmap) dense host weights to free anonymous memory (no gain, reverted)

At decode with a 64 GiB pool the process held 84.5 GiB anonymous memory plus a 34 GB page cache (the "unused" memory seen in monitors is that reclaimable cache, plus the 16-20 GiB guard reserve). About 14 GiB of the anonymous memory is host copies of dense tensors (`resident_layers`, read once and re-read by every graph build, including KV-capacity rebuilds, so they cannot just be freed). Mapping them `MAP_PRIVATE|MAP_POPULATE` from the GGUF (offset 4096-aligned; env-gated) cut anonymous memory 84.5 -> 74.6 GiB and short-prompt decode was unchanged (1.72 first-to-last, 1.63 Rates). But on the 426-token prompt at a 78 GiB pool the MemAvailable low-water was the same (15.7 vs 15.0 GB) and decode was the same (1.37 Rates / 1.43 first-to-last vs 1.36 / 1.41); 82 and 90 GiB were still blocked by the 16 GiB floor. The kernel keeps roughly that much reclaimable-but-not-counted memory regardless, so the usable pool ceiling on this 125 GB box is about 74-78 GiB. Change reverted (core/engine.hpp), not adopted.

### Trial 43: prompt-adaptive resident set, oracle upper bound (negative)

25-token computing-history prompt, recipe, 64 GiB pool, 22 GiB hot set, swap-free, NGEN=80. Baseline run with `ZEROMOE_GLM5NEXT_ROUTE_TRACE` (80 tokens x 75 layers; 9270 distinct (layer, expert) pairs routed, more than the ~7900 resident): 1.44 tok/s (Rates), 1.50 first-to-last, loads 5752/64 tok, split stall 206 ms/tok. Then a usage file built from that same trace (base counts + 3x trace counts, in-sample = best case for any adaptive scheme): 1.55 (Rates), 1.63 first-to-last, loads 4206 (-27%), but split stall 216 ms/tok (unchanged). Result: a 27% cut in SSD loads is worth +7.6% decode; split stall tracks the number of layers with a pending miss times single-miss latency, not the load count. A real prompt-adaptive scheme would gain less than +7.6%, so with the 78 GiB pool (1.37 long) it cannot reach 1.5. Dynamic VRAM promotion not built.

### Trial 44: confidence-based pilot prefetch width (negative, removed)

Added a relative-score filter (keep candidate j if score >= top - delta) over a 6-wide pilot, 22-token prompt, recipe otherwise, 64 GiB pool, swap-free. Router scores of the top 6 candidates are clustered within ~0.3 (e.g. 16.28 16.05 16.05 15.96), so delta 0.5 / 2.0 kept all 6/6. Results (Rates / first-to-last): 6-wide d=0.5 1.62 / 1.71 (loads 3145, stall 182 ms/tok); 6-wide d=2.0 1.52 / 1.59 (loads 3100, stall 213); 3/3 baseline rerun 1.69 / 1.78 (loads 3916, stall 161). 20% fewer demand loads but a slower decode: extra speculative reads contend with demand reads. Consistent with Trial 43 (fewer loads != lower stall). Code removed.

### Trial 45: nsys decode profile (GPU idle structure)

nsys 12 s window of steady decode (22 tokens), `--cuda-graph-trace=node`, 22-token prompt, recipe, 64 GiB pool, c07 toml (1.60 tok/s under the profiler). 5060 Ti (core/dense): 107,548 kernels = 4,889 per token, busy 48.8 ms/token. 3090 (hot experts): 12,553 kernels, busy 12 ms/token (2%). Inter-kernel gaps on the 5060 Ti per token: <10 us 1.0 ms, 10-50 us 1.1 ms, 50-200 us 6.3 ms, 0.2-1 ms 23.2 ms (about one per layer, a per-layer sync), 1-5 ms 138.7 ms (46/token), >5 ms 328.7 ms (37/token, avg 9 ms). So launch overhead is ~8 ms/token (<2%) and ~470 ms/token (85%) is the GPU waiting for the CPU expert lane, which in turn waits on SSD loads and CPU expert math. CUDA graph / launch / kernel tuning has nothing material to give; the lane (236 ms math + ~165 ms split stall per token) is the critical path. The GPU-assist ring (about 106 experts/token) is already PCIe-Gen3-bandwidth balanced against the CPU (slot sweeps flat, Trial 21).

### Trial 46: lane breakdown, and page-freeing eviction is the largest remaining cost

New per-token lane breakdown (22-token prompt, recipe, 64 GiB pool, 1.68-1.72 tok/s): begin_load 50 ms, phase-1 graph 87, phase-2 graph 51, unsplit graph 64, assist wait 15, split stall 157-163. begin_load is host bookkeeping, but 65 ms/token of `madvise(MADV_DONTNEED)` eviction runs (142 evictions/token across lane and prefetch worker) and 15 ms/token of arena-lock wait. Async eviction (victims marked loading, freed by a background task after the read is published; env-gated, removed) was a wash: begin_load 50 -> 17 ms but phase-2 and unsplit graph time rose by about the same (madvise now runs beside the math; TLB-shootdown IPIs hit the compute threads): 1.66 / 1.68 vs 1.68 off.

Upper bound via an experiment-only `ZEROMOE_OFFLOAD_EVICT=none` (never return pages; memory grows, so short runs and small pools only): pool 40 GiB, NGEN=32: 1.32 / 1.28 (default) vs 1.67 / 1.67 (none), +28%; pool 50 GiB, NGEN=24: 1.42 vs 1.91, +34%; begin_load 68 -> 4 ms/token, lane wait 106 -> 29. Process kernel CPU time over 10 s of decode at 50 GiB: 14.4 CPU-s (default) vs 7.5 (none); user time 47 vs 28. At 64 GiB `none` cannot run because prefill's frees are skipped too and the memory guard trips. Evictions per token fall from ~295 (50 GiB) to ~142 (64 GiB), so the gain at 64 GiB should be smaller than +34% but is likely still >10%.

Why it is not simply fixable: the arena is expert-indexed (address = layer base + expert x stride) and eviction bounds RSS by returning pages; the prefill graph also needs that layout. Recycling slots instead of freeing pages needs an expert->slot indirection (lane remap tables, GPU assist, stage worker, pin hooks, hot-set copy, prefill read-in-place), a multi-hour refactor. THP is already disabled for the arena (MADV_NOHUGEPAGE), MADV_FREE was slower (Trial 29), async eviction is a wash. No change adopted; diagnostics (lane breakdown, evict timing) and the `EVICT=none` knob stay (experiment only).

### Trial 47: page-reusing eviction (`ZEROMOE_OFFLOAD_RETARGET=1`) -- peak 2.05 tok/s, long output 1.91 tok/s

Change (`VirtualArena::retarget`, used by `ExpertResidency::begin`): when a miss inserts an expert and the cache evicts a victim, the victim's already-populated pages are moved to the new expert's address with `mremap(MREMAP_MAYMOVE|MREMAP_FIXED|MREMAP_DONTUNMAP)` instead of `madvise(MADV_DONTNEED)` + fresh page faults and zeroing. The vacated range stays an empty anonymous mapping with the same VMA flags (NOHUGEPAGE). Planes of different size move their common prefix and free the tail; any mremap failure falls back to freeing the victim's remaining planes. Prefill (transient path) is untouched. Opt-in via env; 0 failed plane moves in all runs.

Interleaved A/B (22-token prompt, recipe: prefetch 3 / stage 3 / direct model reads, 64 GiB pool, 22 GiB hot set, c07 toml, swap-free, NGEN=80), first-to-last / Rates:
| run | retarget | first-to-last | Rates |
|---|---|---|---|
| rt64 | on / off | 2.03 / 1.75 | 1.91 / 1.66 |
| c_on_a | on | 2.05 | 1.93 |
| c_off_a | off | 1.74 | 1.65 |
| c_on_b | on | 2.06 | 1.92 |
| c_off_b | off | 1.77 | 1.68 |
Lane breakdown (ms/token) on vs off: begin_load 14 vs 48, lane wait 26 vs 62, split stall 139 vs 173, compute 351 vs 394; read 4.7 vs 5.4 ms/load (reads land in warm pages). Gain +16-18%.

Coherence: c_on_a is character-identical to c_off_b (420 chars); other pairs diverge only late (chars 318-410 of ~420), the same near-tie divergence that two retarget-off runs show against each other. 300-token output reads coherently, no repetition.

Long prompt (426-token computing-history prompt, 16 GiB guard floor override, ZEROMOE_MEM_GUARD_FLOOR_MB=16384, host_ram_gb 100): 74 GiB pool: 1.45 and 1.52 Rates (1.52 and 1.59 first-to-last) vs 1.31 without retarget (Trial 41). 78 GiB is refused by the 16 GiB floor, as before.

Long generation (22-token prompt, NGEN=300, 64 GiB pool, retarget on): 1.91 Rates, 1.94 first-to-last (1.80 overall), lane stall flat at ~178 ms/token to the end, 35k mappings at token 80 (bounded by one mapping per (layer, expert, plane), about 115k worst case vs vm.max_map_count 1,048,576).

Recipe for the best numbers: `ZEROMOE_GLM5NEXT_PILOT_PREFETCH=3 ZEROMOE_GLM5NEXT_PILOT_STAGE=3 ZEROMOE_GLM5NEXT_MODEL_DIRECT=1 ZEROMOE_OFFLOAD_RETARGET=1` (long prompts additionally need the guard-floor override and a 74 GiB pool).

Cleanup note: after Trial 47 the measurement-only code was removed (evict/lock timers, retarget counters, `EVICT=none`, balanced split mode, split timing, read chunking, async-eviction); the per-token lane breakdown log line stays. Re-verified on the cleaned build: retarget on 2.09 first-to-last (Rates 1.96) vs off 1.78 (1.68). Memory at peak and the config/scripts are in `docs/GLM5.3-baseline.md` and `bench/glm53/`.

Update (2026-10-20): `ZEROMOE_OFFLOAD_RETARGET` is now on by default whenever SSD offload is active; `=0` disables. See `docs/offload-retarget.md`.

## DeepSeek V4.1 Flash (2026-10-21)
3090 + 5060 Ti, Release, coherent chat-template text at temp 0.8, `configs/deepseek_v41_100gib_ddr_17gib_vram_dual_gpu.toml`:
decode about 4.0-4.5 tok/s (about 230 ms/token), up from 0.88 -> 1.30 -> 3.5 over earlier sessions. 10 tok/s target not met.
Prefill about 29 tok/s on 817 tokens (older measurement). Details, switches and verification:
`docs/deepseek_v41_changelog_2026-10-21.md`.

## MiMo V2.6 Flash and the Ling gate (2026-10-23, RTX 3090 + RTX 5060 Ti, ggml 0.27 patched build)

| Metric | Baseline | Notes |
|---|---|---|
| MiMo decode, 64 tokens, 4 prompts | 13.4–15.7 tok/s | dual-GPU config, 3090 cache 15 GiB + 5060 Ti 14.5 GiB, DDR 100 GiB, 0 SSD reads in decode; `docs/models/mimo26-flash.md` |
| MiMo prefill, 21 / 146 / 516 / 1022 tokens | 24.6 / 43.8 / 77.0 / 80.4 tok/s | 512-token chunks, op-offload from 256 tokens |
| Ling short decode (`[REQUEST-DECODE-OUTPUT]`, 32 tokens) | 10.07 tok/s (10.1–10.4 measured) | gate: `bench/ling/reference-audit/run.sh`, tokens compared within the same tree |

Numbers are warm: the first run after a fresh ggml build JIT-compiles for the sm_120 GPU (Ling 4.5 tok/s,
MiMo's first prefill 45 s). Build and patch recipe: `docs/build-ggml.md`.

Ling long-prompt prefill (1,771 tokens, 64 GiB DDR pool, 2026-10-23): 22.2 tok/s as one graph, 21.4 tok/s in 512-token
chunks, 21.6 tok/s with chunks and `PREFILL_OP_OFFLOAD=1` (no effect: the Ling graph does not engage op-offload).
