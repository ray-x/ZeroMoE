# Qwen 3.8 Flash decode baselines: how they were achieved (2026-10-16)

Two placements, single RTX 3090, both reproduced on the current `graph-framework` build with one script:

```bash
bench/qwen_decode_baselines.sh split  10 400        # 22 layers on GPU + 26 on CPU, UD-IQ1_M
bench/qwen_decode_baselines.sh hybrid 14554 200 10  # usage-ranked hot experts in VRAM, Q2_0
```

| placement | result (script run, 18:41) | earlier runs same day | reference |
|---|---|---|---|
| split 22/26, UD-IQ1_M | **28.66 tok/s** (400 tokens, 13.96 s) | 24.6, 26.0 (clangd indexer using ~1.5 cores) | 26.2-26.9 on 2026-10-02 |
| hybrid K=14,554 of 24,576, Q2_0 | **43.12 tok/s** (196 steps) | 41.28 | 39-42 on 2026-10-05 |

Both beat their earlier references; these are the new baselines in `configs/bench/perf-baselines.md`. Single runs: the spread between
runs of the same config today was 24.6-28.7 (split) and 41.3-43.1 (hybrid), so treat a drop under ~5% as noise and above it
as worth investigating. Loadavg at the start of the 28.66 run was 1.06; at the start of the 43.12 run 5.38 (the CPU part of
the hybrid is small, so it is less sensitive).

## Machine and build

| item | value |
|---|---|
| CPU | Xeon W-2235, 6 cores / 12 threads, governor `performance` |
| RAM | 125 GB (118 GB available at start), zram swap only |
| GPU | RTX 3090 24 GB, PCIe Gen3 x16, power limit 350 W; the 5060 Ti is hidden with `CUDA_VISIBLE_DEVICES=0` |
| source | `graph-framework` at `e0fc5f2` (the binary was built 2026-10-16 03:01:57, the commit is 03:03:33; later commits are docs and the script only) |
| binary | `build_native/zeromoe` (Release, CUDA, `GGML_AVX512=ON`, `GGML_AVX512_VNNI=ON`, `GGML_AVX2=OFF`, OpenMP on) |
| ggml | `ZEROMOE_GGML_ROOT=/home/ray/work/ggml-iq2xxs/ggml` at `7bd457fa1` plus uncommitted env-gated trace code (inactive unless `SCHED_TRACE2`/`SERIAL` are set) |
| engine rule | one `zeromoe` at a time; the script waits for no `zeromoe` / `llama-bench` process |

## Split: 22 layers fully on the GPU, 26 fully on the CPU

Model: `/data/sdb/models/qwen3.8-flash-next/UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf`.
Config: `configs/bench/qwen4exp_tg400_iq1m_22_26.toml`, with three lines changed by the script (`omp_max_threads = 10`,
`max_new_tokens = 400`, `log_path`). Settings that define the placement and the fast path:

| key | value | meaning |
|---|---|---|
| `cpu_affinity_layers` | `[0-7, 14, 20, 21, 22, 24, 28, 30, 33, 34, 36, 38, 40-46]` (26 layers) | these layers run all experts on the CPU; the other 22 are complete on the GPU |
| `qwen4exp_full_graph`, `qwen4exp_full_graph_decode` | `true` | one ggml graph per token across all 48 layers |
| `qwen4exp_cpu_gpu_overlap` | `true` (`..._max_layers = 0`) | CPU experts of one layer overlap GPU work |
| `qwen4exp_decode_device_hc`, `hc_ggml`, `gdn_full_ggml`, `gdn_proj_ggml` | `true` | HC, GDN in the graph |
| `qwen4exp_cpu_ggml_moe`, `qwen4exp_cpu_strided_grouped`, `qwen4exp_cpu_fused_grouped` | `true` | ggml IQ1_M CPU kernels |
| `moe_ggml_mul_mat_id`, `moe_ggml_shared_expert`, `moe_ggml_gpu_router`, `moe_ggml_device` | `true`, `true`, `true`, `0` | |
| `context_size` | `416` | 16-token prompt + 400 generated. Longer pushes the last token onto the legacy path (+1.5 s, fakes an 11% gap) |
| `ram_preload_all_experts`, `host_ram_gb`, `ram_pinned_budget_gb` | `true`, `100`, `45.0` | CPU-layer experts live in pinned DDR (21.2 GiB aliased) |
| `omp_max_threads` | **`10`** (config ships 12) | see "Why 10 threads" |
| `[[gpus]]` | `dense_gb = 4.1`, `kv_cache_gb = 2.0`, `moe_cache_gb = 11.5`, `kv_bits = 4` | legacy cache is disabled under full graph; dense arena actually 3.26 GiB |

Environment (all set by the script):

```text
CUDA_VISIBLE_DEVICES=0
ZEROMOE_QWEN4EXP_FULL_GRAPH=1
ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE=1
ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS=40      # "as many layers as fit"; the 22 non-affinity layers become the 17.74 GiB device arena
ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1
ZEROMOE_BENCH_SILENT=1                         # no token printing / per-token VRAM sampling in the timed path
ZEROMOE_BENCH_IGNORE_STOP=1                    # generate all 400 tokens
```

Input: stdin `Tell a story.` then `:q`; with the chat template that is 16 tokens (`[PREFILL-DONE] 16 tokens`). Greedy
(`temperature = 0.0`), no teacher forcing, no prompt-token file (the original `/tmp` token files are gone).
Metric: `[REQUEST-DECODE-WALL] tokens=400 wall=...` in the engine log (includes ~65 ms decode-graph setup).

Expected engine-log lines (check them before trusting a number):

```text
[MOE-OWNERSHIP] full-graph device block [8, 48) marks 22 non-CPU-affinity layer(s) as VRAM-only, approximately 17.7368 GiB
[GGML-QWEN-FULL] device expert arena used=17.74/17.74 GiB; 22 complete layer(s), 11264 expert records
[GGML-QWEN-FULL] graph built layers=48 nodes=6509 splits=54
[GGML-QWEN-FULL] CPU backend configured with 10 threads (persistent threadpool)
[GGML-QWEN-FULL-DECODE] enabled: one scheduler submission per token, KV capacity=416 ...
```

## Hybrid: usage-ranked hot experts in VRAM

Model: `/data/sdb/models/qwen3.8-flash-next/Q2_0` (directory). Config: `configs/testing/qwen4exp_q2_0_option_b_global_topk48.toml`, with
`qwen4exp_option_b_global_top_k = 14554`, `omp_max_threads = 10`, `log_path` set by the script.

| key | value | meaning |
|---|---|---|
| `qwen4exp_option_b_selected_layers` | all 48 layers | |
| `qwen4exp_option_b_global_top_k` | **14554** (59% of 24,576 experts) | one global ranking from the usage survey; ranked experts go to VRAM, the rest stay in DDR |
| `qwen4exp_option_b_global_floor_per_layer` | `1` | every layer gets at least one hot expert |
| `qwen4exp_option_b_top_n` / `..._right_size_top_n` | `512` / `false` | per-layer ceiling off in global mode |
| `qwen4exp_option_b_min_coverage` | `0.80` | |
| `pin_file`, `prefill_pin_file` | `tests/moe_usage_survey/zeromoe_usage_survey_qwen4exp_usage_tap`, `..._survey_prefill` | the usage rankings; `freeze_usage_stats = true` |
| `cpu_affinity_layers` | `[]` | no layer is CPU-pinned; cold experts of every layer run on the CPU |
| `qwen4exp_full_graph`, `..._full_graph_decode` | `true` | |
| `host_ram_gb` | `90` | `ram_preload_all_experts = true` |
| `context_size` | `2048` | |
| `[[gpus]]` | `dense_gb = 4.1`, `kv_cache_gb = 2.0`, `moe_cache_gb = 11.5` | legacy expert cache is not created under full graph |

Environment (all set by the script):

```text
CUDA_VISIBLE_DEVICES=0
ZEROMOE_QWEN4EXP_FULL_GRAPH=1
ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE=1
ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS=0        # expert residency comes from the global pool, not from a layer count
ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1
ZEROMOE_TOKEN_LOOP_DIAG=1                      # per-step iter_total, the metric source
ZEROMOE_BENCH_IGNORE_STOP=1
```

Input: `Explain briefly the key concepts of neural networks and gradient descent in plain English.` on stdin, `--ngen 200`.
Metric: 1000 / mean(`iter_total`) over steps >= 3 (196 steps). Output starts "Here is a simple explanation of neural networks
and gradient descent using an analogy." (coherent).

Measured resources: expert arena 18.74 GiB; steady GPU memory used 22.38 GiB, free 1.62 GiB (minimum free 1.46 GiB, 250 ms
`nvidia-smi` samples); no fallback to layered decode; survey-predicted hot-set coverage 93.3%. Graph admission needs >= 1.00 GiB
free when the decode graph is built: K above ~14,729 fails on a 24 GiB card unless `ZEROMOE_QWEN4EXP_GRAPH_ALLOC_FLOOR_MIB=512`
is set (then K = 14,746 builds; do not use that for the baseline). A refusal means the run silently decodes all-CPU at ~835 ms/token,
which the script reports as `FELL BACK TO LAYERED DECODE: True`. The engine releases the expert arena before a prefill that needs
more workspace than is free and rebuilds it for decode (about 4 s once); this baseline is the decode figure, not first-request latency.

## Why 10 threads, not 12

The CPU part of the graph is barrier-bound: the ggml thread pool waits for its slowest thread every op, so a thread that shares
a core with another process stalls all of them. With the shipped `omp_max_threads = 12` on a 12-thread CPU, the split config
measured **3.05 tok/s** (400 tokens) and 1.49 tok/s in a 40-token layer-timing run while a Serena `clangd --background-index` (~1.5
cores) and a niced ggml compile from another session were running. At 10 threads, with the indexer still running, it measured
24.6 and 26.0; with the box quieter (loadavg 1.06) 28.66. The hybrid's CPU share is small, so it is much less affected.
Rule: set `omp_max_threads` to (hardware threads - 2) unless the machine is known idle, and read `/proc/loadavg` before a decode
measurement (the script prints it). The per-layer timing that showed it: `ZEROMOE_QWEN4EXP_GRAPH_LAYER_TIMING=1` printed `cpu_total=338 ms` per token
under contention (I did not capture a quiet-run value for comparison).

## What does not reach these numbers

`configs/testing/qwen4exp_3090_only.toml` (the prefill benchmark config: partial residency plus the RAM tier, UD-IQ1_M) decodes at
19.5 tok/s. It was the config used for the Qwen prefill work and for the 19.5 figure in `configs/bench/perf-baselines.md`; it is not
a decode-tuned placement.

## Raw logs (not committed; `logs/` is untracked)

`logs/qwen_decode_split_20261016_1841.{log,engine.log}`, `logs/qwen_decode_hybrid_K14554_20261016_1841.{log,engine.log,csv}`,
earlier same-day runs `logs/qwen_decode_split22_26_*.log`, `logs/optb_probe_K14554.*`. The engine log of the split run with 12
threads is `/tmp/zeromoe_qwen4exp_tg400_iq1m.log` (overwritten by later runs of the stock config).
