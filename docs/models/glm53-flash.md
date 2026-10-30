# GLM 5.3 Flash (GLM5NEXT)

- **Arch / family:** `glm5next` → `GLM5NEXT`. 45 trunk layers (34 KDA + 11 DSA), dense FFN layers
  0-2, MoE layers 3-44 (42) × 288 experts, top-8 + shared expert, `hc_mult=4`.
- **Model:** `/path/to/models/GLM-5.3-Flash/GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf` (+ shard 2).
- **Placement (whole-layer):** attention/HC/router/shared/head on the 3090; routed experts of
  layers 3-10 in VRAM, layers 11-44 on CPU (`cpu_affinity_layers = [11..44]`).

## Canonical recipe

```bash
echo "<prompt>" | CUDA_VISIBLE_DEVICES=0 \
  ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB=512 ZEROMOE_GLM5NEXT_CPU_THREADS=6 \
  scripts/run_capped.sh -- ./build/zeromoe \
  --config configs/bench/glm53flash_AJ_IQ2XXS_wholelayer8_cpu.toml --temp 0 --predict 150 --force-start
```

The registered GLM5Next graph is the default CUDA path; there is no layered-path environment switch.

## Frozen baseline

| Metric | Value |
|---|---|
| Decode (band) | **6.5 – 8.5 tok/s** (measured 7.5–7.6; doc 7.26) |
| Prefill | 0.5 tok/s (29 tokens; dominated by startup) |

## Tricks applied (why)

- **Fused hyper-connection ops** are the default (graph ~6.1k nodes vs ~31k unfused) — this is the
  single biggest decode win on this file (`docs/GLM5.3-Flash-baseline.md` §3.4).
- **LM head inside the graph** removes a ~48 ms/token CPU round-trip.

## Gotchas

- `host_ram_gb=106` needs `106 + 21.5 = 127.5 GiB` of `MemAvailable`; on a 125 GiB box preflight
  refuses, so run under `run_capped.sh` with .
- `ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB=512` is required or the dense arena refuses and decode
  silently falls back to legacy.
- A fresh binary's first run spends ~67 s in CUDA JIT (sm_75 PTX on sm_86); warm it once.

## GSQ-RCO 3.0-bit (117 GB, Q2_K/Q3_K/Q4_K experts), offload-pool mode (2026-10-29)

`GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf` does not fit the whole-layer placement (91 GiB of experts), so it runs in offload-pool
mode: experts live in a DDR pool read from the model file, hot ones in VRAM, the CPU lane computes the cold ones.

### Which model file the numbers are for

The GSQ-RCO file is larger and heavier per token than the AJ IQ2_XXS file used by the older baselines on this page:

| | AJ IQ2_XXS (older baselines) | GSQ-RCO 3.0-bit (this section) |
|---|---|---|
| file size | 81.3 GiB (87.3 GB, 2 shards) | 109.4 GiB (117 GB, 1 file) |
| routed experts | IQ2_XXS (8 layers: down IQ2_S), 2.06 bpw, ~6.6 MB/expert | Q2_K layers 9-44 (2.625 bpw), Q3_K layers 6-8, Q4_K layers 3-5 |
| dense tensors (attention, shared experts, head) | smaller; dense arena ~6.8 GiB | Q8_0; dense arena ~9.0 GiB (8.6 GiB uploaded) |
| expert bytes read per token | baseline | ~27% more for Q2_K layers, more still for layers 3-8 |
| fits whole-layer on the 3090 + CPU | yes | no, offload pool needed (cold set ~88 GiB) |
| dual-GPU decode (this box) | 70.1 ms/token = 14.3 tok/s (2026-10-15, perf-baselines.md) | 70.5-71.1 ms/token = 14.1-14.2 tok/s |

So the larger quant reaches the same speed as the AJ file: the CPU kernels for IQ2_XXS are the AVX-512 ones from the patch series
(0009), while Q2_K runs ggml's stock AVX2 dot, so a Q2_K-specific kernel is the remaining headroom. Output quality of the two
files is not equivalent (teacher-forced top-1 agreement 77%, mean KL 0.35, see the handoff doc): they are different models.

### Performance test: config and procedure

- Config: `configs/bench/glm53flash_gsqrco_offload_pool.toml` (copy of the run config; `graph_offload_first_layer = 3`,
  `graph_offload_ddr_gib = 88`, `host_ram_gb = 16`, one `[[gpus]]` entry for the 3090 with `moe_cache_gb = 12.4` so the dense arena
  fits, `vram_cache_full_layer_prefix = 0`, no GPU assist, `ram_pinned_budget_gb = 16`).
- Launcher: `bench/glm_gsqrco_pool.sh TAG [POOL_GIB] [prompt]` from the repo root. It waits for a quiet box, sets
  `CUDA_VISIBLE_DEVICES`, `ZEROMOE_GLM5NEXT_GPU1_HOT_GIB=14.2` (dual), `ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB=512`,
  `ZEROMOE_GLM5NEXT_PREFILL_CPU_THREADS=6`, runs under `scripts/run_capped.sh --limit-gb 118`, watches MemAvailable at 1 Hz
  (stops the engine below 6 GiB) and prints the steady decode rate and the lane breakdown. `CVD=0` runs the 3090 alone.
- Run: `--temp 0 --predict 100 --force-start`, prompt "Write a Python function that checks whether a string is a palindrome,
  and explain it.", `--graph-offload-usage-file docs/ssd-offload/data/glm53flash_usage_combined_old_plus_survey5.txt`
  (the same file is the pin ranking in the toml; prefill ranking `glm53flash_survey5_usage_prefill.txt`).
- Measurement: steady decode = mean of the engine's `FULL-DECODE-TIME` for tokens 21-99 (the first tokens include graph build);
  the lane's `wait`/`compute` come from the `[GLM5NEXT-LANE] tokens=96` line; `misses=0` confirms the pool holds the cold set.
- Code: master after `e205935` (GLM decode lane 10 threads, prefill threads independent; the dense host copies are kept, see the
  warning below). Built from a clean tree with ggml patches 0001-0017 (+ host-node 0001-0003) on llama.cpp `8baabd53b`.
- Seeding the pool takes ~4 min from the SATA SSD and prefill of 23 tokens ~65 s on the first request (cold experts read
  from the pool through the layered prefill); both are excluded from the decode figures.

Recipe summary:

```toml
graph_offload_first_layer = 3
graph_offload_model = "/path/to/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf"
graph_offload_ddr_gib = 88      # the cold set is ~88 GiB; 80 leaves ~2 SSD misses/token (190 ms/token)
```

| Config (pool 88 GiB, 0 misses) | ms/token | tok/s |
|---|---|---|
| 3090 only, 6 decode threads | 93.8 | 10.7 |
| 3090 + 5060 Ti (14.2 GiB hot), 6 threads | 76.9 | 13.0 |
| same, 10 decode threads (new default) | 70.5-71.1 | **14.1-14.2** |
| same, K-quant GPU assist (1 slot) | 90.7 | 11.0 (worse: leave assist off with two GPUs) |
| 3090 only, pool 80 (2 misses/token, 70 ms of SSD wait) | 190 | 5.3; with K-quant assist 5.8 |

Lane at 14.2 tok/s: wait 21 ms + CPU cold-expert compute 39 ms per token. Measured with the box partly loaded (load 12-14 from
other jobs), so these are slight underestimates.

Changes behind these numbers: decode CPU lane defaults to 10 threads (was 6); the prefill graph's thread count no longer
follows `ZEROMOE_GLM5NEXT_CPU_THREADS` (10 prefill threads grew host memory by ~16 GiB right after pool seeding and tripped the
memory watchdog; prefill stays at 6, override with `ZEROMOE_GLM5NEXT_PREFILL_CPU_THREADS`); the early-ready lane names
the cause (layer, call, input hidden state, weights) before aborting on a non-finite router weight.

### Ranking matched to this model file (2026-10-30, in-sample)

The shipped usage ranking was collected on the AJ IQ2_XXS file; GSQ-RCO routes differently (77% top-1 agreement), so the hot sets
(3090: `pin_file`; 5060 Ti: `--graph-offload-usage-file`) are mismatched. A 6-prompt pilot survey on this file
(`docs/ssd-offload/data/glm53flash_gsqrco_pilot6_usage_decode.txt`, prompts `bench/data_glm_gsqrco_pilot6_prompts.txt`, recorded with
`graph_usage_record_file`) shows the effect on **those same prompts** (dual GPU, 3122 hot experts, pool 88, 550 decode tokens):

| ranking | cold routes (offline) | steady decode (engine) |
|---|---|---|
| shipped (AJ-derived) | 54.1% (engine measured 52%) | 70.8 ms = 14.13 tok/s |
| shipped + pilot counts (`bench/make_mix_ranking.py`) | 41.0% | **63.1 ms = 15.84 tok/s** (+11%) |
| pilot only (offline only, not run) | 37.1% | predicted ~58 ms |

**This is in-sample and must not be quoted as a general gain**: the ranking comes from the prompts it is tested on. Offline,
3090-alone cold routes go 74.2% -> 63.2% the same way. The real gain on unseen prompts needs a held-out survey (the 2026-10-14
AJ refit gave +2-3 tok/s on held-out prompts); `bench/make_survey_sets.py` builds a stratified train/test split of the 100
survey prompts for that. Output text changes slightly with the ranking (different experts run on GPU vs CPU), as with any
placement change. Run: `GLM_RANKING=<file> GLM_EXTRA_ENV=ZEROMOE_GLM5NEXT_FULL_GRAPH_PREFILL=1 bench/glm_gsqrco_pool.sh ...`.

**Full-graph prefill** (`ZEROMOE_GLM5NEXT_FULL_GRAPH_PREFILL=1`) works on this file at pool 88 now: 20-104 token prompts prefill in
0.7-3.6 s (20-31 tok/s) instead of ~46 s for 20 tokens on the layered path; 6 consecutive prompts in one session all decode on the
graph path (`graph_prefill_runs=6`). It is opt-in and was validated only on those 6 prompts.

Gotchas:
- **Memory figures below were measured with a feature that has since been removed (2026-10-30).** Until then the engine freed its
  8.7 GiB of dense host copies once the first decode graph held them on the device. That broke every later request in a session
  (the layered prefill and the legacy logits path read `attn_k_b`/`attn_v_b`/`output.weight` from host memory: zeroed attention, zero
  logits, output `!`); the benchmarks sent one prompt per session, so it went unnoticed until a 6-prompt session failed on prompt 2.
  With the copies kept, expect about 8.7 GiB lower MemAvailable than the figures below: dual GPU pool 88 about 17-21 GiB (above the
  10 GiB guard), 3090 only pool 88 about 7 GiB (**below the guard: the 10.7 tok/s 3090-only figure needs a smaller pool or a graph-only
  prefill; not re-measured**). Any change that frees host memory must be validated with a session of at least two prompts.
- Pool 88 peaks at ~104 GiB anonymous memory on the 3090-only run (15.7 GiB available, guard 10 GiB) and ~89-92 GiB with the
  5060 Ti holding 14.2 GiB of experts (29 GiB available). Run it only on a quiet box: another job using ~10 GiB trips the
  watchdog. Skipping the dense host copies at startup does NOT work: the layer-by-layer prefill builds its MLA caches from
  host `attn_k_b`/`attn_v_b` and the dense arena size is derived from the host copies (a try gave zeroed weights and a 0.65 s fake prefill).
- Benchmarks are invalid under CPU load: the same run gave 0.89 tok/s while a neighbour compiled (lane compute 323-2123 ms/token).
- Open: at pool 80 (3090 only) some runs hit a non-finite hidden state at decode start, `invalid argument` on dense-weight
  uploads, or runaway memory during seeding. Seen once at pool 88 (next bullet); cause not found. Ruled out: lane (messenger only), pool
  pinning, deferred host unregisters, registered upload sources, CUDA API errors (driver log). Notes in the handoff doc.
- Intermittent: 1 of 6 launcher runs at pool 88 (dual GPU, neighbour compiling) aborted right after prefill with
  `cudaMemcpyAsync ... invalid argument` in `ggml_backend_cuda_buffer_set_tensor` (same family as the pool-80 failures above).
  Not explained.
- 20 tok/s has no measured path yet: per token 40 ms CPU expert compute + 21 ms graph. A 512-bit Q2_K dot was tried
  (`bench/q2k_dot_bench.cpp`): bit-exact with ggml's stock AVX2 kernel but not faster (1 thread 8.2 vs 8.3 GB/s, 10 threads
  38.1 vs 37.5 GB/s). The raw DRAM read rate is ~60 GB/s and the dot reaches ~63% of it, in both 256- and 512-bit form, so the
  limit is per-block instruction work (scale/min handling, scalar setup, one row at a time), not vector width or DRAM. A
  restructured kernel could gain up to ~1.6x on the CPU part (71 -> ~55 ms/token); untried, and a changed float summation order needs a
  same-input trace.
