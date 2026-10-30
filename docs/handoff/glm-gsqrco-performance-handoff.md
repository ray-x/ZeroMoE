# GLM-5.3-Flash GSQ-RCO: performance handoff (2026-10-29, status updated 2026-10-30)

## Latest status (2026-10-30)

**Merge status (read first):** the full-graph prefill default for GLM and the 1 GiB prefill-planner margin (`618d454`) are on master. Validated on a binary stamped clean at `618d454` (3090 + 5060 Ti, GSQ-RCO,
pool 88, default settings, no graph-prefill variable): **(1)** a 2,313-token prompt completes (it ran out of memory with the old 0.15 GiB margin): the planner declines the 2048-token batch (1.90 GiB needed, 1.55 GiB free),
retries at 408 tokens, 3090 peak 23,446 of 24,576 MiB; prefill takes **408 s = 5.7 tok/s** (the cold experts run on the CPU; `graph_prefill_op_offload = true` ran the 3090 out of memory again, so it is not enabled),
so long-prompt prefill is correct but slow and remains open; **(2)** the 6 pilot prompts decode at **70.2 ms = 14.25 tok/s**, 6 of 6 on the graph path; **(3)** teacher-forced logits are **bitwise identical** on all 8 steps to the
earlier runs. Not yet re-run: the merged tree itself, which also contains the other agent's later commit `613d9fe` (environment cleanup, 71 lines of `models/` and `core/`), and the Qwen family, which shares the RAM-tier slab
code fixed here. Chunking and memory margins are tuned on this one file and card; other GLM files or GPUs may need their own check.

Code: master `b7d6bd1e027c` and later (clean stamp; ggml llama.cpp `8baabd53b` + patches 0001-0017 + host-node 0001-0003). Model file:
`GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf`, 117.48 GB (109.41 GiB), 3.000 bpw (Q2_K 83.3% / Q3_K 6.9% / Q4_K 6.9% of 313.327 B parameters). Full model-file
comparison with the AJ IQ2_XXS file (87.35 GB, 2.230 bpw): `configs/bench/perf-baselines.md`.

**Decode, 3090 + 5060 Ti (14.2 GiB hot experts on the 5060 Ti), pool 88 GiB, 0 SSD misses**

| measurement | steady decode | note |
|---|---|---|
| shipped settings, clean runs, 6 pilot prompts, 550 tokens | **69.0-70.9 ms = 14.1-14.5 tok/s** | best clean run 68.9 ms (14.51); first-session single prompt 70.5-71.1 ms |
| shipped + ranking matched to the pilot prompts | 63.1 ms = 15.84 tok/s | **in-sample** (ranking built from the same 6 prompts): an upper bound, not a general gain |
| AJ IQ2_XXS, same binary, same prompts | 75.4 ms = 13.26 tok/s | the 70.1 ms record of 2026-10-15 was a different binary, prompt and placement and is not reproduced |
| 3090 only | not re-measured (the 10.7 tok/s of 2026-10-29 needs a smaller pool or more memory than is safe, see the GLM page); two pool-84 runs (about 1 SSD miss per token) gave 9.42 and 8.11 tok/s but both were contaminated by compile load (63 and 711 samples): **no valid 3090-only speed**; memory is valid: min 19.6-20.7 GiB available at pool 84, so pool 88 should leave ~15 GiB | not re-run by decision |

**Changes on master from this work**
- decode CPU lane 10 threads by default; prefill thread count independent (6); pool 88 GiB recipe, config `configs/bench/glm53flash_gsqrco_offload_pool.toml`, launcher
  `bench/glm_gsqrco_pool.sh` (quiet-box gate, memory watchdog, contamination report, `GLM_RANKING`, `MOE_CACHE_GB`, `ZEROMOE_BIN`, CUDA error log, abort backtrace helper);
- **bug fixed (`b7d6bd1`)**: the engine freed its dense host copies after the first decode graph, so every later request in a session ran with zeroed attention and zero
  logits; removed (found with a 6-prompt session; earlier benchmarks sent one prompt);
- **full-graph prefill is now the GLM default, like every other family** (the opt-in variable `ZEROMOE_GLM5NEXT_FULL_GRAPH_PREFILL` was removed on owner instruction; MTP keeps the layered path): on this file at pool 88, 20-104 token prompts take 0.7-3.6 s instead of ~46 s for 20 tokens; validation of the new default is recorded below;
- survey tooling (`bench/make_survey_sets.py`, `bench/make_mix_ranking.py`), the pilot usage counts, the Q2_K dot benchmark, Nsight analysis scripts;
- the early-ready lane names the cause (layer, call, hidden state, weights) before aborting on a non-finite router weight.

**Levers measured and closed (clean runs, same binary and prompts)**: GPU assist (worse), skipping/freeing the dense host copies (broke later requests), Q2_K AVX512-VNNI dot
(bit-exact, 1.9 ms slower: 70.9/71.2 vs 69.0/69.4 ms), more resident experts (5060 Ti 15.0 GiB + 3090 cache 13.4 GB: 68.9 vs 69.7 ms, noise), adaptive promotion (the
graph-owned path cannot work in pool mode; the old replay says not worth building), pool pinning/flush theories for the pool-80 failures.

**Where the time goes (Nsight, section 2.5)**: CPU waits for the 3090 20%, CPU cold experts 54% (both GPUs idle), GPU combine + 5060 Ti branch + token tail 27%.
**Next:** run the 5060 Ti hot-expert branch concurrently with the CPU lane (about 7 ms, to ~15.6 tok/s) with its own copies of ids/hidden state and a same-input
parity check first; re-measure the 3090 alone at pool 88; a held-out ranking check (needs a survey, not run by decision). **20 tok/s is not reached.**
**Open:** the intermittent `cudaMemcpyAsync ... invalid argument` crash (3 of ~15 runs before the host-copy release was removed; none in the ~20 runs since).

---------------------------------------------------------------------------------------------------------------------

Goal that was set: GLM decode from ~10 tok/s toward ~20 tok/s (first recover the ~14 tok/s the project had before, on
the 3090 first, then with the 5060 Ti). Result when this document was first written: 14.1 tok/s on the 3090 + 5060 Ti, 10.7 tok/s on the 3090 alone (see the status above for the current numbers).

Hardware and file: Xeon W-2235 (6 cores / 12 threads, AVX-512 + VNNI, 4-channel DDR4, ~60 GB/s measured read), 125 GB RAM,
RTX 3090 (24 GB, PCIe 3.0 x16) + RTX 5060 Ti (16 GB, x8), model
`GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf` (109.4 GiB; experts Q2_K layers 9-44, Q3_K 6-8, Q4_K 3-5; dense Q8_0 ~9 GiB). The host is
shared with other agents (builds, model runs): every number below was taken with that in mind, see section 2.

---------------------------------------------------------------------------------------------------------------------

## 1. How to improve GLM performance: what was done, what is left

### 1.1 Where the time goes (3090 + 5060 Ti, pool 88 GiB, zero SSD misses)

Per token 70-71 ms (14.1-14.3 tok/s). From the engine's `[GLM5NEXT-LANE]` line (42 MoE layers):

| part | ms/token | what it is |
|---|---|---|
| lane **wait** | ~21 | the CPU lane waiting for the GPU part of each layer (attention/HC/router, D2H of ids+weights) |
| lane **compute** | ~40 | cold experts computed on the CPU (+ output upload); ~200 cold experts x ~8.9 MB (Q2_K) = ~1.8 GB per token |
| rest | ~10 | LM head, graph tail, host sync |

Facts that bound what is possible:
- The box reads DRAM at ~60 GB/s (simple AVX-512 read, 6-12 threads). The Q2_K dot reaches 38-48 GB/s at 10 threads. So the
  CPU part cannot go below ~30 ms; the whole token has a floor of ~61 ms (~16 tok/s) **unless fewer bytes are read per token**.
- 3090 VRAM: 51% used during startup/prefill (12.9 GB, only the expert cache), **92% during decode** (22.7 GB, the dense arena
  8.96 GiB is uploaded when the decode graph is built). 5060 Ti holds 14.2 GiB of hot experts.
- The AJ IQ2_XXS file (81.35 GiB = 87.35 GB, 2.230 bpw) gave 70.1 ms on the 3090 + 5060 Ti on 2026-10-15; GSQ-RCO (109.41 GiB, 3.000 bpw vs AJ 2.230 bpw, 1.345x the expert bytes) gives 70.5-71.1 ms.
  Same speed with more bytes: the CPU kernels for the two quants have different efficiency (IQ2_XXS has the AVX-512 patch 0009;
  Q2_K runs ggml's stock AVX2 dot).

### 1.2 Measured results

| config (all pool 88 GiB unless noted) | steady ms/token | tok/s |
|---|---|---|
| 3090 only, pool 80 (~2 SSD misses/token) | 190 | 5.3 (5.8 with K-quant assist) |
| 3090 only, 6 decode threads | 93.8 | 10.7 |
| 3090 + 5060 Ti (14.2 GiB hot), 6 decode threads | 76.9 | 13.0 |
| same, 10 decode threads (**the default now**) | 70.5-71.1 | **14.1-14.2** |
| same, K-quant GPU assist (1 slot) | 90.7 | 11.0 (worse, leave assist off with the 3090 + 5060 Ti) |
| launcher runs on the shared host, stock lib: 7 normal runs | 70.0-76.4 (70.0, 70.8, 70.8, 72.0, 72.3, 74.9, 76.4) | 13.1-14.3 |
| same, 3 runs overlapped by a neighbour's compile | 85.7, 91.4, 173 | 11.7, 10.9, 5.8 (invalid) |

"Steady" = mean of the engine's `FULL-DECODE-TIME` for tokens 21-99. The same config measured 0.89 tok/s once while a
neighbour compiled: **never quote a number without the contamination report** (the launcher prints it).

### 1.3 Levers already pulled (kept)

| change | effect | where |
|---|---|---|
| pool 88 GiB (the cold set is ~88 GiB) instead of 80 / auto 67.9 | removes ~70 ms/token of SSD waits: 5.3 -> 10.7 tok/s on the 3090 | `graph_offload_ddr_gib` |
| 10 CPU threads for the **decode** lane (was 6) | lane compute 45.7 -> 39.9 ms: 13.0 -> 14.2 tok/s on the 3090 + 5060 Ti | `ggml_glm5next_graph_decode.cpp` |
| prefill threads independent of the decode setting (6) | 10 prefill threads grew host memory ~16 GiB after seeding and tripped the watchdog | `ggml_glm5next_graph.cpp`; since the 2026-10-30 env cleanup the env var is gone and both counts come from `omp_max_threads` (prefill = min(it, 6)) |
| ~~dense host copies freed once the decode graph holds them~~ | **removed 2026-10-30**: it zeroed attention/logits for every request after the first (found with a 6-prompt session) | see the warning in `docs/models/glm53-flash.md` |
| 5060 Ti holds 14.2 GiB of hot experts | 10.7 -> 13.0-14.2 tok/s | `ZEROMOE_GLM5NEXT_GPU1_HOT_GIB=14.2` |

### 1.4 Levers tried and rejected (do not repeat without a new idea)

- **K-quant GPU assist** (`graph_gpu_assist_slots=1` + `ZEROMOE_GLM5NEXT_GPU_ASSIST_ANY_TYPE=1`): +10% on the 3090 alone at pool 80
  (5.27 -> 5.83), **-15% with the 3090 + 5060 Ti** (11.0 vs 13.0). It pulls from the same DRAM, so it cannot beat the 60 GB/s floor.
- **Skipping the dense host copies at startup** (to fit a bigger pool): breaks the layer-by-layer prefill, which builds its MLA
  caches from host `attn_k_b`/`attn_v_b`, and the dense arena size derives from the host copies (zeroed weights, a 0.65 s fake
  prefill). Reverted. The clean fix is a graph-only prefill that does not need the host copies.
- **Wider Q2_K CPU dots** (`bench/q2k_dot_bench.cpp`, bit-exact vs ggml): 512-bit = no gain (1.00x); VNNI 256-bit = 0.9-1.0x; two
  blocks interleaved (`ilv2`) = 1.11x at 10 threads in the microbenchmark, but in the engine it measured **slower or equal**
  (73.0, 74.4 ms vs 70.0 stock; confounded by a differently-built private CPU library and neighbour compiles). The stock kernel is
  latency-bound, not ALU- or DRAM-bound, and the engine reads ~9 MB scattered chunks, not a 1.6 GB stream. Patch 0018 was dropped.
- **Pool pinning off, flushing deferred unregisters, per-layer finite checks**: all done to chase the pool-80 failures
  (section 2.4); none changed anything. Removed.

### 1.5 Remaining levers, ordered by expected value

1. **Overlap GPU and CPU work** (section 3). Potential: the 21 ms "wait" and part of the ~10 ms rest; unknown until profiled.
2. **Fewer cold bytes per token**: adaptive hot-set promotion from the prompt's actual routing (the static usage ranking covers
   ~38% of traffic; Qwen's `expert_promotion` gains came from this). Offload-pool mode has no dynamic promotion; it is new work.
   Measure first: the per-prompt hit rate from the `[LAYER-MISS]` / route logs vs the static ranking.
3. **Speculative decoding** (reads each cold expert once for several tokens, the biggest structural win): this GGUF has no
   draft head (`blk.0`-`blk.44` only, no `nextn`), and there is no small GLM model on the box, so nothing to run today. A model-free
   method (prompt lookup / n-gram) is not implemented in the engine (`spec_type` only has the MTP-style draft path).
4. **Graph-only prefill** so the dense host copies and ~16 GiB of pool headroom can be given back (pool 88 currently leaves 16-30
   GiB of MemAvailable; the guard stops at 10).
5. ~~A restructured Q2_K kernel~~ **measured 2026-10-30, closed**: an AVX512-VNNI two-blocks-in-flight dot (bit-exact, 1.11x in the microbenchmark) was 1.9 ms *slower* in the engine in a controlled A/B (stock 69.0/69.4 ms vs patched 70.9/71.2 ms; two clean rounds) and other agents saw no gain either; original estimate was:
   needs a same-input trace if float summation order changes. Low priority after the failed `ilv2`.
6. ~~A bigger 3090 expert cache / 5060 Ti hot set~~ **measured 2026-10-30, closed**: 5060 Ti hot 14.2 -> 15.0 GiB and 3090 cache 12.4 -> 13.4 GB gave 69.7 -> 68.9 ms (-0.8 ms, inside noise); the dense arena still fits with 1 GB more cache.

Not a lever on this box: more CPU threads past 10 (12 gives nothing: DRAM bound), GPU assist, dense trunk on the 5060 Ti (measured
worse on 2026-10-13).

---------------------------------------------------------------------------------------------------------------------

## 2. Current setup and reproduction

### 2.1 Code and branches

| what | where |
|---|---|
| master with the GLM landing + docs | `a403eaf` (later commits on master belong to other agents) |
| this handoff, the launcher's CUDA-log/backtrace additions, the Q2_K bench variants | branch `q2k-kernel` in `/home/ray/Projects/ZeroMoE-wt-land` |
| earlier investigation history (diagnostics, reverts) | branch `glm-lane-diag` (do not merge; superseded by master) |
| ggml | llama.cpp `8baabd53b` + `patches/ggml` 0001-0017 + `patches/ggml-host-node` 0001-0003; built at `/home/ray/Projects/ZeroMoE-results/ggml-land/ggml-build` (the zeromoe binary's RUNPATH; **do not modify that directory**, other agents link it) |
| binary | `/home/ray/Projects/ZeroMoE-wt-land/build/zeromoe` (`tools/build_provenance.py` stamps the source; commit before building, a dirty tree fails the stamp) |

### 2.2 Files

- `configs/bench/glm53flash_gsqrco_offload_pool.toml`: the run config (offload-pool mode, `graph_offload_ddr_gib = 88`,
  `graph_offload_first_layer = 3`, one `[[gpus]]` entry for the 3090, no GPU assist).
- `bench/glm_gsqrco_pool.sh TAG [POOL_GIB] [prompt]`: the launcher. Needs `GLM_MODEL=/path/to/...gguf`. `CVD=0` for the 3090 alone
  (default `0,1`), `GPU1_HOT_GIB` (14.2), `CPUT` (decode threads, default engine 10), `PREFILL_THREADS` (6). It waits for a quiet box
  (idle GPUs, no other model run, load < 3, > 100 GiB free), runs under `scripts/run_capped.sh --limit-gb 118`, stops the engine below
  6 GiB MemAvailable, prints steady ms/token, the lane line and **which other processes used >= 20% CPU during the run**,
  writes `TAG.log/.out/.mem.csv`, a CUDA driver error log `TAG.cudalog` and preloads `bench/abort_backtrace.c` (prints a stack on
  SIGABRT, no ptrace needed; resolve frames with `addr2line -Cfie build/zeromoe <addr>`).
- `bench/q2k_dot_bench.cpp`: Q2_K dot variants, bit-exactness check and 1/6/10-thread timing.
- `docs/models/glm53-flash.md` (GSQ-RCO section), `configs/bench/perf-baselines.md` (rows), the older notes
  `docs/ssd-offload/11-glm53flash-handoff-2026-10-11.md`.
- Raw logs of this investigation: `/home/ray/Projects/ZeroMoE/build_glm_results/2026-10-28-150f68a-gsqrco-baseline/` (the `dual*`,
  `z88`, `val88_*` series; `*.engine.log`, `*.mem.csv`).

### 2.3 Reproduce the 14 tok/s baseline

```bash
cd /home/ray/Projects/ZeroMoE-wt-land     # branch q2k-kernel (or master + the files above)
GLM_MODEL=/data/sda/models/glm53flash/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf bench/glm_gsqrco_pool.sh /path/to/outdir/run1 88
# 3090 only:  CVD=0 ... ; expected ~10.7 tok/s
```
~8 minutes: ~4 min seeding the pool from the SATA SSD, ~65 s first prefill (23 tokens), then ~13 s of decode. Expect misses=0,
lane wait ~21 ms, compute ~40 ms, MemAvailable minimum 26-30 GiB (3090 + 5060 Ti) or ~16 GiB (3090 only).

### 2.4 Open reliability problem (reproduce with the same launcher)

`cudaMemcpyAsync ... invalid argument` in `ggml_backend_cuda_buffer_set_tensor` (`ggml-cuda.cu:849`), aborting right after the decode
graph is built and **before** `reusable graph prepared` (so during the state seed / dense upload, a host-to-device copy). Seen in
3 of ~15 launcher runs at pool 88 (2 stock-lib runs and 1 with the patched CPU library; the first on 20:39, two back to back at 21:35-21:45), and in earlier pool-80 runs together with non-finite
hidden states (the lane's `isfinite(weights)` assert is only the messenger: the layer-3 input was already NaN in all 4096 elements,
layer 0 attention output NaN in one run), `invalid argument` on dense uploads, and a data mismatch at byte 0 after upload.
Ruled out: the lane itself, pool pinning (`graph_offload_pin_dma=false`), deferred `cudaHostUnregister` (flush found 0 pending), the
dense upload sources being registered, any driver-reported error other than benign `HOST_MEMORY_ALREADY_REGISTERED` (57 per run),
core-dump memory spikes. A `cudaPointerGetAttributes` check once reported ordinary `posix_memalign` buffers as `cudaMemoryTypeHost`
(registered) without our registering them: a stale registration of a reused virtual-address range is the leading suspect (not
proven). Next step: run the launcher repeatedly (`hunt.sh`-style loop) until it crashes with the new backtrace + `TAG.cudalog`
(the driver log names the failing call), then check the failing tensor's pointers.

---------------------------------------------------------------------------------------------------------------------

## 2.4b Pinned-host-memory race in the RAM-tier slab pin (found and fixed 2026-10-30)

A 2,313-token prompt on GSQ-RCO (3090 + 5060 Ti, pool 88) aborted with `CUDA error: invalid argument` in a device-to-host copy; the CUDA driver log said
**"Copy is larger than memobj size, for destination operand"** (`cuMemcpyDtoHAsync`, `CUDA_ERROR_INVALID_VALUE`), next to `cudaHostRegister declined ... already mapped`. An
`LD_PRELOAD` tracer (`bench/hostreg_trace.c`: records every `cudaHostRegister`/`cudaHostUnregister` range with a call stack, reports `free()` of memory inside a live
registration and registrations refused as already registered) logged **203 frees of still-registered memory and 198 refused registrations**, all on 14,155,776-byte (13.5 MiB expert
slab) ranges registered by `StreamingEngine::ensure_slab_capacity` on the prefetcher's worker threads.

Cause: `ensure_slab_capacity` installed the new buffer under `deferred_slab_free_mutex`, released the mutex, then registered `slot.slab_buffer` (not the buffer it had installed)
and set `slot.slab_pinned = true` without checking that the slot still owned it. A concurrent regrow or release unregisters only `if (slot.slab_pinned)`, so it could free the buffer
while the registration was alive. The registration stayed on freed memory that `malloc` handed out again (or covered a range shorter than the buffer in use), and a later copy
into such a block failed. This is probably also the cause of the intermittent `cudaMemcpyAsync ... invalid argument` aborts seen earlier (3 of ~15 launcher runs); the pool-pinning,
deferred-unregister and registered-source theories recorded above did not explain them.

Fix (`core/engine.hpp`): register the buffer this call installed; commit the flag under the mutex only if the slot still owns that buffer (otherwise unregister it); mark the
slot with `slab_pin_target` while a registration is in flight so a regrow/release defers its free (the drain waits for the marker). Verified on the same prompt: **0 tracer
events (was 401), no "memobj" error**; teacher-forced logits **bitwise identical** to the pre-fix binary on all 8 steps; 6-prompt decode **70.8 ms = 14.13 tok/s** (pre-fix 69.8-70.9 ms).
Not re-run on the Qwen family, which shares this code path (the change adds one short lock per slab registration).

**Still open (separate problem)**: the same 2,313-token prompt then fails with a GPU `CUDA_ERROR_OUT_OF_MEMORY` in `ggml_cuda_mul_mat_q` (`mmq.cu:257`) during the second graph-prefill
chunk. The 3090 is ~93% full (12.4 GB expert cache + 8.96 GiB dense arena + the graph workspace; peak 22.7 of 24.6 GiB) and the GLM prefill planner (`ggml_glm5next_graph.cpp`,
margin 0.15 GiB, constants measured on the AJ file) leaves too little for the matmul scratch. With the 3090 cache reduced by 1.5 GB (10.9 GB) the prompt completed, but at **7.6 tok/s
prefill** (304 s for 2,313 tokens; cold experts run on the CPU because GLM has no `prefill_op_offload` default, unlike Qwen), and with `graph_prefill_op_offload = true` it ran out of memory
again (3090 peak 24.1 of 24.6 GiB). Making full-graph prefill the GLM default (like the other families) is therefore held back until the planner has a real margin and long prompts pass.

## 2.5 Measured decode timeline (2026-10-30, 3090 + 5060 Ti, from the node-level Nsight trace of another session)

Source: `/home/ray/Projects/ZeroMoE/build_glm_results/2026-10-30-b6d190-glm-node-trace/retry/` (98 decode tokens, 4158 `lane.run` ranges, 465,902
kernels). Scripts: `bench/glm_nsys_lane_breakdown.py`, `bench/glm_nsys_gpu1_branch.py`. The profiler stretches the token to 87.9 ms (70.8 ms un-instrumented),
so scale the figures below by about 0.8. Medians over tokens 20-90.

| phase of one token | ms | share | what is running |
|---|---|---|---|
| CPU lane waits for the 3090 (attention + router of the layer) | 17.2 | 20% | 3090 busy 86%, 5060 Ti idle |
| **CPU lane computes the cold experts and uploads the result** | **47.2** | **54%** | **3090 busy 6.7% (3.2 ms), 5060 Ti 0%: both GPUs idle** |
| between lane ranges: GPU combine, 5060 Ti hot experts, next layer start | 23.7 | 27% | 3090 busy 17.6%, 5060 Ti 12%; 18.3 ms in the 42 layers + ~5 ms token tail (LM head, sampling, sync) |

Device utilisation per token: 3090 kernels 22.2 ms (25% of wall), 5060 Ti kernels 2.9 ms (3%), CPU lane busy 54%. The three devices work one after
another, not at the same time. A layer cannot start before the previous layer's experts (cold ones included) are done, so one token's layers cannot be
pipelined; parallelism exists only inside a layer.

One layer between two lane ranges (median 489 us, layer 20 of token 50 as an example):
`0 us` upload cold outputs (131 KB H2D, 13 us) -> small 3090 kernels (40-60 us) -> hidden state 3090 -> host -> 5060 Ti (16 KB, two copies, ~35 us + wait) ->
`171-232 us` **5060 Ti computes its hot experts (3 matvecs of ~16 us + small kernels, ~65 us)** -> result 5060 Ti -> host -> 3090 (131 KB, two copies) ->
`379-435 us` 3090 combine (`moe_weighted_reduction`), HC post, rms_norm, next layer's first projection. Of the 489 us: 90 us before the 5060 Ti hop starts,
**211 us the 5060 Ti branch including both host-staged hops**, 186 us on the 3090 afterwards. GPU peer-to-peer is **not supported** between the cards
(`nvidia-smi topo -p2p r`: NS), so the hops via host memory stay.

**What can run in parallel (3090, 5060 Ti, CPU) and what it is worth**

| overlap | status | estimate |
|---|---|---|
| 5060 Ti hot experts **concurrently with** the CPU cold experts of the same layer (both only need the router ids/weights + hidden state, available before the CPU lane starts; today the 5060 Ti branch runs after the lane) | not done; an attempt (`e45e92d`, reverted `e5967f9`) failed parity and gave non-finite values: it needs the ids/hidden state/weights copied into independent buffers because the scheduler recycles the originals | 8.8 ms of 87.9 per token on the critical path (7.1 ms un-instrumented), about 10% -> ~64 ms = ~15.6 tok/s |
| 5060 Ti reduces its own experts first and returns one 16 KB vector instead of 8 x 16 KB | not done | ~10-20 us per layer, 0.4-0.8 ms |
| 3090 post-lane chain (~110 small kernels/layer, 1-5 us each, launch-latency bound; 85 scheduler splits stop CUDA-graph capture around the host node) | not done | ~1-2 ms |
| token tail (LM head, logits readback, sampling, next-token launch) ~5 ms | not done | ~1-2 ms |
| the 3090 attention of layer L+1 with the CPU cold experts of layer L | **impossible without changing numerics**: it needs layer L's output | 0 |
| the CPU phase itself (47 ms, 54%) | memory bound; nothing left to overlap with | only fewer cold experts per token (ranking) shrinks it |

All the exact overlaps together are worth about 9-12 ms per token (about 16-17 tok/s). 20 tok/s (50 ms) also needs the CPU phase to fall from ~38 ms
(un-instrumented) to about 25 ms, i.e. roughly a third fewer cold experts, which overlap cannot give.

---------------------------------------------------------------------------------------------------------------------

## 3. Overlapping GPU and CPU work

### 3.1 The idea

Within one token every layer is a strict chain: GPU attention/HC/router -> (ids, weights) -> cold experts on CPU **in parallel**
with hot experts + shared expert on GPU -> combine -> next layer. Idle time per token: the CPU is idle during the GPU's attention/router
part (the 21 ms "wait"); the GPU is idle (or only running hot/shared experts) during the CPU's ~40 ms. Overlap is possible where
two pieces of work do not depend on each other. Candidates (to be validated by the profile):

| # | overlap | exact? | notes |
|---|---|---|---|
| A | hot + shared experts on the 3090 and hot experts on the 5060 Ti (the engine calls it GPU1) **concurrent with** the CPU cold experts | yes | already designed in (`MoEIngraphPlan.caps.prefer_overlap`); must be verified that the scheduler really issues them before the blocking host node and that GPU0 and GPU1 branches run concurrently |
| B | remove idle gaps inside the GPU part of a layer (many small kernels per layer; ~8189 graph nodes) | yes | launch overhead / sync points between splits (`splits=169`, `transitions=85`) |
| C | start the layer-(L+1) CPU work before layer L is fully combined | **no** (needs the layer output) | only possible with speculative routing/inputs; changes numerics, forbidden without a same-input A/B |
| D | precompute W*A for the next attention while the CPU works (split by linearity) | no (reassociation) and reads the dense weights twice | rejected in principle |

So A and B are the exact ones; C/D are not. The honest expectation: A+B can shave part of the 21 ms wait and the ~10 ms tail; they
cannot touch the ~40 ms CPU part (that needs fewer bytes, section 1.5).

### 3.1b Parity harness for the overlap change: determinism of the unchanged build (measured 2026-10-30)

Two runs of the unchanged build (master `b7d6bd1`, 3090 + 5060 Ti, pool 88, prompt "The capital of France is", the same 8 teacher-forced tokens, logits dumped per step with
`ZEROMOE_DECODE_LOGITS_DUMP`, compared with `bench/cmp_decode_logits.py`):

| prefill | run 1 vs run 2 |
|---|---|
| layered prefill (the **fallback** path, used whenever the full-graph prefill is not enabled) | **not reproducible**: cosine 0.992 at step 0, 0.92-0.95 at steps 1-7, max abs 1.8-7.8, top-1 flips at step 1 (13 vs 624) |
| full-graph prefill (now the default; before 2026-10-30 it needed `ZEROMOE_GLM5NEXT_FULL_GRAPH_PREFILL=1`; the path that runs when the memory plan fits, project rule: no silent layered/legacy execution) | **bitwise identical on all 8 steps** |

The layered prefill is a fallback path for all models, so its run-to-run noise is not a defect of the intended path. For GLM the full-graph prefill used to be opt-in; on owner instruction it is now the default for GLM like for the other families (deterministic, 20-100x faster than the layered path). A parity test of the overlap change must use the full-graph prefill and then requires bitwise-equal logits (any difference is a real change). With the layered prefill
the unchanged build differs from itself by as much as the earlier overlap attempt `e45e92d` differed from its parent (cosine 0.86-0.98), so that attempt's recorded
"parity failure" cannot be attributed to the reorder (probable cause of the layered noise: experts placed on the GPU or CPU depending on upload timing; the two kernels quantize
activations differently). The earlier revert was based on evidence that could not distinguish a bug from noise; the NaN diagnostic seen once in that attempt was a separate symptom.

### 3.2 Data we have

- Lane totals per token: wait ~21 ms, compute ~40 ms (`[GLM5NEXT-LANE] tokens=96 ...`, also `breakdown` with `unsplit_graph`).
- Graph shape: 8189 nodes, 169 scheduler splits, 85 transitions, workspace 0.15 GiB (`[GGML-GLM5NEXT-FULL] graph built`).
- GPU memory per phase (see 1.1) and the CPU kernel/DRAM bandwidth numbers (`bench/q2k_dot_bench.cpp`).
- GPU assist (a different overlap, CPU+GPU split of the cold experts) was measured: worse. It tells us DRAM, not compute, is
  the shared resource.

### 3.3 Data we need (not collected yet)

`nsys` is installed (`/usr/bin/nsys`). Capture ~8 s of steady decode on the 3090 + 5060 Ti setup:

```bash
# run via the launcher with nsys wrapped around the engine; start collecting after the first request's decode begins
nsys profile --trace=cuda,nvtx,osrt --sample=none --delay 400 --duration 10 -o /path/to/glm_decode \
  <the engine command line printed in TAG.out / the launcher's bash -c>
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace,cuda_api_sum /path/to/glm_decode.nsys-rep
```
(decode starts ~395-400 s after launch: ~247 s seeding + ~65 s prefill + ~6 s graph build.) What to extract from the trace:
1. GPU busy fraction per device within a token (sum of kernel time / 70 ms), and the gap histogram on the 3090 and the 5060 Ti.
2. For one MoE layer: the order and duration of {attention kernels, router, D2H copies, host-node (CPU) interval, hot-expert kernels
   on GPU0 (3090), hot-expert kernels on GPU1 (5060 Ti), shared expert, combine}. In particular: do GPU0 (3090) / GPU1 (5060 Ti) hot kernels overlap the CPU interval, or
   do they run before/after it?
3. The CPU interval per layer (host node) vs the GPU kernels' time in the same layer.
4. Time between the last kernel of layer L and the first kernel of layer L+1 (launch / sync gap).
Also useful: the per-layer timing the engine prints at DEBUG log level (`layer_timing_env()`); note any eval
callback breaks the early-ready scheduler bridge for the layers after the first lane, so rely on nsys, not callbacks).

### 3.4 Code map and change plan

Where the overlap is built:
- `models/glm/ggml_glm5next_graph_build.cpp` ~330-470: per-layer FFN build; `partial->hot/cold/hot1`, `lane_wanted`,
  `plan.caps.early_ready/prefer_overlap`, `hooks.run_cold`, `hooks.build_shared`.
- `models/graph/moe_ingraph.hpp`, `moe_block.hpp`, `cold_moe_strategy_early_ready.hpp`, `early_ready_graph_builder.hpp`,
  `early_ready_offload_strategy.hpp`: how the plan resolves to a strategy and orders the nodes.
- `models/graph/early_ready_lane.hpp`: the lane (D2H of input/ids/weights, `ready` event, CPU graph, output upload, GPU assist).
- `patches/ggml-host-node/0001-0003`: CUDA host nodes and the scheduler's early-ready private CPU path (the mechanism that lets a CPU
  node sit inside a GPU graph).
- `models/glm/ggml_glm5next_graph_settings.cpp`: `early_ready_wanted`, `lane_with_gpu1_wanted`, `gpu_assist_slots`, `second_gpu_device`.

Plan:
1. **Profile** (3.3). Decide A vs B from where the idle time is. No code change before this.
2. If A fails (hot/shared kernels are placed after the host node or GPU0 (3090) and GPU1 (5060 Ti) branches serialize): change the node order in the
   plan so the GPU branches are expanded before the host node (`ggml_build_forward_expand` order matters for the scheduler), and make
   GPU1's (5060 Ti) hot branch use its own stream/backend so it overlaps GPU0 (3090); verify with nsys that they now overlap the host interval.
3. If B (gaps inside the GPU part): look at splits/transitions around each layer, fuse small elementwise ops (hyper-connection fusion
   already exists, patch 0015), and check whether the CUDA graph capture (patch 0003) covers the layer body or is disabled by the host node.
4. Every change goes through the project's rules: same-input A/B of the layer-boundary hidden state and routed IDs vs the unchanged
   build (`ZEROMOE_TEACHER_FORCE_TOKEN_FILE`, `scripts/validate_vs_llama.sh`), then 3+ launcher runs on a quiet box; keep only if the
   steady ms/token improves beyond the ~3 ms run-to-run noise, with the contamination report clean.
5. Record the result in `configs/bench/perf-baselines.md` and `docs/models/glm53-flash.md`; remove any temporary toggle after the A/B
   (standing project rule).

### 3.5 Expectation, stated plainly

Even a perfect overlap cannot cross the ~61 ms floor set by DRAM (30 ms of CPU reads + the unavoidable GPU chain). 20 tok/s (50 ms)
needs fewer bytes per token: adaptive promotion (1.5 item 2) or speculative decoding (item 3), plus whatever overlap recovers.
