# GLM-5.3-Flash GSQ-RCO: performance handoff (2026-10-29)

Goal that was set: GLM decode from ~10 tok/s toward ~20 tok/s (first recover the ~14 tok/s the project had before, on
the 3090 first, then with the 5060 Ti). Result so far: **14.1 tok/s on the 3090 + 5060 Ti, 10.7 tok/s on the 3090 alone.** 20 tok/s has
no measured path yet; sections 1 and 3 list what is left.

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
| prefill threads independent of the decode setting (6) | 10 prefill threads grew host memory ~16 GiB after seeding and tripped the watchdog | `ggml_glm5next_graph.cpp`, `ZEROMOE_GLM5NEXT_PREFILL_CPU_THREADS` |
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
5. **A restructured Q2_K kernel** (rows in flight, fewer scalar ops per block): ceiling ~1.6x on the CPU part, realistic far less;
   needs a same-input trace if float summation order changes. Low priority after the failed `ilv2`.
6. A bigger 3090 expert cache (12.4 -> ~14 GB): +1.5% hot experts, ~1 ms/token, and the arena must still fit. Cheap but tiny.

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
