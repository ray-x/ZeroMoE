# GLM-5.3-Flash decode performance: results record (2026-10-14)

Box: Xeon W-2235 (6C/12T, Cascade Lake, AVX-512 VNNI), 125 GB DDR4, RTX 3090 (PCIe Gen3 x16, ~12.2 GB/s host->device) + RTX 5060 Ti (PCIe Gen3 x8, ~6.9 GB/s); the W-2235 platform is PCIe 3.0 only, so the 3090 never runs Gen4.
Model: GLM-5.3-Flash AJ-IQ2_XXS (87 GB), experts offloaded (SSD -> DDR pool -> VRAM hot sets), full-graph decode.
Branch `graph-framework` (master fast-forwarded to `1842a5c`).

## Method (how the numbers were taken)
- tok/s = 1000 / mean wall-clock ms per token over the **last 30 decoded tokens**, from consecutive `FULL-DECODE-TIME` log lines.
  Averages over all 40-48 tokens include slow warm-up tokens and read ~0.3-0.5 tok/s lower.
- Compared configurations always decode **identical tokens** (teacher forcing: `MOESTREAM_TEACHER_FORCE_TOKEN_FILE`;
  `bench/data_glm_tf48.tokens`, or tokens from a free run of the same prompt via `bench/ranking_ab_prompt.sh`).
  Free-running text alone moves the same configuration between 13.8 and 16.0 tok/s.
- Held-out = prompt not in the ranking file used (train-only file `usage_train30`); benchmark sequence = "Explain how a
  mixture-of-experts transformer routes tokens ...", 48 tokens.

## Results
Default binary (`build_gk`: in-tree ggml with the AVX-512VL IQ2_XXS kernel + patch 0006), no library overrides, recipe env
(`bench/glm_recipe_env.sh`: hot set 15.0 GiB on the 5060 Ti, 10 CPU threads, auto pool ~76 GiB, second assist ring on the 5060 Ti):

| input | before | after |
|---|---|---|
| benchmark sequence | 14.85 tok/s | **15.31** (15.43 with the kernel library) |
| held-out 094 dockerfile | 12.32 | **15.03** |
| held-out 046 css | 11.69 | 13.19 |

Peak (best windows found in the logs, recipe): best 30-token stretch 17.1 tok/s, best 10-token stretch 19.0, fastest single token
~24 tok/s (42 ms). Typical sustained: 13-15 tok/s depending on how well the reply matches the resident set (per-token time
42-99 ms with the number of routed experts outside VRAM).

Qwen3.8-Flash-Next (Q2_0, 3090 only, 96 tokens, tokens identical): baseline 29.81 / 30.66 vs new 30.94 / 30.81 tok/s (+2%, within
noise); GPU-trunk bound, unaffected by the GLM changes. Identity gate on the final binary: Qwen logits max KL 0.0, free-running
tokens identical; GLM `lru` and `lane` logits and tokens identical vs the frozen baseline.

## Single 3090 (no 5060 Ti), from existing logs
Not re-measured with the final recipe: these are older-config runs (lane + GPU assist on, 58 GiB pool, 8 CPU threads, previous ranking,
mixed ggml builds), taken from `logs/perf/sweepA_A_1`, `sweepB_*`, `assist_tf`, `assist3_*`; steady = last 30 tokens.

| window | single 3090 | dual GPU (recipe) |
|---|---|---|
| sustained (last 30 tokens) | 12.2-13.3 tok/s | 15.0-15.4 (benchmark-like text) |
| best 30-token stretch | 13.3 | 17.1 |
| best 10-token stretch | 14.8 | 19.0 |
| fastest single token | 52 ms = ~19 tok/s | ~42 ms = ~24 tok/s |
| 3090-only without assist (older runs) | 9.9-10.6 | - |

So a single 3090 exceeds 15 tok/s only over short bursts (single tokens, ~10 tokens at best 14.8) and averages ~12-13 over a reply.
Reason: the 3090 alone keeps ~14.7 GiB of hot experts (vs 28.9 GiB with the 5060 Ti), so more routed experts go through the CPU lane
and assist. The final recipe (refit ranking, kernel in the default binary, 10 threads) on a single 3090 has not been measured; an
estimate of +0.5 to +1 tok/s over these numbers is a guess.

## What produced the gain (this session, GLM, dual GPU)
| change | effect |
|---|---|
| split load, IQ2_XXS kernel, bigger DDR pool, graph prefill | earlier work: 4.3 -> ~10 tok/s on the 3090 alone |
| 5060 Ti hot set 14.2 GiB, lane on its layers, GPU assist (3090 ring) | 3090-only 10.1 -> 14.7 tok/s |
| assist only for IQ/Q2_0 expert layers | GSQ-RCO file 88.2 -> 70 ms/token (K-quants stay on the CPU) |
| hot set 15.0 GiB, second assist ring (5060 Ti), threads 10 | +0.3 tok/s (66.9-67.3 ms vs 68.3) |
| ranking refit on 100 prompts (decode-only routing) | held-out +1.4 / +2.5 tok/s; neutral on the benchmark prompt |
| engine linked a ggml without the AVX-512 kernel | +0.5 tok/s (14.91 -> 15.43) |
| GSQ-RCO 3.0-bit file (Q2_K/Q3_K/Q4_K, Q8_0 dense) | parity with AJ: 68.9 ms with assist 1 (default now: assist off for K-quants, 70.1 ms) |

## Tried and not worth it (details in doc 11/12)
Dense on the 5060 Ti (-8 ms), Q4_0/Q2_K cold tier (no faster at GLM shape / 30% noise), THP, dynamic resident set (-1% cold routes at
8 promotions/token), assist slots > 3, cost-model assist split, IQ2_XXS micro-restructurings (+8% microbench, noise end to end),
prompt-lookup speculation (1.03 words/pass), MTP (no head in either checkpoint).

## Open
- The user's shared build `~/Projects/moestream-build-upstream-master` still links the Oct 4 ggml (no kernel): ~14.9 tok/s until
  rebuilt from `llama.cpp-moestream-integration` (`7bd457fa1`).
- Prompts far from the surveyed corpus stay at 13-14 tok/s; a usage ranking from the real workload is the remaining lever
  (`bench/make_survey_batches.py`, `bench/glm_route_survey.sh`, `bench/ranking_holdout_eval.py --write`).

## 5060 Ti efficiency round (2026-10-14; code kept on branch `assist-prestage-experiment`, nothing merged)
Where the 5060 Ti's time goes: ~3 ms/token of kernels out of ~65 (4-5% busy), its link carries ~1 assist expert per layer at 6.9 GB/s.
Its +20% over a single 3090 is residency (28.9 vs 14.7 GiB of hot experts), not compute. Same fixed sequence, last 30 tokens, recipe env:

| experiment | result |
|---|---|
| layer split, 5060 Ti holds only layers >= 24 / 31 / 36 (3090 the first layers) | 65.3 / 64.5 / 64.4 ms vs 65.4 (15.31 / 15.51 / 15.52 vs 15.30 tok/s); trunk wait -1.2 / -2.5 / -6.2 ms, lane compute +2.5 / +3.9 / +4.5 ms |
| non-contiguous layer sets from an offline optimiser (15 and 19 layers incl. 3-12) | 65.1 / 64.7 ms vs 64.6 / 64.9: neutral; the model predicted 1.4-4.7 ms better, the benchmark prompt's lane cost rose by as much as the hops saved |
| assist prestaging: pilot predicts next layer's cold experts, copied to the ring during the trunk | lane compute 38.1 -> 30.3 ms, 68% of assisted experts already in the ring, but wall 64.7 -> 95 ms (an unexplained ~0.6 ms/layer per staged expert outside the lane's wait/compute); default off |
| first prestage build | 381 ms: the extra predictor tensors overflowed the exactly sized dense arena and the LM head silently fell back to the CPU (fixed in the experiment branch) |

Kernels: with the 5060 Ti busy ~3 ms/token, kernel tuning there cannot move the total; the native-arch (sm_120) build was not tried.
Pin layout: assist copies run at the PCIe limits (3090 ~12 GB/s per 2 MB plane copy, 5060 Ti 6.9 GB/s), planes are separate regions,
three copies per expert; no layout change was found that helps. Communication: cross-GPU hop removal is real (-6 ms of trunk wait) but is
paid back by lane compute because the 5060 Ti's set then matches the hot experts worse.

### Prestage: root cause of the stall, and why it still does not pay (same day, branch `assist-prestage-experiment`)
- **Stall cause (found):** layers that have a second-GPU branch do cross-GPU peer copies (hidden to the 5060 Ti and back) inside their
  trunk. The staged expert DMA (1-1.6 ms per layer) occupies the same copy engines/PCIe path, so those small copies queue behind it:
  between-lane host time went 10.2 -> 38.0 ms/token (+0.67 ms per layer). Not the LM head after the arena fix, not `cudaFree` (those
  are the legacy prefill), not `CUDA_DEVICE_MAX_CONNECTIONS`, and not the order of the lane's output upload.
- **Mitigation:** prestage only layers without a second-GPU branch (layer-split layout): gap back to 8.1 ms, 70.1 ms vs 64.5 ms base.
- **Why no gain:** with 61% of assisted experts pre-staged, lane compute did not fall (42.7 vs 41.4 ms): the in-window copies are not
  the straggler, the CPU graph is. Letting staged hits assist on top of the normal share moves time back to the assist join
  (compute 46-49 ms, 74 ms/token). The lane is on a balanced plateau; prestage cannot shift it.
- Net: no measured improvement in 5060 Ti efficiency from layout, prestage or kernels; the experiments and tools stay on the branch.

## Layer split adopted: the 5060 Ti holds layers >= 31 (2026-10-14)
Order-balanced repeats on identical tokens, last 30 tokens, tok/s, binary with the AVX-512 kernel, recipe env (hot set 15.0 GiB, 10
threads), ranking built without the prompt:

| input | current layout | 5060 Ti holds layers >= 31 | 5060 Ti holds layers >= 36 |
|---|---|---|---|
| benchmark sequence (n=4 / 2 / 4) | 15.37 (15.25-15.42) | 15.40 (15.38-15.43) | 15.40 (15.37-15.48) |
| held-out css 046 (n=3 each) | 14.71 (14.70-14.71) | **15.20** (15.19-15.22), +3.4% | 14.84 |
| held-out dockerfile 094 (n=3 each) | 15.31 (15.29-15.33) | **15.76** (15.67-15.83), +3.0% | 15.26 |

Mechanism: a layer with hot experts on both GPUs pays cross-GPU copies of the hidden state inside its trunk; with the cut, layers 3-30
are 3090-only (no hop; trunk wait 21.5 -> 19.0 ms/token) and layers 31-44 are 5060 Ti-only, at the price of a slightly worse hot-set match
(lane compute 37.7 -> 41.7 ms). The trade is neutral for in-domain text and positive for out-of-domain text, whose routes are less
concentrated. Shipped: `MOESTREAM_GLM5NEXT_GPU1_MIN_LAYER=31` (default in `bench/glm_recipe_env.sh`), the 3090 ranking with layers >= 31
zeroed (`glm53flash_usage_base_plus_decode100_3090_lt31.txt`) and `pin_global_floor_per_layer = 0` in the shipped config.

### Pin layout / PCIe links (2026-10-14, `nvidia-smi` + a standalone copy microbenchmark)
Both links negotiate PCIe **Gen3** (`pcie.link.gen.max = 3`; 3090 x16, 5060 Ti x8); the Xeon W-2235 has no Gen4. Host->device copies of
2.16 MB expert planes from 2 GiB of buffers: 12.2 GB/s on the 3090 and 6.9 GB/s on the 5060 Ti for every host layout tried
(`cudaMallocHost`, anonymous 4 KiB pages + `cudaHostRegister`, 2 MiB THP + register, randomly scattered 4 KiB pages + register),
i.e. the copy rate is the link limit and the arena's page layout / pinning method cannot improve it. Earlier notes that call the 3090
"PCIe4" are wrong; the assist copies were already running at the Gen3 x16 limit.

### Kernel efficiency: native SASS instead of JIT'd PTX (2026-10-14)
The engine's ggml-cuda was built with `CMAKE_CUDA_ARCHITECTURES=75` (PTX only), so both GPUs ran code JIT-compiled from compute 7.5.
Building `86;120` (sm_86 for the 3090, sm_120 for the 5060 Ti; `bench/build_native.sh`) with the layer-split recipe, order-balanced,
last-30 tok/s, identical tokens:

| input | PTX/JIT | native SASS | trunk wait |
|---|---|---|---|
| benchmark sequence | 15.50, 15.40 (mean 15.45) | 15.68, 15.66 (mean **15.67**, +1.4%) | 19.6 -> 18.9 ms/token |
| held-out dockerfile | 15.85, 15.72 (mean 15.79) | 15.88, 15.89 (mean **15.89**, +0.6%) | 18.8 -> 18.7 ms/token |

The 5060 Ti's own kernels are ~3 ms/token, so most of the gain is the 3090 trunk; the native build costs one longer compile and no code change.

## Prefill (805-token generic prompt, 2026-10-14)

| mode | prefill |
|---|---|
| legacy whole-prompt (default; graph prefill is opt-in) | ~1 tok/s (killed after 9.6 min) |
| `MOESTREAM_GLM5NEXT_FULL_GRAPH_PREFILL=1`, chunk 128, `build_native` (has patch 0006), 3090 | **5.7 tok/s** (140 s) |
| same, chunk 256, dual GPU | 4.6 tok/s (175 s; 1.39 GiB workspace, builds; chunk 512 is declined, OOM) |
| same with `build_gk` / upstream-master ggml-cuda (no patch 0006) | CUDA illegal memory access in `mul_mat_q<IQ2_XXS>` |

- Only the first `PREFILL_CHUNK` tokens go through the batched graph (~5 s for 128); the other ~680 run as S=1
  decode-graph steps at 160-260 ms each (prompt tokens miss the decode-ranked hot set), which is ~95% of the time.
- Qwen runs the whole prompt in one graph; GLM cannot (workspace ~6.5 MB/token, ~1.7 GiB free) and its graph is
  position-zero only (`pos_base != 0` is rejected), the same as Qwen's prepare.
- The fix is a chunked prefill graph at `pos_base > 0` (KDA state carry-in, DSA indexer + MLA over cached KV, as
  llama.cpp does with ubatch 512). Not built yet. Quick win available: enable graph prefill by default once the
  patched ggml-cuda is the shipped one (1 -> ~5.7 tok/s).

### Prefill workspace breakdown and fix

`MOESTREAM_GLM5NEXT_WS_DUMP=1` prints the intermediate live-set peak at graph build. At S=128 the peak was 754 MiB, of
which ~420 MiB were the 34 KDA layers' conv-input concats (`[S+3, 24576]` f32, 12.3 MiB each): the exported conv-history
and state copies were expanded into the graph only at its end, which kept each layer's conv input and scan output live for
the whole graph. Expanding them right after the layer (`ggml_build_forward_expand`, `MOESTREAM_GLM5NEXT_EARLY_STATE_EXPORT=0`
turns it off) cut the live peak 754 -> 64 MiB and the workspace 0.83 -> 0.22 GiB at S=128 (6.5 -> ~0.8 MB/token).
The whole 805-token prompt now fits one graph (0.62 GiB): prefill **5.7 -> 21.1 tok/s** (38 s), same output start.
Default `MOESTREAM_GLM5NEXT_PREFILL_CHUNK` is now 1024. Graph prefill is still opt-in
(`MOESTREAM_GLM5NEXT_FULL_GRAPH_PREFILL=1`) and needs the ggml-cuda with patch 0006 (`bench/build_native.sh`).

### Where the 38 s of whole-prompt prefill goes (S=805, `MOESTREAM_GLM5NEXT_OP_PROFILE=1`)

| op | time |
|---|---|
| `MUL_MAT_ID(iq2_xxs)` on CPU (cold experts, 118 nodes) | 26.4 s |
| `MAP_CUSTOM1` host nodes (offload miss loader, 42 nodes) | 8.2 s |
| `MUL_MAT_ID(iq2_s)` on CPU | 1.7 s |
| everything on the GPU (MMQ hot experts, attention, KDA, dense) | < 1.5 s |

The GPU is idle for ~90% of prefill. Next step: run the cold experts on the GPU during prefill by uploading the used ones
per layer into VRAM (the expert cache region can be overwritten while prefilling and refilled before decode), instead of the
CPU `MUL_MAT_ID`. Tried: `MOESTREAM_GLM5NEXT_PREFILL_OP_OFFLOAD=1` (scheduler op-offload, needs ~1.7 GiB workspace and
a smaller cache; ggml-backend needed negative-id tolerance) - the scheduler's expert copy fails with `cudaMemcpyAsync:
invalid argument` from the offload-store host tensors, so the upload has to be done by the graph's own miss node.

### llama.cpp reference for GLM prefill (neurall fork `~/Projects/neurall/llama.cpp`, build 11401, RTX 3090, pp805, AJ IQ2_XXS)

| llama-bench config | pp805 |
|---|---|
| `-ncmoe 45 -nopo 1` (experts on CPU, no op-offload) | 24.7 t/s |
| `-ncmoe 45 -ub 512` (op-offload on) | 56 t/s |
| `-ncmoe 45 -ub 4096` (op-offload on) | **95.3 t/s** |
| `-ncmoe 41 -ub 4096` | **104.3 t/s** |
| `-ncmoe 36` | does not fit (context creation fails) |

MoEStream graph prefill (CPU cold experts) is 21 t/s: it matches llama.cpp *without* op-offload (24.7). The 4x gap is the
per-layer copy of used experts to the GPU with the big ubatch amortising it, exactly the staging path proposed above.

### Cold experts on the GPU during prefill: 20.2 -> 47.8 tok/s (805 tokens, greedy tokens identical)

`MOESTREAM_GLM5NEXT_PREFILL_OP_OFFLOAD=1` (+ `MOESTREAM_GLM5NEXT_FULL_GRAPH_PREFILL=1`) lets the scheduler copy only the
experts a layer's cold branch uses into VRAM and run `MUL_MAT_ID` there, like llama.cpp `-ncmoe N -ub 4096`.
`patches/ggml/0007` makes the scheduler's expert copy safe for the virtual arena (negative ids, per-slot copies,
zero padding instead of reading past the last expert). Per-op profile after: CUDA `MUL_MAT_ID` 7.1 s (PCIe-bound),
miss node `MAP_CUSTOM1` 8.7 s (store/DDR -> arena), rest < 1.5 s; total 16.8 s vs llama.cpp 7.7 s (104 t/s).
Needs ~1.7 GiB workspace, so it was run with `moe_cache_gb = 14.0` (the cache otherwise leaves 1.2 GiB free).
Open: release/shrink the cache for the prefill (as the Qwen path releases its arena before prefill), overlap the
miss-node load of layer L+1 with layer L compute, or read experts straight from the model mapping.

### Read-ahead of transient experts: 47.8 -> 98 tok/s prefill (805 tokens, greedy tokens identical)

The miss node used to load a layer's cold experts on the compute path (8.7 s of 16.8 s). Now a background thread reads the
next `MOESTREAM_GLM5NEXT_PREFILL_AHEAD` (default 2) layers' non-cached experts while the GPU computes the current one
(`ExpertResidency::prefetch_transient`). Per run (3090, 58 GiB DDR pool, `moe_cache_gb = 14.0`, prefill graph + op-offload):

| | prefill |
|---|---|
| CPU cold experts | 20.2 tok/s |
| GPU cold experts (miss node loads inline), 32 GiB pool | 47.8 |
| + read-ahead, 32 GiB pool | 74.3 |
| + read-ahead, 58 GiB pool | **98-99** (8.1-8.2 s) |
| + registering transient experts for DMA (`MOESTREAM_OFFLOAD_PIN_TRANSIENT=1`) | 87.5 (worse; off) |
| llama.cpp `-ncmoe 41 -ub 4096` | 104 |

AHEAD=4 and TRANSIENT_PAR=16 gave no change (72-73 at the 32 GiB pool). Needs ~1.7 GiB of free VRAM for the workspace and
expert copy buffers, hence the smaller cache; releasing/shrinking the cache for the prefill only is still open.

### Full cache + batched prefill: 100 tok/s without shrinking the cache

The prefill needs ~1.7 GiB of free VRAM beyond the dense arena and cache. The legacy GPU attention mirrors
(`attn_qabsorb_gpu`, `attn_score_gpu`; ~0.4 GiB) are unused by the full graph, so the shipped config turns them off and
`moe_cache_gb = 15.8` (14.71 GiB aliased) stays intact: 805-token prefill 100.3 tok/s (7.9-8.0 s), greedy tokens
identical to the CPU-cold path, no cache release/refill needed. `bench/glm_recipe_env.sh` now enables
`MOESTREAM_GLM5NEXT_FULL_GRAPH_PREFILL=1` and `..._PREFILL_OP_OFFLOAD=1` and the `build_native` ggml-cuda when it exists.

### Regression found and fixed: graph prefill turned on sparse KPool decode (-8% decode)

Same teacher-forced benchmark, dual-GPU recipe: legacy prefill 15.75-15.96 tok/s decode; graph prefill 14.1-14.5. The decode graph
had +693 nodes (11 DSA indexer layers x 63): after a graph prefill the prompt's indexer seeds exist, so decode switched to the
sparse KPool path, which costs ~6 ms/token and is a no-op until the context passes `top_k` (2048). It is now used only when
`prompt + MOESTREAM_GLM5NEXT_SPARSE_DECODE_RESERVE (1024) > top_k`; decode with graph prefill is back to 15.85-15.87.
Also: read-ahead is skipped for prompts under 64 tokens (24-token prefill 9.3 s -> 2.8 s) and never reads experts a VRAM
branch serves (`OffloadSite::not_cold`).

Dual-GPU recipe prefill (805 tokens, 76 GiB pool, 3090 + 5060 Ti): 67-69 tok/s (single 3090: 98-101). OFFLOAD_PINNED 0 / 0.5 / 1.0 =
56.6 / 67 / 69.1 tok/s. Nothing is read ahead there (all cold experts are seeded in the pool); the time is the PCIe copy of the
cold experts, which pinned memory speeds up.

### Why the dual-GPU prefill was slow (67 vs 118 tok/s) and the fix

Same config, same 76 GiB pool, 805 tokens: 3090 only 117.6 tok/s, dual 67.1 tok/s; the whole difference was the miss node
(`MAP_CUSTOM1` 4.7 s). A batched graph has no GPU1 branch, so the 5060 Ti's hot experts (2426, not seeded in the DDR pool) go
through the cold branch and were read inline, ~58 per layer. The read-ahead skip list wrongly treated them as VRAM-served.
Fixed (`not_cold` = 3090 hot set only for batched graphs): dual **96.9 tok/s**, 3090 only **118.9**, greedy tokens identical
between the two. The remaining 1.5 s is the PCIe copy of the extra ~15 GiB of experts the 5060 Ti would serve in decode.
llama.cpp `-ncmoe 41 -ub 4096` on the same prompt: 104 tok/s.

### Prefill: two-link split and speculative expert prefetch (2026-10-15, dual GPU, 805 tokens, tokens identical)

Both are opt-in and **off by default** because neither beats the plain graph prefill (159 tok/s, 5.05 s).

| config | prefill | note |
|---|---|---|
| plain graph prefill (recipe) | 159.3 tok/s (5.05 s) | reference |
| `PREFILL_SPLIT=0.63` (37% of cold experts computed on the 5060 Ti) | 157.8 | both PCIe links copy concurrently (verified with nsys), wall time unchanged |
| `PREFILL_PREFETCH=1`, 2 sets, pool 15.0 GiB | 155.8 | copies are complete before the layer needs them (wait 0 ms) |
| `PREFILL_PREFETCH=1`, 3 sets, chunk 8, pool 14.0 GiB | **164.7 (4.89 s)** | +3.5%, costs 1.8 GiB of the 3090 expert pool |

Findings: the miss node costs 0.2 ms; the layer period of 128-140 ms (layers 3-30; 63-70 ms for layers >= 31) is not copy-bound
once the copies are prefetched, and the GPU kernels add up to ~26 ms per layer. The host thread sits in `cudaStreamSynchronize`
for 60-75 ms per layer, behind the activations that cross between the GPUs (no P2P: staged through the host, queued behind the
weight copies on the single H2D engine; queueing the copies a few experts at a time recovered only part of it). The missing
~60-100 ms per layer is not attributed yet (nsys under 3 sets runs out of memory on the 3090).
Code: `src/ggml_glm5next_graph_offload.cpp` (`prefetch_*`), cold-branch in `ggml_glm5next_graph_build.cpp`, scheduler events
in `patches/ggml/0008-*`. Env: `MOESTREAM_GLM5NEXT_PREFILL_PREFETCH{,_SETS,_CHUNK}`, `PREFILL_SPLIT`, `PREFETCH_DEBUG`.
Pool resizing after prefill was not built: with `moe_cache_gb = 17.0` the decode graph itself no longer fits (0.23 tok/s).

### Prefill: the copy queue starved the compute graph (2026-10-15): 159 -> 177 tok/s, now the default

Host tracing (`MOESTREAM_SCHED_TRACE2`) showed the unsplit prefill blocked 3.6 s of 5.05 s issuing weight copies (one link at
11.8 GB/s), and with the split + pinned pool the wait moved to the miss node's input hop (107 ms per layer) with the same wall
time. nsys of the prefetch config then showed both links idle 50% of the time: one layer's copies (~72 ms, 1.55 GiB at 21 GB/s
over two links) are followed by ~65 ms of compute, and the compute only starts when the copies end. Microbenchmark
(`scratchpad/lat.cu`): a small H2D (the rewritten ids, KDA history, the activations that cross between the GPUs) submitted on
another stream while a run of weight copies is in flight is not served until the run ends, **even when only two chunks are
queued at a time**; waiting for each chunk to finish before queueing the next empties the queue at chunk boundaries and the
small copy then waits 0.1-0.75 ms, with the run's duration unchanged (67.5 ms).

Fix: the prefetch worker waits for each chunk (`MOESTREAM_GLM5NEXT_PREFILL_PREFETCH_DRAIN`, default 1) and queues in chunks of
2 experts (`_CHUNK`, default 2), alternating between the two GPUs (`PREFILL_INTERLEAVE`, default 1). The whole DDR pool must be
pinned (`OFFLOAD_PINNED=1.0`: a pageable source makes the copy synchronous, 4.82 s vs 4.55 s).

| config (805 tokens, dual GPU, tokens identical) | prefill |
|---|---|
| plain graph prefill | 159.3 tok/s (5.05 s) |
| prefetch 3 sets, chunk 8, pool 14.0 (previous best) | 164.7 (4.89 s) |
| + drain, chunk 2, 3 sets, pool 14.0 | 182.5 (4.41 s) |
| drain, chunk 2, 2 sets, pool 15.0, pinned 1.0, one worker thread for both GPUs | 176.9 (4.55 s) |
| same, **one worker thread per GPU (shipped)** | **178.8 (4.50 s)** |

The prefetch ring is now on by default when the second GPU takes part (off for a single GPU, where it does not fit next to the
3090 pool); the shipped config uses `moe_cache_gb = 15.0` (decode within 1%: 70.1 -> 70.7 ms/token over tokens 16-32). If the
graph does not fit with the ring it is built once more without it (previously: legacy prefill, 25 min). Remaining distance to
the link limit (2.6 s for ~43 GB over two links): the first layers (3 only gets its experts by fixup), the 14 layers >= 31, and
the compute chain itself. With a thread per GPU the copies of a layer take 86 ms (1.55 GiB at 18 GB/s, the two links' limit) but
the layer period of layers 4-30 is still 118 ms (65 ms for layers >= 31, whose copies take 24 ms): the miss node no longer waits
for copies, the host waits ~77 ms per layer for the GPU chain (attention, router, cold experts on both GPUs, activations crossing
the host). That chain, not the links, is what is left.

### Prefill: expert sum on the expert GPU, early first layer: 178.8 -> 251 tok/s (2026-10-15)

Per-split timing (`MOESTREAM_SCHED_TRACE2` + `MOESTREAM_SCHED_SERIAL`, env-gated instrumentation in the ggml fork) showed the
layer chain was dominated by two cross-GPU hops through the host: the activation to the 5060 Ti (13 MB, 25 ms) and the
**unreduced** `MUL_MAT_ID` output coming back (`[D, topk, S]` = 105 MB, 56 ms). The weighting by the route weights and the sum over
the routed experts were placed on the 3090 by the scheduler. Pinning those two nodes to the expert GPU
(`run_branch(..., pin_be)`, prefill graphs only; `MOESTREAM_GLM5NEXT_PIN_BRANCH_SUM=0` disables) sends the 13 MB sum instead:

| 805 tokens, dual GPU, tokens identical | prefill |
|---|---|
| copy thread per GPU, drain (previous) | 178.8 tok/s (4.50 s) |
| + weighting and sum on the expert GPU | 244.9 (3.29 s) |
| + first offload layer's copies start at graph prepare | **251.3 (3.20 s)** |

Pinning the same nodes in the decode graph costs ~1% decode (74.9 -> 75.8 ms/token over tokens 16-96), so decode is unchanged.
Layers 4-30 now take 85 ms each = the 1.55 GiB of cold experts of a layer over the two links (84 ms issue time, 19.8 GB/s):
at the PCIe limit. Layers >= 31 take 45 ms for a 24 ms copy; the rest is startup (~0.35 s). 805 tokens move ~43 GB, so the copy
floor is ~2.3 s; the measured 3.20 s is within 1.4x of it.

Prompts longer than the graph batch: see the next section.

### Prefill of long prompts: chunks continue each other through the graph (2026-10-15)

Before, a prompt longer than `MOESTREAM_GLM5NEXT_PREFILL_CHUNK` (1024) ran the first chunk through the graph and the rest as
S=1 steps (1604 tokens: 53.7 s, 30 tok/s; 1998 tokens: 86 s). Now every chunk is a graph (`pos_base > 0`): its KDA state and
conv history come from the engine buffers the previous chunk exported, the DSA layers get the earlier latent rows as a host input
(`glm5next_prev_latent_<layer>`, concatenated before the chunk's own), the mask is offset by `pos_base`, and the indexer seeds
of the chunks are concatenated (chunk boundaries are multiples of 8). The generation loop plans the chunks (last chunk >= 64
tokens), halves the batch if the graph declines for memory, and falls back to S=1 steps for what a chunk declines.

| prompt | before | now |
|---|---|---|
| 805 tokens | 251 tok/s | 251 |
| 1604 | 29.8 tok/s (53.7 s) | 268.9 (5.97 s) |
| 1998 | 23.2 (86 s) | 313.6 (6.37 s) |
| 2403 (> top_k 2048) | - | 59.9 (40 s; the last 355 tokens are stepped) / 198.7 with `PREFILL_DENSE_BEYOND_TOPK=1` (12.1 s) |

The default chunk is now 2048 (= the sparse top-k): every chunk streams all cold experts again, so a bigger chunk is faster
(1998 tokens as 1024 + 974: 11.8 s). Chunks that continue beyond top_k would need the indexer's per-query selection; the dense
attention used by the prefill graph is the model's attention only while the context fits top_k, so that is opt-in
(`MOESTREAM_GLM5NEXT_PREFILL_DENSE_BEYOND_TOPK=1`, not the model's exact semantics).

Correctness: the hidden state of a graph prefill differs from another batch shape by relL2 ~0.1 even inside the first chunk
(rows 100..500 of 512+293 vs 805: 0.08-0.19), so tokens of a chunked prompt can legitimately differ from the single-graph
run (1604 tokens: chunks 800/736/68 and 1200/404 reproduce the single-graph tokens, 1024/580 does not). Zeroing the carried
KDA state, the transposition or the earlier latent rows gives relL2 0.5-0.8, so the carry is exercised and right.
A chunk whose pooled indexer keys or seeds do not line up (pos_base not a multiple of the pool size) is declined.

### Prefill batch size, memory pre-checks, Qwen chunked prefill (2026-10-16)

**Config.** `prefill_batch_tokens` (toml, 0 = family default) is the prefill graph batch (llama.cpp's n_ubatch);
`MOESTREAM_GLM5NEXT_PREFILL_CHUNK` / `MOESTREAM_QWEN4EXP_PREFILL_CHUNK` override. GLM defaults to 2048 (512 prefills a
1065-token prompt at 90 tok/s against ~250: every batch restreams all cold experts), Qwen to the whole prompt.

**Memory pre-check.** A batch the device cannot hold is refused before any allocation, with the size that fits, and the
generation loop retries with it: GLM `1.0 GiB`-class estimate from measured free memory (device 0 ~0.85 MiB/token, ring
admitted only when ring + workspace fit), Qwen 7.8 MiB/token device + 1.1 MiB/token host + 6 GiB kept for the decode graph.
A 200000-token request on a 3343-token prompt ran as 1432-token batches (14 s). The prompt must also fit `context_size`
(`engine.max_t`): a longer prompt is refused with the value to raise (before, the KV caches would be written past).

**Qwen chunked prefill.** Batches after the first continue the previous one: recurrent state and conv history come from the
engine buffers (transposed to the scan's layout, which the first batch hid because it starts from zeros), the earlier K/V rows
come from the dense KV cache as a host input concatenated before the batch's own, positions and the causal mask are offset by
`pos_base`, and the PLE n-gram window continues from the previous batch. A 1611-token prompt in 512/800/1024 batches gives
the same tokens as one graph, last-row hidden within relL2 0.16-0.27 of it and of each other (the shape-to-shape noise level;
the unfixed transposition gave ~1.0). A 3989-token prompt (config with `context_size = 8192`) runs as 1448 + 1448 + 1093:
65.8 s (61 tok/s), decode works. `configs/qwen4exp_3090_only.toml` has `context_size = 2048`.

**Not done.** A non-graph prefill for the case where even the smallest graph batch cannot be built: with the graph-only
expert placement the layered MoE path finds no staged expert blocks (`no RAM block`, tokens dropped), so it is not enabled;
the memory-driven batching above covers the VRAM case. GLM's selection beyond top-k is checked only against dense
prefill (same tokens on a 2192-token prompt), not against the decode graph, whose own beyond-top-k path is dense in the
stepped case.

**Selection beyond top-k, consistency check (2026-10-16).** Prompt of 2192 tokens (graph batches 2048 + 144, so the last
144 queries and every decode step have more context than top_k). Run 1 generated 14 tokens with the decode graph. Run 2
replayed prompt + the first 6 generated tokens as a prefill (2198 tokens, batches 2048 + 150) and generated 8: the same
8 tokens as run 1's 7th-14th (also with `DENSE_BEYOND_TOPK=1`). The prefill's selection therefore agrees with the decode
graph's on this sequence; it does not separate sparse from dense this close to top_k, and the stepped-prefill reference is
itself dense beyond the seeded 2048 tokens, so it cannot be used as a trusted reference.

**Size sweep (2026-10-16, merged binary, every batch through the graph, no stepped or declined continuation):**

| model | prompt tokens | graph batches | prefill |
|---|---|---|---|
| GLM | 1152 | 1152 | 5.8 s (198 tok/s) |
| GLM | 1689 | 1689 | 6.4 s (264) |
| GLM | 2240 | 2048 + 192 | 11.0 s (204) |
| GLM | 2798 | 2048 + 750 | 13.3 s (210) |
| GLM | 3867 | 2048 + 1819 | 14.7 s (263) |
| Qwen | 1389 | 1389 | 22.4 s (62) |
| Qwen | 2033 | 1448 + 585 | 34.1 s (60) |
| Qwen | 2969 | 1448 + 1448 + 73 | 51.2 s (58) |
| Qwen | 3989 | 1448 + 1448 + 1093 | 65.6 s (61) |

Bounds that remain by design: a prompt longer than `context_size` (GLM 4096, Qwen 8192 in the shipped configs) is refused with
the value to raise, and a batch that cannot fit even at 64 tokens (VRAM taken by another process) is refused before
allocating; there is no non-graph prefill for Qwen under the graph-only expert placement.

GLM at `context_size = 8192` (shipped config raised from 4096): 5117 tokens ran as 2048 + 2048 + 1021, all in the graph, 22.0 s
(232 tok/s). Dense (`DENSE_BEYOND_TOPK=1`) and per-query selection continue the document with the same heading
and agree on 11 of the first 12 tokens.
