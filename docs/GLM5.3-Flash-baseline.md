# GLM-5.3-Flash baseline (single RTX 3090, AJ-IQ2_XXS)

Recorded 2026-09-30. Reference point for further GLM-5.3-Flash work.

**Headline:** 7.26 tok/s steady-state decode (137.7 ms/token) with 8 MoE layers resident in VRAM and the
remaining 34 MoE layers' experts on the CPU. neurall's llama.cpp on the identical file and placement:
7.37 tok/s (135.7 ms/token). Gap about 1.5%.

## 1. Hardware / software

| Item | Value |
|---|---|
| CPU | Intel Xeon W-2235 @ 3.80 GHz, 6 cores / 12 threads, AVX-512 (incl. VNNI) |
| RAM | 125 GB (`host_ram_gb = 106`) |
| GPU used | RTX 3090, 24 GB, device 0 (single GPU) |
| GPU not used | RTX 5060 Ti 16 GB (idle in this baseline; see section 8) |
| Storage | model on NVMe (`/home/ray/vllm-workspace`, nvme1n1p3) |
| Driver / CUDA | 610.57.04 / CUDA 13.3 (nvcc), OS Linux 7.2 |
| ZeroMoE | commit `2a06c0a` plus uncommitted working-tree changes (see section 9) |
| ggml (recommended) | `/home/ray/Projects/llama.cpp-moestream-b11401`, branch `zeromoe-ggml-b11401`, HEAD `9fe8a156f` (neurall's b11401 + ZeroMoE fork commit + negative-id-skip commit) |
| ggml (older, ~1% slower) | `/home/ray/Projects/llama.cpp-moestream-integration`, b11062 + fork commit |

## 2. Model file

`/home/ray/vllm-workspace/GLM-5.3-Flash/GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf` (+ `-00002-of-00002.gguf`)

- Shards: 44,665,098,240 B + 42,680,908,320 B, about 87.3 GB total. File type `IQ2_XXS` (2.0625 bpw).
- Architecture: neurall reports `glm5-next 320B.A18B`; 45 trunk layers (34 KDA + 11 DSA), dense FFN in layers 0-2,
  MoE in layers 3-44 (42 layers), 288 routed experts/layer, top-8, plus a shared expert; hyper-connections hc_mult=4.
- Routed experts: 34 layers gate/up/down all IQ2_XXS; 8 layers gate/up IQ2_XXS with down IQ2_S. Embedding Q6_K.
- Always pass the explicit first-shard path. Pointing `model_path` at a directory silently merges sibling `.gguf` files.
- Other GLM-5.3-Flash files on this box (not baseline): `UD-IQ1_S` (93 GB, /data/sdb/models/GLM-5.3-Flash),
  `UD-IQ3_XXS` (partial), `GSQ-RCO-3.0bit` (117 GB).

## 3. Placement (what runs where)

| On the 3090 | On the CPU |
|---|---|
| attention of all 45 layers, hyper-connections, router, shared experts, dense FFN (layers 0-2), LM head (`output_norm` + `output.weight`), routed experts of MoE layers 3-10 (all 288 experts each, ~15.5 GB) | routed-expert matmuls (`MUL_MAT_ID`, IQ2_XXS/IQ2_S kernels) of MoE layers 11-44 (34 layers) |

Per token the scheduler runs 34 CPU splits and 35 GPU splits. The GPU is idle about 85% of each token.

## 4. Build

```
cmake -S . -B build_b11401 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON \
  -DZEROMOE_ENABLE_CUDA_GRAPHS=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DZEROMOE_GGML_OPENMP=ON \
  -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16" \
  -DZEROMOE_ENABLE_GGML_HOST_NODE_PATCHES=OFF -DZEROMOE_ENABLE_AVX512_VNNI=OFF \
  -DZEROMOE_GGML_ROOT=/home/ray/Projects/llama.cpp-moestream-b11401/ggml
cmake --build build_b11401 --target zeromoe -j12
```

- `HOST_NODE_PATCHES` must be OFF (the host-node scheduler-gate patch does not apply to b11401).
- Do NOT point `ZEROMOE_GGML_ROOT` at another project's tree (neurall): CMake applies patches to it in place.
- Compile flags match neurall's: `-O3 -DNDEBUG -march=native`, Release, LTO off, OpenMP on. `GGML_LLAMAFILE` is OFF
  (tested ON: 0-1% slower, no benefit).
- The first run of a freshly built binary spends about 67 s in CUDA JIT in step 0; discard that run for timing.

## 5. Configuration

Config file: `configs/bench/glm53flash_AJ_IQ2XXS_wholelayer8_cpu.toml`. Root keys must stay before the first `[[gpus]]` table.

Keys that define this baseline:

| Key | Value | Why |
|---|---|---|
| `model_path` | first shard of AJ-IQ2_XXS | explicit path, see section 2 |
| `vram_cache_full_layer_prefix` | 46 | give layers `[0,46)` complete 288-expert VRAM regions first |
| `vram_preload_prefix_layers` | 46 | fill the VRAM cache in ascending layer order (whole layers) |
| `pinned_vram_capacity` | 2400 | must be >= total VRAM slots, else the prefix preload is capped (default 240 filled only layer 3) |
| `cpu_affinity_layers` | `[11 ... 44]` | experts of these layers stay on the CPU, no VRAM slots |
| `moe_cache_gb` (gpus[0]) | 15.8 | fits 8 whole layers (2304 slots, 15.55 GB) |
| `pin_strategy` | `global_topn` | usage-ranked preload for the RAM tier (unused for GPU layers here) |
| `freeze_usage_stats` | true | usage files are read, never updated |
| `pin_file` / `prefill_pin_file` | `.zeromoe_usage_glm53flash` / `.zeromoe_usage_prefill_glm53flash` | routing counts, last written 2026-09-08 |
| `host_ram_gb` | 106 | do not reduce; tied to a 2026-09-09 crash (also keep `mem_guard_stop_mb = 10240`) |
| `ram_preload_all_experts` | true | ~80 GB of experts resident in RAM |
| `mtp_enabled` | false | the reusable full graph is disabled when MTP is on |
| `context_size` | 4096 | |

Environment variables:

| Variable | Value | Effect |
|---|---|---|
| `ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB` | `512` | AJ needs it: dense arena 6.65 GiB leaves 0.87 GiB free, below the 1 GiB / 2 GiB defaults, so the graph is declined and decode silently falls back to legacy. Real graph workspace is 0.15 GiB |
| `ZEROMOE_GLM5NEXT_CPU_THREADS` | `6` | ggml CPU threads. 6 and 8 are equal; 10 is slower; default was 4 |
| `ZEROMOE_GLM5NEXT_UNFUSED_HC` | unset | fused hyper-connection ops are the default (graph 6,131 nodes vs 31,061) |
| `ZEROMOE_GLM5NEXT_CPU_LM_HEAD` | unset | set to 1 to put the LM head back on the CPU (about -48 ms/token) |
| `ZEROMOE_GLM5NEXT_GPU1_LAYERS` | unset | e.g. `11-17` uploads those layers' experts to the 5060 Ti (not part of this baseline) |
| `GGML_SCHED_PROF` | optional `1` | neurall-fork scheduler profiler; logs a per-graph CPU/GPU/copy breakdown |
| `ZEROMOE_GLM5NEXT_GRAPH_LAYER_TIMING` | optional `1` | per-layer attention/FFN timing every 100 tokens (adds sync overhead) |

## 6. How to run

The engine reads the prompt from stdin.

```
cd /home/ray/Projects/ZeroMoE
nvidia-smi; pgrep -af zeromoe     # GPU co-existence guard: nothing else on the GPU
export ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB=512 ZEROMOE_GLM5NEXT_CPU_THREADS=6
echo "Write an extremely long and detailed 4000-word essay about the history, mathematics and engineering of mixture of experts models. Do not stop early." \
  | ./build_b11401/zeromoe --config configs/bench/glm53flash_AJ_IQ2XXS_wholelayer8_cpu.toml --ngen 1500 \
  > logs/glm53flash_aj_run.log 2>&1
```

- Engine log: `/tmp/zeromoe_glm53flash_aj.log` (`log_path`). Copy it into `logs/` after the run.
- Startup takes about 112 s (open shards, preload experts to RAM and the 8 layers to VRAM, build the graph); decode of 700 tokens about 100 s.
- Healthy start: `graph built layers=45 nodes=6131 splits=69` and `[EXPERT-TIER-MAP]` showing `vram=288` for layers 3-10 and `vram=0` for 11+.
  If you see `graph construction declined; using legacy decode`, decode will be about 6x slower (check the allocator-floor variable).

## 7. Baseline performance (AJ-IQ2_XXS, single 3090, layers 3-10 in VRAM)

Steady-state (median of `[TOKEN-LOOP-DIAG] iter_total`, steps 50-700, quiet machine, 6 threads, ABBA-ordered):

| Build | ms/token | tok/s |
|---|---|---|
| **ggml b11401 (`build_b11401`, recommended)** | **137.1, 138.3 (avg 137.7)** | **7.23-7.30 (7.26)** |
| ggml b11062 (`build`) | 141.4, 141.9 | 7.05-7.07 |
| b11401 + GGML_LLAMAFILE=ON | 139.5, 138.9 | 7.17-7.20 |

1500-token run (wall includes startup graph-build step): 7.14 tok/s (b11062, 8 threads, 210 s), 6.97 tok/s (b11401, first-token
cost 2.9 s, `GGML_SCHED_PROF` on). Output read for coherence: 1,129 words of on-topic MoE history and math, no degeneration.

Reference (neurall llama.cpp b11401, same file, `CUDA_VISIBLE_DEVICES=0`, `-ngl 99`, experts of layers 11+ on CPU):
`llama-bench -n 1500 -r 2` = 7.30 +/- 0.03 tok/s; `llama-cli` steady 135.68 ms/token (7.37 tok/s). Thread count 6.

Per-token breakdown (ms), same profiler categories:

| Category | ZeroMoE | neurall |
|---|---|---|
| CPU compute (34 CPU splits, experts) | 118.4-118.6 (117.8 on b11062) | 115.6-116.6 |
| Wait + input copy before CPU splits (exposed GPU attention/routing) | 17.6-18.1 | 18.1 |
| Copy back before GPU splits | 1.7-2.0 | 1.35 |
| GPU launch | 0.4-1.3 | 0.5-1.0 |
| Scheduler total | 138.3 | 135.2-136.5 |
| Host work outside the graph (sample, logits readback) | ~1.3 | ~0 |

Per layer (fused HC): attention 0.27 ms x 45; FFN 0.26 ms on a GPU-resident layer, about 3.6 ms on a CPU-expert layer.
CPU experts move 54 MB/layer in about 3.46 ms (about 15.6 GB/s; single-core IQ2_XXS dot microbenchmark 3.05 GB/s/core).

Other file for reference, `GLM-5.3-Flash-UD-IQ1_S` (93 GB, `configs/testing/glm53flash_IQ1_wholelayer8_cpu.toml`, allocator floor 1024):
ZeroMoE 6.87 tok/s over 1,500 tokens vs neurall 6.99 +/- 0.03. With 7 more layers on the 5060 Ti: 6.85 tok/s at 1,500 tokens (the 5060 Ti helps because more
layers leave the CPU; single-GPU comparisons were requested for the record).

History on this file/box: 2.70 tok/s (first working full-graph decode) -> 4.7 (LM head into graph, CPU threads 4 -> 6) ->
5.4-5.8 (whole-layer VRAM placement fixed) -> 6.87 (fused hyper-connection ops).

## 8. Gotchas when measuring

- A fresh binary spends about 67 s in CUDA JIT in step 0 (sm_75 PTX on an sm_86 GPU). Warm it once, never time run 1.
- Other sessions' CPU jobs (e.g. a Python entropy audit) inflate CPU compute by about 15% (117.8 -> 135 ms). Check `ps`/load before timing; use ABBA order.
- The engine is not bit-deterministic across equivalent op orders; fused vs composed hyper-connection ops differ in hidden state by
  visible amounts (near-tied expert flips) although each op matches its reference to 1e-7. Judge by long-run coherence.
- Never `GGML_CUDA_DISABLE_GRAPHS` A/B without checking; CUDA graphs already replay about 98% of splits.

## 9. Code and config changes behind this baseline (uncommitted in the ZeroMoE repo)

- `src/ggml_glm5next_graph.cpp`: LM head (`output_norm` + `output.weight`) inside the decode graph; fused hyper-connections (comb/pre/post) default on;
  CPU thread setting; allocator-floor override; optional 5060 Ti expert path (`GPU1_LAYERS`); layer-timing instrumentation; VRAM-reference expert
  stride/offset fixes.
- `include/gpu_topology.hpp`: `compact_layer_slot_counts` now skips dense layers (slab size 0) instead of `break`, so `vram_cache_full_layer_prefix` works.
- Configs: `configs/bench/glm53flash_AJ_IQ2XXS_wholelayer8_cpu.toml` (baseline), `configs/testing/glm53flash_IQ1_wholelayer8_cpu.toml` and friends.
- Not done: the reusable-graph 12 tok/s target and the Qwen conditional CUDA node (neurall itself reaches about 7.4 tok/s on this file and hardware);
  the Qwen4Exp smoke test on the rebased ggml.

## 9b. Partial layers (top 50% of a layer's experts in VRAM) - added 2026-10-01

Same VRAM budget as the 8-whole-layer baseline (15.8 GB = 2,341 slots), spent as 7 whole layers (3-9) plus two half layers (18 and 19, about 162 experts each,
the most concentrated layers in the fresh routing profile).

- Config: `configs/testing/glm53flash_AJ_IQ2XXS_7full_2half.toml` (`vram_cache_full_layer_prefix = 10` so only layers 0-9 get whole regions, `cpu_affinity_layers` = 10-17 and 20-44,
  `pin_file = .zeromoe_usage_glm53flash_aj_tap`).
- Result (quiet-machine ABBA, median ms/token): 8 whole layers 139.4 / 140.4 (7.15 tok/s) vs 7 whole + 2 half **136.75 / 136.79 (7.31 tok/s), 2.2% faster**.
- Requires the `-1` sentinel fix in `partial_map` (`src/ggml_glm5next_graph.cpp`): routes a branch does not own used to map to expert 0 (computed then masked), so the CPU cold
  branch always did full work and the same layout measured 1.0% SLOWER. Output is bit-identical with the fix. `ZEROMOE_GLM5NEXT_PARTIAL_ZERO_SENTINEL=1` restores the old behavior.
- Hit rates on unseen prompts: about 76% (essay) to 83-89% (code review) per route. Usage tap tooling: `ZEROMOE_GLM5NEXT_ROUTE_TAP=<file>` with `ZEROMOE_GLM5NEXT_CACHE_STATS=1`
  writes per-layer expert counts (usage-file format) and logs `[GLM5NEXT-PARTIAL-HIT]`. The old `.zeromoe_usage_glm53flash` was far too optimistic (predicted 99% for layer 18, real ~60%).
- CPU model (standalone, no model load): a `MUL_MAT_ID` node costs about 0.16 ms fixed (if any id is valid) plus 0.122 ms per valid id on 6 threads with cold experts.

## 10. Where the time goes and the remaining levers

CPU expert compute is about 85% of a token. Ranked levers: more layers in VRAM (about 3.4 ms per layer moved; the 5060 Ti holds 7 layers, roughly 8.5 tok/s),
faster CPU expert kernels (at most about 20%; a gather-based AVX-512 IQ3_XXS kernel was 2.5x slower and was reverted), memory-layout tuning of the
CPU-resident experts (untested), and trimming the roughly 1.3 ms of host work per token.
